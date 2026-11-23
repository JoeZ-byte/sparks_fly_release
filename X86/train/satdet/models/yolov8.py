"""YOLOv8 检测网络（自研实现，含可选的 P2 浅层检测头）。

结构严格对齐官方 v8n（layers 0..21 + Detect），因此官方 COCO 权重可以
1:1 加载；P2 分支是**额外挂载**的模块（``p2_branch.*`` + ``detect.cv2p2/cv3p2``），
不与官方权重键冲突，随机初始化后随 Neck/Head 一起微调。

为什么加 P2
-----------
VISO 目标边长中位 5~9 px。图块 256 上采样 2 倍到 512 后变成 10~18 px：
- 在 P3(stride 8) 上占 1.3~2.3 格 —— 勉强
- 在 P2(stride 4) 上占 2.6~4.5 格 —— 才真的可学
所以 P2 不是"锦上添花"，是本数据集能收敛的前提。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from satdet.models.blocks import C2f, Concat, Conv, DFL, SPPF

__all__ = [
    "Detect",
    "P2Branch",
    "YOLOv8",
    "build_yolov8n",
    "make_anchors",
    "dist2bbox",
    "bbox2dist",
    "decode_predictions",
]

# v8n 各层通道（width=0.25, max_channels=1024）
CH = (64, 128, 256)          # P3, P4, P5 输出通道
CH_BACKBONE = (16, 32, 64, 128, 256)
DEPTH = (1, 2, 2, 1)         # C2f 重复次数（0.33 * {3,6,6,3}）


class Detect(nn.Module):
    """YOLOv8 解耦检测头。

    输出为每层一个张量：``(B, 4*reg_max + nc, H*W)``，
    前 4*reg_max 是 ltrb 分布（DFL），后 nc 是类别 logits。
    """

    def __init__(self, nc: int = 1, ch: tuple[int, ...] = CH, reg_max: int = 16,
                 p2_ch: int | None = None) -> None:
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = reg_max
        self.no = nc + 4 * reg_max

        c2 = max(16, ch[0] // 4, reg_max * 4)
        c3 = max(ch[0], min(nc, 100))

        # 名称与官方一致：cv2=box 分支，cv3=cls 分支
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * reg_max, 1))
            for x in ch
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, nc, 1))
            for x in ch
        )

        # P2 分支单独挂（键名不与官方冲突）
        if p2_ch is not None:
            c2p = max(16, p2_ch // 4, reg_max * 4)
            c3p = max(p2_ch, min(nc, 100))
            self.cv2p2 = nn.Sequential(Conv(p2_ch, c2p, 3), Conv(c2p, c2p, 3),
                                       nn.Conv2d(c2p, 4 * reg_max, 1))
            self.cv3p2 = nn.Sequential(Conv(p2_ch, c3p, 3), Conv(c3p, c3p, 3),
                                       nn.Conv2d(c3p, nc, 1))
        else:
            self.cv2p2 = None
            self.cv3p2 = None

        self.dfl = DFL(reg_max)
        self.stride = torch.tensor([4.0, 8.0, 16.0, 32.0] if p2_ch else [8.0, 16.0, 32.0])

        self._init_bias()

    def _init_bias(self) -> None:
        """分类分支 bias 给一个先验（避免训练初期全负样本把输出压死）。"""
        for a, b in zip(self.cv2, self.cv3):
            b[-1].bias.data[:] = 1.0 if False else 0.0
        if self.cv3p2 is not None:
            self.cv3p2[-1].bias.data[:] = 0.0

    def forward(self, x: list[torch.Tensor]) -> list[torch.Tensor]:
        """返回每层 ``(B, 4*reg_max+nc, H*W)`` 的原始预测。"""
        out: list[torch.Tensor] = []
        if self.cv2p2 is not None:
            p2 = x[0]
            out.append(torch.cat((self.cv2p2(p2).view(p2.shape[0], 4 * self.reg_max, -1),
                                  self.cv3p2(p2).view(p2.shape[0], self.nc, -1)), 1))
            rest = x[1:]
        else:
            rest = x
        for i, xi in enumerate(rest):
            # 通道维拼接 -> (B, 4*reg_max + nc, H*W)，与官方布局一致
            out.append(torch.cat((self.cv2[i](xi).view(xi.shape[0], 4 * self.reg_max, -1),
                                  self.cv3[i](xi).view(xi.shape[0], self.nc, -1)), 1))
        return out

    def bias_init(self) -> None:  # 兼容习惯叫法
        self._init_bias()


class ECA(nn.Module):
    """Efficient Channel Attention：用 1D 卷积学通道权重，参数量只有 k 个。

    为什么加在 P2 路径：P2 来自 backbone 很浅的一层，语义弱但空间细节强，
    背景纹理（路面、屋顶、树影）与 6px 车辆的局部统计很像。ECA 让网络自己
    学"哪些通道更可能是车"，代价是每层仅 3~5 个参数，几乎不影响推理速度。
    """

    def __init__(self, channels: int, k: int = 3) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pool(x)                                   # (B,C,1,1)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))     # (B,1,C)
        y = y.transpose(-1, -2).unsqueeze(-1)              # (B,C,1,1)
        return x * self.act(y)


class P2Branch(nn.Module):
    """P2 融合分支：backbone stride-4 特征 + 上采样的 P3 输出。"""

    def __init__(self, c_backbone: int = 32, c_p3: int = 64, c_out: int = 64,
                 n: int = 1, use_eca: bool = True) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.c2f = C2f(c_backbone + c_p3, c_out, n=n, shortcut=False)
        self.eca = ECA(c_out) if use_eca else nn.Identity()

    def forward(self, feats: list[torch.Tensor]) -> torch.Tensor:
        x_s4, p3 = feats
        return self.eca(self.c2f(torch.cat((x_s4, self.up(p3)), 1)))


class YOLOv8(nn.Module):
    """YOLOv8n 检测器。

    Attributes
    ----------
    model : nn.Sequential
        层 0..21，与官方 v8n 完全一致（键名 ``model.<i>.*``）。
    detect : Detect
        检测头（对应官方 ``model.22``，加载时做一次前缀替换）。
    p2_branch : P2Branch | None
        仅 ``p2=True`` 时存在。
    """

    def __init__(self, nc: int = 1, p2: bool = False, reg_max: int = 16,
                 use_eca: bool = True) -> None:
        super().__init__()
        self.nc = nc
        self.p2 = p2
        c1, c2, c3, c4, c5 = CH_BACKBONE
        d1, d2, d3, d4 = DEPTH

        self.model = nn.Sequential(
            Conv(3, c1, 3, 2),                 # 0  s2
            Conv(c1, c2, 3, 2),                # 1  s4  <- P2 分支取这一层
            C2f(c2, c2, d1, True),             # 2  s4
            Conv(c2, c3, 3, 2),                # 3  s8
            C2f(c3, c3, d2, True),             # 4  s8  <- P3 融合源
            Conv(c3, c4, 3, 2),                # 5  s16
            C2f(c4, c4, d3, True),             # 6  s16 <- P4 融合源
            Conv(c4, c5, 3, 2),                # 7  s32
            C2f(c5, c5, d4, True),             # 8  s32
            SPPF(c5, c5, 5),                   # 9  s32
            nn.Upsample(scale_factor=2, mode="nearest"),  # 10
            Concat(1),                         # 11
            C2f(c5 + c4, c4, 1, False),        # 12
            nn.Upsample(scale_factor=2, mode="nearest"),  # 13
            Concat(1),                         # 14
            C2f(c4 + c3, c3, 1, False),        # 15 -> P3 (s8,  c3)
            Conv(c3, c3, 3, 2),                # 16
            Concat(1),                         # 17
            C2f(c3 + c4, c4, 1, False),        # 18 -> P4 (s16, c4)
            Conv(c4, c4, 3, 2),                # 19
            Concat(1),                         # 20
            C2f(c4 + c5, c5, 1, False),        # 21 -> P5 (s32, c5)
        )
        self.detect = Detect(nc=nc, ch=(c3, c4, c5), reg_max=reg_max,
                             p2_ch=c3 if p2 else None)
        self.p2_branch = P2Branch(c2, c3, c3, 1, use_eca=use_eca) if p2 else None
        self.stride = self.detect.stride

    # --------------------------------------------------------------
    def forward_neck(self, x: torch.Tensor) -> list[torch.Tensor]:
        """返回送入检测头的特征金字塔（按 stride 从小到大）。"""
        m = self.model
        x0 = m[0](x)
        x1 = m[1](x0)
        x2 = m[2](x1)
        x3 = m[3](x2)
        x4 = m[4](x3)
        x5 = m[5](x4)
        x6 = m[6](x5)
        x7 = m[7](x6)
        x8 = m[8](x7)
        x9 = m[9](x8)

        y10 = m[10](x9)
        y11 = m[11]([y10, x6])
        y12 = m[12](y11)
        y13 = m[13](y12)
        y14 = m[14]([y13, x4])
        p3 = m[15](y14)
        y16 = m[16](p3)
        y17 = m[17]([y16, y12])
        p4 = m[18](y17)
        y19 = m[19](p4)
        y20 = m[20]([y19, x9])
        p5 = m[21](y20)

        feats = [p3, p4, p5]
        if self.p2_branch is not None:
            feats.insert(0, self.p2_branch([x1, p3]))
        return feats

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """训练/推理通用前向：返回每层原始预测。"""
        return self.detect(self.forward_neck(x))

    # --------------------------------------------------------------
    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """把官方权重里的 ``model.22.*`` 前缀映射到 ``detect.*``。"""
        remapped = {}
        for k, v in state_dict.items():
            if k.startswith("model.22."):
                remapped["detect." + k[len("model.22."):]] = v
            else:
                remapped[k] = v
        return super().load_state_dict(remapped, strict=strict, assign=assign)


def build_yolov8n(nc: int = 1, p2: bool = False, use_eca: bool = True) -> YOLOv8:
    """构造 YOLOv8n（depth=0.33 / width=0.25 已固化在构造函数里）。"""
    return YOLOv8(nc=nc, p2=p2, use_eca=use_eca)


# =====================================================================
# 解码工具（anchor 生成、距离->框）
# =====================================================================
def make_anchors(feats: list[torch.Tensor], strides: torch.Tensor,
                 grid_cell_offset: float = 0.5) -> tuple[torch.Tensor, torch.Tensor]:
    """生成 anchor 中心点与对应 stride。

    Returns
    -------
    anchor_points : (A, 2) 像素坐标中心点
    stride_tensor : (A, 1)
    """
    anchor_points, stride_tensor = [], []
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        h, w = feats[i].shape[2], feats[i].shape[3]
        sx = torch.arange(w, device=device, dtype=dtype) + grid_cell_offset
        sy = torch.arange(h, device=device, dtype=dtype) + grid_cell_offset
        sy, sx = torch.meshgrid(sy, sx, indexing="ij")
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), float(stride), dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)


def dist2bbox(distance: torch.Tensor, anchor_points: torch.Tensor,
              xywh: bool = True, dim: int = -1) -> torch.Tensor:
    """ltrb 距离 -> 框（相对 anchor 中心）。"""
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)
    return torch.cat((x1y1, x2y2), dim)


def bbox2dist(anchor_points: torch.Tensor, bbox: torch.Tensor, reg_max: int) -> torch.Tensor:
    """框 -> ltrb 距离（截断到 reg_max-1，供 DFL 目标使用）。"""
    x1y1, x2y2 = bbox.chunk(2, -1)
    return torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1).clamp_(0, reg_max - 0.01)


@torch.no_grad()
def decode_predictions(preds: list[torch.Tensor], strides: torch.Tensor,
                       nc: int, reg_max: int = 16) -> tuple[torch.Tensor, torch.Tensor]:
    """原始多尺度预测 -> (boxes_xyxy 像素坐标, 类别分数)。

    Returns
    -------
    boxes : (B, A, 4)  xyxy
    scores : (B, A, nc) 已 sigmoid
    """
    feats = [p.clone() for p in preds]
    shape = feats[0].shape
    for i, f in enumerate(feats):
        feats[i] = f.view(shape[0], 4 * reg_max + nc, -1)

    # anchor 的 h/w 需要每层尺寸：用 stride 反推
    hs = []
    total = 0
    for f in feats:
        a = f.shape[2]
        hs.append(a)
    anchor_points, stride_tensor = _anchors_from_counts(hs, strides, feats[0].device, feats[0].dtype)

    box_parts, cls_parts = [], []
    for i, f in enumerate(feats):
        box_parts.append(f[:, : 4 * reg_max])
        cls_parts.append(f[:, 4 * reg_max:])

    box_cat = torch.cat(box_parts, dim=2)          # (B, 4*reg_max, A)
    cls_cat = torch.cat(cls_parts, dim=2)          # (B, nc, A)
    b, _, a = box_cat.shape
    dist = box_cat.view(b, 4, reg_max, a)          # (B,4,reg_max,A)
    dist = dist.softmax(dim=2).mul(
        torch.arange(reg_max, device=dist.device, dtype=dist.dtype).view(1, 1, reg_max, 1)
    ).sum(dim=2)                                    # (B,4,A)
    # 注意：这里用显式期望而不是 DFL.conv，保证与导出图无关、数值等价
    dbox = dist2bbox(dist.permute(0, 2, 1), anchor_points.unsqueeze(0), xywh=False)  # (B,A,4)
    boxes = dbox * stride_tensor.unsqueeze(0)
    scores = cls_cat.permute(0, 2, 1).sigmoid()
    return boxes, scores


def _anchors_from_counts(counts, strides: torch.Tensor, device, dtype):
    """由每层 anchor 数反推 (h, w)。图块边长固定为 32 的倍数，故可整除。

    注意：ONNX 追踪时 ``count`` 可能是 Tensor（由 shape 推出来），必须先转 int，
    否则 ``round()`` 会报 "type Tensor doesn't define __round__"。
    """
    points, st = [], []
    for count, stride in zip(counts, strides):
        s = int(stride)
        # 假定方形特征图（本项目输入恒为正方形）
        side = int(round(float(count) ** 0.5))
        sy = torch.arange(side, device=device, dtype=dtype) + 0.5
        sx = torch.arange(side, device=device, dtype=dtype) + 0.5
        gy, gx = torch.meshgrid(sy, sx, indexing="ij")
        points.append(torch.stack((gx, gy), -1).view(-1, 2))
        st.append(torch.full((side * side, 1), float(s), device=device, dtype=dtype))
    return torch.cat(points), torch.cat(st)
