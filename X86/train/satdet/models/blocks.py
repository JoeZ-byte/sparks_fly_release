"""YOLOv8 基础算子。

命名规则**刻意与 ultralytics 官方实现保持一致**（Conv 用 ``conv/bn/act``，
C2f 用 ``cv1/cv2/m``），这样官方 COCO 权重的 state_dict 可以 1:1 灌进来，
不需要做任何键名映射（358 个参数张量全部对上）。见
``scripts/01_convert_weights.py`` 与 ``satdet/models/weight_loader.py``。

为什么自研而不是直接用 ultralytics
----------------------------------
1. 运行时依赖越少越好：ultralytics 安装会顺带升级 torch/torchvision，
   把 cu128 的 CUDA 支持搞坏（已踩过）。
2. 性能优化需要完全掌控算子与内存布局（后续 INT8 量化、算子融合、
   ONNX 导出都要改图），改第三方库比自己写更难维护。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

__all__ = ["autopad", "Conv", "Bottleneck", "C2f", "SPPF", "Concat", "DFL"]


def autopad(k: int, p: int | None = None, d: int = 1) -> int:
    """same padding：保证 stride=1 时特征图尺寸不变。"""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    return k // 2 if p is None else p


class Conv(nn.Module):
    """Conv2d + BN + SiLU，YOLOv8 的基本积木。

    注意：``bias=False`` —— BN 后面接 bias 是浪费，而且融合时会被吸收。
    """

    default_act: nn.Module = nn.SiLU()

    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, p: int | None = None,
                 g: int = 1, d: int = 1, act: bool | nn.Module = True) -> None:
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        if act is True:
            self.act: nn.Module = self.default_act
        elif isinstance(act, nn.Module):
            self.act = act
        else:
            self.act = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x: torch.Tensor) -> torch.Tensor:
        """推理期使用：Conv+BN 已融合成单层卷积。"""
        return self.act(self.conv(x))


class Bottleneck(nn.Module):
    """标准残差块；``shortcut`` 只在输入输出通道一致时才真的加。"""

    def __init__(self, c1: int, c2: int, shortcut: bool = True, g: int = 1,
                 k: tuple[int, int] = (3, 3), e: float = 0.5) -> None:
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    """CSP 结构的跨阶段特征融合（v8 的主干/颈部单元）。"""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False,
                 g: int = 1, e: float = 0.5) -> None:
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, 2 * c_, 1, 1)
        self.cv2 = Conv((2 + n) * c_, c2, 1)
        self.m = nn.ModuleList(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class SPPF(nn.Module):
    """空间金字塔池化（3 个 5x5 串行最大池化等价于大感受野）。"""

    def __init__(self, c1: int, c2: int, k: int = 5) -> None:
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class Concat(nn.Module):
    """按通道拼接（占位模块，无参数，但保持 Sequential 的下标对齐）。"""

    def __init__(self, dimension: int = 1) -> None:
        super().__init__()
        self.d = dimension

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        return torch.cat(x, self.d)


class DFL(nn.Module):
    """Distribution Focal Loss 的积分层：把 4×reg_max 个分布期望成 4 个距离。

    实现成 ``Conv2d(c1, 1, 1)`` 而不是矩阵乘，是为了让 ONNX 导出后只剩
    一个 1x1 卷积（便于算子融合与 INT8 量化）。权重是固定的 [0,1,...,15]。
    """

    def __init__(self, c1: int = 16) -> None:
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, a = x.shape
        # (b, 4*reg_max, A) -> softmax over reg_max -> 期望值
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
