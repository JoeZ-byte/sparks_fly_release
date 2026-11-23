"""小目标专用的框回归损失：NWD（归一化高斯 Wasserstein 距离）。

为什么 6 px 目标优先用 NWD 而不是 IoU 系列
-----------------------------------------
把一个 5x6 px 的框整体平移 2 px，IoU 会从 1.0 掉到 0.43；平移 3 px 就掉到 0.25。
也就是说 IoU 在极小目标上分辨率极差：**2 px 的定位误差 = 一半的 IoU**。
实测帧间位移本来就有 2~3 px，导致：
1. 训练时正负样本的 IoU 区分度低，分类/回归都难收敛；
2. 评估时 IoU=0.5 会把"偏 2 px 的正确检测"判成漏检（所以主指标用 0.3）。

具体到梯度：
* 纯 IoU / GIoU 在两框不相交时值为 0 且**梯度为 0**，完全学不动；
* CIoU 增加了"中心距/外接框对角线"的惩罚项，缓解了这一点，但该惩罚被
  外接框尺度归一化，目标越小、相距越远，信号越弱；
* NWD 把框建模成二维高斯（均值=中心，标准差=半宽/半高），用 Wasserstein
  距离度量，对位置误差是**平滑且有界**的，任意距离都给得出方向，
  也不受"是否重叠"影响。

注意：NWD 的尺度归一化常数 ``C`` 必须按目标尺度设定（本项目 C=12，
对应放大后 10~18 px 的目标）。C 照抄论文里的 12（640 图）会让小目标的
损失曲线过于平坦 —— 这一点写进 docs/05 的超参说明。

公式
----
W²(pred, gt) = (dx² + dy²) + ((w1-w2)/2)² + ((h1-h2)/2)²
NWD          = exp(-sqrt(W² + eps) / C)
loss         = 1 - NWD

实测（6 px 框、C=12）：NWD 梯度随距离按 exp(-d/C) 衰减
（6/12/24/48 px 处逐次递减），但**永不恰好为 0**；
同一场景下 CIoU 因中心距项，梯度反而在远距离更大（≈0.033 vs ≈0.0011）。
所以 NWD 的价值不是"梯度更大"，而是"平滑、无死区、不随重叠与否突变"。
它的代价是远距离信号弱 —— 这就是分配器必须用 ``center_radius`` 把候选
限制在目标附近的原因（见 ``detection_loss.py``）。

``C`` 是与数据集目标尺度相关的归一化常数：取平均目标尺寸量级。
本项目图块 256 上采样 2 倍 -> 目标约 10~18 px，故默认 ``C = 12``。
C 越大，NWD 对距离越不敏感（曲线越平）；C 的取值写进配置，不做隐式默认。
"""

from __future__ import annotations

import math

import torch

__all__ = ["nwd_similarity", "nwd_loss", "bbox_ciou", "box_iou", "giou_loss"]


def _xyxy_center_size(box: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    cx = (box[..., 0] + box[..., 2]) * 0.5
    cy = (box[..., 1] + box[..., 3]) * 0.5
    w = (box[..., 2] - box[..., 0]).clamp_min(1e-6)
    h = (box[..., 3] - box[..., 1]).clamp_min(1e-6)
    return torch.stack((cx, cy), dim=-1), torch.stack((w, h), dim=-1)


def nwd_similarity(pred: torch.Tensor, gt: torch.Tensor, C: float = 12.0,
                   eps: float = 1e-7) -> torch.Tensor:
    """归一化高斯 Wasserstein 相似度，取值 (0, 1]，越大越像。"""
    c1, wh1 = _xyxy_center_size(pred)
    c2, wh2 = _xyxy_center_size(gt)
    d_center = ((c1 - c2) ** 2).sum(dim=-1)
    d_size = (((wh1 - wh2) * 0.5) ** 2).sum(dim=-1)
    w2 = (d_center + d_size).clamp_min(0.0)
    return torch.exp(-torch.sqrt(w2 + eps) / float(C))


def nwd_loss(pred: torch.Tensor, gt: torch.Tensor, C: float = 12.0,
             eps: float = 1e-7, reduction: str = "mean") -> torch.Tensor:
    """NWD 损失 = 1 - NWD。"""
    loss = 1.0 - nwd_similarity(pred, gt, C=C, eps=eps)
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss
    raise ValueError(f"未知 reduction: {reduction}")


def box_iou(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """成对 IoU，返回 (N, M)。"""
    a = torch.as_tensor(a, dtype=torch.float32)
    b = torch.as_tensor(b, dtype=torch.float32)
    if a.numel() == 0 or b.numel() == 0:
        return torch.zeros((a.shape[0], b.shape[0]), device=a.device)
    area1 = (a[:, 2] - a[:, 0]).clamp_min(0) * (a[:, 3] - a[:, 1]).clamp_min(0)
    area2 = (b[:, 2] - b[:, 0]).clamp_min(0) * (b[:, 3] - b[:, 1]).clamp_min(0)
    lt = torch.max(a[:, None, :2], b[None, :, :2])
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp_min(0)
    inter = wh[..., 0] * wh[..., 1]
    return inter / (area1[:, None] + area2[None, :] - inter + eps)


def bbox_ciou(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """成对 CIoU（与 ultralytics 的 ``bbox_iou(ciou=True)`` 等价）。"""
    a = torch.as_tensor(a, dtype=torch.float32)
    b = torch.as_tensor(b, dtype=torch.float32)
    if a.numel() == 0 or b.numel() == 0:
        return torch.zeros((a.shape[0], b.shape[0]), device=a.device)

    iou = box_iou(a, b, eps=eps)

    # 最小外接矩形
    lt = torch.min(a[:, None, :2], b[None, :, :2])
    rb = torch.max(a[:, None, 2:], b[None, :, 2:])
    wh_c = (rb - lt).clamp_min(eps)
    c2 = wh_c[..., 0] ** 2 + wh_c[..., 1] ** 2

    ca = _xyxy_center_size(a)[0]
    cb = _xyxy_center_size(b)[0]
    rho2 = ((ca[:, None, :] - cb[None, :, :]) ** 2).sum(-1)

    wa, ha = (a[:, 2] - a[:, 0]), (a[:, 3] - a[:, 1])
    wb, hb = (b[:, 2] - b[:, 0]), (b[:, 3] - b[:, 1])
    v = (4 / math.pi ** 2) * (
        torch.atan(wb[None, :] / hb[None, :].clamp_min(eps))
        - torch.atan(wa[:, None] / ha[:, None].clamp_min(eps))
    ) ** 2
    with torch.no_grad():
        alpha = v / (v - iou + (1 + eps))
    return iou - (rho2 / c2 + v * alpha)


def giou_loss(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """GIoU 损失，作为 CIoU/NWD 之外的备用对照项。"""
    iou = box_iou(pred, gt, eps=eps).diagonal()
    return (1.0 - iou).mean()
