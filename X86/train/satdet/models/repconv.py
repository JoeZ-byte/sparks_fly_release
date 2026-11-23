"""MobileOne 式重参数化卷积（训练多分支 / 推理单分支）。

依据：MobileOne（Apple, CVPR2022）。核心思想是"训练时用多分支提升可训练性，
推理时通过代数等价把分支合并成一个普通卷积"，因此**推理零额外开销**。

实现要点
--------
* 训练分支：3x3 + 1x1 + identity（三路），每路都带 BN；
* 合并（fuse）：把 BN 吸收进卷积权重（`w' = w * gamma / sqrt(var+eps)`，
  bias 由 BN 的 beta/gamma/均值/方差算出），再把 1x1 用 zero-padding 补成 3x3、
  identity 用"单位核"表示，三者逐元素相加；
* 合并后结构与原 `Conv`（Conv2d+BN+SiLU）**完全等价**，所以可以直接替换。

用法::

    rep = RepConv(c1, c2, k=3, s=1)      # 训练时
    ...
    rep.fuse()                            # 推理前合并（原地）
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from satdet.models.blocks import Conv, autopad

__all__ = ["RepConv", "reparameterize_model", "convert_model_to_repconv"]


class RepConv(nn.Module):
    """训练多分支、推理单分支的 Conv+BN+SiLU。"""

    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1, p: int | None = None,
                 g: int = 1, act: bool | nn.Module = True) -> None:
        super().__init__()
        self.c1, self.c2, self.k, self.s = c1, c2, k, s
        self.p = autopad(k, p)
        self.g = g
        # 3x3 主分支
        self.conv3 = nn.Conv2d(c1, c2, k, s, self.p, groups=g, bias=False)
        self.bn3 = nn.BatchNorm2d(c2)
        # 1x1 辅助分支（只有 k==3 且 stride==1 时才有意义）
        self.has_1x1 = (k == 3 and s == 1 and g == 1)
        if self.has_1x1:
            self.conv1 = nn.Conv2d(c1, c2, 1, 1, 0, bias=False)
            self.bn1 = nn.BatchNorm2d(c2)
        # identity 分支（仅当通道数与步长匹配）
        self.has_id = (c1 == c2 and s == 1 and g == 1)
        if self.has_id:
            self.bn_id = nn.BatchNorm2d(c2)
        if act is True:
            self.act: nn.Module = nn.SiLU()
        elif isinstance(act, nn.Module):
            self.act = act
        else:
            self.act = nn.Identity()
        self.fused: Conv | None = None

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fused is not None:            # 已合并：走单分支
            return self.fused(x)
        y = self.bn3(self.conv3(x))
        if self.has_1x1:
            y = y + self.bn1(self.conv1(x))
        if self.has_id:
            y = y + self.bn_id(x)
        return self.act(y)

    # ------------------------------------------------------------------
    @staticmethod
    def _fuse_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> tuple[torch.Tensor, torch.Tensor]:
        """把 BN 吸收进卷积：返回 (weight, bias)。"""
        w = conv.weight.detach().clone()
        if conv.groups > 1:                   # depthwise：逐通道缩放
            w = w * (bn.weight / torch.sqrt(bn.running_var + bn.eps)).view(-1, 1, 1, 1)
        else:
            w = w * (bn.weight / torch.sqrt(bn.running_var + bn.eps)).view(-1, 1, 1, 1)
        bias = bn.bias - bn.running_mean * bn.weight / torch.sqrt(bn.running_var + bn.eps)
        return w, bias

    def fuse(self) -> Conv:
        """把多分支合并成一个普通 Conv（Conv2d+BN+SiLU），原地替换 forward。"""
        w3, b3 = self._fuse_bn(self.conv3, self.bn3)
        W = w3.clone()
        B = b3.clone()
        if self.has_1x1:
            w1, b1 = self._fuse_bn(self.conv1, self.bn1)
            # 1x1 核补零成 kxk（中心放原权重）
            Wk = torch.zeros_like(W)
            off = (self.k - 1) // 2
            Wk[:, :, off:off + 1, off:off + 1] += w1
            W = W + Wk
            B = B + b1
        if self.has_id:
            bn = self.bn_id
            scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
            b_id = bn.bias - bn.running_mean * scale
            # identity = 每通道的单位核（仅 depthwise 情形）
            Wk = torch.zeros_like(W)
            off = (self.k - 1) // 2
            for i in range(self.c2):
                Wk[i, i, off, off] += scale[i]
            W = W + Wk
            B = B + b_id

        fused = Conv(self.c1, self.c2, self.k, self.s, self.p, self.g, act=self.act)
        # 关键：新建模块默认在 CPU，要跟随原权重所在 device/dtype
        fused.to(device=W.device, dtype=W.dtype)
        fused.conv.weight.data.copy_(W)
        # 把 bias 放进 BN（BN 初始为恒等：gamma=1, beta=bias, mean=0, var=1-eps）
        fused.bn.weight.data.fill_(1.0)
        fused.bn.bias.data.copy_(B)
        fused.bn.running_mean.zero_()
        fused.bn.running_var.fill_(1.0 - fused.bn.eps)
        fused.eval()
        self.fused = fused
        return fused


def convert_model_to_repconv(model: nn.Module) -> int:
    """把检测头 / P2 分支里的 3x3 Conv 换成 RepConv（权重做等价初始化）。

    只换 groups==1 的 3x3：这是 MobileOne 论文里收益最大、也最容易合并的形态。
    1x1 分支权重初始化为 0、identity 的 BN gamma 初始化为 0，
    因此**初始化时与替换前完全等价**（可逐位验证）。
    """
    n = 0

    def walk(module: nn.Module) -> None:
        nonlocal n
        for name, child in list(module.named_children()):
            if isinstance(child, Conv) and child.conv.kernel_size == (3, 3) \
                    and child.conv.groups == 1:
                rep = RepConv(child.conv.in_channels, child.conv.out_channels, k=3,
                              s=child.conv.stride[0], p=child.conv.padding[0])
                rep.to(device=child.conv.weight.device, dtype=child.conv.weight.dtype)
                rep.conv3.weight.data.copy_(child.conv.weight.data)
                rep.bn3.load_state_dict(child.bn.state_dict())
                if rep.has_1x1:
                    rep.conv1.weight.data.zero_()
                    rep.bn1.weight.data.zero_()
                if rep.has_id:
                    rep.bn_id.weight.data.zero_()
                setattr(module, name, rep)
                n += 1
            else:
                walk(child)

    # 只动 neck/head 相关的子模块
    if getattr(model, "detect", None) is not None:
        walk(model.detect)
    if getattr(model, "p2_branch", None) is not None:
        walk(model.p2_branch)
    return n


def reparameterize_model(model: nn.Module) -> int:
    """把模型里所有 RepConv 合并成普通 Conv，返回合并数量。"""
    n = 0
    for name, mod in list(model.named_modules()):
        if isinstance(mod, RepConv) and mod.fused is None:
            mod.fuse()
            n += 1
    return n