"""量化感知实验：per-channel 权重量化 + per-tensor 激活量化 + AdaRound 式学习舍入。

为什么在 PyTorch 里做
--------------------
ONNX Runtime 的 quantize_static 只支持几种校准器（MinMax/Percentile/Entropy），
**不支持 AdaRound/BRECQ 那种"学习权重舍入"**。而衡量 AdaRound 的收益需要可信的
mAP 评测，只有 torch 侧的 harness 是被反复验证过的（docs/10 的 mAP 都出自它）。
所以这里先把量化过程"仿真"成 nn.Module，跑 harness 拿到相对增益，
再把学到的舍入结果写回权重、导出 ONNX 给板端用。

两个层次
--------
1. ``QuantConv2d``：per-output-channel 对称 INT8 权重 + per-tensor 非对称 uint8 激活；
   权重舍入有两种模式：
     * RTN（round-to-nearest）：``round(w/s)``
     * AdaRound：``round(w/s) + clamp(h(a) - 0.5, -0.5, 0.5)``，其中 h 是 sigmoid，
       a 是每层可学习参数（STE 反传）。初始化 h(a)=0.5 时等价 RTN。
2. ``adaround_layer``：逐层（block-wise，BRECQ 的最简形式）最小化"量化后输出 vs
   浮点输出"的 MSE，只优化该层的 a。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["QuantConv2d", "quantize_model", "adaround_layer", "QConfig"]


class QConfig:
    def __init__(self, bits: int = 8, adaround: bool = False,
                 act_percentile: float | None = None) -> None:
        self.bits = bits
        self.adaround = adaround
        # 激活校准：None = MinMax；给定时用对称百分位裁剪（抑制长尾离群点）
        self.act_percentile = act_percentile


class QuantConv2d(nn.Conv2d):
    """把 nn.Conv2d 换成带 fake-quant 的版本（结构与参数保持不变）。"""

    def __init__(self, conv: nn.Conv2d, cfg: QConfig) -> None:
        super().__init__(conv.in_channels, conv.out_channels, conv.kernel_size,
                         conv.stride, conv.padding, conv.dilation, conv.groups,
                         conv.bias is not None, conv.padding_mode)
        self.weight.data.copy_(conv.weight.data)
        if conv.bias is not None:
            self.bias.data.copy_(conv.bias.data)
        self.cfg = cfg
        self.bits = cfg.bits
        self.qmax = 2 ** (cfg.bits - 1) - 1
        # 激活量化参数（校准后填入）
        self.act_scale = 1.0
        self.act_zero = 0.0
        self.act_quant = True
        # AdaRound 可学习参数（与权重同形）
        self.alpha = None
        if cfg.adaround:
            self.alpha = nn.Parameter(torch.zeros_like(self.weight))

    # ---- 权重 fake-quant ----
    def _w_scale(self) -> torch.Tensor:
        w = self.weight.detach()
        s = w.abs().amax(dim=(1, 2, 3), keepdim=True) / self.qmax
        return s.clamp_min(1e-8)

    def _wq(self) -> torch.Tensor:
        s = self._w_scale()
        w = self.weight
        if self.alpha is None:
            r = torch.round(w / s)                       # RTN
        else:
            # AdaRound：r = floor(w/s + h)，h∈(0,1) 是学习出来的"舍入阈值"
            #   h=0.5 且初始化时等价 RTN；
            #   STE：前向离散取整，反向把梯度透传给 h（系数 1）
            h = torch.sigmoid(self.alpha) * 0.98 + 0.01
            base = torch.floor(w / s + h.detach())
            r = base + (h - h.detach())
        return torch.clamp(r, -self.qmax - 1, self.qmax) * s

    # ---- 激活 fake-quant ----
    def _aq(self, x: torch.Tensor) -> torch.Tensor:
        if not self.act_quant or self.act_scale <= 0:
            return x
        q = torch.clamp(torch.round(x / self.act_scale) + self.act_zero, 0, 255)
        return (q - self.act_zero) * self.act_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(self._aq(x), self._wq(), self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


@torch.no_grad()
def _calibrate_act(model: nn.Module, batches: list[torch.Tensor],
                   percentile: float | None = None) -> None:
    """统计每层 Conv 输入的激活范围。

    percentile=None 时用 MinMax；给定时用「对称百分位」——取 |x| 的该分位数作为上限，
    抑制长尾离群点（DFQ/Percentile 量化的核心思想）。
    """
    stats = {}
    samples: dict[str, list[torch.Tensor]] = {}

    def hook(name):
        def fn(mod, inp, out):
            x = inp[0]
            if percentile is None:
                lo, hi = float(x.min()), float(x.max())
                cur = stats.get(name)
                stats[name] = (min(lo, cur[0]), max(hi, cur[1])) if cur else (lo, hi)
            else:
                # 抽样：每隔 8 个元素取一个，控制内存
                samples.setdefault(name, []).append(x.detach().abs().flatten()[::8].float())
        return fn

    handles = []
    for name, mod in model.named_modules():
        if isinstance(mod, QuantConv2d):
            handles.append(mod.register_forward_hook(hook(name)))
    for b in batches:
        model(b)
    for h in handles:
        h.remove()
    for name, mod in model.named_modules():
        if not isinstance(mod, QuantConv2d):
            continue
        if percentile is not None and name in samples:
            v = torch.cat(samples[name])
            if v.numel() == 0:
                continue
            hi = float(torch.quantile(v, percentile / 100.0))
            hi = max(hi, 1e-6)
            scale = max(2.0 * hi / 255.0, 1e-8)      # 对称范围 [-hi, hi]
            mod.act_scale = scale
            mod.act_zero = 128
        elif name in stats:
            lo, hi = stats[name]
            scale = max((hi - lo) / 255.0, 1e-8)
            mod.act_scale = scale
            mod.act_zero = round(-lo / scale)


def quantize_model(model: nn.Module, cfg: QConfig,
                   calib: list[torch.Tensor] | None = None) -> nn.Module:
    """把模型里的 Conv2d 全换成 QuantConv2d，并用校准集定激活范围。"""
    def swap(module: nn.Module) -> None:
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Conv2d) and not isinstance(child, QuantConv2d):
                new = QuantConv2d(child, cfg)
                # 关键：新建的模块默认在 CPU，要跟随原模块的 device/dtype
                new.to(device=child.weight.device, dtype=child.weight.dtype)
                if new.bias is not None:
                    new.bias.data = new.bias.data.to(child.weight.device)
                if new.alpha is not None:
                    new.alpha.data = new.alpha.data.to(child.weight.device)
                setattr(module, name, new)
            else:
                swap(child)

    # 先冻结激活量化，便于用浮点范围做校准
    swap(model)
    if calib:
        _calibrate_act(model, calib, percentile=cfg.act_percentile)
    return model


def adaround_layer(model: nn.Module, calib: list[torch.Tensor], iters: int = 60,
                   lr: float = 1e-2, only_trainable: bool = True) -> int:
    """逐层学习舍入（block-wise 重构）：最小化该层量化输出与浮点输出的 MSE。

    为了控制成本，只处理"有可训练参数"的层（即我们训练过的 neck/head）；
    骨干是冻结的，用的是官方权重，本来就更耐量化。
    """
    n_done = 0
    # 记录浮点参考输出
    for name, mod in list(model.named_modules()):
        if not isinstance(mod, QuantConv2d) or mod.alpha is None:
            continue
        if only_trainable and all(p.requires_grad is False for p in mod.parameters()):
            continue
        # 收集该层的输入
        inputs = []

        def hook(m, inp, out, store=inputs):
            store.append(inp[0].detach())

        h = mod.register_forward_hook(hook)
        with torch.no_grad():
            for b in calib:
                model(b)
        h.remove()
        if not inputs:
            continue
        # 浮点参考（关量化）
        mod.act_quant = False
        w_backup = mod.weight.detach().clone()
        with torch.no_grad():
            ref = [F.conv2d(x, w_backup, mod.bias, mod.stride, mod.padding,
                            mod.dilation, mod.groups) for x in inputs]
        mod.act_quant = True
        opt = torch.optim.Adam([mod.alpha], lr=lr)
        for _ in range(iters):
            total = 0.0
            for x, r in zip(inputs, ref):
                out = F.conv2d(mod._aq(x), mod._wq(), mod.bias, mod.stride,
                               mod.padding, mod.dilation, mod.groups)
                total = total + F.mse_loss(out, r)
            opt.zero_grad()
            (total / max(1, len(inputs))).backward()
            opt.step()
        n_done += 1
    return n_done