"""导出"只到原始预测"的 ONNX，把 decode 留给 numpy。

配合 ``satdet/infer/decode_fast.py``：只解码过阈值的 anchor，
把后处理从 22 ms（全量 DFL）降到 ~0.5 ms。
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["RawModel"]


class RawModel(nn.Module):
    """``input (B,3,H,W) -> preds (B, 4*reg_max+nc, A)``。"""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = self.model(x)
        return torch.cat([o.view(o.shape[0], o.shape[1], -1) for o in outs], dim=2)
