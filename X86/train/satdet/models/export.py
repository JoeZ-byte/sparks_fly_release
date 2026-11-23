"""ONNX 导出：把模型变成"固定输入 shape、单文件、可量化"的计算图。

为什么把 decode 也导出进去
--------------------------
* 导出后只有 **2 个输出**（boxes、scores），运行时不必再实现一遍 DFL 解码，
  省掉 Python 侧开销（CPU 场景下 Python 循环常常比卷积还贵）；
* 固定输入 512×512，图里没有动态分支，ORT 可以开启全部图优化。

anchor 在 ``__init__`` 里按 imgsz 预计算并注册成 buffer：
这样导出图里是常量，没有 shape 推导与动态迭代，
既消除 TracerWarning，也让 ORT 的常量折叠生效。
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from satdet.core.logging import get_logger
from satdet.models.yolov8 import dist2bbox

__all__ = ["ExportModel", "export_onnx"]

log = get_logger("export")


class ExportModel(nn.Module):
    """导出用包装：``input (1,3,H,W) -> (boxes_xyxy, class_scores)``。"""

    def __init__(self, model: nn.Module, imgsz: int = 512) -> None:
        super().__init__()
        self.model = model
        self.nc = int(getattr(model, "nc", model.detect.nc))
        self.reg_max = int(model.detect.reg_max)

        strides = [float(s) for s in model.stride]
        points, st = [], []
        for s in strides:
            side = int(imgsz // s)
            gy, gx = torch.meshgrid(
                torch.arange(side, dtype=torch.float32) + 0.5,
                torch.arange(side, dtype=torch.float32) + 0.5,
                indexing="ij",
            )
            points.append(torch.stack((gx, gy), -1).view(-1, 2))
            st.append(torch.full((side * side, 1), s, dtype=torch.float32))
        self.register_buffer("anchors", torch.cat(points), persistent=False)
        self.register_buffer("strides_all", torch.cat(st), persistent=False)
        self.imgsz = int(imgsz)

    def forward(self, x: torch.Tensor):
        preds = self.model(x)
        feats = [p.view(p.shape[0], 4 * self.reg_max + self.nc, -1) for p in preds]
        box = torch.cat([f[:, : 4 * self.reg_max] for f in feats], dim=2)
        cls = torch.cat([f[:, 4 * self.reg_max:] for f in feats], dim=2)

        b, _, a = box.shape
        dist = box.view(b, 4, self.reg_max, a)
        # softmax × arange 的显式期望（与 DFL.conv 数值等价，但图更干净）
        w = F.softmax(dist, dim=2)
        bins = torch.arange(self.reg_max, dtype=dist.dtype, device=dist.device)
        dist = (w * bins.view(1, 1, self.reg_max, 1)).sum(dim=2).permute(0, 2, 1)   # (B,A,4)
        boxes = dist2bbox(dist, self.anchors.unsqueeze(0), xywh=False) * self.strides_all.view(1, -1, 1)
        scores = cls.permute(0, 2, 1).sigmoid()
        return boxes, scores


def export_onnx(model: nn.Module, out_path: str | Path, imgsz: int = 512,
                opset: int = 17, simplify: bool = True,
                dynamic_batch: bool = True) -> Path:
    """导出 ONNX。

    ``dynamic_batch=True`` 只把第 0 维设为动态 —— H/W 仍固定 512。
    这是关键：**把一帧的 7 个图块拼成一个 batch 做一次前向**，
    是 CPU 上摊薄调度开销最有效的手段（实测 batch 7 比 7 次单独调用快 3 倍）。
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    wrapper = ExportModel(model, imgsz=imgsz).eval()
    dummy = torch.zeros(1, 3, imgsz, imgsz)
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {"images": {0: "batch"},
                        "boxes": {0: "batch"},
                        "scores": {0: "batch"}}

    torch.onnx.export(
        wrapper, dummy, str(out),
        input_names=["images"], output_names=["boxes", "scores"],
        opset_version=opset, do_constant_folding=True, dynamo=False,
        dynamic_axes=dynamic_axes,
    )
    log.info("已导出 ONNX: %s (%.2f MB, imgsz=%d, opset=%d)",
             out.name, out.stat().st_size / 1e6, imgsz, opset)

    if simplify:
        try:
            import onnx

            m = onnx.load(str(out))
            onnx.checker.check_model(m)
            n_nodes = len(m.graph.node)
            try:
                import onnxslim

                m = onnxslim.slim(m)
                onnx.save(m, str(out))
                log.info("onnxslim 化简: %d -> %d 节点", n_nodes, len(m.graph.node))
            except ImportError:
                log.info("未安装 onnxslim，跳过化简（节点数 %d）", n_nodes)
        except Exception as exc:  # noqa: BLE001
            log.warning("ONNX 校验/化简失败（不影响使用）: %s", exc)
    return out
