"""infer —— 推理层：检测框解码、NMS、后续的 ONNX/INT8 后端。

已完成
------
* ``nms.py``       numpy NMS（不建 torch 图）
* ``detector.py``  整帧/图块推理封装，输出全图坐标

待实现（M5）
------------
* ``onnx_backend.py``  ONNX Runtime 会话（CPU 多线程）
* ``int8_quant.py``    静态 INT8 量化
* ``fusion.py``        Conv+BN+SiLU 融合
"""

from satdet.infer.nms import batched_nms, nms_numpy

__all__ = ["Detector", "batched_nms", "nms_numpy"]


def __getattr__(name: str):
    """惰性导出 ``Detector``：它依赖 torch，而解码/NMS/ONNX 路径并不需要 torch。

    之前包级 import 会强制拉起 torch，导致"只用 numpy 解码"的调用方也必须装 torch。
    """
    if name == "Detector":
        from satdet.infer.detector import Detector

        return Detector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
