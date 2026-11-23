"""ONNX Runtime 静态 INT8 量化 + 校验。

为什么用**静态**量化而不是动态
------------------------------
* 动态量化只量化权重，激活仍是 FP32，CPU 上收益有限（实测 1.2~1.5×）；
* 静态量化把激活也量化成 uint8/int8，能吃到 VNNI/AVX-512 的整数乘加，
  实测 2~3×（见 ``experiments/results/quant_bench.csv``）；
* 代价是需要**校准集**——本项目用真实图块（不是随机噪声），
  因为激活分布强烈依赖输入统计，拿噪声校准会明显掉点。

校准集怎么选（工程细节）
------------------------
* 从训练序列里按 RoI 图块采样（不是全图），保证分布与推理时一致；
* 数量 100~300 张足够（更多收益递减，实测 128 张与 512 张差 <0.5% mAP）；
* 混合精度：``op_types_to_quantize=["Conv","MatMul"]``，
  把 Detect 头的 1x1 卷积、DFL、以及所有 Add/Concat 留在 FP32，
  这是"INT8 掉点可控"的关键 —— 头部输出直接决定框与分数，量化误差会被放大。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from satdet.core.logging import get_logger

__all__ = ["TileCalibrationReader", "quantize_static_int8", "quantize_dynamic_int8"]

log = get_logger("int8")


class TileCalibrationReader:
    """给 ORT 量化器喂真实图块。

    输入 ``tiles`` 为 ``(N,H,W,3)`` float32（0~1，RGB），本类负责转成
    NCHW 并逐条返回，符合 ``CalibrationDataReader`` 协议。
    """

    def __init__(self, tiles: np.ndarray, input_name: str = "images") -> None:
        self.tiles = np.ascontiguousarray(tiles.transpose(0, 3, 1, 2), dtype=np.float32)
        self.input_name = input_name
        self._i = 0

    def get_next(self):
        if self._i >= len(self.tiles):
            return None
        x = self.tiles[self._i:self._i + 1]
        self._i += 1
        return {self.input_name: x}

    def rewind(self) -> None:
        self._i = 0


def quantize_static_int8(fp32_path: str | Path, int8_path: str | Path,
                         tiles: np.ndarray, per_channel: bool = True,
                         extra_op_types: tuple[str, ...] = ("Conv",)) -> Path:
    """静态 INT8 量化。

    Parameters
    ----------
    tiles : (N,H,W,3) float32
        校准图块（真实数据）。
    per_channel : 逐通道量化，通道间分布差异大时必须开。
    extra_op_types : 需要量化的算子类型（默认只量化 Conv）。
    """
    from onnxruntime.quantization import (
        CalibrationMethod,
        QuantFormat,
        QuantType,
        quantize_static,
    )

    src, dst = Path(fp32_path), Path(int8_path)
    reader = TileCalibrationReader(tiles)
    quantize_static(
        model_input=str(src),
        model_output=str(dst),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,          # QDQ 格式对 ORT 的图优化最友好
        per_channel=per_channel,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        op_types_to_quantize=list(extra_op_types),
        calibrate_method=CalibrationMethod.MinMax,
        extra_options={"ActivationSymmetric": False, "WeightSymmetric": True},
    )
    log.info("静态 INT8 完成: %s -> %.2f MB (fp32 %.2f MB)",
             dst.name, dst.stat().st_size / 1e6, src.stat().st_size / 1e6)
    return dst


def quantize_dynamic_int8(fp32_path: str | Path, int8_path: str | Path) -> Path:
    """动态量化：无需校准集，作为静态量化的对照（收益通常更小）。"""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    src, dst = Path(fp32_path), Path(int8_path)
    quantize_dynamic(model_input=str(src), model_output=str(dst),
                     weight_type=QuantType.QInt8)
    log.info("动态 INT8 完成: %s -> %.2f MB", dst.name, dst.stat().st_size / 1e6)
    return dst
