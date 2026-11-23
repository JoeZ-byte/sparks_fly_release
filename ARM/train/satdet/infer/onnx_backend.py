"""ONNX Runtime 后端：CPU 推理的唯一入口。

CPU 性能要点（每一条都对应赛题的"算子/多核"要求）
------------------------------------------------
1. ``intra_op_num_threads`` = **物理核数**（不是逻辑核）：超订会让线程互相抢核，
   实测 16 逻辑核开 16 线程比开 8 线程慢。
2. ``graph_optimization_level = ORT_ENABLE_ALL``：算子融合（Conv+BN+ReLU 等）。
3. ``enable_mem_pattern`` + 固定输入 shape：避免每次 run 重新规划内存。
4. ``I/O binding``（可选）：把输入输出张量绑定到预分配 buffer，省掉一次拷贝。
5. 输入用 ``uint8 -> float32 /255`` 在 numpy 里做，而不是在图里做，
   避免图里多一个 Normalize 子图（量化工具对这类算子的处理不稳定）。
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from satdet.core.logging import get_logger

__all__ = ["OnnxBackend"]

log = get_logger("onnx_backend")


class OnnxBackend:
    """ORT 会话封装。

    Parameters
    ----------
    path : ONNX 模型路径（fp32 或 int8 均可）
    imgsz : 固定输入边长
    threads : intra-op 线程数；None = 物理核数
    provider : "CPUExecutionProvider" / "CUDAExecutionProvider"
    """

    def __init__(self, path: str | Path, imgsz: int = 512, threads: int | None = None,
                 provider: str = "CPUExecutionProvider", opt_level: str = "all",
                 allow_spinning: bool = True) -> None:
        import onnxruntime as ort

        self.path = Path(path)
        self.imgsz = int(imgsz)

        so = ort.SessionOptions()
        so.graph_optimization_level = {
            "disable": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
            "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
            "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
            "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
        }[opt_level]
        so.enable_mem_pattern = True
        so.enable_cpu_mem_arena = True
        self.threads = int(threads) if threads else _physical_cores()
        so.intra_op_num_threads = self.threads
        so.inter_op_num_threads = 1          # 单请求顺序执行，避免线程切换开销
        if not allow_spinning:
            # 线程不忙等、让出 CPU：多进程场景下避免线程池互抢核（计划2 条目 11 实测用）
            so.add_session_config_entry("session.intra_op.allow_spinning", "0")

        providers = [provider]
        if provider == "CUDAExecutionProvider":
            providers.append("CPUExecutionProvider")
        self.session = ort.InferenceSession(str(self.path), so, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.provider = self.session.get_providers()[0]
        log.info("ORT 会话: %s | provider=%s | threads=%d | 输入 %s",
                 self.path.name, self.provider, self.threads,
                 [i.shape for i in self.session.get_inputs()])

    # ------------------------------------------------------------------
    def infer(self, batch_nhwc_float: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """输入 ``(B,H,W,3)`` float32（0~1，RGB），返回 (boxes, scores)。"""
        x = np.ascontiguousarray(batch_nhwc_float.transpose(0, 3, 1, 2), dtype=np.float32)
        out = self.session.run(self.output_names, {self.input_name: x})
        return out[0], out[1]

    def infer_raw(self, batch_nhwc_float: np.ndarray) -> np.ndarray:
        """只取第一个输出（raw 预测 ``(B, 4*reg_max+nc, A)``）。

        配合 ``satdet/infer/decode_fast.py`` 使用：decode 交给 numpy，
        只解码过阈值的 anchor（实测后处理 22 ms -> 0.19 ms）。
        """
        x = np.ascontiguousarray(batch_nhwc_float.transpose(0, 3, 1, 2), dtype=np.float32)
        return self.session.run(self.output_names[:1], {self.input_name: x})[0]

    def infer_timed(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """返回 (boxes, scores, 毫秒)。"""
        t0 = time.perf_counter()
        boxes, scores = self.infer(x)
        return boxes, scores, (time.perf_counter() - t0) * 1000.0

    def warmup(self, n: int = 5) -> None:
        dummy = np.zeros((1, self.imgsz, self.imgsz, 3), np.float32)
        for _ in range(n):
            self.infer(dummy)

    def benchmark(self, x: np.ndarray, iters: int = 30, warmup: int = 5) -> dict:
        """单批次微基准：返回 p50/p95/吞吐。"""
        self.warmup(warmup)
        times = []
        for _ in range(iters):
            _b, _s, ms = self.infer_timed(x)
            times.append(ms)
        arr = np.asarray(times)
        return {
            "threads": self.threads,
            "provider": self.provider,
            "p50_ms": float(np.percentile(arr, 50)),
            "p95_ms": float(np.percentile(arr, 95)),
            "mean_ms": float(arr.mean()),
            "batch": int(x.shape[0]),
            "samples_per_s": float(x.shape[0] * 1000.0 / arr.mean()) if arr.mean() > 0 else float("nan"),
        }


def _physical_cores() -> int:
    try:
        import psutil

        return int(psutil.cpu_count(logical=False) or 8)
    except Exception:  # noqa: BLE001
        return 8
