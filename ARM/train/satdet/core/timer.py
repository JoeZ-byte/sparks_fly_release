"""计时工具：流水线逐阶段耗时与分位数统计。

设计要点
--------
* 用 ``time.perf_counter``（单调时钟，不受系统时间调整影响）。
* 分位数用线性插值，避免小样本下 p95 抖动过大。
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

__all__ = ["Stopwatch", "LatencyRecorder"]


class Stopwatch:
    """上下文计时器。

    >>> with Stopwatch() as sw:
    ...     do_work()
    >>> sw.ms
    """

    __slots__ = ("_t0", "ms")

    def __init__(self) -> None:
        self._t0 = 0.0
        self.ms = 0.0

    def __enter__(self) -> "Stopwatch":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.ms = (time.perf_counter() - self._t0) * 1000.0

    @property
    def s(self) -> float:
        return self.ms / 1000.0


def _percentile(values: np.ndarray, q: float) -> float:
    """线性插值分位数（与 numpy 默认一致）。"""
    return float(np.percentile(values, q)) if values.size else float("nan")


@dataclass
class LatencyRecorder:
    """记录一串耗时样本，输出 p50/p95/p99 与吞吐。

    用法见 ``satdet/eval/bench_harness.py``：每个流水线阶段一个 recorder。
    """

    name: str = "latency"
    samples_ms: list[float] = field(default_factory=list)

    def add(self, ms: float) -> None:
        self.samples_ms.append(float(ms))

    def extend(self, values: Iterable[float]) -> None:
        self.samples_ms.extend(float(v) for v in values)

    def reset(self) -> None:
        self.samples_ms.clear()

    def summary(self) -> dict[str, float]:
        """返回统计字典；空样本时给 NaN，避免下游除零。"""
        if not self.samples_ms:
            return {"n": 0, "mean": float("nan"), "p50": float("nan"),
                    "p95": float("nan"), "p99": float("nan"), "max": float("nan"),
                    "fps": float("nan")}
        arr = np.asarray(self.samples_ms, dtype=np.float64)
        mean = float(arr.mean())
        return {
            "n": int(arr.size),
            "mean": mean,
            "p50": _percentile(arr, 50),
            "p95": _percentile(arr, 95),
            "p99": _percentile(arr, 99),
            "max": float(arr.max()),
            "fps": 1000.0 / mean if mean > 0 else float("nan"),
        }

    def to_row(self, **extra: object) -> dict[str, object]:
        """展平为 CSV 行：``{"stage": name, "mean_ms":..., ...}``。"""
        s = self.summary()
        row: dict[str, object] = {"stage": self.name}
        row.update({f"{k}": v for k, v in s.items()})
        row.update(extra)
        return row

    @staticmethod
    def describe(values: Iterable[float]) -> str:
        """一行文字摘要，方便日志打点。"""
        arr = list(values)
        if not arr:
            return "no samples"
        return (f"n={len(arr)} mean={statistics.fmean(arr):.2f}ms "
                f"max={max(arr):.2f}ms")
