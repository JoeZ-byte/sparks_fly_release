"""纯 Python 核心：接口与 C 加速库（native/src/c_api.h）一一对应。

分层
----
stream.py      只负责 I/O：命令行、解码、绘制、CSV/密封、输出模式与异步调度
py_core.py     纯 Python 算法核心（本文件）——与 C 封装同构
native_bridge  C++ 算法核心的 ctypes 封装——与本文件同构

两边都提供：
    Config(...)            配置项与 SflyConfig 字段一一对应
    Pipeline(model, cfg)   构造
    .step(frame) -> list[Track]     Track 字段与 SflyTrack 一一对应
    .stats()  -> dict      与 SflyStats 一一对应
    .close()
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pipeline_py import PyPipeline as _PyCore


@dataclass
class Config:
    """与 native/src/c_api.h 的 SflyConfig 字段一一对应。"""
    model_path: str = ""
    imgsz: int = 320
    max_cells: int = 6
    threads: int = 4
    warmup: int = 0
    max_coast: int = 8
    conf: float = 0.2
    pre_mode: int = 1          # 1=白顶帽, 0=关闭
    min_hits: int = 5
    gate: float = 0.7
    app_w: float = 0.0
    flow_w: float = 1.0
    diff_mode: int = 0
    bg_mode: int = 0


@dataclass
class Track:
    """与 SflyTrack 一一对应。"""
    track_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    px_per_frame: float = -1.0
    speed_ready: int = 0
    kmh: float = 0.0

    @property
    def box(self):
        return (self.x1, self.y1, self.x2, self.y2)


class Pipeline:
    """纯 Python 实现，行为与 NativePipeline 对齐。"""

    def __init__(self, model: str | Path, cfg: Config | None = None, **kw) -> None:
        if cfg is None:
            fields = {k: v for k, v in kw.items() if k in Config.__dataclass_fields__}
            cfg = Config(**fields)
        self.cfg = cfg
        self._core = _PyCore(
            Path(model), imgsz=cfg.imgsz, max_cells=cfg.max_cells, threads=cfg.threads,
            conf=cfg.conf, flow_w=cfg.flow_w, pre_mode=cfg.pre_mode,
            min_hits=cfg.min_hits, gate=cfg.gate, max_coast=cfg.max_coast,
            warmup=cfg.warmup,
        )
        self._frames = 0
        self._tiles = 0
        self._dets = 0

    def step(self, frame) -> list[Track]:
        raw = self._core.step(frame)
        self._frames = self._core.frame
        self._tiles = getattr(self._core, "tiles", 0)
        self._dets = getattr(self._core, "dets", 0)
        # 与 native_bridge.NativePipeline.step 返回完全相同的结构
        return [{"track_id": int(x["track_id"]),
                 "box": tuple(float(v) for v in x["box"]),
                 "px": float(x.get("px", -1.0)),
                 "ready": bool(x.get("ready", False)),
                 "kmh": float(x.get("kmh", 0.0))} for x in raw]

    def stats(self) -> dict:
        return {"motion_ms": 0.0, "infer_ms": 0.0, "track_ms": 0.0,
                "tiles": self._tiles, "dets": self._dets, "frames": self._frames}

    def close(self) -> None:
        self._core.close()