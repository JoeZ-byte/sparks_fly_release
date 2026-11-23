"""多进程推理流水线：帧级数据并行 + 每进程少量线程。

形态依据（实测，见 docs/06 第 4 节）
----------------------------------
* 单会话开 16 线程反而比 8 线程慢 30%（小张量 + 内存带宽受限 + 超订）；
* **4 进程 × 4 线程**吞吐最高：66.5 图块/s，比 1 进程 × 16 线程快 1.37×；
* batch 拼接更慢，所以每个进程一次只算一帧内的若干图块。

流水线结构
----------
```
主进程                                        工作进程 (×N)
 ├ 读 JPEG 字节 ────────────▶ 共享内存池 ────▶ cv2 解码
 ├ 1/4 分辨率解码 → 运动提案 → 图块原点 ─────▶ 裁剪 / resize / 推理
 │                                              ↓
 └ 跟踪 + 车速 + 叠加 ◀──── 结果队列 ◀────── 快速解码 + NMS
```

主进程只在 1/4 分辨率上做运动提案（约 1 ms），避免"为了提案把整帧解码两遍"。
"""

from __future__ import annotations

import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from satdet.core.logging import get_logger
from satdet.infer.decode_fast import decode_topk

__all__ = ["PipelineConfig", "PipelineStats", "InferenceWorker", "run_frame_batch"]

log = get_logger("pipeline")

_STATE: dict = {}


@dataclass
class PipelineConfig:
    """流水线参数。"""

    onnx_path: str = ""
    imgsz: int = 512
    tile: int = 256
    n_workers: int = 4
    threads_per_worker: int = 4
    allow_spinning: bool = False   # 实测关闭自旋后 4x2 吞吐是 1x8 的 2.17 倍（scripts/24_spin_bench.py）
    pre_blur: int = 0             # 送入网络前的中值滤波核（0=关）；计划2 条目 7，需实测定夺
    dedup_px: float = 0.0         # 相邻重复框合并半径。实测 3px 会合并相邻车道真车，mAP -7.6~-9.1，默认关
    sr_noise: float = 0.0         # 随机共振实验：推理前注入调谐高斯噪声(0~1)，0=关
    conf_thr: float = 0.25
    nms_iou: float = 0.5
    max_nms: int = 1000
    max_det: int = 300


@dataclass
class PipelineStats:
    """流水线统计（写报告用）。"""

    frames: int = 0
    tiles: int = 0
    wall_s: float = 0.0
    per_stage_ms: dict = field(default_factory=dict)

    @property
    def fps(self) -> float:
        return self.frames / self.wall_s if self.wall_s > 0 else float("nan")

    @property
    def tiles_per_frame(self) -> float:
        return self.tiles / self.frames if self.frames else float("nan")


class InferenceWorker:
    """工作进程：持有自己的 ORT 会话，按图块原点裁剪并推理。"""

    def __init__(self, cfg: PipelineConfig | dict) -> None:
        if isinstance(cfg, dict):
            cfg = PipelineConfig(**cfg)
        self.cfg = cfg
        from satdet.infer.decode_fast import make_anchors_np
        from satdet.infer.onnx_backend import OnnxBackend

        self.backend = OnnxBackend(cfg.onnx_path, imgsz=cfg.imgsz,
                                   threads=cfg.threads_per_worker,
                                   allow_spinning=cfg.allow_spinning)
        self.anchors, self.strides = make_anchors_np(cfg.imgsz, [4.0, 8.0, 16.0, 32.0])

    def run(self, jpeg: bytes, origins: list[tuple[int, int]]) -> list[tuple[float, float, float, float, float]]:
        """返回 ``[(x1,y1,x2,y2,score), ...]``（全图坐标）。"""
        img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return []
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tile, imgsz = self.cfg.tile, self.cfg.imgsz
        scale = imgsz / float(tile)

        chunks = []
        for x0, y0 in origins:
            crop = rgb[y0:y0 + tile, x0:x0 + tile]
            if crop.shape[0] != tile or crop.shape[1] != tile:
                pad = np.full((tile, tile, 3), 114, np.uint8)
                pad[:crop.shape[0], :crop.shape[1]] = crop
                crop = pad
            if self.cfg.sr_noise > 0:
                # 论文《二维随机共振》的忠实简化：给图块加受控噪声，
                # 验证"噪声托举弱目标过检测阈值"是否成立。固定种子可复现。
                rng = np.random.default_rng(42)
                crop = np.clip(crop.astype(np.float32) / 255.0
                               + rng.normal(0.0, self.cfg.sr_noise, crop.shape), 0, 1)
                crop = (crop * 255).astype(np.uint8)
            if self.cfg.pre_blur >= 3:
                # 中值滤波：平滑 JPEG 压缩噪点同时保住边缘（不用高斯，6px 目标经不起模糊）
                crop = cv2.medianBlur(crop, self.cfg.pre_blur)
            chunks.append(cv2.resize(crop, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR))
        if not chunks:
            return []

        # 关键实测：CPU 上**逐图块推理（batch=1）比拼 batch 快 3.5 倍**。
        # 原因：小通道卷积在 batch 放大后中间张量变大、L2/L3 溢出，
        # 而 batch=1 时整层数据能留在缓存里（见 docs/06 第 3 节）。
        results = []
        for (x0, y0), chunk in zip(origins, chunks):
            batch = chunk[None].astype(np.float32) / 255.0
            preds = self.backend.infer_raw(batch)
            boxes, scores = decode_topk(preds, self.anchors, self.strides, nc=1,
                                        reg_max=16, conf_thr=self.cfg.conf_thr,
                                        max_nms=self.cfg.max_nms)[0]
            if len(boxes) == 0:
                continue
            b = boxes / scale
            if self.cfg.dedup_px > 0 and len(b) > 1:
                from satdet.infer.decode_fast import merge_near_duplicates

                b2, _s2 = merge_near_duplicates(b, scores, self.cfg.dedup_px)
                b = b2
            for j in range(len(b)):
                results.append((float(b[j, 0] + x0), float(b[j, 1] + y0),
                                float(b[j, 2] + x0), float(b[j, 3] + y0),
                                float(scores[j])))
        return _nms_all(results, self.cfg.nms_iou, self.cfg.max_det)



def _nms_all(results, iou_thr: float, max_det: int):
    if not results:
        return []
    from satdet.infer.nms import nms_numpy

    arr = np.asarray(results, np.float32)
    idx = nms_numpy(arr[:, :4], arr[:, 4], iou_thr, max_det)
    return [tuple(r) for r in arr[idx].tolist()]


# =====================================================================
# 进程池入口
# =====================================================================
def _init_worker(cfg_dict: dict) -> None:
    _STATE["worker"] = InferenceWorker(cfg_dict)


def _work(job: tuple[bytes, list[tuple[int, int]]]) -> list[tuple[float, float, float, float, float]]:
    return _STATE["worker"].run(*job)


class FrameBatchRunner:
    """帧级并行的执行器封装。"""

    def __init__(self, cfg: PipelineConfig) -> None:
        self.cfg = cfg
        self._pool = ProcessPoolExecutor(
            max_workers=cfg.n_workers,
            initializer=_init_worker,
            initargs=(cfg.__dict__.copy(),),
        )

    def map(self, jobs):
        return self._pool.map(_work, jobs)

    def close(self) -> None:
        self._pool.shutdown(wait=True)

    def __enter__(self) -> "FrameBatchRunner":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def run_frame_batch(jobs, cfg: PipelineConfig) -> tuple[list, PipelineStats]:
    """一次性跑完一批 ``(jpeg, origins)``。"""
    t0 = time.perf_counter()
    with FrameBatchRunner(cfg) as runner:
        out = list(runner.map(jobs))
    stats = PipelineStats(frames=len(jobs),
                         tiles=int(sum(len(j[1]) for j in jobs)),
                         wall_s=time.perf_counter() - t0)
    return out, stats
