"""pipeline —— L5 多核流水线层。

* ``shm.py``           共享内存帧池（跨进程零拷贝 + 背压）
* ``orchestrator.py``  帧级数据并行（4 进程 × 4 线程，依据 docs/06 实测）
"""

from satdet.pipeline.orchestrator import (
    FrameBatchRunner,
    InferenceWorker,
    PipelineConfig,
    PipelineStats,
    run_frame_batch,
)
from satdet.pipeline.shm import FrameHandle, SharedFramePool

__all__ = [
    "FrameBatchRunner",
    "InferenceWorker",
    "PipelineConfig",
    "PipelineStats",
    "run_frame_batch",
    "FrameHandle",
    "SharedFramePool",
]
