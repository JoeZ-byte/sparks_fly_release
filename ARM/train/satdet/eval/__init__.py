"""eval —— 评测层：检测/跟踪指标与性能基准（口径见 06_评测口径.md）。"""

from satdet.eval.det_metrics import DetEvalResult, evaluate_detections
from satdet.eval.track_metrics import TrackEvalResult, evaluate_tracking

__all__ = ["DetEvalResult", "evaluate_detections",
           "TrackEvalResult", "evaluate_tracking"]
