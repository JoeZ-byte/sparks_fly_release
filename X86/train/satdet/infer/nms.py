"""numpy 版 NMS：推理后处理不建 torch 图。

为什么不用 ``torchvision.ops.nms``
--------------------------------
* 每个图块都要建一次图 → 大量小的 kernel launch，实测在 CPU 上是纯亏；
* 目标是 6 px，单块预测框通常 <20 个，numpy 向量化完全够快；
* 去掉 torchvision 依赖，运行时只剩 torch/opencv/onnxruntime。
"""

from __future__ import annotations

import numpy as np

__all__ = ["nms_numpy", "batched_nms"]


def nms_numpy(boxes: np.ndarray, scores: np.ndarray, iou_thr: float = 0.5,
              max_out: int = 300) -> np.ndarray:
    """单类别 NMS，返回保留的下标（按分数降序）。"""
    b = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    s = np.asarray(scores, dtype=np.float32).reshape(-1)
    if b.shape[0] == 0:
        return np.zeros(0, dtype=np.int64)

    x1, y1, x2, y2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    areas = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    order = np.argsort(-s, kind="stable")

    keep: list[int] = []
    while order.size > 0 and len(keep) < max_out:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        union = areas[i] + areas[rest] - inter
        iou = inter / np.maximum(union, 1e-9)
        order = rest[iou <= iou_thr]
    return np.asarray(keep, dtype=np.int64)


def batched_nms(boxes: np.ndarray, scores: np.ndarray, class_ids: np.ndarray,
                iou_thr: float = 0.5, max_out: int = 300) -> np.ndarray:
    """多类别 NMS：按类别分别做，最后合并排序。"""
    if len(boxes) == 0:
        return np.zeros(0, dtype=np.int64)
    keep_all: list[np.ndarray] = []
    for c in np.unique(class_ids):
        idx = np.where(class_ids == c)[0]
        k = nms_numpy(boxes[idx], scores[idx], iou_thr, max_out)
        keep_all.append(idx[k])
    keep = np.concatenate(keep_all) if keep_all else np.zeros(0, dtype=np.int64)
    return keep[np.argsort(-scores[keep], kind="stable")][:max_out]
