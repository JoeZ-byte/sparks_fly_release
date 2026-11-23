"""标注质检：把"数据里到底有什么坑"变成可复现的检查项。

为什么单独做这件事
------------------
赛题要求"真实场景"，评审一定会问：标注可信吗？规模够吗？
本模块把每个疑点都落成一条可重跑的检查，输出 CSV，
报告里直接引用，避免"我印象里好像是……"。

检查项
------
bounds        框是否越出图像边界
frame_gap     是否存在无标注的空帧
duplicate     同一帧内是否有完全重复的框
label_gt2     同一序列 label.txt 与 gt2.txt 的逐帧框集合差异（数据集自清洗的痕迹）
tiny          面积 < 4 px^2 的极小框（可能是压缩噪声被标成目标）
overlap       同帧内 IoU > 0.8 的重叠框（重复标注风险）
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from satdet.core.logging import get_logger
from satdet.data.ann_parsers import Frames, parse_label_txt, parse_mot_txt
from satdet.data.viso_index import SequenceInfo, VisoIndex

__all__ = ["run_checks", "check_bounds", "check_frame_gap", "check_duplicates",
           "check_tiny", "check_overlap", "check_label_gt2"]

log = get_logger("verify")


def _row(seq: str, check: str, status: str, count: int, detail: str) -> dict:
    return {"seq": seq, "check": check, "status": status, "count": int(count), "detail": detail}


def check_bounds(info: SequenceInfo) -> dict:
    gt = info.load_gt()
    if not len(gt):
        return _row(info.seq_id, "bounds", "skip", 0, "无标注")
    b = gt.boxes
    bad = (b[:, 0] < 0) | (b[:, 1] < 0) | (b[:, 2] > info.width) | (b[:, 3] > info.height)
    n = int(bad.sum())
    detail = "全部在图像内" if n == 0 else f"越界 {n} 个，样例 frame={gt.frame[bad][:3].tolist()}"
    return _row(info.seq_id, "bounds", "ok" if n == 0 else "warn", n, detail)


def check_frame_gap(info: SequenceInfo) -> dict:
    gt = info.load_gt()
    if not len(gt):
        return _row(info.seq_id, "frame_gap", "skip", 0, "无标注")
    counts = np.bincount(gt.frame, minlength=info.num_frames + 1)
    empty = np.flatnonzero(counts[1:info.num_frames + 1] == 0) + 1
    n = int(empty.size)
    detail = "每帧都有标注" if n == 0 else f"{n} 个空帧，前几个: {empty[:8].tolist()}"
    return _row(info.seq_id, "frame_gap", "ok" if n == 0 else "info", n, detail)


def check_duplicates(info: SequenceInfo) -> dict:
    gt = info.load_gt()
    if not len(gt):
        return _row(info.seq_id, "duplicate", "skip", 0, "无标注")
    keys = np.column_stack([gt.frame, np.round(gt.boxes, 2)])
    _, idx, cnt = np.unique(keys, axis=0, return_index=True, return_counts=True)
    n = int((cnt > 1).sum())
    return _row(info.seq_id, "duplicate", "ok" if n == 0 else "warn", n,
                "无重复框" if n == 0 else f"{n} 组完全重复的框")


def check_tiny(info: SequenceInfo, min_area: float = 4.0) -> dict:
    gt = info.load_gt()
    if not len(gt):
        return _row(info.seq_id, "tiny", "skip", 0, "无标注")
    wh = gt.wh
    area = wh[:, 0] * wh[:, 1]
    n = int((area < min_area).sum())
    pct = 100.0 * n / max(1, len(gt))
    return _row(info.seq_id, "tiny", "warn" if pct > 5 else "ok", n,
                f"面积<{min_area:g}px^2 的框 {n} 个 ({pct:.2f}%)")


def check_overlap(info: SequenceInfo, iou_thr: float = 0.8, max_frames: int = 120) -> dict:
    gt = info.load_gt()
    if not len(gt):
        return _row(info.seq_id, "overlap", "skip", 0, "无标注")
    total = 0
    frames = np.unique(gt.frame)[:max_frames]
    for f in frames:
        b = gt.boxes[gt.frame == f]
        if len(b) < 2:
            continue
        x1 = np.maximum(b[:, None, 0], b[None, :, 0])
        y1 = np.maximum(b[:, None, 1], b[None, :, 1])
        x2 = np.minimum(b[:, None, 2], b[None, :, 2])
        y2 = np.minimum(b[:, None, 3], b[None, :, 3])
        inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        union = area[:, None] + area[None, :] - inter
        iou = inter / np.clip(union, 1e-6, None)
        iu = np.triu_indices(len(b), k=1)
        total += int((iou[iu] > iou_thr).sum())
    return _row(info.seq_id, "overlap", "ok" if total == 0 else "info", total,
                f"同帧 IoU>{iou_thr} 的框对 {total} 组（前 {len(frames)} 帧）")


def check_label_gt2(info: SequenceInfo) -> dict:
    """对比 label.txt 与 gt2.txt：量化数据集作者的清洗动作。"""
    lb_path = info.root / "label.txt"
    gt_path = info.root / "gt2.txt"
    if not (lb_path.is_file() and gt_path.is_file()):
        return _row(info.seq_id, "label_gt2", "skip", 0, "缺少 label.txt 或 gt2.txt")
    lb = parse_label_txt(lb_path)
    gt = parse_mot_txt(gt_path)
    if not len(lb) or not len(gt):
        return _row(info.seq_id, "label_gt2", "skip", 0, "文件为空")

    def keyset(fr: Frames) -> set:
        return {(int(a), round(float(b), 2), round(float(c), 2), round(float(d), 2), round(float(e), 2))
                for a, b, c, d, e in zip(fr.frame, fr.boxes[:, 0], fr.boxes[:, 1],
                                         fr.boxes[:, 2], fr.boxes[:, 3])}

    a, b = keyset(lb), keyset(gt)
    only_label = len(a - b)
    only_gt = len(b - a)
    status = "ok" if only_label == 0 and only_gt == 0 else "info"
    return _row(info.seq_id, "label_gt2", status, only_label + only_gt,
                f"label 独有 {only_label} 个框，gt2 独有 {only_gt} 个框（gt2 为清洗版）")


def run_checks(index: VisoIndex, seq_ids: Iterable[str] | None = None) -> list[dict]:
    """跑全部检查，返回扁平记录列表。"""
    wanted = list(seq_ids) if seq_ids is not None else index.order
    rows: list[dict] = []
    for seq_id in wanted:
        if seq_id not in index.sequences:
            log.warning("跳过不存在的序列 %s", seq_id)
            continue
        info = index[seq_id]
        rows.extend([
            check_bounds(info),
            check_frame_gap(info),
            check_duplicates(info),
            check_tiny(info),
            check_overlap(info),
            check_label_gt2(info),
        ])
    return rows
