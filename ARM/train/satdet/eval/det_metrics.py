"""检测指标：mAP / F1 / 分尺寸召回 / 中心距召回。

**主指标是 mAP30**，不是 mAP50。原因（见 docs/08 第 1.1 节）：
目标边长中位 5~9 px，中心偏 2 px 时 IoU 就从 1.0 掉到 0.43；
用 IoU=0.5 会把"定位差 2 px 的正确检测"判成漏检，指标失去意义。
实测帧间位移本来就有 2~3 px，所以主口径必须放宽到 0.3。

本模块只做纯计算（numpy），不碰 IO，方便单测与在无 GPU 环境跑。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "DEFAULT_IOU_THRESHOLDS",
    "DetEvalResult",
    "size_bin_of",
    "evaluate_detections",
    "iou_matrix",
]

DEFAULT_IOU_THRESHOLDS = (0.3, 0.5, 0.6, 0.7, 0.8, 0.9)

#: 面积分档（px^2），与 configs/dataset/viso.yaml 的 eval.size_bins_edges 对齐
#: 标签按 **等效边长** 命名（面积 = 边长²），边界为左闭右开：
#: ``<4px`` 表示面积 < 16，``4-6px`` 表示 16 <= 面积 < 36，依此类推。
SIZE_BINS = ((0.0, 16.0, "lt4px"), (16.0, 36.0, "4to6px"), (36.0, 64.0, "6to8px"),
             (64.0, 256.0, "8to16px"), (256.0, np.inf, "gt16px"))


def iou_matrix(a: np.ndarray, b: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """成对 IoU，返回 (len(a), len(b))。"""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)

    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)

    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    return inter / (area_a[:, None] + area_b[None, :] - inter + eps)


def size_bin_of(boxes: np.ndarray) -> np.ndarray:
    """返回每个框所属面积档的索引（对应 SIZE_BINS）。"""
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    area = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    idx = np.zeros(len(b), dtype=np.int64)
    for i, (lo, hi, _) in enumerate(SIZE_BINS):
        idx[(area >= lo) & (area < hi)] = i
    return idx


def _match_frame(pred_boxes: np.ndarray, pred_scores: np.ndarray,
                 gt_boxes: np.ndarray, iou_thr: float
                 ) -> tuple[np.ndarray, np.ndarray]:
    """单帧贪心匹配。

    Returns
    -------
    tp : (P,) bool  每个预测是否命中
    matched_gt : (G,) bool  每个 GT 是否被命中
    """
    n_p, n_g = len(pred_boxes), len(gt_boxes)
    tp = np.zeros(n_p, dtype=bool)
    matched_gt = np.zeros(n_g, dtype=bool)
    if n_p == 0 or n_g == 0:
        return tp, matched_gt

    iou = iou_matrix(pred_boxes, gt_boxes)
    order = np.argsort(-pred_scores, kind="stable")
    for pi in order:
        if iou[pi].max() < iou_thr:
            continue
        # 只在尚未匹配的 GT 里选 IoU 最大的
        cand = np.where(~matched_gt)[0]
        if cand.size == 0:
            break
        best = cand[np.argmax(iou[pi, cand])]
        if iou[pi, best] >= iou_thr:
            tp[pi] = True
            matched_gt[best] = True
    return tp, matched_gt


def _ap_from_tp(tp: np.ndarray, n_gt: int) -> float:
    """COCO 风格所有点插值 AP。tp 必须已按分数降序排列。"""
    if n_gt == 0:
        return float("nan")
    if tp.size == 0:
        return 0.0          # 有 GT 但一个预测都没有 -> AP=0（不是 NaN）

    ctp = np.cumsum(tp, dtype=np.float64)
    cfp = np.cumsum(~tp, dtype=np.float64)
    recall = ctp / n_gt
    precision = ctp / np.maximum(ctp + cfp, 1e-12)

    # COCO 的 all-points 插值：recall 先生成阶梯，precision 再从右向左取累计最大
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


@dataclass
class DetEvalResult:
    """一次评测的全部结果（可直接转 CSV 行）。"""

    map30: float = float("nan")
    map50: float = float("nan")
    map50_95: float = float("nan")
    precision: float = float("nan")
    recall: float = float("nan")
    f1: float = float("nan")
    conf: float = float("nan")                 # F1 最优的置信度阈值
    recall_by_size: dict[str, float] = field(default_factory=dict)
    recall_center2px: float = float("nan")
    n_gt: int = 0
    n_pred: int = 0
    n_frames: int = 0
    # 调阈值曲线，便于画 P-R / F1-conf 图
    conf_curve: list[dict] = field(default_factory=list)

    def to_row(self, **extra: object) -> dict:
        row: dict[str, object] = {
            "mAP30": self.map30,
            "mAP50": self.map50,
            "mAP50_95": self.map50_95,
            "Precision": self.precision,
            "Recall": self.recall,
            "F1": self.f1,
            "conf": self.conf,
            "recall_center2px": self.recall_center2px,
            "n_gt": self.n_gt,
            "n_pred": self.n_pred,
            "n_frames": self.n_frames,
        }
        for k, v in self.recall_by_size.items():
            row[f"recall_{k}"] = v
        row.update(extra)
        return row


def evaluate_detections(
    preds: list[dict],
    gts: dict[tuple[str, int], np.ndarray],
    conf_thr: float | None = None,
    conf_scan: tuple[float, float, int] = (0.05, 0.95, 19),
    iou_thresholds: tuple[float, ...] = DEFAULT_IOU_THRESHOLDS,
) -> DetEvalResult:
    """计算 mAP / F1 / 分尺寸召回。

    Parameters
    ----------
    preds : list[dict]
        每个框集合一条：``{"seq": str, "frame": int, "boxes": (N,4) xyxy 全图,
        "scores": (N,)}``。可以一帧多条（图块推理后合并）——内部会按 (seq,frame) 聚合。
    gts : dict[(seq, frame)] -> (M,4)
    conf_thr : float | None
        给定时直接用该阈值算 P/R/F1；None 则在 ``conf_scan`` 上扫 F1 最优点。
    """
    # ---- 按 (seq, frame) 聚合预测 ----
    grouped: dict[tuple[str, int], list[tuple[np.ndarray, np.ndarray]]] = {}
    for p in preds:
        key = (str(p["seq"]), int(p["frame"]))
        grouped.setdefault(key, []).append(
            (np.asarray(p["boxes"], dtype=np.float64).reshape(-1, 4),
             np.asarray(p["scores"], dtype=np.float64).reshape(-1))
        )

    frames = sorted(set(gts.keys()) | set(grouped.keys()))
    merged: list[tuple[tuple[str, int], np.ndarray, np.ndarray, np.ndarray]] = []
    n_gt_total = 0
    for key in frames:
        gt = np.asarray(gts.get(key, np.zeros((0, 4))), dtype=np.float64).reshape(-1, 4)
        parts = grouped.get(key, [])
        if parts:
            pb = np.concatenate([x[0] for x in parts], axis=0)
            ps = np.concatenate([x[1] for x in parts], axis=0)
        else:
            pb = np.zeros((0, 4))
            ps = np.zeros(0)
        merged.append((key, pb, ps, gt))
        n_gt_total += len(gt)

    n_pred_total = int(sum(len(m[1]) for m in merged))
    res = DetEvalResult(n_gt=n_gt_total, n_pred=n_pred_total, n_frames=len(frames))
    if n_gt_total == 0:
        return res

    # ---- 多 IoU 阈值算 AP ----
    aps: dict[float, float] = {}
    for thr in iou_thresholds:
        tps, scores = [], []
        for _key, pb, ps, gt in merged:
            tp, _ = _match_frame(pb, ps, gt, thr)
            tps.append(tp)
            scores.append(ps)
        tp_all = np.concatenate(tps) if tps else np.zeros(0, dtype=bool)
        sc_all = np.concatenate(scores) if scores else np.zeros(0)
        order = np.argsort(-sc_all, kind="stable")
        aps[thr] = _ap_from_tp(tp_all[order], n_gt_total)

    res.map30 = aps.get(0.3, float("nan"))
    res.map50 = aps.get(0.5, float("nan"))
    rng = [t for t in iou_thresholds if 0.5 - 1e-9 <= t <= 0.95 + 1e-9]
    res.map50_95 = float(np.nanmean([aps[t] for t in rng])) if rng else float("nan")

    # ---- F1 / P / R（在 IoU=0.3 上扫置信度）----
    if conf_thr is not None:
        candidates = [float(conf_thr)]
    else:
        lo, hi, n = conf_scan
        candidates = list(np.linspace(lo, hi, n))

    best = (-1.0, float("nan"), float("nan"), float("nan"))
    for c in candidates:
        matched = 0
        n_keep = 0
        for _key, pb, ps, gt in merged:
            sel = ps >= c
            n_keep += int(sel.sum())
            if not sel.any() or len(gt) == 0:
                continue
            _tp, matched_gt = _match_frame(pb[sel], ps[sel], gt, 0.3)
            matched += int(matched_gt.sum())
        precision = matched / n_keep if n_keep else 1.0
        recall = matched / n_gt_total
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        res.conf_curve.append({"conf": float(c), "precision": precision,
                               "recall": recall, "f1": f1, "n_pred": n_keep})
        if f1 > best[0]:
            best = (f1, precision, recall, float(c))

    res.f1, res.precision, res.recall, res.conf = best

    # ---- 分尺寸召回（IoU=0.3，使用最优或给定阈值）----
    use_conf = res.conf if conf_thr is None else float(conf_thr)
    hit_per_bin = np.zeros(len(SIZE_BINS), dtype=np.int64)
    tot_per_bin = np.zeros(len(SIZE_BINS), dtype=np.int64)
    for _key, pb, ps, gt in merged:
        if len(gt) == 0:
            continue
        bins = size_bin_of(gt)
        tot_per_bin += np.bincount(bins, minlength=len(SIZE_BINS))
        sel = ps >= use_conf
        if not sel.any():
            continue
        _tp, matched_gt = _match_frame(pb[sel], ps[sel], gt, 0.3)
        hit_per_bin += np.bincount(bins[matched_gt], minlength=len(SIZE_BINS))

    for i, (_, _, name) in enumerate(SIZE_BINS):
        res.recall_by_size[name] = (float(hit_per_bin[i]) / tot_per_bin[i]
                                    if tot_per_bin[i] else float("nan"))

    # ---- 中心距 <= 2px 召回（小目标最贴近"看到没有"的指标）----
    hit = 0
    total = 0
    for _key, pb, ps, gt in merged:
        if len(gt) == 0:
            continue
        total += len(gt)
        sel = ps >= use_conf
        if not sel.any():
            continue
        p = pb[sel]
        pc = np.stack([(p[:, 0] + p[:, 2]) / 2, (p[:, 1] + p[:, 3]) / 2], 1)
        gc = np.stack([(gt[:, 0] + gt[:, 2]) / 2, (gt[:, 1] + gt[:, 3]) / 2], 1)
        d = np.linalg.norm(gc[:, None, :] - pc[None, :, :], axis=2)
        hit += int((d.min(axis=1) <= 2.0).sum())
    res.recall_center2px = hit / total if total else float("nan")
    return res
