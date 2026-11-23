"""跟踪指标：MOTA / IDF1 / IDSW（"连续识别"的量化证据）。

口径
----
* 逐帧 IoU 匹配（阈值默认 0.3，理由同检测：目标只有 6 px）；
* MOTA = 1 - (FN + FP + IDSW) / |GT|
* IDF1 = 2·IDTP / (2·IDTP + IDFP + IDFN)，IDTP 来自**全局 ID 一一映射**的最优解
  （对 gt_id × pred_id 的共现计数矩阵做匈牙利匹配），这是标准 IDF1 做法；
* IDSW：同一真实目标被切换成另一个预测 ID 的次数。

只有 ``mot/gt.txt``（带轨迹 ID）的序列才能算这些指标；早期外部视频 只能用检测指标。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from satdet.eval.det_metrics import iou_matrix

__all__ = ["TrackEvalResult", "evaluate_tracking"]


@dataclass
class TrackEvalResult:
    mota: float = float("nan")
    idf1: float = float("nan")
    idsw: int = 0
    fn: int = 0
    fp: int = 0
    n_gt: int = 0
    n_pred: int = 0
    n_frames: int = 0
    id_precision: float = float("nan")
    id_recall: float = float("nan")
    motp: float = float("nan")          # 平均匹配 IoU
    per_frame: list[dict] = field(default_factory=list)

    def to_row(self, **extra: object) -> dict:
        row = {
            "MOTA": self.mota, "IDF1": self.idf1, "IDSW": self.idsw,
            "FN": self.fn, "FP": self.fp, "n_gt": self.n_gt, "n_pred": self.n_pred,
            "IDP": self.id_precision, "IDR": self.id_recall, "MOTP": self.motp,
            "n_frames": self.n_frames,
        }
        row.update(extra)
        return row


def _match_cost(gb: np.ndarray, pb: np.ndarray, mode: str,
                iou_thr: float, center_px: float) -> tuple[np.ndarray, np.ndarray]:
    """返回 (cost, valid) 两个 (G,P) 矩阵。

    ``mode="iou"``：代价 = -IoU，IoU < iou_thr 视为不可匹配（MOT 标准口径）。
    ``mode="center"``：代价 = 中心距，超过 ``center_px`` 视为不可匹配。
    6 px 的目标偏 2 px 时 IoU 只剩 0.5，IoU 口径会把大量"其实跟住了"的
    情况判成不匹配；中心距口径更贴近该数据的实际精度水平，两者同时报告。
    """
    if mode == "center":
        gc = np.stack([(gb[:, 0] + gb[:, 2]) / 2, (gb[:, 1] + gb[:, 3]) / 2], 1)
        pc = np.stack([(pb[:, 0] + pb[:, 2]) / 2, (pb[:, 1] + pb[:, 3]) / 2], 1)
        d = np.linalg.norm(gc[:, None, :] - pc[None, :, :], axis=2)
        return d, d <= center_px
    iou = iou_matrix(gb, pb)
    return -iou, iou >= iou_thr


def evaluate_tracking(
    preds: list[dict],
    gts: dict[int, dict[str, np.ndarray]],
    iou_thr: float = 0.3,
    match_mode: str = "iou",
    center_px: float = 3.0,
) -> TrackEvalResult:
    """评测跟踪结果。

    Parameters
    ----------
    preds : list[dict]
        每帧一条，须按帧号升序：``{"frame": int, "ids": (N,), "boxes": (N,4)}``。
    gts : dict[int, dict]
        ``{frame: {"ids": (M,), "boxes": (M,4)}}``。
    """
    res = TrackEvalResult()
    res.n_frames = len(preds)

    # gt_id -> pred_id 共现计数（IDF1 用）
    cooc: dict[tuple[int, int], int] = {}
    last_pred_of_gt: dict[int, int] = {}
    iou_sum = 0.0
    iou_cnt = 0

    for p in preds:
        f = int(p["frame"])
        pb = np.asarray(p["boxes"], np.float64).reshape(-1, 4)
        pid = np.asarray(p["ids"], np.int64).reshape(-1)
        g = gts.get(f)
        if g is None:
            res.fp += len(pb)
            res.n_pred += len(pb)
            continue
        gb = np.asarray(g["boxes"], np.float64).reshape(-1, 4)
        gid = np.asarray(g["ids"], np.int64).reshape(-1)
        res.n_gt += len(gb)
        res.n_pred += len(pb)

        tp = 0
        if len(gb) and len(pb):
            cost, valid = _match_cost(gb, pb, match_mode, iou_thr, center_px)
            rows, cols = linear_sum_assignment(cost)
            for r, c in zip(rows, cols):
                if not valid[r, c]:
                    continue
                tp += 1
                iou_sum += float(-cost[r, c]) if match_mode == "iou" else float(cost[r, c])
                iou_cnt += 1
                gt_i = int(gid[r])
                pr_i = int(pid[c])
                cooc[(gt_i, pr_i)] = cooc.get((gt_i, pr_i), 0) + 1
                if gt_i in last_pred_of_gt and last_pred_of_gt[gt_i] != pr_i:
                    res.idsw += 1
                last_pred_of_gt[gt_i] = pr_i

        res.fn += len(gb) - tp
        res.fp += len(pb) - tp

    res.motp = iou_sum / iou_cnt if iou_cnt else float("nan")
    res.mota = (1.0 - (res.fn + res.fp + res.idsw) / res.n_gt) if res.n_gt else float("nan")

    # ---- IDF1：全局 ID 一一映射 ----
    if cooc:
        gt_ids = sorted({k[0] for k in cooc})
        pr_ids = sorted({k[1] for k in cooc})
        gi = {v: i for i, v in enumerate(gt_ids)}
        pi = {v: i for i, v in enumerate(pr_ids)}
        m = np.zeros((len(gt_ids), len(pr_ids)), dtype=np.int64)
        for (g, p_), c in cooc.items():
            m[gi[g], pi[p_]] = c
        rows, cols = linear_sum_assignment(-m)
        idtp = int(m[rows, cols].sum())
        res.idf1 = 2.0 * idtp / (res.n_gt + res.n_pred) if (res.n_gt + res.n_pred) else float("nan")
        res.id_precision = idtp / res.n_pred if res.n_pred else float("nan")
        res.id_recall = idtp / res.n_gt if res.n_gt else float("nan")
    else:
        res.idf1 = 0.0 if (res.n_gt or res.n_pred) else float("nan")
        res.id_precision = 0.0
        res.id_recall = 0.0
    return res
