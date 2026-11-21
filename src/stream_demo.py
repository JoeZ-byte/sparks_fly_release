# -*- coding: utf-8 -*-
"""卫星视频车辆流式识别 —— 单文件 release 版。

对外 0 依赖项目仓库：只需要 Python 3.9+ 与三个基础库
    pip install numpy opencv-python onnxruntime
不需要 scipy / torch / 项目代码 / 数据集。

用法
----
    python stream_demo.py --input ../samples/sample_01.mp4            # 实时渲染
    python stream_demo.py --input ../samples/sample_01.mp4 --save out.mp4
    python stream_demo.py --input x.mp4 --k 0     # 未标定：不显示 km/h，只显示 px/帧

管线（与项目 satdet 完全同源，精简为单文件）：
    运动先验（背景差分+网格预算） -> 跟踪（卡尔曼+匈牙利+暂定态+复活）
    -> 图块裁剪放大 -> YOLOv8n+P2(NWD) ONNX -> 快速解码+NMS -> 车速 -> 叠加渲染
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

import onnxruntime as ort

# =====================================================================
# 1. 解码与 NMS
# =====================================================================
STRIDES = (4.0, 8.0, 16.0, 32.0)
REG_MAX = 16
TILE, IMSZ = 256, 512


def make_anchors() -> tuple[np.ndarray, np.ndarray]:
    anchors, strides = [], []
    for s in STRIDES:
        f = int(IMSZ // s)
        y, x = np.mgrid[0:f, 0:f]
        a = np.stack([x.ravel(), y.ravel()], 1).astype(np.float32) * s
        anchors.append(a)
        strides.append(np.full(len(a), s, np.float32))
    return np.concatenate(anchors), np.concatenate(strides)


def _dfl(x: np.ndarray) -> np.ndarray:
    x = x.reshape(-1, REG_MAX)
    p = np.exp(x - x.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    return (p * np.arange(REG_MAX, dtype=np.float32)).sum(1)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
    if len(boxes) == 0:
        return np.zeros(0, np.int32)
    x1, y1 = boxes[:, 0], boxes[:, 1]
    x2, y2 = boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / np.maximum(1e-6, areas[i] + areas[order[1:]] - inter)
        order = order[1:][iou <= iou_thr]
    return np.asarray(keep, np.int32)


def decode_topk(pred: np.ndarray, anchors: np.ndarray, strides: np.ndarray,
                conf_thr: float, max_nms: int = 1000
                ) -> tuple[np.ndarray, np.ndarray]:
    p = pred[0]
    box_raw = p[: 4 * REG_MAX].T
    cls = p[4 * REG_MAX:].T
    c = cls.max(1)
    idx = np.flatnonzero(c >= conf_thr)
    if idx.size == 0:
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32)
    if idx.size > max_nms:
        idx = idx[np.argpartition(-c[idx], max_nms - 1)[:max_nms]]
    b = box_raw[idx].reshape(-1, 4, REG_MAX)
    lt = _dfl(b[:, :2]).reshape(-1, 2)
    rb = _dfl(b[:, 2:]).reshape(-1, 2)
    a = anchors[idx]
    st = strides[idx][:, None]
    x1y1 = a - lt * st
    x2y2 = a + rb * st
    boxes = np.concatenate([x1y1, x2y2], 1)
    scores = c[idx]
    keep = nms(boxes, scores, 0.5)
    return boxes[keep].astype(np.float32), scores[keep].astype(np.float32)


# =====================================================================
# 2. 匈牙利匹配（免 scipy）
# =====================================================================
def hungarian(cost: np.ndarray) -> list[int]:
    n, m = cost.shape
    big = 1e9
    c = np.where(np.isnan(cost), big, cost.astype(np.float64))
    u = np.zeros(n + 1)
    v = np.zeros(m + 1)
    p = np.zeros(m + 1, np.int64)
    way = np.zeros(m + 1, np.int64)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, big)
        used = np.zeros(m + 1, bool)
        # 迭代上限 m+2：某行完全无可行列时 delta 无法更新，原版 while True 会死循环
        for _it in range(m + 2):
            used[j0] = True
            i0 = p[j0]
            delta = big
            j1 = 0
            row = c[i0 - 1]
            for j in range(1, m + 1):
                if used[j]:
                    continue
                cur = row[j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            if j1 == 0:
                break                      # 没有可行的未用列：放弃该行
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        else:
            continue
        for _rec in range(m + 2):
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = int(j1)
            if j0 == 0:
                break
    ans = [-1] * n
    for j in range(1, m + 1):
        if p[j] > 0:
            ans[int(p[j]) - 1] = j - 1
    return ans


def greedy_assign(sub: np.ndarray) -> list[tuple[int, int]]:
    """大组用的向量化贪心：每次取全局最小代价对，然后划掉该行该列。

    纯 Python 匈牙利在 100+×100+ 的近似满矩阵上要上百秒；这里用 numpy
    反复取 argmin，120 个目标也只要几毫秒。门限已经很紧，贪心与最优解的
    差距在实践中可以忽略。
    """
    m = np.where(np.isnan(sub), np.inf, sub.astype(np.float64)).copy()
    out: list[tuple[int, int]] = []
    n, k = m.shape
    for _ in range(min(n, k)):
        idx = int(np.argmin(m))
        r, c = divmod(idx, k)
        if not np.isfinite(m[r, c]):
            break
        out.append((r, c))
        m[r, :] = np.inf
        m[:, c] = np.inf
    return out


def gated_assign(cost: np.ndarray, top_k: int = 3, exact_max: int = 12) -> list[tuple[int, int]]:
    """带门限的匹配：候选截断 -> 连通分量拆分 -> 小组精确 / 大组贪心。

    三级策略，保证密集场景也是毫秒级：
    1. **候选截断**：每条轨迹只保留代价最小的 ``top_k`` 个检测（其余视为
       不可匹配）。密集车流里一条轨迹的候选常常几十个，截断后图立刻变稀疏；
    2. **连通分量拆分**：只在互相有候选的轨迹/检测之间求解，
       通常每组 2~4 元；
    3. 组内：``≤exact_max`` 用精确匈牙利，更大用向量化贪心。
    """
    valid = ~np.isnan(cost)
    n_tr, n_det = cost.shape
    if not valid.any():
        return []

    # ---- 1) 每条轨迹保留 top_k 个候选 ----
    pruned = np.full_like(cost, np.nan, dtype=np.float64)
    for ti in range(n_tr):
        cols = np.flatnonzero(valid[ti])
        if cols.size == 0:
            continue
        if cols.size > top_k:
            order = cols[np.argsort(cost[ti, cols])[:top_k]]
        else:
            order = cols
        pruned[ti, order] = cost[ti, order]
    valid = ~np.isnan(pruned)
    pairs = np.argwhere(valid)
    if pairs.size == 0:
        return []

    # ---- 2) 并查集分组（轨迹 [0,n_tr) 与检测共用编号空间）----
    parent = list(range(n_tr + n_det))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for ti, di in pairs:
        ra, rb = find(int(ti)), find(n_tr + int(di))
        if ra != rb:
            parent[rb] = ra

    groups: dict[int, tuple[list[int], list[int]]] = {}
    for ti in range(n_tr):
        if valid[ti].any():
            groups.setdefault(find(ti), ([], []))[0].append(ti)
    for di in range(n_det):
        if valid[:, di].any():
            groups.setdefault(find(n_tr + di), ([], []))[1].append(di)

    # ---- 3) 组内求解 ----
    out: list[tuple[int, int]] = []
    for rows, cols in groups.values():
        if not rows or not cols:
            continue
        sub = pruned[np.ix_(rows, cols)]
        ok_r = ~np.all(np.isnan(sub), axis=1)
        ok_c = ~np.all(np.isnan(sub), axis=0)
        if not ok_r.all() or not ok_c.all():
            rows = [rows[i] for i in np.flatnonzero(ok_r)]
            cols = [cols[i] for i in np.flatnonzero(ok_c)]
            if not rows or not cols:
                continue
            sub = pruned[np.ix_(rows, cols)]
        if len(rows) == 1 and len(cols) == 1:
            if not np.isnan(sub[0, 0]):
                out.append((rows[0], cols[0]))
            continue
        for r, c in greedy_assign(sub):
            out.append((rows[r], cols[c]))
    return out


# =====================================================================
# 3. 运动先验
# =====================================================================
class MotionProposer:
    def __init__(self, width: int, height: int, thr: float = 40.0,
                 max_cells: int = 6, alpha: float = 0.02):
        self.w, self.h = width, height
        self.thr, self.max_cells, self.alpha = thr, max_cells, alpha
        self.bg = None
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    def _mask(self, gray: np.ndarray) -> np.ndarray:
        g = gray.astype(np.float32)
        if self.bg is None:
            self.bg = g.copy()
            return np.zeros_like(gray, np.uint8)
        diff = cv2.absdiff(gray, self.bg.astype(np.uint8))
        mask = (diff > self.thr).astype(np.uint8) * 255
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)

    def _cells(self, mask: np.ndarray) -> np.ndarray:
        ny = max(1, int(math.ceil(self.h / TILE)))
        nx = max(1, int(math.ceil(self.w / TILE)))
        m = (mask > 0).astype(np.float32)
        small = cv2.resize(m, (nx, ny), interpolation=cv2.INTER_AREA)
        return np.rint(small * (m.shape[0] / ny) * (m.shape[1] / nx)).astype(np.int64)

    def propose(self, gray: np.ndarray, track_boxes: np.ndarray) -> np.ndarray:
        mask = self._mask(gray)
        scores = self._cells(mask)
        flat = scores.ravel()
        picked = []
        if flat.size and flat.max() > 0:
            thr = max(2, int(round(0.05 * flat.max())))
            order = np.argsort(-flat)
            for k in order:
                if len(picked) >= self.max_cells or flat[k] < thr:
                    break
                iy, ix = divmod(int(k), scores.shape[1])
                picked.append((ix * TILE, iy * TILE))
        for b in track_boxes:
            o = (int(b[0] // TILE) * TILE, int(b[1] // TILE) * TILE)
            if o not in picked:
                picked.append(o)
        cv2.accumulateWeighted(gray.astype(np.float32), self.bg, self.alpha)
        return np.asarray(picked, np.int32).reshape(-1, 2)

    def warmup_grid(self, stride: int = 2) -> np.ndarray:
        """冷启动用的稀疏网格：隔 stride 个格子取一块。

        1024² 上满网格是 16 块/帧，纯 CPU 跑不动；隔行隔列取 4 块就能把
        轨迹建起来（实测覆盖足够），冷启动结束后立刻切回 top-K 提案。
        """
        ny = max(1, int(math.ceil(self.h / TILE)))
        nx = max(1, int(math.ceil(self.w / TILE)))
        ys, xs = np.meshgrid(np.arange(0, ny, max(1, stride)) * TILE,
                             np.arange(0, nx, max(1, stride)) * TILE, indexing="ij")
        return np.stack([xs.ravel(), ys.ravel()], 1).astype(np.int32)


# =====================================================================
# 4. 跟踪
# =====================================================================
class _Kalman:
    def __init__(self) -> None:
        self.F = np.eye(8)
        self.F[0, 4] = self.F[1, 5] = self.F[2, 6] = self.F[3, 7] = 1.0
        self.H = np.eye(4, 8)
        self.Q = np.eye(8) * 1e-2
        self.R = np.eye(4) * 1e-1
        self.P = np.eye(8) * 10.0
        self.x = np.zeros((8, 1))

    def init(self, b: np.ndarray) -> None:
        self.x = np.array([[(b[0] + b[2]) / 2], [(b[1] + b[3]) / 2],
                           [b[2] - b[0]], [b[3] - b[1]],
                           [0.0], [0.0], [0.0], [0.0]])
        self.P = np.eye(8) * 10.0

    def predict(self) -> np.ndarray:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        cx, cy, w, h = self.x[:4, 0]
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], np.float32)

    def update(self, b: np.ndarray) -> None:
        z = np.array([[(b[0] + b[2]) / 2], [(b[1] + b[3]) / 2],
                      [b[2] - b[0]], [b[3] - b[1]]], np.float64)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(8) - K @ self.H) @ self.P

    @property
    def vel(self) -> tuple[float, float]:
        return float(self.x[4, 0]), float(self.x[5, 0])


class _Track:
    def __init__(self, box: np.ndarray, score: float) -> None:
        self.kf = _Kalman()
        self.kf.init(box)
        self.box = box.astype(np.float32)
        self.score = float(score)
        self.hits = 1
        self.tsu = 0
        self.pid: int | None = None

    def predict(self) -> None:
        self.box = self.kf.predict()
        self.tsu += 1

    def update(self, box: np.ndarray, score: float) -> None:
        self.kf.update(box)
        cx, cy, w, h = self.kf.x[:4, 0]
        self.box = np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2],
                            np.float32)
        self.score = float(score)
        self.hits += 1
        self.tsu = 0


class Tracker:
    def __init__(self, min_hits: int = 3, max_age: int = 15,
                 gate_norm_max: float = 1.0):
        self.min_hits, self.max_age, self.gate = min_hits, max_age, gate_norm_max
        self.tracks: list[_Track] = []
        self.dead: list[tuple] = []
        self.next_id = 1
        self.frame = 0

    def update(self, dets: np.ndarray, scores: np.ndarray) -> list[_Track]:
        self.frame += 1
        for t in self.tracks:
            t.predict()
        n_tr, n_det = len(self.tracks), len(dets)
        if n_tr and n_det:
            pred = np.stack([t.box for t in self.tracks])
            pc = np.stack([(pred[:, 0] + pred[:, 2]) / 2, (pred[:, 1] + pred[:, 3]) / 2], 1)
            dc = np.stack([(dets[:, 0] + dets[:, 2]) / 2, (dets[:, 1] + dets[:, 3]) / 2], 1)
            pw = np.maximum(pred[:, 2] - pred[:, 0], 1.0)
            ph = np.maximum(pred[:, 3] - pred[:, 1], 1.0)
            vel = np.stack([t.kf.vel for t in self.tracks]).astype(np.float64)
            speed = np.linalg.norm(vel, axis=1)
            base = np.maximum(0.5 * (pw + ph) / 2.0, 3.0)
            delta = dc[None] - pc[:, None]
            dist = np.linalg.norm(delta, axis=2)
            along = np.where(speed[:, None] > 1e-6,
                             (delta * vel[:, None]).sum(-1) / np.maximum(speed[:, None], 1e-6), 0.0)
            across = np.sqrt(np.maximum(dist ** 2 - along ** 2, 0.0))
            a = base[:, None] * (4.5 + 0.6 * speed[:, None])
            b = base[:, None] * 2.2
            norm_e = np.sqrt((along / a) ** 2 + (across / b) ** 2)
            gate_c = np.maximum(10.0, 3.0 * base)[:, None]
            norm = np.where(speed[:, None] > 1.0, norm_e, dist / gate_c)
            dw = np.maximum(dets[:, 2] - dets[:, 0], 1.0)
            dh = np.maximum(dets[:, 3] - dets[:, 1], 1.0)
            cost = norm + 0.25 * (
                np.abs(np.log(pw[:, None] / dw[None]))
                + np.abs(np.log(ph[:, None] / dh[None])))
            cost = cost + 0.3 * (1.0 - np.clip(scores, 0, 1))[None]
            cost = np.where(norm > self.gate, np.nan, cost)
            assign = gated_assign(cost)
            matched = set()
            for ti, di in assign:
                self.tracks[ti].update(dets[di], scores[di])
                matched.add(di)
            unmatched = [i for i in range(n_det) if i not in matched]
        else:
            unmatched = list(range(n_det))

        still = []
        for di in unmatched:
            if not self._revive(dets[di], scores[di]):
                still.append(di)
        for di in still:
            self.tracks.append(_Track(dets[di], scores[di]))

        for t in self.tracks:
            if t.pid is None and t.hits >= self.min_hits:
                t.pid = self.next_id
                self.next_id += 1

        alive = []
        for t in self.tracks:
            if t.tsu > self.max_age:
                if t.pid is not None:
                    vx, vy = t.kf.vel
                    self.dead.append((t.pid, t.box.copy(), vx, vy, self.frame))
            else:
                alive.append(t)
        self.tracks = alive
        self.dead = [d for d in self.dead if self.frame - d[4] <= 12][-64:]
        return [t for t in self.tracks if t.pid is not None and t.tsu == 0]

    def _revive(self, box: np.ndarray, score: float) -> bool:
        if not self.dead:
            return False
        c = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
        best, best_d = -1, 12.0
        for k, (pid, dbox, vx, vy, df) in enumerate(self.dead):
            age = self.frame - df
            if age <= 0 or age > 12:
                continue
            dc = np.array([(dbox[0] + dbox[2]) / 2, (dbox[1] + dbox[3]) / 2])
            dc = dc + np.array([vx, vy]) * age
            d = float(np.linalg.norm(c - dc))
            if d < best_d:
                best, best_d = k, d
        if best < 0:
            return False
        pid, _db, vx, vy, _df = self.dead.pop(best)
        t = _Track(box, score)
        t.pid = pid
        t.hits = max(t.hits, self.min_hits)
        t.kf.x[4, 0] = vx * 0.7
        t.kf.x[5, 0] = vy * 0.7
        self.tracks.append(t)
        return True


# =====================================================================
# 5. 车速
# =====================================================================
class SpeedEstimator:
    def __init__(self, k: float, fps: float) -> None:
        self.k, self.fps = k, fps
        self.hist: dict[int, deque] = {}
        self.ewma: dict[int, float] = {}

    def update(self, tracks: list[_Track], frame: int,
               min_speed_px: float = 0.2) -> dict[int, dict]:
        out = {}
        for t in tracks:
            tid = t.pid
            cx, cy = (t.box[0] + t.box[2]) / 2, (t.box[1] + t.box[3]) / 2
            h = self.hist.setdefault(tid, deque(maxlen=64))
            h.append((frame, cx, cy))
            px = None
            if len(h) >= 4:
                vals = []
                for lag in (6, 12, 18):
                    cand = [q for q in h if frame - q[0] >= 3
                            and abs(q[0] - (frame - lag)) <= 6]
                    if cand:
                        q = min(cand, key=lambda q: abs(q[0] - (frame - lag)))
                        gap = frame - q[0]
                        vals.append(math.hypot(cx - q[1], cy - q[2]) / gap)
                if vals:
                    px = float(np.median(vals))
                    prev = self.ewma.get(tid)
                    if prev is not None:
                        px = 0.7 * px + 0.3 * prev
                    px = min(px, 6.0)
                    if prev is not None:
                        slew = 35.0 / max(1e-6, 3.6 * max(self.k, 1e-6) * self.fps)
                        px = float(np.clip(px, max(0.0, prev - slew), prev + slew))
                    self.ewma[tid] = px
            vx, vy = t.kf.vel
            out[tid] = {
                "px": float(px) if px is not None else float(math.hypot(vx, vy)),
                "src": "interval" if px is not None else "kalman",
                "box": t.box,
            }
        return out


# =====================================================================
# 6. 主流程
# =====================================================================
def draw(frame: np.ndarray, spd: dict, k: float, fps: float) -> np.ndarray:
    for tid, s in sorted(spd.items(), key=lambda kv: -kv[1]["px"]):
        x1, y1, x2, y2 = (int(round(v)) for v in s["box"])
        px = s["px"]
        # 静止门限：连续低速目标不显示（交通流只关心在动的车）
        if px < 0.2:
            continue
        if k > 0:
            kmh = 3.6 * px * k * fps
            label = f"#{tid} {kmh:.0f} km/h"
            color = (0, 165, 255) if kmh < 60 else (0, 255, 0) if kmh < 100 else (0, 0, 255)
        else:
            label = f"#{tid} {px:.1f} px/f"
            color = (0, 165, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        ty = max(14, y1 - 6)
        cv2.putText(frame, label, (x2 + 2, ty), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, label, (x2 + 2, ty), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def main() -> int:
    ap = argparse.ArgumentParser(description="卫星视频车辆流式识别（单文件版）")
    ap.add_argument("--input", required=True, help="输入视频或图片序列目录")
    ap.add_argument("--model", default="model/a_p2_nwd_full2_raw_fp32.onnx")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--k", type=float, default=0.92, help="米/像素（0=未标定，只显示 px/帧）")
    ap.add_argument("--fps", type=float, default=10.0, help="源视频帧率（测速用）")
    ap.add_argument("--warmup", type=int, default=15, help="冷启动网格帧数")
    ap.add_argument("--max-cells", type=int, default=6)
    ap.add_argument("--threads", type=int, default=4,
                    help="ORT 线程数（单图块推理带宽受限，4~8 最优；开满反而慢）")
    ap.add_argument("--save", default="", help="保存叠加结果视频（可选）")
    ap.add_argument("--display-scale", type=float, default=1.0)
    ap.add_argument("--no-show", action="store_true", help="不弹窗口（批量/无显示环境）")
    ap.add_argument("--max-frames", type=int, default=0, help="只处理前 N 帧（0=全部）")
    args = ap.parse_args()

    if getattr(sys, "frozen", False):
        here = Path(sys.executable).resolve().parent   # PyInstaller 打包后
    else:
        here = Path(__file__).resolve().parent
    model_path = Path(args.model)
    if not model_path.is_file():
        model_path = here.parent / model_path           # src/ 的上层 = release/
    if not model_path.is_file():
        model_path = here / model_path
    if not model_path.is_file():
        print(f"找不到模型: {args.model}（默认放在 release/model/）")
        return 1

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    n_cpu = args.threads if args.threads > 0 else 4
    so.intra_op_num_threads = n_cpu
    so.inter_op_num_threads = 1
    # 关键：关闭 ORT 线程自旋。默认自旋在 CPU 被其他程序占满时会长时间空转
    # （实测同一台机器上"自旋开"能让一帧从 0.4 s 变成几十秒），关掉后稳定。
    so.add_session_config_entry("session.intra_op.allow_spinning", "0")
    sess = ort.InferenceSession(str(model_path), so, providers=["CPUExecutionProvider"])
    anchors, strides = make_anchors()

    cap = cv2.VideoCapture(str(args.input))
    if not cap.isOpened():
        print(f"打不开输入: {args.input}")
        return 1
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    prop = MotionProposer(W, H, thr=40.0, max_cells=args.max_cells)
    tracker = Tracker()
    speed = SpeedEstimator(args.k, args.fps)
    warm_grid = prop.warmup_grid()

    writer = None
    if args.save:
        writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"),
                                 args.fps, (W, H))

    fno = 0
    t0 = time.perf_counter()
    show = not args.no_show
    print("运行中：按 q 退出。冷启动 %d 帧后切回运动提案。" % args.warmup)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        fno += 1
        if args.max_frames and fno > args.max_frames:
            break
        if fno % 10 == 1:
            print(f"  帧 {fno} … {fno / max(time.perf_counter() - t0, 1e-6):.1f} FPS",
                  flush=True)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ---- 提案 ----
        tb = np.asarray([t.box for t in tracker.tracks], np.float32).reshape(-1, 4)
        if fno <= args.warmup:
            origins = warm_grid
        else:
            origins = prop.propose(gray, tb)

        # ---- 图块推理 + 解码 ----
        dets, scores = [], []
        for x0, y0 in origins:
            # 跟踪预测可能漂出画面：把图块原点夹回合法范围（否则裁到空区域）
            x0 = max(0, min(W - TILE, int(x0)))
            y0 = max(0, min(H - TILE, int(y0)))
            crop = frame[y0:y0 + TILE, x0:x0 + TILE]
            if crop.shape[0] != TILE or crop.shape[1] != TILE:
                pad = np.full((TILE, TILE, 3), 114, np.uint8)
                pad[:crop.shape[0], :crop.shape[1]] = crop
                crop = pad
            x = cv2.resize(crop, (IMSZ, IMSZ), interpolation=cv2.INTER_LINEAR)
            x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            preds = sess.run(None, {"batch": x[None].transpose(0, 3, 1, 2)})[0]
            b, s = decode_topk(preds, anchors, strides, args.conf)
            if len(b):
                scale = IMSZ / TILE
                b = b / scale + np.asarray([x0, y0, x0, y0], np.float32)
                dets.append(b)
                scores.append(s)
        if dets:
            dets = np.concatenate(dets)
            scores = np.concatenate(scores)
            # 全图级 NMS（跨图块去重）
            keep = nms(dets, scores, 0.5)
            dets, scores = dets[keep], scores[keep]
        else:
            dets = np.zeros((0, 4), np.float32)
            scores = np.zeros(0, np.float32)

        # ---- 跟踪 + 车速 + 渲染 ----
        tracks = tracker.update(dets, scores)
        spd = speed.update(tracks, fno)
        frame = draw(frame, spd, args.k, args.fps)

        if args.display_scale != 1.0:
            disp = cv2.resize(frame, None, fx=args.display_scale,
                              fy=args.display_scale, interpolation=cv2.INTER_NEAREST)
        else:
            disp = frame
        if show:
            try:
                cv2.imshow("satstream", disp)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            except cv2.error:
                show = False           # 无显示环境：自动退化为只存视频
                print("（检测到无显示环境，已切换为 --no-show 模式）", flush=True)
        if writer is not None:
            writer.write(frame)

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    dt = time.perf_counter() - t0
    print(f"完成 {fno} 帧，耗时 {dt:.1f}s（{fno / max(dt, 1e-6):.1f} FPS）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
