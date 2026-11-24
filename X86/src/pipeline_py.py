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
import os
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
# 2.4 PyTorch GPU 后端（有 NVIDIA 显卡时最快）
# =====================================================================
# 为什么不用 onnxruntime-gpu：ORT 的 CUDA EP 版本与 CUDA 运行时强绑定
# （1.31/1.32 要 CUDA 13，而本机装的是 CUDA 12.8），装不匹配就直接静默回退
# 到 CPU。PyTorch 的 CUDA 12.8 wheel 原生支持 Blackwell（sm_120），
# 且模型本来就是 PyTorch 训练的，直接加载 .pt 最省事。
class TorchBackend:
    """在 CUDA 上跑原始预测（批量），解码仍交给 numpy。"""

    def __init__(self, ts_path: Path, imgsz: int = 512, use_half: bool = True,
                 device: str = "cuda", batch_hint: int = 6) -> None:
        """加载 TorchScript 模型（纯 torch 即可，不需要项目里的 satdet 包）。"""
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("没有可用的 CUDA 设备")

        self.torch = torch
        self.device = torch.device(device)
        self.net = torch.jit.load(str(ts_path), map_location=self.device).eval()
        self.half = bool(use_half) and self.device.type == "cuda"
        if self.half:
            self.net.half()
        self.imgsz = imgsz
        self.anchors, self.strides = make_anchors()
        # 固定输入缓冲区：每帧新建张量会让 TorchScript 反复重建执行计划
        # （实测前 10 次每次卡约 2 秒）。复用同一个张量后稳定在 6 ms。
        self.batch_hint = max(1, batch_hint)
        self._buf = torch.zeros(self.batch_hint, 3, imgsz, imgsz,
                                device=self.device,
                                dtype=torch.float16 if self.half else torch.float32)
        # 上传原始 uint8 图块（196 KB/块），在 GPU 上做缩放与归一化：
        # 比在 CPU 上 resize+astype 再传 1.5 MB 快得多（实测 24.8 ms -> 2 ms）
        self._u8 = np.empty((self.batch_hint, TILE, TILE, 3), np.uint8)
        self._reg = torch.arange(REG_MAX, device=self.device)
        a, s = make_anchors()
        self._anchors_t = torch.from_numpy(np.asarray(a, np.float32)).to(self.device)
        self._strides_t = torch.from_numpy(np.asarray(s, np.float32)).to(self.device)
        # 预热：按目标批量跑几次，把 cuDNN/TorchScript 的首次编译与
        # 算法选择全部挪到启动阶段（否则前几帧会出现 2~6 秒的停顿）
        with torch.no_grad():
            for _ in range(3):
                self.net(self._buf)
        torch.cuda.synchronize()

    def infer(self, crops: list[np.ndarray], conf: float):
        """一批图块 -> [(boxes, scores)]（图块内坐标，已换算回 TILE 尺度）。

        全流程都压在 GPU 上：uint8 上传 -> GPU 缩放/归一化 -> 前向 ->
        GPU 上 top-k + DFL 解码 -> 只把最终框传回 CPU。
        """
        if not crops:
            return []
        torch = self.torch
        n = min(len(crops), self.batch_hint)
        for i in range(n):
            self._u8[i] = crops[i]                       # 196 KB/块
        raw = torch.from_numpy(self._u8[:n]).to(self.device, non_blocking=True)
        with torch.no_grad():
            x = raw.permute(0, 3, 1, 2).float().div_(255.0).flip(1)   # BGR->RGB
            x = torch.nn.functional.interpolate(
                x, size=(self.imgsz, self.imgsz), mode="bilinear",
                align_corners=False).contiguous()   # 模型里有 view()，必须连续
            if self.half:
                x = x.half()
            self._buf[:n].copy_(x)          # 固定缓冲区：避免每帧重建执行计划
            preds = self.net(self._buf)     # (n, 4*reg_max+nc, A)
            boxes, scores = self._decode(preds, conf)
        b_all = (boxes / (self.imgsz / TILE)).float().cpu().numpy()
        s_all = scores.float().cpu().numpy()
        return [(b_all[i][s_all[i] > conf], s_all[i][s_all[i] > conf]) for i in range(n)]

    def _decode(self, preds, conf: float, k: int = 160):
        """GPU 上的 top-k + DFL 解码，只回传少量候选。"""
        torch = self.torch
        n = preds.shape[0]
        cls = preds[:, REG_MAX * 4, :]                      # (n, A)
        kk = min(k, cls.shape[1])
        top_s, top_i = torch.topk(cls, kk, dim=1)
        box = preds[:, :REG_MAX * 4, :].gather(
            2, top_i.unsqueeze(1).expand(-1, REG_MAX * 4, -1))
        box = box.view(n, 4, REG_MAX, kk)
        prob = torch.softmax(box, dim=2)
        dist = (prob * self._reg.view(1, 1, -1, 1)).sum(2)   # (n,4,kk)
        a = self._anchors_t[top_i].permute(0, 2, 1)         # (n,2,kk)
        st = self._strides_t[top_i].unsqueeze(1)            # (n,1,kk)
        lt = dist[:, :2] * st                               # (n,2,kk)
        rb = dist[:, 2:] * st
        boxes = torch.cat([a - lt, a + rb], dim=1).permute(0, 2, 1)   # (n,kk,4)
        return boxes, top_s


# =====================================================================
# 2.5 多进程图块推理池
# =====================================================================
# 为什么用多进程而不是多线程：Python 有 GIL，线程只能等 C++ 内核放锁；
# 而 ONNX Runtime 单会话在 4 线程时每块 ~36 ms 就到顶（再开线程反而慢，
# 属于内存带宽受限）。所以真正的并行只能靠"多个进程各自开一个会话"。
# 实测：4 进程 x 4 线程，6 块/帧可以稳在 20+ FPS（单进程同配置只有 4.6）。
_W_SESS = None
_W_IMSZ = 512
_W_CONF = 0.35
_W_ANCHORS = None
_W_STRIDES = None


def _worker_init(model_path: str, threads: int, imgsz: int, conf: float) -> None:
    """子进程初始化：建自己的 ORT 会话（只做一次，之后常驻）。"""
    global _W_SESS, _W_IMSZ, _W_CONF, _W_ANCHORS, _W_STRIDES
    import onnxruntime as _ort

    so = _ort.SessionOptions()
    so.graph_optimization_level = _ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = max(1, threads)
    so.inter_op_num_threads = 1
    so.add_session_config_entry("session.intra_op.allow_spinning", "0")
    _W_SESS = _ort.InferenceSession(model_path, so, providers=["CPUExecutionProvider"])
    _W_ANCHORS, _W_STRIDES = make_anchors()
    _W_IMSZ = imgsz
    _W_CONF = conf


def _worker_infer(crop: np.ndarray):
    """子进程里做：缩放 → 归一化 → 推理 → 解码，只回传框（很小）。"""
    x = _prep_tile(crop, _W_IMSZ)
    preds = _W_SESS.run(None, {"batch": x[None]})[0]
    b, s = decode_topk(preds, _W_ANCHORS, _W_STRIDES, _W_CONF)
    return b / (_W_IMSZ / TILE), s


class TilePool:
    """进程池：一帧的所有图块并行推理（结果顺序与输入一致）。"""

    def __init__(self, model_path: Path, workers: int, threads: int, imgsz: int,
                 conf: float) -> None:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor

        self._ex = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=mp.get_context("spawn"),
            initializer=_worker_init,
            initargs=(str(model_path), threads, imgsz, conf),
        )
        self.workers = workers
        # 预热：让子进程把会话建好，避免第一帧卡几秒
        warm = np.zeros((TILE, TILE, 3), np.uint8)
        list(self._ex.map(_worker_infer, [warm] * workers))

    def infer(self, crops: list[np.ndarray]):
        if not crops:
            return []
        return list(self._ex.map(_worker_infer, crops))

    def close(self) -> None:
        self._ex.shutdown(wait=False, cancel_futures=True)


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
        # 每格上次被检查的帧号：没看过的格子自动加分，保证全场轮转覆盖
        ny = max(1, int(math.ceil(height / TILE)))
        nx = max(1, int(math.ceil(width / TILE)))
        self.last_seen = np.zeros((ny, nx), np.float32)
        self.frame = 0

    def _mask(self, gray: np.ndarray) -> np.ndarray:
        """运动掩膜。半分辨率计算：格子只用来投票，不需要全分辨率精度。"""
        small = cv2.resize(gray, (self.w // 2, self.h // 2), interpolation=cv2.INTER_AREA)
        if self.bg is None:
            self.bg = small.astype(np.float32)
            return np.zeros_like(small, np.uint8)
        diff = cv2.absdiff(small, self.bg.astype(np.uint8))
        mask = (diff > self.thr).astype(np.uint8) * 255
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)

    def _cells(self, mask: np.ndarray) -> np.ndarray:
        ny = max(1, int(math.ceil(self.h / TILE)))
        nx = max(1, int(math.ceil(self.w / TILE)))
        m = (mask > 0).astype(np.float32)
        small = cv2.resize(m, (nx, ny), interpolation=cv2.INTER_AREA)
        return np.rint(small * (m.shape[0] / ny) * (m.shape[1] / nx)).astype(np.int64)

    def propose(self, gray: np.ndarray, track_boxes: np.ndarray,
                track_bonus: float = 0.6) -> np.ndarray:
        """选本帧要推理的图块：运动最激烈的 K 个格子 + 有轨迹的格子，总数硬上限 K。

        之前的实现会"每有一条轨迹就加一个图块"，密集场景直接炸到 42 块/帧
        （1024² 上每块 36 ms，一帧就是 1.5 秒）。现在改成：
        1. 运动得分归一化到 0~1；
        2. 轨迹中心所在格子加一个先验分（保证慢车/静止车不漏）；
        3. 只取全局前 ``max_cells`` 个格子——**总图块数恒等于预算**。
        """
        mask = self._mask(gray)
        scores = self._cells(mask).astype(np.float32)
        mx = float(scores.max()) if scores.size else 0.0
        if mx > 0:
            scores = scores / mx
        for b in track_boxes:
            cx = int(np.clip((b[0] + b[2]) * 0.5, 0, self.w - 1)) // TILE
            cy = int(np.clip((b[1] + b[3]) * 0.5, 0, self.h - 1)) // TILE
            scores[cy, cx] += track_bonus
        # 陈旧度：越久没检查的格子分越高（8 帧封顶），保证全画面轮转覆盖。
        # 这是"边缘漏检"的主因修复：边缘格子运动少，原来永远排不进 top-K。
        self.frame += 1
        stale = np.clip((self.frame - self.last_seen) / 8.0, 0.0, 1.5)
        scores = scores + 0.9 * stale
        flat = scores.ravel()
        k = min(self.max_cells, flat.size)
        order = np.argsort(-flat)[:k]
        nx = scores.shape[1]
        picked = [(int(i) % nx * TILE, int(i) // nx * TILE) for i in order]
        for i in order:
            self.last_seen[int(i) // nx, int(i) % nx] = self.frame
        self.update_bg(gray)          # 背景统一在半分辨率上推进
        return np.asarray(picked, np.int32).reshape(-1, 2)

    def update_bg(self, gray: np.ndarray) -> None:
        """只推进背景模型（非检测帧用，保持掩膜时序一致）。"""
        g = cv2.resize(gray, (self.w // 2, self.h // 2),
                       interpolation=cv2.INTER_AREA).astype(np.float32)
        if self.bg is None or self.bg.shape != g.shape:
            self.bg = g.copy()
            return
        cv2.accumulateWeighted(g, self.bg, self.alpha)

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
                 gate_norm_max: float = 1.0, display_coast: int = 2):
        self.min_hits, self.max_age, self.gate = min_hits, max_age, gate_norm_max
        self.display_coast = display_coast
        self.tracks: list[_Track] = []
        self.dead: list[tuple] = []
        self.next_id = 1
        self.frame = 0

    def seed(self, items: list[tuple]) -> None:
        """用"回溯"得到的 (id, box, score, vx, vy) 播种轨迹。

        速度要**取负**：回溯是倒着想时间跑的，其速度方向与实际相反；
        不播种速度的话，轨迹会被当成静止目标（卡尔曼预测不动 → 测速≈0 →
        被静止门限过滤），表现就是"头几帧没有框"。
        """
        self.tracks = []
        self.dead = []
        top = 0
        for item in items:
            tid, box, score = item[0], item[1], item[2]
            vx = float(item[3]) if len(item) > 3 else 0.0
            vy = float(item[4]) if len(item) > 4 else 0.0
            tk = _Track(np.asarray(box, np.float32), float(score))
            tk.kf.x[4, 0] = -vx          # 反向 -> 正向
            tk.kf.x[5, 0] = -vy
            tk.pid = int(tid)
            tk.hits = max(self.min_hits, 1)      # 直接视为已确认
            tk.tsu = 0
            self.tracks.append(tk)
            top = max(top, int(tid))
        self.next_id = top + 1
        self.frame = 0

    def coast(self) -> list[_Track]:
        """不做检测的帧：只让卡尔曼外推一帧，用于检测周期化。

        实测帧间位移只有 2~3 px，隔帧检测 + 外推的框位置误差远小于目标尺寸，
        所以用它换吞吐是安全的（项目里 N=4 时单帧 60.4 -> 16.3 ms）。
        """
        self.frame += 1
        for t in self.tracks:
            t.predict()
        self.tracks = [t for t in self.tracks if t.tsu <= self.max_age]
        return self.tracks

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
        # 输出所有已确认轨迹：tsu<=3 允许短时外推（车辆被遮挡/本帧没覆盖到的格子时
        # 仍按卡尔曼预测显示，播放时不会"闪没"；超过 3 帧才隐藏，避免幽灵框）。
        return [t for t in self.tracks if t.pid is not None
                and t.tsu <= self.display_coast]

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
                # ready=False 表示速度还没测出来（历史不足 4 帧），此时不应该
                # 套用"静止目标不画"的门限，否则新出现的车会整帧消失
                "ready": px is not None,
                "box": t.box,
            }
        return out


# =====================================================================
# 6. 主流程
# =====================================================================
def draw(frame: np.ndarray, spd: dict, k: float, fps: float,
         max_labels: int = 60, scale: float = 1.0) -> np.ndarray:
    """画框 + 标签。密集场景（>max_labels）只画框不写标签，省一半绘制耗时。"""
    if len(spd) > max_labels:
        spd = dict(sorted(spd.items(), key=lambda kv: -kv[1]["px"])[:max_labels])
        labels = False
    else:
        labels = True
    for tid, s in sorted(spd.items(), key=lambda kv: -kv[1]["px"]):
        x1, y1, x2, y2 = (int(round(v * scale)) for v in s["box"])
        px = s["px"]
        # 静止门限：连续低速目标不显示（交通流只关心在动的车）。
        # 但"速度还没测出来"的轨迹（刚出现/刚播种）必须先画出来，
        # 否则播放头几帧会是空的。
        if s.get("ready", True) and px < 0.2:
            continue
        if k > 0:
            kmh = 3.6 * px * k * fps
            label = f"#{tid} {kmh:.0f} km/h"
            color = (0, 165, 255) if kmh < 60 else (0, 255, 0) if kmh < 100 else (0, 0, 255)
        else:
            label = f"#{tid} {px:.1f} px/f"
            color = (0, 165, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        if labels:
            ty = max(14, y1 - 6)
            cv2.putText(frame, label, (x2 + 2, ty), cv2.FONT_HERSHEY_SIMPLEX,
                        0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def _prep_tile(crop: np.ndarray, imgsz: int) -> np.ndarray:
    """图块预处理：缩放 + BGR->RGB + 归一化，一次完成（blobFromImage 内部融合）。"""
    return cv2.dnn.blobFromImage(
        crop, scalefactor=1.0 / 255.0, size=(imgsz, imgsz),
        mean=None, swapRB=True, crop=False)[0]


def _infer_batch(sess, crops: list[np.ndarray], imgsz: int, conf: float,
                 anchors, strides):
    """一次 run 推理整批图块（INT8 下实测比逐块快约 18%）。"""
    if not crops:
        return []
    batch = np.stack([_prep_tile(c, imgsz) for c in crops])
    preds = sess.run(None, {"batch": batch})[0]
    scale = imgsz / TILE
    out = []
    for i in range(len(crops)):
        b, s = decode_topk(preds[i:i + 1], anchors, strides, conf)
        out.append((b / scale, s))
    return out


def prewarm_lookback(cap, prop, pool, sess, gpu, anchors, strides, args,
                     n_frames: int, show_black=None):
    """黑屏预热 + 倒序回溯：先正向算 N 帧，再反向跑跟踪，得到第 1 帧的车辆 ID。

    为什么要反向：正向从空轨迹开始，车要跑 3 帧才"转正"（min_hits=3），
    所以直接播放时前几帧是无号的。把 N 帧的检测结果**倒序**喂给跟踪器，
    ID 会从后往前传播，回到第 1 帧时每辆车都已经有号。
    """
    frames, dets = [], []
    for i in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prop.propose(gray, np.zeros((0, 4), np.float32))     # 推进背景模型
        origins = prop.warmup_grid() if i < 12 else prop.propose(
            gray, np.zeros((0, 4), np.float32))
        crops, rects, seen = [], [], set()
        for x0, y0 in origins:
            x0 = max(0, min(frame.shape[1] - TILE, int(x0)))
            y0 = max(0, min(frame.shape[0] - TILE, int(y0)))
            if (x0, y0) in seen:
                continue
            seen.add((x0, y0))
            crops.append(frame[y0:y0 + TILE, x0:x0 + TILE])
            rects.append((x0, y0))
        if gpu is not None:
            res = gpu.infer(crops, args.conf)
        elif pool is not None:
            res = pool.infer(crops)
        else:
            res = _infer_batch(sess, crops, args.imgsz, args.conf, anchors, strides)
        db, sb = [], []
        for (x0, y0), (b, s) in zip(rects, res):
            if len(b):
                db.append(b + np.asarray([x0, y0, x0, y0], np.float32))
                sb.append(s)
        if db:
            d = np.concatenate(db)
            s = np.concatenate(sb)
            keep = nms(d, s, 0.5)
            dets.append((d[keep], s[keep]))
        else:
            dets.append((np.zeros((0, 4), np.float32), np.zeros(0, np.float32)))
        if show_black is not None and i % 3 == 0:
            try:
                img = np.zeros((360, 640, 3), np.uint8)
                cv2.putText(img, f"initializing {i + 1}/{n_frames}", (40, 180),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow("satstream", img)
                cv2.waitKey(1)
            except cv2.error:
                pass

    # 反向跑跟踪，让 ID 从最后一帧往回流
    tr = Tracker()
    first_ids = []
    for i in range(len(dets) - 1, -1, -1):
        tr.update(dets[i][0], dets[i][1])
        if i == 0:
            first_ids = [(t.pid, t.box.copy(), t.score,
                          float(t.kf.vel[0]), float(t.kf.vel[1]))
                         for t in tr.tracks if t.pid is not None]
    return frames, first_ids


class _BatchJob:
    """在后台线程里跑一次"预处理 + 推理 + 解码 + NMS"，返回 (boxes, scores)。"""

    def __init__(self, crops, rects, args, anchors, strides, gpu, sess):
        self.crops, self.rects, self.args = crops, rects, args
        self.anchors, self.strides = anchors, strides
        self.gpu, self.sess = gpu, sess

    def __call__(self):
        a = self.args
        if self.gpu is not None:
            res = self.gpu.infer(self.crops, a.conf)
        elif self.sess is not None:
            res = _infer_batch(self.sess, self.crops, a.imgsz, a.conf,
                               self.anchors, self.strides)
        else:
            res = []
        dets, scores = [], []
        for (x0, y0), (b, s) in zip(self.rects, res):
            if len(b):
                dets.append(b + np.asarray([x0, y0, x0, y0], np.float32))
                scores.append(s)
        if not dets:
            return np.zeros((0, 4), np.float32), np.zeros(0, np.float32)
        d = np.concatenate(dets)
        s = np.concatenate(scores)
        keep = nms(d, s, 0.5)
        return d[keep], s[keep]


# =====================================================================
# 1.5 数据密封（AES-256-GCM，硬件加速）
# =====================================================================
# 卫星影像是涉密数据：输入流在落地/进内存前必须解密，输出结果落盘前必须加密。
# 这里用 cryptography（OpenSSL 底层）的 AES-GCM，在飞腾派上走 **ARMv8 Crypto
# Extensions**（/proc/cpuinfo 有 aes/pmull/sha2），实测：
#     硬件 1017 MB/s  vs  软件（OPENSSL_armcap=0）48 MB/s  → 21 倍
# 文件格式（流式，可边解密边喂解码器）：
#     MAGIC(8) | chunk_size(4) | { nonce(12) | ciphertext(chunk+16) } *
_MAGIC = b"SFLYSEAL"


def _aesgcm(key: bytes):
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except ImportError as exc:      # 板端未装 cryptography 时给出明确提示
        raise SystemExit(
            "需要 cryptography 才能用密封功能：pip install cryptography"
        ) from exc
    return AESGCM(key)


def seal_file(src: Path, dst: Path, key: bytes, chunk: int = 1 << 20) -> int:
    """把 src 加密成 dst（AES-256-GCM，分块流式）。返回明文字节数。"""
    import os
    import struct

    aead = _aesgcm(key)
    total = 0
    with open(src, "rb") as fin, open(dst, "wb") as fout:
        fout.write(_MAGIC)
        fout.write(struct.pack("<I", chunk))
        while True:
            data = fin.read(chunk)
            if not data:
                break
            nonce = os.urandom(12)
            ct = aead.encrypt(nonce, data, None)
            fout.write(nonce)
            fout.write(ct)
            total += len(data)
    return total


def unseal_file(src: Path, dst: Path, key: bytes) -> int:
    """把加密文件解密到 dst（内存文件系统优先，便于"解密→解码"零磁盘残留）。"""
    import struct

    aead = _aesgcm(key)
    total = 0
    with open(src, "rb") as fin, open(dst, "wb") as fout:
        magic = fin.read(8)
        if magic != _MAGIC:
            raise SystemExit("不是本系统产生的密封文件（magic 不匹配）")
        (chunk,) = struct.unpack("<I", fin.read(4))
        while True:
            nonce = fin.read(12)
            if len(nonce) < 12:
                break
            blob = fin.read(chunk + 16)
            if not blob:
                break
            plain = aead.decrypt(nonce, blob, None)
            fout.write(plain)
            total += len(plain)
    return total


def wipe_file(path: Path) -> None:
    """安全擦除：先用 0 覆写再删除（防止明文残留在磁盘/共享内存）。"""
    try:
        size = path.stat().st_size
        with open(path, "r+b") as fh:
            left = size
            zero = b"\0" * min(1 << 20, max(1, size))
            while left > 0:
                n = min(len(zero), left)
                fh.write(zero[:n])
                left -= n
            fh.flush()
        path.unlink()
    except FileNotFoundError:
        pass


class SealedVideoSource:
    """加密视频 -> 内存解密 -> OpenCV 解码（退出时擦除明文）。

    演示"专网加密下发"：地面站只发 .enc 文件，板子用硬件 AES 解密到
    内存文件系统 /dev/shm，再交给解码器；播放结束把明文覆写并删除。
    """

    def __init__(self, enc_path: Path, key: bytes, shm_dir: str = "/dev/shm") -> None:
        import os
        import time as _t

        self.enc_path = enc_path
        base = Path(shm_dir) if os.path.isdir(shm_dir) else Path(os.environ.get("TEMP", "/tmp"))
        self.plain_path = base / f"sfly_{os.getpid()}_{int(_t.time())}.mp4"
        t0 = _t.perf_counter()
        size = unseal_file(enc_path, self.plain_path, key)
        dt = _t.perf_counter() - t0
        mb = size / (1 << 20)
        print(f"  解密 {mb:.1f} MB -> {self.plain_path}（{mb / max(dt, 1e-9):.0f} MB/s，"
              f"AES-256-GCM 硬件路径）", flush=True)

    def open(self) -> "cv2.VideoCapture":
        return cv2.VideoCapture(str(self.plain_path))

    def close(self) -> None:
        wipe_file(self.plain_path)
        print(f"  明文已擦除: {self.plain_path.name}", flush=True)


class AsyncDetector:
    """把"检测"放到后台线程，显示循环不再等它。

    飞腾派上单帧检测要 300~1000 ms，同步跑的话显示帧率就是检测帧率（1~3 FPS）。
    用异步后：显示维持 10+ FPS（靠卡尔曼外推），检测以自己能跑的速度持续更新。
    """

    def __init__(self, fn) -> None:
        from concurrent.futures import ThreadPoolExecutor

        self._pool = ThreadPoolExecutor(max_workers=1)
        self._fn = fn
        self._future = None

    @property
    def busy(self) -> bool:
        return self._future is not None

    def submit(self, job) -> bool:
        if self._future is not None:
            return False                      # 上一帧还没算完，本帧跳过检测
        self._future = self._pool.submit(self._fn, job)
        return True

    def poll(self):
        if self._future is None or not self._future.done():
            return None
        fut, self._future = self._future, None
        try:
            return fut.result()
        except Exception:
            return None

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="卫星视频车辆流式识别（单文件版）")
    ap.add_argument("--input", default="", help="输入视频路径（与 --seal-input 二选一）")
    ap.add_argument("--seal-input", default="",
                    help="加密视频（.enc）：先硬件解密到内存再解码，退出时擦除明文")
    ap.add_argument("--seal-key", default="",
                    help="AES-256 密钥（64 位 hex）；密封功能必填")
    ap.add_argument("--seal-csv", default="", help="把跟踪结果加密写入该 .enc 文件")
    ap.add_argument("--seal-video", default="", help="把叠加视频加密写入该 .enc 文件")
    ap.add_argument("--model", default="model/a_p2_nwd_v3_raw_fp32.onnx")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--k", type=float, default=0.92, help="米/像素（0=未标定，只显示 px/帧）")
    ap.add_argument("--fps", type=float, default=10.0, help="源视频帧率（测速用）")
    ap.add_argument("--warmup", type=int, default=15, help="冷启动网格帧数")
    ap.add_argument("--prewarm", type=int, default=20,
                    help="黑屏预热帧数：先正算再倒序回溯，让第 1 帧就带稳定 ID（0=关闭）")
    ap.add_argument("--max-cells", type=int, default=6,
                    help="每帧最多推理的图块数（算力预算，越小越快）")
    ap.add_argument("--backend", choices=["auto", "onnx", "torch"], default="auto",
                    help="auto=有 CUDA 用 PyTorch GPU，否则 onnxruntime CPU")
    ap.add_argument("--no-int8", action="store_true", help="CPU 上不用 INT8 模型")
    ap.add_argument("--fp32", action="store_true",
                    help="GPU 上用 fp32（默认 fp16，更快；精度差异可忽略）")
    ap.add_argument("--tmodel", default="model/a_p2_nwd_v3_raw.ts",
                    help="GPU 后端用的 TorchScript 权重")
    ap.add_argument("--workers", type=int, default=1,
                    help="推理进程数（默认 1；实测多进程在 CPU 上无收益）")
    ap.add_argument("--imgsz", type=int, default=0,
                    help="图块送入网络的边长（384 更快、精度略降）")
    ap.add_argument("--threads", type=int, default=5,
                    help="ORT 线程数（8 是实测甜点；再开因内存带宽反而变慢）")
    ap.add_argument("--max-coast", type=int, default=2,
                    help="未更新轨迹最多外推显示多少帧（板端检测慢时应调大，如 15）")
    ap.add_argument("--play-fps", type=float, default=0.0,
                    help="显示节流目标帧率（0=不限速；异步模式建议设成视频帧率）")
    ap.add_argument("--async-detect", action=argparse.BooleanOptionalAction,
                    default=False,
                    help="检测放后台线程，显示用卡尔曼外推（慢机器上显著提升观感）")
    ap.add_argument("--det-every", type=int, default=1,
                    help="检测周期：N 表示每 N 帧检测一次（>1 会明显掉召回，仅供极限提速）")
    ap.add_argument("--auto-tiles", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="按实测帧率自动调整每帧图块数（默认开）")
    ap.add_argument("--target-fps", type=float, default=0.0,
                    help="自适应目标帧率（0=自动：GPU 25、CPU 12）")
    ap.add_argument("--min-cells", type=int, default=4,
                    help="自适应时每帧图块下限（4 是精度拐点，3 会明显掉召回）")
    ap.add_argument("--int8", action="store_true",
                    help="CPU 用 INT8 量化模型（约 2.3 倍提速，精度不降）")
    ap.add_argument("--save", default="", help="保存叠加结果视频（可选）")
    ap.add_argument("--csv", default="", help="把逐帧结果写成 CSV（评测用）")
    ap.add_argument("--profile", action="store_true", help="打印各阶段耗时")
    ap.add_argument("--display-scale", type=float, default=0.7)
    ap.add_argument("--no-show", action="store_true", help="不弹窗口（批量/无显示环境）")
    ap.add_argument("--max-frames", type=int, default=0, help="只处理前 N 帧（0=全部）")
    ap.add_argument("--loop", action="store_true", help="循环播放（演示用）")
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

    if not args.imgsz:
        args.imgsz = 512                 # 必须在建后端之前定，否则 GPU 会拿到 0
    backend_kind = args.backend
    if backend_kind == "auto":
        backend_kind = "onnx"
        try:
            import torch as _t

            if _t.cuda.is_available():
                backend_kind = "torch"
        except Exception:
            pass
    gpu = None
    if backend_kind == "torch":
        ts = Path(args.tmodel)
        if not ts.is_file():
            ts = here.parent / args.tmodel            # release/
        if not ts.is_file():
            ts = model_path.with_suffix(".ts")
        if not ts.is_file():
            print(f"找不到 GPU 用的 TorchScript: {ts}；回退到 onnxruntime CPU")
            backend_kind = "onnx"
        else:
            try:
                gpu = TorchBackend(ts, args.imgsz, not args.fp32, "cuda",
                                   batch_hint=max(1, args.max_cells))
                print(f"后端: PyTorch { 'fp16' if gpu.half else 'fp32'} @ CUDA "
                      f"({gpu.torch.cuda.get_device_name(0)})", flush=True)
            except Exception as exc:
                print(f"GPU 后端不可用（{type(exc).__name__}: {exc}）；回退 CPU")
                backend_kind = "onnx"

    n_cpu = args.threads if args.threads > 0 else min(5, os.cpu_count() or 4)
    # CPU 路径默认优先用 INT8 量化模型（同精度、约 2.3 倍提速）
    if backend_kind == "onnx" and not args.no_int8:
        # 优先用与 --imgsz 匹配的 INT8 模型（量化后的图尺寸是固定的）
        base = model_path.stem.replace("_fp32", "").replace("_448", "").replace("_384", "")
        for cand in (model_path.with_name(f"{base}_int8_{args.imgsz}.onnx"),
                     model_path.with_name(f"{base}_int8.onnx")):
            if cand.is_file():
                model_path = cand
                print(f"使用 INT8 量化模型: {cand.name}", flush=True)
                break
        else:
            if args.imgsz != 512:
                print(f"警告：没有 {args.imgsz} 尺寸的 INT8 模型，回退到 512", flush=True)
    # 进程数：自动取 核心数/threads（至少 1、最多 4）。每进程 4 线程是甜点，
    # 再多线程反而变慢（内存带宽），所以要靠"多进程"堆吞吐。
    default_workers = max(1, min(4, (os.cpu_count() or 4) // max(1, n_cpu)))
    workers = args.workers if args.workers > 0 else default_workers
    pool = None
    sess = anchors = strides = None
    if gpu is not None:
        pass                                  # GPU 后端已就绪，无需 CPU 推理器
    elif workers > 1:
        print(f"启动 {workers} 个推理进程（每个 {n_cpu} 线程）…", flush=True)
        pool = TilePool(model_path, workers, n_cpu, args.imgsz, args.conf)
    else:
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.intra_op_num_threads = n_cpu
        so.inter_op_num_threads = 1
        # 关键：关闭 ORT 线程自旋，否则 CPU 被抢占时会长时间空转
        so.add_session_config_entry("session.intra_op.allow_spinning", "0")
        sess = ort.InferenceSession(str(model_path), so, providers=["CPUExecutionProvider"])
        anchors, strides = make_anchors()

    if not args.seal_input and not args.input:
        print("需要 --input <视频> 或 --seal-input <加密视频>")
        return 1
    sealed_src = None
    if args.seal_input:
        if not args.seal_key:
            print("--seal-input 需要配合 --seal-key <64位hex>")
            return 1
        sealed_src = SealedVideoSource(Path(args.seal_input), bytes.fromhex(args.seal_key))
        cap = sealed_src.open()
    else:
        cap = cv2.VideoCapture(str(args.input))
    if not cap.isOpened():
        print(f"打不开输入: {args.seal_input or args.input}")
        return 1
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    prop = MotionProposer(W, H, thr=40.0, max_cells=args.max_cells)
    tracker = Tracker(display_coast=args.max_coast)
    speed = SpeedEstimator(args.k, args.fps)
    warm_grid = prop.warmup_grid()

    prof = {} if args.profile else None

    def _tick(name, t0):
        if prof is None:
            return
        prof[name] = prof.get(name, 0.0) + time.perf_counter() - t0
    det_every = max(1, args.det_every)
    target_fps = args.target_fps if args.target_fps > 0 else (25.0 if gpu is not None else 12.0)
    csv_fh = open(args.csv, "w", encoding="utf-8", newline="") if args.csv else None
    if csv_fh is not None:
        csv_fh.write("track_id,frame,x1,y1,x2,y2,px_per_frame,score\n")
    writer = None
    if args.save:
        writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"),
                                 args.fps, (W, H))

    # ---- 黑屏预热 + 倒序回溯：让播放第一帧起每辆车都有编号 ----
    if args.prewarm > 0:
        try:
            cv2.namedWindow("satstream", cv2.WINDOW_AUTOSIZE)
            img = np.zeros((360, 640, 3), np.uint8)
            cv2.putText(img, "initializing...", (40, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow("satstream", img)
            cv2.waitKey(1)
        except cv2.error:
            pass
        _frames, first_ids = prewarm_lookback(
            cap, prop, pool, sess, gpu, anchors, strides, args, args.prewarm,
            show_black=None if args.no_show else True)
        if first_ids:
            tracker.seed(first_ids)
            for _tid, _box, _sc, _vx, _vy in first_ids:
                speed.ewma[int(_tid)] = float(math.hypot(-_vx, -_vy))
            print(f"  回溯完成：第 1 帧已有 {len(first_ids)} 辆车带 ID", flush=True)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        prop = MotionProposer(W, H, thr=40.0, max_cells=prop.max_cells)
        speed = SpeedEstimator(args.k, args.fps)
        warm_grid = prop.warmup_grid()
        pace_fps = args.play_fps if args.play_fps > 0 else (args.fps if args.async_detect else 0.0)
    full_grid = prop.warmup_grid(stride=1)      # 全网格：开播头几帧认领所有播种轨迹

    pace_fps = args.play_fps if args.play_fps > 0 else (args.fps if args.async_detect else 0.0)
    full_grid = prop.warmup_grid(stride=1)
    full_claim = 0
    # 有回溯时不走冷启动网格：播种轨迹本身就是"车辆在哪"的先验，
    # 再用 4 格稀疏网格会在第 8~15 帧出现明显空窗（实测框数从 44 掉到 6）。
    warm_frames = 0 if args.prewarm > 0 else args.warmup
    async_det = AsyncDetector(lambda job: job()) if args.async_detect else None
    det_times: list[float] = []
    fno = 0
    t0 = time.perf_counter()
    window: list[float] = []
    win_ready = False
    show = not args.no_show
    print("运行中：按 q 退出。冷启动 %d 帧后切回运动提案。" % args.warmup)
    while True:
        ok, frame = cap.read()
        if not ok:
            if args.loop:                      # 演示模式：播完回到第一帧继续
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                prop = MotionProposer(W, H, thr=40.0, max_cells=args.max_cells)
                tracker = Tracker()
                speed = SpeedEstimator(args.k, args.fps)
                warm_grid = prop.warmup_grid()
                fno = 0
                t0 = time.perf_counter()
                continue
            break
        fno += 1
        if args.max_frames and fno > args.max_frames:
            break
        window.append(time.perf_counter())
        if pace_fps > 0:
            _due = t0 + fno / pace_fps
            _lag = _due - time.perf_counter()
            if _lag > 0:
                time.sleep(_lag)
        # ---- 自适应图块预算：按最近 10 帧的实际帧率调 K ----
        if (args.auto_tiles and async_det is None
                and fno % 20 == 0 and len(window) >= 21):
            span = window[-1] - window[-21]
            fps_now = 20.0 / span if span > 0 else 0.0
            if fps_now < target_fps * 0.8 and prop.max_cells > args.min_cells:
                prop.max_cells -= 1
                warm_grid = prop.warmup_grid()
                print(f"  [自动] {fps_now:.1f} FPS 偏低 -> 图块数降到 {prop.max_cells}", flush=True)
            elif fps_now > target_fps * 1.4 and prop.max_cells < 8:
                prop.max_cells += 1
                warm_grid = prop.warmup_grid()
                print(f"  [自动] {fps_now:.1f} FPS 富余 -> 图块数升到 {prop.max_cells}", flush=True)
        if len(window) > 31:
            window.pop(0)
        if fno % 10 == 1:
            inst = 1.0 / max(time.perf_counter() - window[-2], 1e-6) if len(window) > 1 else 0.0
            span = window[-1] - window[0]
            roll = (len(window) - 1) / span if span > 0 else 0.0
            det_fps = (len(det_times) - 1) / (det_times[-1] - det_times[0]) if len(det_times) > 1 else 0.0
            print(f"  帧 {fno} … 显示 {roll:.1f} FPS（瞬时 {inst:.1f}）  检测 {det_fps:.2f} 次/秒",
                  flush=True)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ---- 检测周期化：非检测帧只外推，不推理 ----
        if det_every > 1 and (fno % det_every) != 0 and tracker.tracks:
            prop.update_bg(gray)
            spd = speed.update(tracker.coast(), fno)
            frame = draw(frame, spd, args.k, args.fps)
            if writer is not None:
                writer.write(frame)
            if show:
                try:
                    cv2.imshow("satstream", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                except cv2.error:
                    show = False
            continue

        # ---- 提案 ----
        tb = np.asarray([t.box for t in tracker.tracks], np.float32).reshape(-1, 4)
        if fno <= full_claim:
            origins = full_grid          # 可选：头几帧全画面扫描
        elif fno <= warm_frames:
            origins = warm_grid
        else:
            origins = prop.propose(gray, tb)

        # ---- 裁图块（夹回画面内 + 去重）----
        crops, rects, seen = [], [], set()
        for x0, y0 in origins:
            x0 = max(0, min(W - TILE, int(x0)))
            y0 = max(0, min(H - TILE, int(y0)))
            if (x0, y0) in seen:       # 轨迹漂出画面后夹紧会撞到同一块，去重
                continue
            seen.add((x0, y0))
            crop = frame[y0:y0 + TILE, x0:x0 + TILE]
            if crop.shape[0] != TILE or crop.shape[1] != TILE:
                pad = np.full((TILE, TILE, 3), 114, np.uint8)
                pad[:crop.shape[0], :crop.shape[1]] = crop
                crop = pad
            crops.append(crop)
            rects.append((x0, y0))

        # ---- 异步检测：本帧只提交，结果稍后合并 ----
        if async_det is not None:
            det_res = async_det.poll()
            if det_res is not None:
                det_times.append(time.perf_counter())
                if len(det_times) > 10:
                    det_times.pop(0)
                d, s = det_res
                tracks = tracker.update(d, s)
                spd = speed.update(tracks, fno)
            else:
                tracks = tracker.coast()
                spd = speed.update(tracks, fno)
            if async_det.busy is False:
                # 复用本帧已算好的 origins/crops（不再二次 propose：避免背景模型重复推进）
                crops, rects, seen = [], [], set()
                for x0, y0 in origins:
                    x0 = max(0, min(W - TILE, int(x0)))
                    y0 = max(0, min(H - TILE, int(y0)))
                    if (x0, y0) in seen:
                        continue
                    seen.add((x0, y0))
                    c = frame[y0:y0 + TILE, x0:x0 + TILE]
                    if c.shape[0] != TILE or c.shape[1] != TILE:
                        pad = np.full((TILE, TILE, 3), 114, np.uint8)
                        pad[:c.shape[0], :c.shape[1]] = c
                        c = pad
                    crops.append(c)
                    rects.append((x0, y0))
                async_det.submit(_BatchJob(crops, rects, args, anchors, strides, gpu, sess))
            if writer is not None:
                writer.write(draw(frame, spd, args.k, args.fps))
            if show:
                try:
                    if not win_ready:
                        cv2.namedWindow("satstream", cv2.WINDOW_AUTOSIZE)
                        win_ready = True
                    cv2.imshow("satstream", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                except cv2.error:
                    show = False
            if csv_fh is not None:
                for tid, st in spd.items():
                    b = st["box"]
                    csv_fh.write(f"{tid},{fno},{b[0]:.2f},{b[1]:.2f},{b[2]:.2f},{b[3]:.2f},"
                                 f"{st['px']:.4f},{st.get('conf', 1.0):.4f}\n")
            continue

        # ---- 推理 + 解码 ----
        _ip0 = time.perf_counter()
        if gpu is not None:
            # 恒定批量：TorchScript 对每个 batch 形状都要重新准备执行计划，
            # 4 块/6 块来回切换会出现秒级卡顿（实测单帧 2~6 秒）。
            # 这里把不足的部分用已有图块补齐，形状恒定后再无抖动；
            # 重复图块产生的重复框会被最后的 NMS 去掉。
            target = max(1, args.max_cells)
            padded = list(crops)
            while padded and len(padded) < target:
                padded.append(padded[len(padded) % len(crops)])
            results = gpu.infer(padded, args.conf)[:len(crops)] if padded else []
        elif pool is not None:
            results = pool.infer(crops)
        else:
            results = _infer_batch(sess, crops, args.imgsz, args.conf, anchors, strides)

        _tick("infer", _ip0)
        dets, scores = [], []
        for (x0, y0), (b, s) in zip(rects, results):
            if len(b):
                b = b + np.asarray([x0, y0, x0, y0], np.float32)
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
        _tp0 = time.perf_counter()
        tracks = tracker.update(dets, scores)
        _tick("track", _tp0)
        _sp0 = time.perf_counter()
        spd = speed.update(tracks, fno)
        _tick("speed", _sp0)
        if csv_fh is not None:
            for tid, st in spd.items():
                b = st["box"]
                csv_fh.write(f"{tid},{fno},{b[0]:.2f},{b[1]:.2f},{b[2]:.2f},{b[3]:.2f},"
                             f"{st['px']:.4f},{st.get('conf', 1.0):.4f}\n")
        # 显示用缩略图（先缩放再画框：省掉全分辨率绘制 + 一次缩放）
        sc = args.display_scale
        if writer is None and sc != 1.0:
            disp = cv2.resize(frame, None, fx=sc, fy=sc,
                             interpolation=cv2.INTER_AREA)
            draw(disp, spd, args.k, args.fps, scale=sc)
        else:
            frame = draw(frame, spd, args.k, args.fps)
            disp = (frame if sc == 1.0 else
                    cv2.resize(frame, None, fx=sc, fy=sc,
                               interpolation=cv2.INTER_NEAREST))

        # 左上角叠加实时帧率 / 后端 / 图块数（现场演示自证）
        if len(window) > 2:
            span = window[-1] - window[-2]
            fps_show = 1.0 / span if span > 0 else 0.0
            tag = "GPU" if gpu is not None else "CPU"
            info = f"{fps_show:4.1f} FPS  {tag}  tiles={len(crops)}"
            cv2.rectangle(disp, (0, 0), (300, 26), (0, 0, 0), -1)
            cv2.putText(disp, info, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 255), 2, cv2.LINE_AA)
        if show:
            try:
                _dp0 = time.perf_counter()
                if not win_ready:
                    # WINDOW_AUTOSIZE + 与图像同尺寸：避免 imshow 每帧内部缩放
                    cv2.namedWindow("satstream", cv2.WINDOW_AUTOSIZE)
                    win_ready = True
                cv2.imshow("satstream", disp)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                _tick("display", _dp0)
            except cv2.error:
                show = False           # 无显示环境：自动退化为只存视频
                print("（检测到无显示环境，已切换为 --no-show 模式）", flush=True)
        if writer is not None:
            writer.write(frame)

    cap.release()
    if writer is not None:
        writer.release()
    if prof:
        print("  阶段耗时(ms/帧): " + "  ".join(
            f"{k}={v / max(1, fno) * 1000:.1f}" for k, v in
            sorted(prof.items(), key=lambda kv: -kv[1])), flush=True)
    if pool is not None:
        pool.close()
    if csv_fh is not None:
        csv_fh.close()

    # ---- 输出密封：CSV / 视频先落临时文件，再用硬件 AES 加密，最后擦除临时文件 ----
    if args.seal_key and (args.seal_csv or args.seal_video):
        key = bytes.fromhex(args.seal_key)
        for src_attr, dst_path, label in (
                (args.csv, args.seal_csv, "CSV"),
                (args.save, args.seal_video, "视频")):
            if not dst_path or not src_attr:
                continue
            src = Path(src_attr)
            if not src.is_file():
                continue
            _seal_t0 = time.perf_counter()
            n = seal_file(src, Path(dst_path), key)
            _seal_dt = time.perf_counter() - _seal_t0
            print(f"  {label}已加密: {dst_path}（{n/2**20:.1f} MB, "
                  f"{n/(1<<20)/max(_seal_dt,1e-9):.0f} MB/s，AES-256-GCM）", flush=True)
            wipe_file(src)

    if sealed_src is not None:
        sealed_src.close()
    try:
        cv2.destroyAllWindows()          # headless 版 OpenCV 没有 GUI，会抛错
    except cv2.error:
        pass
    dt = time.perf_counter() - t0
    print(f"完成 {fno} 帧，耗时 {dt:.1f}s（{fno / max(dt, 1e-6):.1f} FPS）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# =====================================================================
# 面向 release 的流水线封装（纯 Python 后端）
# ---------------------------------------------------------------------
# 与 C/C++ 加速库完全同构：白顶帽增强 -> 运动提案 -> 图块 -> INT8 推理
# -> DFL 解码 -> NMS -> 光流速度先验 -> 卡尔曼跟踪 -> 测速
# =====================================================================

TILE = 256
SPEED_K = 0.92      # 米/像素
SPEED_FPS = 10.0    # 帧率（速度换算用）


class PyPipeline:
    """纯 Python 版流水线。接口与 NativePipeline 一致：step(frame) -> list[dict]。"""

    def __init__(self, model, *, imgsz: int = 512, max_cells: int = 6, threads: int = 4,
                 conf: float = 0.2, flow_w: float = 1.0, pre_mode: int = 1,
                 min_hits: int = 5, gate: float = 0.7, max_coast: int = 8,
                 warmup: int = 0) -> None:
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.intra_op_num_threads = int(threads)
        so.inter_op_num_threads = 1
        so.add_session_config_entry("session.intra_op.allow_spinning", "0")
        self.sess = ort.InferenceSession(str(model), so, providers=["CPUExecutionProvider"])
        self.anchors, self.strides = make_anchors()
        self.imgsz = int(imgsz)
        self.max_cells = int(max_cells)
        self.conf = float(conf)
        self.flow_w = float(flow_w)
        self.pre_mode = int(pre_mode)
        self.warmup = int(warmup)
        self.prop = None
        self.tracker = None
        self.speed = None
        self.min_hits = int(min_hits)
        self.gate = float(gate)
        self.max_coast = int(max_coast)
        self.frame = 0
        self.prev_gray = None
        self.tiles = 0
        self.dets = 0

    # ---------------- 光流：块匹配（SAD），返回像素/帧 ----------------
    @staticmethod
    def _flow(prev_gray, gray, box, r: int = 6, pw: int = 5):
        cx = int((box[0] + box[2]) * 0.5)
        cy = int((box[1] + box[3]) * 0.5)
        H, W = gray.shape[:2]
        if cx - r - pw < 0 or cy - r - pw < 0 or cx + r + pw >= W or cy + r + pw >= H:
            return 0.0, 0.0
        patch = prev_gray[cy - pw:cy + pw + 1, cx - pw:cx + pw + 1].astype(np.int16)
        best = (1 << 30, 0, 0)
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                cand = gray[cy + dy - pw:cy + dy + pw + 1,
                            cx + dx - pw:cx + dx + pw + 1].astype(np.int16)
                sad = int(np.abs(cand - patch).sum())
                if sad < best[0]:
                    best = (sad, dx, dy)
        return float(best[1]), float(best[2])

    def _apply_flow(self, gray) -> None:
        for t in self.tracker.tracks:
            fx, fy = self._flow(self.prev_gray, gray, t.box)
            if abs(fx) + abs(fy) > 0.01:
                k = self.flow_w * 0.5
                t.kf.x[4, 0] = (1 - k) * t.kf.x[4, 0] + k * fx
                t.kf.x[5, 0] = (1 - k) * t.kf.x[5, 0] + k * fy

    # ---------------- 单帧 ----------------
    def step(self, frame) -> list:
        self.frame += 1
        H, W = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.prop is None:
            self.prop = MotionProposer(W, H, thr=40.0, max_cells=self.max_cells)
            self.tracker = Tracker(min_hits=self.min_hits, max_age=15,
                                   display_coast=self.max_coast)
            self.speed = SpeedEstimator(SPEED_K, SPEED_FPS)

        g = gray
        if self.pre_mode == 1:
            g = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, np.ones((3, 3), np.uint8))

        tb = np.asarray([t.box for t in self.tracker.tracks], np.float32).reshape(-1, 4)
        if self.frame <= self.warmup:
            self.prop.propose(g, np.zeros((0, 4), np.float32))
            self.prev_gray = gray
            return []
        origins = self.prop.propose(g, tb)

        crops, rects, seen = [], [], set()
        for x0, y0 in origins:
            x0 = max(0, min(W - TILE, int(x0)))
            y0 = max(0, min(H - TILE, int(y0)))
            if (x0, y0) in seen:
                continue
            seen.add((x0, y0))
            c = frame[y0:y0 + TILE, x0:x0 + TILE]
            if c.shape[0] != TILE or c.shape[1] != TILE:
                pad = np.full((TILE, TILE, 3), 114, np.uint8)
                pad[:c.shape[0], :c.shape[1]] = c
                c = pad
            crops.append(c)
            rects.append((x0, y0))

        dets, scores = [], []
        if crops:
            results = _infer_batch(self.sess, crops, self.imgsz, self.conf,
                                   self.anchors, self.strides)
            for (x0, y0), (b, s) in zip(rects, results):
                if len(b):
                    dets.append(b + np.asarray([x0, y0, x0, y0], np.float32))
                    scores.append(s)
        if dets:
            d = np.concatenate(dets)
            sc = np.concatenate(scores)
            keep = nms(d, sc, 0.5)
            d, sc = d[keep], sc[keep]
        else:
            d = np.zeros((0, 4), np.float32)
            sc = np.zeros(0, np.float32)

        if self.flow_w > 0 and self.prev_gray is not None and len(self.tracker.tracks):
            self._apply_flow(gray)
        self.prev_gray = gray

        tracks = self.tracker.update(d, sc)
        states = self.speed.update(tracks, self.frame)
        self.tiles = len(crops)
        self.dets = len(d)

        out = []
        for tid, st in states.items():
            b = st["box"]
            px = float(st["px"])
            out.append({"track_id": int(tid),
                        "box": (float(b[0]), float(b[1]), float(b[2]), float(b[3])),
                        "px": px,
                        "kmh": 3.6 * px * SPEED_K * SPEED_FPS,
                        "ready": bool(st.get("ready", True))})
        return out

    def close(self) -> None:
        self.sess = None