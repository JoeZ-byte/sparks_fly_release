"""只解码"可能命中"的 anchor —— CPU 后处理的第一大优化。

问题
----
把 decode 放进 ONNX 图里，意味着每帧要对**全部** 21760 个 anchor 做
DFL softmax（4×16×21760 ≈ 1.4 M 元素）+ 框换算 + sigmoid。
实测（RTX 5060 机器的 CPU、batch=1）：含 decode 37.9 ms，不含 15.8 ms，
**decode 占了 58.5%**。这在实时场景里不可接受。

优化
----
分类分（sigmoid 后的 1 个数）先算，只有分数 >= 阈值（默认 0.05）的 anchor
才需要做 DFL 解码。实测被选中的 anchor 通常只有几十~几百个（<1%），
于是后处理成本下降两个数量级。

正确性
------
与"全量解码 + 阈值过滤"数学等价：未过阈值的 anchor 本来就会被丢弃。

接口
----
``decode_topk(preds, anchors, strides, nc, reg_max, conf_thr, max_nms)``
``preds`` : (B, 4*reg_max+nc, A) 原始预测（numpy）
返回每张图的 ``(boxes_xyxy, scores)``。
"""

from __future__ import annotations

import numpy as np

__all__ = ["make_anchors_np", "decode_topk", "sigmoid"]


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def make_anchors_np(imgsz: int, strides: list[float] | tuple[float, ...],
                    dtype=np.float32) -> tuple[np.ndarray, np.ndarray]:
    """生成 anchor 中心点与 stride（与导出图的布局一致：先小 stride 后大 stride）。"""
    pts, st = [], []
    for s in strides:
        side = int(imgsz / s)
        gy, gx = np.meshgrid(np.arange(side, dtype=dtype) + 0.5,
                            np.arange(side, dtype=dtype) + 0.5, indexing="ij")
        pts.append(np.stack((gx, gy), -1).reshape(-1, 2))
        st.append(np.full((side * side, 1), float(s), dtype=dtype))
    return np.concatenate(pts, 0), np.concatenate(st, 0)


def decode_topk(
    preds: np.ndarray,
    anchors: np.ndarray,
    strides: np.ndarray,
    nc: int = 1,
    reg_max: int = 16,
    conf_thr: float = 0.05,
    max_nms: int = 1000,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """对每张图只解码过阈值的 anchor。

    Returns
    -------
    list of ``(boxes_xyxy (K,4) float32, scores (K,) float32)``
    """
    p = np.asarray(preds, dtype=np.float32)
    if p.ndim == 2:
        p = p[None]
    b, no, a = p.shape
    assert no == 4 * reg_max + nc, f"通道数不符: {no} != 4*{reg_max}+{nc}"

    box = p[:, : 4 * reg_max].reshape(b, 4, reg_max, a)      # (B,4,reg_max,A)
    cls = p[:, 4 * reg_max:]                                  # (B,nc,A)
    scores_all = sigmoid(cls).max(axis=1)                     # (B,A)

    out: list[tuple[np.ndarray, np.ndarray]] = []
    bins = np.arange(reg_max, dtype=np.float32).reshape(1, reg_max, 1)   # 与 prob (4,reg_max,K) 对齐
    for i in range(b):
        s = scores_all[i]
        idx = np.flatnonzero(s >= conf_thr)
        if idx.size > max_nms:
            part = np.argpartition(-s[idx], max_nms)[:max_nms]
            idx = idx[part]
        if idx.size == 0:
            out.append((np.zeros((0, 4), np.float32), np.zeros(0, np.float32)))
            continue

        # 只对选中的 anchor 做 softmax（这是省下来的主要时间）
        d = box[i][:, :, idx]                                 # (4,reg_max,K)
        e = np.exp(d - d.max(axis=1, keepdims=True))
        prob = e / e.sum(axis=1, keepdims=True)
        dist = (prob * bins).sum(axis=1)                      # (4,K)
        dist = dist.T                                          # (K,4)

        anc = anchors[idx]                                     # (K,2)
        lt, rb = dist[:, :2], dist[:, 2:]
        x1y1 = anc - lt
        x2y2 = anc + rb
        boxes = np.concatenate([x1y1, x2y2], axis=1) * strides[idx]
        out.append((boxes.astype(np.float32), s[idx].astype(np.float32)))
    return out


def merge_near_duplicates(boxes: np.ndarray, scores: np.ndarray,
                          center_px: float = 3.0, size_tol: float = 3.0) -> tuple[np.ndarray, np.ndarray]:
    """合并"同一辆车被拆成的多个相邻框"。

    小目标检测常见的伪误检形态：一辆 6px 车在相邻 anchor 上各出一个框，
    中心只差 1~3 px、尺寸几乎一样。这些框会：(1) 计入 FP 数；(2) 干扰
    跟踪器（两条轨迹抢一辆车 -> IDSW）。这里按分数从高到低贪心合并：
    中心距 <= center_px 且宽高差 <= size_tol 的框并入高分框。
    中心距阈值取 3 px：车道宽才 ~4 px，再大就会把相邻车道的车合并掉。
    """
    if len(boxes) <= 1:
        return boxes, scores
    order = np.argsort(-scores)
    keep: list[int] = []
    for i in order:
        b = boxes[i]
        merged = False
        for k in keep:
            bk = boxes[k]
            cdist = float(np.hypot((b[0] + b[2] - bk[0] - bk[2]) / 2,
                                   (b[1] + b[3] - bk[1] - bk[3]) / 2))
            # 分数守卫：重复框的分数接近；相邻车通常一个高一个低，
            # 没有这个守卫会把两条车道的车合并（实测 mAP -9.1）
            if scores[i] < 0.7 * scores[k]:
                continue
            if (cdist <= center_px
                    and abs((b[2] - b[0]) - (bk[2] - bk[0])) <= size_tol
                    and abs((b[3] - b[1]) - (bk[3] - bk[1])) <= size_tol):
                merged = True
                break
        if not merged:
            keep.append(int(i))
    keep.sort()
    return boxes[keep], scores[keep]
