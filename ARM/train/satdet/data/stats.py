"""数据集统计：报告与 PPT 里的每一个数字都由这里产出，禁止手抄。

产物
----
* ``experiments/results/dataset_stats.csv``    逐序列统计表
* ``experiments/results/tiling_curve.csv``     图块边长 -> 像素占比/覆盖率
* ``experiments/figures/*.png``                目标尺寸、密度、轨迹长、运动速度图

图默认用中文字体；环境里没有 CJK 字体时自动退回英文标签，避免出现方框。
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from satdet.core.logging import get_logger
from satdet.core.paths import ensure_dir, relpath
from satdet.data.tiling import tiling_curve
from satdet.data.viso_index import SequenceInfo, VisoIndex

__all__ = [
    "SIZE_BINS",
    "sequence_stats",
    "compute_stats",
    "tracks_table",
    "write_csv",
    "make_figures",
    "boxes_per_frame",
    "setup_cjk_font",
]

log = get_logger("stats")

#: 目标面积分档（px^2）。等效边长见括号。
SIZE_BINS: tuple[tuple[str, float, float], ...] = (
    ("<=4px", 0.0, 16.0),          # 面积<=16, 边长<=4
    ("5-6px", 16.0, 36.0),         # 边长 4~6
    ("7-8px", 36.0, 64.0),         # 边长 6~8
    ("9-16px", 64.0, 256.0),       # 边长 8~16
    (">16px", 256.0, float("inf")),
)

_CJK_FONTS = ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC", "SimSun")


# =====================================================================
# 字体
# =====================================================================
def setup_cjk_font() -> bool:
    """尝试启用中文字体，返回是否成功。"""
    try:
        import matplotlib
        from matplotlib import font_manager

        available = {f.name for f in font_manager.fontManager.ttflist}
    except Exception as exc:  # noqa: BLE001
        log.warning("matplotlib 不可用，跳过作图: %s", exc)
        return False
    for name in _CJK_FONTS:
        if name in available:
            matplotlib.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            matplotlib.rcParams["axes.unicode_minus"] = False
            return True
    return False


# =====================================================================
# 统计
# =====================================================================
def _track_speeds(gt, max_tracks: int = 300) -> np.ndarray:
    """逐轨迹计算相邻帧位移中位数（px/frame），用于衡量运动先验难度。"""
    if not gt.has_ids:
        return np.zeros(0, dtype=np.float64)
    centers = gt.centers
    speeds: list[float] = []
    ids = np.unique(gt.ids[gt.ids >= 0])
    for tid in ids[:max_tracks]:
        m = gt.ids == tid
        if m.sum() < 3:
            continue
        fr = gt.frame[m]
        c = centers[m]
        order = np.argsort(fr)
        fr, c = fr[order], c[order]
        step = np.diff(fr)
        d = np.linalg.norm(np.diff(c, axis=0), axis=1)
        ok = step == 1                     # 只统计相邻帧，避免丢帧造成虚高
        if ok.any():
            speeds.append(float(np.median(d[ok])))
    return np.asarray(speeds, dtype=np.float64)


def tracks_table(gt, min_len: int = 1) -> list[dict]:
    """逐轨迹明细（长度、平均尺寸、速度），供深度分析使用。"""
    if not gt.has_ids:
        return []
    wh = gt.wh
    centers = gt.centers
    rows: list[dict] = []
    for tid in np.unique(gt.ids[gt.ids >= 0]):
        m = gt.ids == tid
        if m.sum() < min_len:
            continue
        fr = gt.frame[m]
        order = np.argsort(fr)
        c = centers[m][order]
        fr = fr[order]
        d = np.linalg.norm(np.diff(c, axis=0), axis=1) if len(c) > 1 else np.zeros(0)
        step = np.diff(fr) if len(fr) > 1 else np.zeros(0)
        ok = step == 1
        rows.append({
            "track_id": int(tid),
            "length": int(m.sum()),
            "w_med": float(np.median(wh[m, 0])),
            "h_med": float(np.median(wh[m, 1])),
            "speed_med": float(np.median(d[ok])) if ok.any() else 0.0,
        })
    return rows


def sequence_stats(info: SequenceInfo, role: str = "-") -> dict:
    """单个序列的统计字典。"""
    gt = info.load_gt()
    n = len(gt)
    wh = gt.wh if n else np.zeros((0, 2))
    area = wh[:, 0] * wh[:, 1] if n else np.zeros(0)
    per_frame = np.bincount(gt.frame, minlength=info.num_frames + 1)[1:] if n else np.zeros(1)

    speeds = _track_speeds(gt)
    trows = tracks_table(gt)
    lengths = np.asarray([r["length"] for r in trows], dtype=np.float64) if trows else np.zeros(0)

    row: dict[str, object] = {
        "seq": info.seq_id,
        "role": role,
        "frames": info.num_frames,
        "annotated_frames": gt.num_frames,
        "resolution": f"{info.width}x{info.height}",
        "instances": n,
        "tracks": gt.num_tracks,
        "obj_per_frame": round(n / max(1, gt.num_frames), 2),
        "obj_per_frame_max": int(per_frame.max()) if n else 0,
        "w_med": round(float(np.median(wh[:, 0])), 2) if n else 0.0,
        "h_med": round(float(np.median(wh[:, 1])), 2) if n else 0.0,
        "side_med": round(float(np.median(np.sqrt(area))), 2) if n else 0.0,
        "side_max": round(float(np.sqrt(area.max())), 2) if n else 0.0,
        "track_len_med": round(float(np.median(lengths)), 1) if lengths.size else 0.0,
        "track_len_max": int(lengths.max()) if lengths.size else 0,
        "speed_med": round(float(np.median(speeds)), 2) if speeds.size else 0.0,
        "speed_p95": round(float(np.percentile(speeds, 95)), 2) if speeds.size else 0.0,
        "gt_source": info.gt_path.name if info.gt_path else "-",
    }
    for label, lo, hi in SIZE_BINS:
        cnt = int(((area >= lo) & (area < hi)).sum())
        row[f"area_{label}"] = cnt
        row[f"area_{label}_pct"] = round(100.0 * cnt / max(1, n), 2)
    return row


def compute_stats(index: VisoIndex, role_map: dict[str, str] | None = None) -> list[dict]:
    """全部序列的统计表。"""
    role_map = role_map or {}
    rows = [sequence_stats(info, role_map.get(info.seq_id, "-")) for info in index]
    return rows


def boxes_per_frame(info: SequenceInfo, stride: int = 1, max_frames: int | None = None) -> list[np.ndarray]:
    """按帧切分 GT 框，供 tiling 评估使用。"""
    gt = info.load_gt()
    if len(gt) == 0:
        return []
    n = info.num_frames
    out: list[np.ndarray] = []
    order = np.argsort(gt.frame)
    fr = gt.frame[order]
    bx = gt.boxes[order]
    lo = 0
    for f in range(1, n + 1):
        hi = int(np.searchsorted(fr, f, side="right"))
        if (f - 1) % stride == 0:
            out.append(bx[lo:hi])
        lo = hi
        if max_frames is not None and len(out) >= max_frames:
            break
    return out


# =====================================================================
# 输出
# =====================================================================
def write_csv(rows: Sequence[dict], path: str | Path) -> Path:
    """按首行键顺序写 CSV（UTF-8 无 BOM，Excel 需另存编码）。"""
    p = Path(path)
    ensure_dir(p.parent)
    if not rows:
        p.write_text("", encoding="utf-8")
        return p
    fields: list[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with p.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    log.info("已写出 %s (%d 行)", relpath(p), len(rows))
    return p


def make_figures(
    index: VisoIndex,
    rows: Sequence[dict],
    outdir: str | Path,
    tiling_rows: Sequence[dict] | None = None,
    sample_frames: int = 120,
) -> list[Path]:
    """生成报告用图，返回文件列表。缺 matplotlib 时静默跳过。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        log.warning("matplotlib 不可用，跳过作图: %s", exc)
        return []

    zh = setup_cjk_font()
    t = (lambda cn, en: cn) if zh else (lambda cn, en: en)

    out = ensure_dir(outdir)
    made: list[Path] = []
    seqs = [s for s in index]

    # ---- 图1：目标尺寸分布 ----
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    all_side = []
    for info in seqs:
        gt = info.load_gt()
        if len(gt) == 0:
            continue
        side = np.sqrt(gt.wh[:, 0] * gt.wh[:, 1])
        all_side.append(side)
        ax.hist(side, bins=np.arange(0, 40, 1), histtype="step", label=info.seq_id, linewidth=1.4)
    if all_side:
        ax.axvline(float(np.median(np.concatenate(all_side))), color="k", ls="--", lw=1,
                   label=t("总体中位数", "all median"))
    ax.set_xlabel(t("等效边长 (px)", "equivalent side (px)"))
    ax.set_ylabel(t("实例数", "#instances"))
    ax.set_title(t("VISO 目标尺寸分布：绝大多数只有几个像素", "VISO object size distribution"))
    ax.set_xlim(0, 32)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    p = out / "fig_dataset_size_hist.png"
    fig.tight_layout(); fig.savefig(p); plt.close(fig)
    made.append(p)

    # ---- 图2：逐帧目标密度 ----
    fig, ax = plt.subplots(figsize=(9, 4), dpi=150)
    for info in seqs[:6]:
        gt = info.load_gt()
        if len(gt) == 0:
            continue
        dens = np.bincount(gt.frame, minlength=info.num_frames + 1)[1:]
        ax.plot(np.arange(1, len(dens) + 1), dens, lw=1.0, label=f"{info.seq_id} (med={np.median(dens):.0f})")
    ax.set_xlabel(t("帧号", "frame"))
    ax.set_ylabel(t("每帧目标数", "boxes per frame"))
    ax.set_title(t("逐帧目标密度：城市快速路场景", "Per-frame object density"))
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    p = out / "fig_dataset_density.png"
    fig.tight_layout(); fig.savefig(p); plt.close(fig)
    made.append(p)

    # ---- 图3：轨迹长度与运动速度 ----
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), dpi=150)
    all_len, all_spd = [], []
    for info in seqs:
        tr = tracks_table(info.load_gt())
        if not tr:
            continue
        all_len.append(np.array([r["length"] for r in tr], dtype=float))
        all_spd.append(np.array([r["speed_med"] for r in tr], dtype=float))
    if all_len:
        axes[0].hist(np.concatenate(all_len), bins=30, color="steelblue")
        axes[0].set_xlabel(t("轨迹长度 (帧)", "track length (frames)"))
        axes[0].set_ylabel(t("轨迹数", "#tracks"))
        axes[0].set_title(t("轨迹长度分布", "Track length"))
        axes[1].hist(np.concatenate(all_spd), bins=30, color="darkorange")
        axes[1].set_xlabel(t("帧间位移中位数 (px)", "median step (px)"))
        axes[1].set_title(t("目标运动速度（卡尔曼参数依据）", "Object speed"))
        for a in axes:
            a.grid(alpha=0.3)
    p = out / "fig_dataset_tracks.png"
    fig.tight_layout(); fig.savefig(p); plt.close(fig)
    made.append(p)

    # ---- 图4：尺寸分档堆叠柱状图 ----
    if rows:
        fig, ax = plt.subplots(figsize=(9, 4), dpi=150)
        labels = [r["seq"] for r in rows]
        x = np.arange(len(labels))
        bottom = np.zeros(len(labels))
        for key, _, _ in SIZE_BINS:
            vals = np.array([float(r.get(f"area_{key}_pct", 0) or 0) for r in rows])
            ax.bar(x, vals, bottom=bottom, label=key)
            bottom += vals
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_ylabel(t("占比 (%)", "share (%)"))
        ax.set_title(t("各序列目标尺寸分档（面积等效边长）", "Object size bins per sequence"))
        ax.legend(fontsize=7, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.12))
        p = out / "fig_dataset_size_bins.png"
        fig.tight_layout(); fig.savefig(p); plt.close(fig)
        made.append(p)

    # ---- 图5：tiling 曲线（B 组核心论据）----
    if tiling_rows:
        fig, ax = plt.subplots(figsize=(8, 4.6), dpi=150)
        tiles = [r["tile"] for r in tiling_rows]
        roi_px = [100 * r["roi_pixel_ratio"] for r in tiling_rows]
        roi_px2 = [100 * r["roi_pixel_ratio_vs_upscaled"] for r in tiling_rows]
        grid_px = [100 * r["grid_pixel_ratio"] for r in tiling_rows]

        ax.plot(tiles, roi_px, "o-", lw=2.2, label=t("RoI 图块 / 整帧 1024", "RoI tiles / full 1024"))
        ax.plot(tiles, roi_px2, "^-", lw=2.0, color="tab:green",
                label=t("RoI 图块 / 等效 2048 整帧", "RoI tiles / equivalent 2048"))
        ax.plot(tiles, grid_px, "s--", lw=1.6, color="gray", label=t("均匀网格", "uniform grid"))
        ax.axhline(100, color="crimson", ls=":", lw=1.3,
                   label=t("整帧 1024 推理（基线）", "full-frame 1024 (baseline)"))
        for r, y in zip(tiling_rows, roi_px):
            ax.annotate(f'{100 * r["roi_coverage"]:.1f}%', (r["tile"], y),
                        textcoords="offset points", xytext=(0, 8), ha="center", fontsize=7)
        ax.set_ylim(0, max(roi_px + grid_px) * 1.2)
        ax.set_xlabel(t("图块边长 (px)", "tile size (px)"))
        ax.set_ylabel(t("相对基线的输入像素 (%)", "input pixels vs baseline (%)"))
        ax.set_title(t("自适应网格 tiling：像素成本与覆盖率（标注=RoI 覆盖率）",
                       "Adaptive grid tiling: pixel cost and coverage"))
        ax.legend(fontsize=7.5, loc="upper left")
        ax.grid(alpha=0.3)
        p = out / "fig_tiling_curve.png"
        fig.tight_layout(); fig.savefig(p); plt.close(fig)
        made.append(p)
    log.info("已生成 %d 张图 -> %s", len(made), relpath(out))
    return made
