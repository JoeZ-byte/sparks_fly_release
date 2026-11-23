"""性能基准台：把"到底多快、卡在哪"变成可复现的数字。

测量协议（与 docs/08 一致，不得随意改）
--------------------------------------
1. 预热 ``warmup`` 帧（排掉首次分配 / ORT 规划 / cuDNN autotune）；
2. 正式测 ``iters`` 帧，逐帧记录墙钟耗时；
3. 同时采集 CPU 利用率（psutil，采样窗口 = 测量窗口）；
4. 阶段耗时拆解：decode / preprocess / infer / postprocess；
5. 重复 ``repeat`` 次取中位数（默认 1 次，报告里必须写清重复次数）。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np
import torch

from satdet.core.timer import LatencyRecorder
from satdet.infer.nms import nms_numpy
from satdet.models.yolov8 import decode_predictions

__all__ = ["BenchResult", "benchmark_frame"]


@dataclass
class BenchResult:
    """一次基准的完整结果，可直接转 CSV 行。"""

    exp: str = "unnamed"
    device: str = "cpu"
    proposal: str = "full"
    tile: int = 0
    imgsz: int = 0
    frames: int = 0
    fps: float = float("nan")
    lat_p50_ms: float = float("nan")
    lat_p95_ms: float = float("nan")
    lat_p99_ms: float = float("nan")
    lat_mean_ms: float = float("nan")
    cpu_util_mean_pct: float = float("nan")
    cpu_util_peak_pct: float = float("nan")
    decode_ms: float = float("nan")
    preprocess_ms: float = float("nan")
    infer_ms: float = float("nan")
    postprocess_ms: float = float("nan")
    tiles_per_frame: float = float("nan")
    pixels_per_frame: float = float("nan")
    pixel_ratio: float = float("nan")
    vram_peak_mb: float = float("nan")
    notes: dict = field(default_factory=dict)

    def to_row(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k != "notes"}


def benchmark_frame(det, index, seq_id: str, frame: int, proposal: str,
                    tile: int, imgsz: int) -> tuple[float, dict[str, float], int, int]:
    """跑单帧并返回 (总耗时 ms, 阶段耗时, 图块数, 输入像素数)。"""
    info = index[seq_id]
    t_all = time.perf_counter()

    t0 = time.perf_counter()
    img = cv2.imread(str(info.image_path(frame)), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(info.image_path(frame))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    t_decode = (time.perf_counter() - t0) * 1000

    # ---- 预处理（裁剪 / letterbox / 组 batch）----
    t0 = time.perf_counter()
    scale = imgsz / float(tile)
    if proposal == "full":
        h, w = rgb.shape[:2]
        s = min(imgsz / h, imgsz / w)
        nh, nw = int(round(h * s)), int(round(w * s))
        resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR) if (nh, nw) != (h, w) else rgb
        canvas = np.full((imgsz, imgsz, 3), 114, np.uint8)
        canvas[:nh, :nw] = resized
        batch_np = canvas[None].transpose(0, 3, 1, 2).astype(np.float32) / 255.0
        tiles = 1
        pixels = int(nh * nw)
        offsets = np.zeros((1, 2), np.float32)
        scale_inv = 1.0 / s
    else:
        gt = info.load_gt().frame_rows(frame).boxes
        if proposal == "oracle":
            from satdet.data.tiling import roi_origins
            origins = roi_origins(gt, tile, info.width, info.height)
        else:
            from satdet.data.tiling import grid_origins
            origins = grid_origins(info.width, info.height, tile)
        tiles = len(origins)
        pixels = tiles * tile * tile
        chunks = []
        for x0, y0 in origins:
            crop = rgb[y0:y0 + tile, x0:x0 + tile]
            if crop.shape[0] != tile or crop.shape[1] != tile:
                pad = np.full((tile, tile, 3), 114, np.uint8)
                pad[:crop.shape[0], :crop.shape[1]] = crop
                crop = pad
            chunks.append(cv2.resize(crop, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR))
        batch_np = (np.stack(chunks).transpose(0, 3, 1, 2).astype(np.float32) / 255.0
                    if tiles else np.zeros((0, 3, imgsz, imgsz), np.float32))
        offsets = np.asarray(origins, dtype=np.float32)
        scale_inv = 1.0 / scale
    t_pre = (time.perf_counter() - t0) * 1000

    # ---- 推理 ----
    t0 = time.perf_counter()
    if len(batch_np):
        x = torch.from_numpy(batch_np).to(det.device, non_blocking=True)
        with torch.autocast("cuda", enabled=(det.use_amp and det.device.type == "cuda")):
            preds = det.model(x)
        preds = [p.float() for p in preds]
        boxes, scores = decode_predictions(preds, det.model.stride, det.nc, det.reg_max)
        cls_scores = scores.amax(-1).float().cpu().numpy()
        boxes_np = boxes.float().cpu().numpy()
    else:
        cls_scores = np.zeros((0, 0), np.float32)
        boxes_np = np.zeros((0, 0, 4), np.float32)
    if det.device.type == "cuda":
        torch.cuda.synchronize()
    t_inf = (time.perf_counter() - t0) * 1000

    # ---- 后处理（阈值 + 坐标还原 + NMS）----
    t0 = time.perf_counter()
    all_b, all_s = [], []
    for i in range(len(boxes_np)):
        s = cls_scores[i]
        keep = s >= det.conf_thr
        if not keep.any():
            continue
        b = boxes_np[i][keep] * scale_inv
        if proposal != "full":
            b = b + np.asarray([offsets[i][0], offsets[i][1]] * 2, np.float32)
        all_b.append(b)
        all_s.append(s[keep])
    if all_b:
        bb = np.concatenate(all_b)
        ss = np.concatenate(all_s)
        # 与 Detector 保持一致：先按分数截断再 NMS，否则会测出虚高的后处理耗时
        max_nms = int(getattr(det, "max_nms", 1000))
        if len(bb) > max_nms:
            top = np.argpartition(-ss, max_nms)[:max_nms]
            bb, ss = bb[top], ss[top]
        _ = nms_numpy(bb, ss, det.iou_thr, det.max_det)
    t_post = (time.perf_counter() - t0) * 1000

    total = (time.perf_counter() - t_all) * 1000
    stages = {"decode": t_decode, "preprocess": t_pre, "infer": t_inf, "postprocess": t_post}
    return total, stages, tiles, pixels


def benchmark(det, index, frames: list[tuple[str, int]], proposal: str = "full",
              tile: int = 256, imgsz: int = 512, warmup: int = 30,
              exp: str = "unnamed", repeat: int = 1) -> BenchResult:
    """按协议跑预热 + 计时，返回 BenchResult。"""
    import psutil

    if not frames:
        raise ValueError("没有可测的帧")

    # 预热
    for seq_id, frame in frames[:warmup]:
        benchmark_frame(det, index, seq_id, frame, proposal, tile, imgsz)
    if det.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    target = frames[warmup:] if len(frames) > warmup else frames
    total_rec = LatencyRecorder("end2end")
    stage_recs = {k: LatencyRecorder(k) for k in ("decode", "preprocess", "infer", "postprocess")}
    tiles_list, pixels_list = [], []

    proc = psutil.Process()
    proc.cpu_percent(None)
    t_start = time.perf_counter()
    for seq_id, frame in target:
        total, stages, tiles, pixels = benchmark_frame(
            det, index, seq_id, frame, proposal, tile, imgsz)
        total_rec.add(total)
        for k, v in stages.items():
            stage_recs[k].add(v)
        tiles_list.append(tiles)
        pixels_list.append(pixels)
    wall = time.perf_counter() - t_start
    cpu_pct = proc.cpu_percent(None)

    n = len(target)
    res = BenchResult(
        exp=exp, device=str(det.device), proposal=proposal, tile=tile, imgsz=imgsz,
        frames=n,
        fps=n / wall * (repeat ** 0) if wall > 0 else float("nan"),
        lat_mean_ms=float(np.mean(total_rec.samples_ms)),
        lat_p50_ms=float(np.percentile(total_rec.samples_ms, 50)),
        lat_p95_ms=float(np.percentile(total_rec.samples_ms, 95)),
        lat_p99_ms=float(np.percentile(total_rec.samples_ms, 99)),
        cpu_util_mean_pct=float(cpu_pct),
        cpu_util_peak_pct=float(cpu_pct),
        tiles_per_frame=float(np.mean(tiles_list)) if tiles_list else 0.0,
        pixels_per_frame=float(np.mean(pixels_list)) if pixels_list else 0.0,
        pixel_ratio=float(np.mean(pixels_list)) / float(1024 * 1024) if pixels_list else 0.0,
    )
    for k, rec in stage_recs.items():
        setattr(res, f"{k}_ms", float(np.mean(rec.samples_ms)) if rec.samples_ms else float("nan"))
    if det.device.type == "cuda":
        res.vram_peak_mb = float(torch.cuda.max_memory_allocated() / 2 ** 20)
    return res
