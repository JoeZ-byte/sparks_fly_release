"""卫星视频车辆检测与测速 · 主程序。

两种后端：
  --backend native   调用 C/C++ 加速库（默认，推荐）
  --backend python   纯 Python 流水线（无编译依赖，便于对照）

平台相关：
  * 飞腾派（2 大核 + 2 小核）：默认 2 线程并绑定大核；支持密封（AES-256-GCM）通道。
  * 其它平台：默认 4 线程、不绑核；密封通道需显式 --force-seal。

示例：
  python stream.py --input samples/sample_08_tiny_033.mp4 --csv out.csv
  python stream.py --input samples/sample_01_highway_002.mp4 --backend python --frames 60
  python stream.py --seal-input video.enc --seal-key <64hex> --seal-csv out.csv
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

from seal import (big_cores, is_phytium, parse_key, seal_file, seal_path_for,  # noqa: E402
                  tmp_plain_path, unseal_file, wipe)


def load_config() -> dict:
    p = HERE / "config.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def build_parser(cfg: dict, phytium: bool) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="卫星视频车辆检测与测速")
    ap.add_argument("--input", default="", help="输入视频（与 --seal-input 二选一）")
    ap.add_argument("--seal-input", default="", help="密封视频（.enc），仅飞腾派")
    ap.add_argument("--seal-key", default="", help="AES-256 密钥，64 位十六进制")
    ap.add_argument("--seal-csv", default="", help="把轨迹 CSV 加密写入该路径（.enc）")
    ap.add_argument("--seal-video", default="", help="把叠加视频加密写入该路径（.enc）")
    ap.add_argument("--force-seal", action="store_true", help="在非飞腾派平台也允许密封")
    ap.add_argument("--model", default=cfg.get("model", ""), help="ONNX 模型路径")
    ap.add_argument("--backend", default="native", choices=["native", "python"],
                    help="native=链接 C/C++ 加速库；python=纯 Python 流水线")
    ap.add_argument("--imgsz", type=int, default=cfg.get("imgsz", 512), help="网络输入边长")
    ap.add_argument("--max-cells", type=int, default=cfg.get("max_cells", 6),
                    help="每帧最多推理的图块数")
    ap.add_argument("--threads", type=int,
                    default=cfg.get("threads_phytium" if phytium else "threads",
                                    cfg.get("threads", 4)),
                    help="推理线程数（飞腾派默认 2：大核只有 2 个）")
    ap.add_argument("--pin-cores", default="auto",
                    help="绑核：auto=飞腾派自动绑大核；none=不绑；或写 '2,3'")
    ap.add_argument("--conf", type=float, default=0.2, help="置信度阈值（对 logit）")
    ap.add_argument("--frames", type=int, default=0, help="只处理前 N 帧（0=全部）")
    ap.add_argument("--warmup", type=int, default=0, help="前 N 帧只更新背景")
    ap.add_argument("--flow-w", type=float, default=cfg.get("flow_w", 1.0),
                    help="光流速度先验权重（默认 1.0；--no-flow 关闭）")
    ap.add_argument("--no-flow", action="store_true", help="关闭光流速度先验")
    ap.add_argument("--no-tophat", action="store_true", help="关闭提案前的白顶帽增强")
    ap.add_argument("--min-hits", type=int, default=cfg.get("min_hits", 5), help="轨迹确认命中帧数")
    ap.add_argument("--gate", type=float, default=cfg.get("gate", 0.7), help="关联门限倍率")
    ap.add_argument("--max-coast", type=int, default=cfg.get("max_coast", 8), help="最大外推帧数")
    ap.add_argument("--csv", default="", help="轨迹 CSV 输出路径（明文）")
    ap.add_argument("--save-video", default="", help="叠加结果视频（明文 mp4）")
    ap.add_argument("--show", action="store_true", help="实时显示窗口")
    ap.add_argument("--native-lib", default="", help="显式指定加速库路径")
    ap.add_argument("--print-every", type=int, default=30, help="每 N 帧打印一次进度")
    return ap


def apply_cores(spec: str, phytium: bool) -> list[int]:
    """按参数绑定 CPU；返回实际绑定的核列表（空表示未绑）。"""
    if spec == "none":
        return []
    cores: list[int] = []
    if spec == "auto":
        if phytium:
            cores = big_cores()
    else:
        try:
            cores = [int(x) for x in spec.replace(" ", "").split(",") if x != ""]
        except ValueError:
            cores = []
    if not cores:
        return []
    try:
        os.sched_setaffinity(0, set(cores))
        return cores
    except (AttributeError, OSError):
        return []


def draw(frame: np.ndarray, tracks: list[dict]) -> np.ndarray:
    out = frame.copy()
    for t in tracks:
        x1, y1, x2, y2 = [int(v) for v in t["box"]]
        kmh = float(t.get("kmh", 0.0))
        color = (0, 165, 255) if kmh < 60 else ((0, 255, 0) if kmh < 100 else (0, 0, 255))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, "#%d %d km/h" % (t["track_id"], int(kmh)), (x1, max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def main() -> int:
    cfg = load_config()
    phytium = is_phytium()
    args = build_parser(cfg, phytium).parse_args()

    sealing = bool(args.seal_input or args.seal_csv or args.seal_video)
    if sealing and not phytium and not args.force_seal:
        print("密封通道仅在飞腾派上启用（当前平台未识别）；如确需在此平台测试，加 --force-seal",
              file=sys.stderr)
        return 2
    key = parse_key(args.seal_key) if sealing else b""

    cores = apply_cores(args.pin_cores, phytium)
    if cores:
        print("绑核：%s（当前平台%s）" % (",".join(map(str, cores)),
                                     "识别为飞腾派" if phytium else "手动指定"))

    if not args.model:
        print("缺少 --model（或在 config.json 里配置 model）", file=sys.stderr)
        return 2
    model = Path(args.model)
    if not model.is_absolute():
        model = (HERE / model).resolve()
    if not model.exists():
        print("模型不存在: %s" % model, file=sys.stderr)
        return 2

    # ---- 输入：明文或密封 ----
    tmp_in = None
    if args.seal_input:
        tmp_in = tmp_plain_path("in")
        n = unseal_file(Path(args.seal_input), tmp_in, key)
        print("已解密输入 %.1f MB -> %s" % (n / 2 ** 20, tmp_in))
        video_path = tmp_in
    elif args.input:
        video_path = Path(args.input)
    else:
        print("需要 --input 或 --seal-input", file=sys.stderr)
        return 2

    flow_w = 0.0 if args.no_flow else float(args.flow_w)
    if args.backend == "native":
        from native_bridge import NativePipeline
        pipe = NativePipeline(model, imgsz=args.imgsz, max_cells=args.max_cells,
                              threads=args.threads, conf=args.conf, flow_w=flow_w,
                              pre_mode=0 if args.no_tophat else 1, min_hits=args.min_hits,
                              gate=args.gate, max_coast=args.max_coast,
                              warmup=args.warmup, lib_path=args.native_lib or None)
    else:
        from pipeline_py import PyPipeline
        pipe = PyPipeline(model, imgsz=args.imgsz, max_cells=args.max_cells,
                          threads=args.threads, conf=args.conf, flow_w=flow_w,
                          pre_mode=0 if args.no_tophat else 1, min_hits=args.min_hits,
                          gate=args.gate, max_coast=args.max_coast, warmup=args.warmup)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("打不开视频: %s" % video_path, file=sys.stderr)
        return 2

    # ---- 输出：明文或密封（先写临时明文，结束后加密并擦除）----
    csv_plain = Path(args.csv) if args.csv else None
    vid_plain = Path(args.save_video) if args.save_video else None
    tmp_csv = tmp_plain_path("csv", ".csv") if args.seal_csv else None
    tmp_vid = tmp_plain_path("vid", ".mp4") if args.seal_video else None
    if tmp_csv is not None:
        csv_plain = tmp_csv
    if tmp_vid is not None:
        vid_plain = tmp_vid

    writer = None
    if vid_plain is not None:
        fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(str(vid_plain), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        if not writer.isOpened():   # 无 mp4v 编码器时退回 MJPEG/AVI
            alt = vid_plain.with_suffix(".avi")
            writer = cv2.VideoWriter(str(alt), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
            if writer.isOpened():
                vid_plain = alt
            else:
                print("警告：当前 OpenCV 没有可用的视频编码器，跳过视频输出", file=sys.stderr)
                writer = None

    csv_fh = open(csv_plain, "w", encoding="utf-8", newline="") if csv_plain else None
    if csv_fh:
        csv_fh.write("track_id,frame,x1,y1,x2,y2,px_per_frame,kmh\n")

    t0 = time.time()
    n = 0
    total_trk = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        if args.frames and n > args.frames:
            n -= 1
            break
        tracks = pipe.step(frame)
        total_trk += len(tracks)
        if csv_fh:
            for t in tracks:
                csv_fh.write("%d,%d,%.2f,%.2f,%.2f,%.2f,%.4f,%.1f\n" % (
                    t["track_id"], n, t["box"][0], t["box"][1], t["box"][2], t["box"][3],
                    t["px"], t["kmh"]))
        if writer is not None or args.show:
            vis = draw(frame, tracks)
            if writer is not None:
                writer.write(vis)
            if args.show:
                cv2.imshow("satstream", vis)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
        if args.print_every and n % args.print_every == 0:
            print("  帧 %d … %.1f FPS，轨迹 %d" % (n, n / max(time.time() - t0, 1e-6), total_trk))

    elapsed = time.time() - t0
    print("\n完成 %d 帧，耗时 %.1fs（%.1f ms/帧，%.2f FPS）" % (
        n, elapsed, elapsed / max(n, 1) * 1000, n / max(elapsed, 1e-6)))
    if hasattr(pipe, "stats") and args.backend == "native":
        st = pipe.stats()
        print("  运动提案 %.1f ms  推理 %.1f ms  跟踪+测速 %.1f ms  图块/帧 %.1f  检测/帧 %.1f" % (
            st["motion_ms"], st["infer_ms"], st["track_ms"], st["tiles"] / max(n, 1),
            st["dets"] / max(n, 1)))
    print("  平均轨迹数/帧 %.1f" % (total_trk / max(n, 1)))

    cap.release()
    if writer is not None:
        writer.release()
    if csv_fh:
        csv_fh.close()
    if args.show:
        cv2.destroyAllWindows()
    if hasattr(pipe, "close"):
        pipe.close()

    # ---- 收尾：加密输出、擦除明文、清理解密输入 ----
    if tmp_csv is not None:
        dst = seal_path_for(args.seal_csv)
        nb = seal_file(tmp_csv, dst, key)
        wipe(tmp_csv)
        print("轨迹已密封：%s（%.1f MB）" % (dst, nb / 2 ** 20))
    if tmp_vid is not None:
        dst = seal_path_for(args.seal_video)
        nb = seal_file(tmp_vid, dst, key)
        wipe(tmp_vid)
        print("视频已密封：%s（%.1f MB）" % (dst, nb / 2 ** 20))
    if tmp_in is not None:
        wipe(tmp_in)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())