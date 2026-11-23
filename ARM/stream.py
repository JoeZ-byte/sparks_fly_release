"""卫星视频车辆检测与测速 · 主程序。

两种后端：
  --backend native   调用 C/C++ 加速库（默认，推荐）
  --backend python   纯 Python 流水线（无编译依赖，便于对照与教学）

示例：
  python stream.py --input samples/sample_08_tiny_033.mp4 --csv out.csv
  python stream.py --input samples/sample_08_tiny_033.mp4 --backend python --frames 60
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))


def load_config() -> dict:
    p = HERE / "config.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}


def build_parser(cfg: dict) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="卫星视频车辆检测与测速")
    ap.add_argument("--input", required=True, help="输入视频（mp4/mjpeg）")
    ap.add_argument("--model", default=cfg.get("model", ""), help="ONNX 模型路径")
    ap.add_argument("--backend", default="native", choices=["native", "python"],
                    help="native=链接 C/C++ 加速库；python=纯 Python 流水线")
    ap.add_argument("--imgsz", type=int, default=cfg.get("imgsz", 512), help="网络输入边长")
    ap.add_argument("--max-cells", type=int, default=cfg.get("max_cells", 6),
                    help="每帧最多推理的图块数")
    ap.add_argument("--threads", type=int, default=cfg.get("threads", 4), help="推理线程数")
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
    ap.add_argument("--csv", default="", help="轨迹 CSV 输出路径")
    ap.add_argument("--save-video", default="", help="叠加结果视频（mp4）")
    ap.add_argument("--show", action="store_true", help="实时显示窗口（需要图形界面）")
    ap.add_argument("--native-lib", default="", help="显式指定加速库路径")
    ap.add_argument("--print-every", type=int, default=30, help="每 N 帧打印一次进度")
    return ap


def draw(frame: np.ndarray, tracks: list[dict]) -> np.ndarray:
    out = frame.copy()
    for t in tracks:
        x1, y1, x2, y2 = [int(v) for v in t["box"]]
        kmh = float(t.get("kmh", 0.0))
        color = (0, 165, 255) if kmh < 60 else ((0, 255, 0) if kmh < 100 else (0, 0, 255))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = "#%d %d km/h" % (t["track_id"], int(kmh))
        cv2.putText(out, label, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def main() -> int:
    cfg = load_config()
    args = build_parser(cfg).parse_args()
    if not args.model:
        print("缺少 --model（或在 config.json 里配置 model）", file=sys.stderr)
        return 2
    model = Path(args.model)
    if not model.is_absolute():
        model = (HERE / model).resolve()
    if not model.exists():
        print("模型不存在: %s" % model, file=sys.stderr)
        return 2

    flow_w = 0.0 if args.no_flow else float(args.flow_w)

    if args.backend == "native":
        from native_bridge import NativePipeline
        pipe = NativePipeline(model, imgsz=args.imgsz, max_cells=args.max_cells,
                              threads=args.threads, conf=args.conf, flow_w=flow_w,
                              pre_mode=0 if args.no_tophat else 1,
                              min_hits=args.min_hits, gate=args.gate,
                              max_coast=args.max_coast, warmup=args.warmup,
                              lib_path=args.native_lib or None)
    else:
        from pipeline_py import PyPipeline
        pipe = PyPipeline(model, imgsz=args.imgsz, max_cells=args.max_cells,
                          threads=args.threads, conf=args.conf, flow_w=flow_w,
                          pre_mode=0 if args.no_tophat else 1,
                          min_hits=args.min_hits, gate=args.gate,
                          max_coast=args.max_coast, warmup=args.warmup)

    cap = cv2.VideoCapture(str(args.input))
    if not cap.isOpened():
        print("打不开视频: %s" % args.input, file=sys.stderr)
        return 2

    writer = None
    if args.save_video:
        fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(args.save_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    csv_fh = open(args.csv, "w", encoding="utf-8", newline="") if args.csv else None
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
            dt = time.time() - t0
            print("  帧 %d … %.1f FPS，轨迹 %d" % (n, n / max(dt, 1e-6), total_trk))

    elapsed = time.time() - t0
    fps = n / elapsed if elapsed > 0 else 0.0
    print("\n完成 %d 帧，耗时 %.1fs（%.1f ms/帧，%.2f FPS）" % (
        n, elapsed, elapsed / max(n, 1) * 1000, fps))
    if hasattr(pipe, "stats") and args.backend == "native":
        st = pipe.stats()
        print("  运动提案 %.1f ms  推理 %.1f ms  跟踪+测速 %.1f ms  图块/帧 %.1f  检测/帧 %.1f" % (
            st["motion_ms"], st["infer_ms"], st["track_ms"],
            st["tiles"] / max(n, 1), st["dets"] / max(n, 1)))
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())