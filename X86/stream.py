"""卫星视频车辆检测与测速 · 主程序。

两种后端：
  --backend native   调用 C/C++ 加速库（默认，推荐）
  --backend python   纯 Python 流水线（无编译依赖，便于对照）

三种输出模式（--mode）：
  csv    只记 CSV，不开窗口、不写视频（板端后台跑、批量评测）
  video  渲染叠加视频（可选同时记 CSV）
  show   实时窗口演示（可选同时记 CSV / 录视频）
  all    三者全开；auto=按 --show / --save-video 推断

平台相关：
  * 飞腾派（2 大核 + 2 小核）：默认 2 线程并绑定大核；密封（AES-256-GCM）通道可用。
  * 其它平台（含树莓派）：默认 4 线程、不绑核；密封通道默认拒绝（--force-seal 可测）。

示例：
  python stream.py --input in.mp4 --mode csv                     # 只出 CSV
  python stream.py --input in.mp4 --mode video --save-video r.mp4
  python stream.py --input in.mp4 --mode show                    # 实时演示
  python stream.py --input in.mp4 --mode show --async            # 板端异步演示
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
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
    ap.add_argument("--mode", default="auto", choices=["auto", "csv", "video", "show", "all"],
                    help="输出模式：csv=只记 CSV（无窗口/无视频）；video=渲染视频；"
                         "show=实时窗口；all=三者全开；auto=按参数推断")
    ap.add_argument("--async", dest="async_mode", action="store_true", default=bool(cfg.get("async", False)),
                    help="异步检测：后台线程推理，显示/视频按速度外推（板端低帧率时保持流畅）")
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
    ap.add_argument("--csv", default="", help="轨迹 CSV 路径（明文）")
    ap.add_argument("--save-video", default="", help="叠加结果视频（明文 mp4）")
    ap.add_argument("--show", action="store_true", help="实时显示窗口（等价 --mode show）")
    ap.add_argument("--show-backend", default="auto", choices=["auto", "cv2", "tk"],
                    help="窗口后端：auto=有 GUI 的 cv2 则用 cv2，否则用 tkinter")
    ap.add_argument("--fullscreen", action="store_true", help="窗口全屏（演示用）")
    ap.add_argument("--display-scale", type=float, default=1.0,
                    help="显示缩放（窗口模式用，1024x600 屏建议 0.55）")
    ap.add_argument("--out-fps", type=float, default=0.0,
                    help="输出/显示限速（0=不限；板端异步演示可用 10）")
    ap.add_argument("--native-lib", default="", help="显式指定加速库路径")
    ap.add_argument("--print-every", type=int, default=30, help="每 N 帧打印一次进度")
    return ap


def apply_cores(spec: str, phytium: bool) -> list:
    if spec == "none":
        return []
    cores = []
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


class AsyncRunner:
    """后台线程跑整条流水线；主线程按显示帧率渲染，用上一帧结果线性外推。

    最新帧优先：上一帧还没跑完就丢弃当前帧（跳帧），保证显示不积压。
    """

    def __init__(self, pipe) -> None:
        self.pipe = pipe
        self._cv = threading.Condition()
        self._job = None
        self._out = None
        self._stop = False
        self.busy = False
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def submit(self, frame) -> bool:
        with self._cv:
            if self._job is not None:
                return False
            self._job = frame.copy()
            self.busy = True
            self._cv.notify()
            return True

    def poll(self):
        with self._cv:
            if self._out is None:
                return None
            out, self._out = self._out, None
            return out

    def _loop(self) -> None:
        while True:
            with self._cv:
                while self._job is None and not self._stop:
                    self._cv.wait(0.05)
                if self._stop:
                    return
                job, self._job = self._job, None
            res = self.pipe.step(job)
            with self._cv:
                self._out = res
                self.busy = False

    def wait(self, timeout: float = 2.0):
        """阻塞直到有结果（头less 模式用，保证每帧都被真正处理）。"""
        end = time.time() + timeout
        while time.time() < end:
            out = self.poll()
            if out is not None:
                return out
            time.sleep(0.002)
        return self.poll()

    def close(self) -> None:
        with self._cv:
            self._stop = True
            self._cv.notify()


def cv2_has_gui() -> bool:
    """探测 OpenCV 是否带 GUI（headless 版 imshow 会抛错）。"""
    try:
        cv2.namedWindow("__probe__", cv2.WINDOW_AUTOSIZE)
        cv2.destroyWindow("__probe__")
        return True
    except Exception:
        return False


class TkWindow:
    """tkinter 显示窗口（OpenCV headless 时的替代，依赖 python3-tk + Pillow）。"""

    def __init__(self, title: str = "satstream", scale: float = 1.0,
                 fullscreen: bool = False) -> None:
        import tkinter as tk
        from PIL import Image, ImageTk
        self._Image = Image
        self._ImageTk = ImageTk
        self.root = tk.Tk()
        self.root.title(title)
        self.scale = float(scale) if scale else 1.0
        self.label = tk.Label(self.root, bg="black", bd=0)
        self.label.pack(expand=True)
        if fullscreen:
            self.root.attributes("-fullscreen", True)
            self.root.configure(bg="black")
        self.photo = None
        self.quit_flag = False
        self.root.bind("<Escape>", lambda _e: self._close())
        self.root.protocol("WM_DELETE_WINDOW", self._close)

    def _close(self) -> None:
        self.quit_flag = True

    def show(self, bgr) -> bool:
        img = self._Image.fromarray(bgr[:, :, ::-1])
        if self.scale != 1.0:
            img = img.resize((max(1, int(img.width * self.scale)),
                              max(1, int(img.height * self.scale))))
        self.photo = self._ImageTk.PhotoImage(img)
        self.label.configure(image=self.photo)
        self.root.update()
        return not self.quit_flag

    def close(self) -> None:
        try:
            self.root.destroy()
        except Exception:
            pass


def draw(frame: np.ndarray, tracks: list) -> np.ndarray:
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
        tmp_in = tmp_plain_path("in", ".mp4")
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

    # ---- 输出模式 ----
    mode = args.mode
    if mode == "auto":
        if args.show:
            mode = "all" if (args.save_video or args.seal_video) else "show"
        elif args.save_video or args.seal_video:
            mode = "video"
        else:
            mode = "csv"
    want_show = mode in ("show", "all") or args.show
    want_video = mode in ("video", "all") or bool(args.save_video) or bool(args.seal_video)
    want_csv = mode in ("csv", "video", "all") or bool(args.csv) or bool(args.seal_csv)
    if not (want_csv or want_video or want_show):
        want_csv = True
    print("输出模式：%s（CSV=%s 视频=%s 窗口=%s 异步=%s）" % (
        mode, want_csv, want_video, want_show, args.async_mode))

    csv_plain = Path(args.csv) if args.csv else None
    vid_plain = Path(args.save_video) if args.save_video else None
    tmp_csv = tmp_plain_path("csv", ".csv") if args.seal_csv else None
    tmp_vid = tmp_plain_path("vid", ".mp4") if args.seal_video else None
    if tmp_csv is not None:
        csv_plain = tmp_csv
    elif want_csv:
        csv_plain = csv_plain or Path("tracks.csv")
    else:
        csv_plain = None
    if tmp_vid is not None:
        vid_plain = tmp_vid
    elif want_video:
        vid_plain = vid_plain or Path("out.mp4")
    else:
        vid_plain = None

    writer = None
    if vid_plain is not None:
        fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(str(vid_plain), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        if not writer.isOpened():          # 无 mp4v 编码器时退回 MJPEG/AVI
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

    if args.async_mode and args.out_fps <= 0:
        if want_video:
            args.out_fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
        elif want_show:
            args.out_fps = 10.0
        print("异步模式：输出限速 %.1f fps（0 表示无显示/视频时按帧阻塞等结果）" % args.out_fps)

    win = None
    if want_show:
        backend = args.show_backend
        if backend == "auto":
            backend = "cv2" if cv2_has_gui() else "tk"
        if backend == "tk":
            win = TkWindow("satstream", args.display_scale, args.fullscreen)
            args.display_scale = 1.0          # 缩放已在 TkWindow 内完成
        print("窗口后端：%s" % backend)

    runner = AsyncRunner(pipe) if args.async_mode else None
    last_tracks = []
    last_by_id = {}
    vx = {}
    vy = {}

    t0 = time.time()
    n = 0
    total_trk = 0
    out_period = (1.0 / args.out_fps) if args.out_fps > 0 else 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        if args.frames and n > args.frames:
            n -= 1
            break

        if runner is not None:
            runner.submit(frame)
            res = runner.poll() if out_period > 0 else runner.wait(5.0)
            if res is not None:
                tracks = res
                for t in tracks:                       # 记录逐 ID 位移，供外推
                    prev = last_by_id.get(t["track_id"])
                    if prev is not None:
                        vx[t["track_id"]] = t["box"][0] - prev[0]
                        vy[t["track_id"]] = t["box"][1] - prev[1]
                    last_by_id[t["track_id"]] = t["box"]
                last_tracks = tracks
            elif last_tracks:                          # 检测未就绪：按速度外推
                tracks = []
                for t in last_tracks:
                    tid = t["track_id"]
                    dx, dy = vx.get(tid, 0.0), vy.get(tid, 0.0)
                    b = t["box"]
                    tracks.append(dict(t, box=(b[0] + dx, b[1] + dy, b[2] + dx, b[3] + dy)))
                last_tracks = tracks
            else:
                tracks = []
        else:
            tracks = pipe.step(frame)
        total_trk += len(tracks)

        if csv_fh:
            for t in tracks:
                csv_fh.write("%d,%d,%.2f,%.2f,%.2f,%.2f,%.4f,%.1f\n" % (
                    t["track_id"], n, t["box"][0], t["box"][1], t["box"][2], t["box"][3],
                    t["px"], t["kmh"]))
        if writer is not None or want_show:
            vis = draw(frame, tracks)
            if writer is not None:
                writer.write(vis)
            if want_show:
                if win is not None:
                    if not win.show(vis):
                        break
                else:
                    shown = vis
                    if args.display_scale and args.display_scale != 1.0:
                        shown = cv2.resize(vis, None, fx=args.display_scale,
                                           fy=args.display_scale, interpolation=cv2.INTER_AREA)
                    cv2.imshow("satstream", shown)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break
        if out_period > 0:
            dt = time.time() - t0
            if dt < n * out_period:
                time.sleep(n * out_period - dt)
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
    if win is not None:
        win.close()
    elif want_show:
        cv2.destroyAllWindows()
    if runner is not None:
        runner.close()
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