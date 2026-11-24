"""可视化演示程序：实时渲染 + 逐帧/分阶段算法效果展示 + 解说演示片导出。"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
RELEASE = HERE.parent / "rel_test"
for _cand in (HERE / "rel_test", HERE.parent / "rel_test", HERE.parent.parent / "rel_test"):
    if (_cand / "src" / "pipeline_py.py").exists():
        RELEASE = _cand
        break
sys.path.insert(0, str(RELEASE / "src"))

from pipeline_py import MotionProposer, PyPipeline  # noqa: E402

_orig_propose = MotionProposer.propose


def _propose_wrapped(self, *a, **k):
    out = _orig_propose(self, *a, **k)
    self.last_origins = [tuple(map(int, o)) for o in out]
    return out


MotionProposer.propose = _propose_wrapped

STAGES = [
    ("阶段 1  原始输入", "1024x1024 卫星视频流 @10fps，逐帧读入"),
    ("阶段 2  运动掩膜", "半分辨率灰度 + 3x3 白顶帽 -> 帧差(阈值40) -> 3x3 椭圆闭运算降噪"),
    ("阶段 3  运动提案", "4x4 格子投票 + 轨迹先验 -> 取前 6 格，每格一个 256x256 图块"),
    ("阶段 4  图块检测", "图块缩放到 320 -> INT8 推理 -> DFL 解码 -> 跨图块 NMS"),
    ("阶段 5  跟踪与测速", "8 状态卡尔曼 + 三级指派 + 光流先验 -> 多滞后中值测速 km/h"),
]


def compose(frame, stage, st):
    vis = frame.copy()
    h, w = vis.shape[:2]
    if stage == 1 and st.get("mask") is not None:
        m = cv2.resize(st["mask"], (w, h), interpolation=cv2.INTER_NEAREST)
        vis = cv2.addWeighted(vis, 0.35, cv2.applyColorMap(m, cv2.COLORMAP_BONE), 0.65, 0)
    elif stage == 2:
        for (x, y) in st.get("origins", []):
            cv2.rectangle(vis, (x, y), (x + 256, y + 256), (0, 0, 255), 2)
    elif stage >= 3:
        for t in st.get("tracks", []):
            x1, y1, x2, y2 = [int(v) for v in t["box"]]
            kmh = float(t.get("kmh", 0.0))
            col = (0, 165, 255) if kmh < 60 else ((0, 255, 0) if kmh < 100 else (0, 0, 255))
            cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2)
            if stage == 4:
                cv2.putText(vis, "#%d %d km/h" % (t["track_id"], int(kmh)),
                            (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (255, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(vis, (0, 0), (w, 34), (0, 0, 0), -1)
    cv2.putText(vis, STAGES[stage][0], (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)
    info = "frame %d  tiles %d  tracks %d  %.1f ms" % (
        st.get("frame", 0), st.get("tiles", 0), len(st.get("tracks", [])), st.get("ms", 0.0))
    cv2.putText(vis, info, (w - 440, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (200, 255, 200), 1, cv2.LINE_AA)
    cv2.rectangle(vis, (0, h - 56), (w, h), (0, 0, 0), -1)
    cv2.putText(vis, STAGES[stage][1][:92], (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (220, 220, 220), 1, cv2.LINE_AA)
    return vis


def card(title, sub):
    img = np.zeros((1024, 1024, 3), np.uint8)
    cv2.putText(img, title, (60, 470), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255),
                3, cv2.LINE_AA)
    y = 540
    for line in sub.split("|"):
        cv2.putText(img, line, (60, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (200, 200, 200),
                    2, cv2.LINE_AA)
        y += 46
    cv2.putText(img, "satstream", (60, 960), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (120, 120, 120), 1, cv2.LINE_AA)
    return img


class TkView:
    def __init__(self, scale, fullscreen):
        import tkinter as tk
        from PIL import Image, ImageTk
        self._Image, self._ImageTk = Image, ImageTk
        self.root = tk.Tk()
        self.root.title("satstream showcase")
        self.scale = scale
        self.label = tk.Label(self.root, bg="black", bd=0)
        self.label.pack(expand=True)
        if fullscreen:
            self.root.attributes("-fullscreen", True)
            self.root.configure(bg="black")
        self.photo = None
        self.closed = False
        self.root.protocol("WM_DELETE_WINDOW", self._close)

    def _close(self):
        self.closed = True

    def show(self, bgr):
        img = self._Image.fromarray(bgr[:, :, ::-1])
        if self.scale != 1.0:
            img = img.resize((max(1, int(img.width * self.scale)),
                              max(1, int(img.height * self.scale))))
        self.photo = self._ImageTk.PhotoImage(img)
        self.label.configure(image=self.photo)
        self.root.update()
        return not self.closed


def read_ctl(path):
    if not path.exists():
        return None
    try:
        txt = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not txt:
        return None
    path.write_text("", encoding="utf-8")
    return txt.split("\n")[-1].split()


def dump_demo(args):
    pipe = PyPipeline(Path(args.model), imgsz=args.imgsz, max_cells=6, threads=4, conf=0.2,
                      flow_w=1.0, pre_mode=1, min_hits=5, gate=0.7, max_coast=8)
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print("打不开视频", file=sys.stderr)
        return 2
    out = Path(args.dump)
    out.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (1024, 1024))
    per = int(args.dump_per_stage * 10)
    card_f = int(args.dump_card * 10)
    total = 0
    for stage in range(len(STAGES)):
        img = card(STAGES[stage][0], STAGES[stage][1].replace(" -> ", " -> ").replace("，", "，"))
        for _ in range(card_f):
            vw.write(img)
            total += 1
        done = 0
        st = {}
        while done < per:
            ok, f = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            t0 = time.time()
            tracks = pipe.step(f)
            st = {"frame": pipe.frame, "mask": getattr(pipe.prop, "last_mask", None),
                  "origins": getattr(pipe.prop, "last_origins", []), "tracks": tracks,
                  "tiles": getattr(pipe, "tiles", 0), "ms": (time.time() - t0) * 1000.0}
            vw.write(compose(f, stage, st))
            done += 1
            total += 1
            if done % 100 == 0:
                print("  %s  %d/%d  总 %d" % (STAGES[stage][0], done, per, total), flush=True)
    vw.release()
    print("完成 %s：%d 帧, %.1f 分钟, %.1f MB" % (
        out, total, total / 600.0, out.stat().st_size / 2 ** 20))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--model", default=str(RELEASE / "models" / "a_p2_nwd_kd320_raw_int8.onnx"))
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--scale", type=float, default=0.55)
    ap.add_argument("--fullscreen", action="store_true")
    ap.add_argument("--ctl", default=str(HERE / "ctl"))
    ap.add_argument("--stage-seconds", type=float, default=6.0)
    ap.add_argument("--dump", default="")
    ap.add_argument("--dump-per-stage", type=float, default=50.0)
    ap.add_argument("--dump-card", type=float, default=6.0)
    args = ap.parse_args()

    if args.dump:
        return dump_demo(args)

    pipe = PyPipeline(Path(args.model), imgsz=args.imgsz, max_cells=6, threads=4, conf=0.2,
                      flow_w=1.0, pre_mode=1, min_hits=5, gate=0.7, max_coast=8)
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print("打不开视频", file=sys.stderr)
        return 2
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    view = TkView(args.scale, args.fullscreen)
    ctl = Path(args.ctl)
    ctl.write_text("", encoding="utf-8")

    stage, paused, single, auto_adv = 0, False, False, True
    last_adv = time.time()
    st, frame, n = {}, None, 0
    while True:
        blank = frame if frame is not None else np.zeros((1024, 1024, 3), np.uint8)
        if not view.show(compose(blank, stage, st)):
            break
        cmd = read_ctl(ctl)
        if cmd:
            c = cmd[0]
            if c == "quit":
                break
            if c == "stage" and len(cmd) > 1:
                stage = max(0, min(len(STAGES) - 1, int(cmd[1])))
                auto_adv = False
            elif c == "next":
                stage, auto_adv = (stage + 1) % len(STAGES), False
            elif c == "prev":
                stage, auto_adv = (stage - 1) % len(STAGES), False
            elif c == "pause":
                paused = True
            elif c == "play":
                paused, auto_adv, last_adv = False, True, time.time()
            elif c == "step":
                paused, single = True, True
            elif c == "auto":
                auto_adv, last_adv = True, time.time()
        if paused and not single:
            time.sleep(0.03)
            continue
        t0 = time.time()
        ok, f = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        frame = f
        tracks = pipe.step(f)
        st = {"frame": pipe.frame, "mask": getattr(pipe.prop, "last_mask", None),
              "origins": getattr(pipe.prop, "last_origins", []), "tracks": tracks,
              "tiles": getattr(pipe, "tiles", 0), "ms": (time.time() - t0) * 1000.0}
        n += 1
        single = False
        if auto_adv and time.time() - last_adv > args.stage_seconds:
            stage = (stage + 1) % len(STAGES)
            last_adv = time.time()
        dt = time.time() - t0
        if dt < 1.0 / fps:
            time.sleep(1.0 / fps - dt)
    try:
        view.root.destroy()
    except Exception:
        pass
    print("展示结束，共 %d 帧" % n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())