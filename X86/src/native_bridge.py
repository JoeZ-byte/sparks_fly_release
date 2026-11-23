"""ctypes 绑定：Python 主程序加载 C/C++ 加速库（satstream.dll / libsatstream.so）。

加速库负责：运动提案、图块预处理、INT8 推理、DFL 解码、NMS、卡尔曼跟踪、测速、光流先验。
Python 侧只负责：视频读写、绘制、CSV。
"""
from __future__ import annotations

import ctypes
import sys
from ctypes import POINTER, Structure, byref, c_char_p, c_double, c_float, c_int, c_int64
from ctypes import c_void_p
from pathlib import Path

__all__ = ["NativePipeline", "library_path"]


class SflyConfig(Structure):
    _fields_ = [
        ("model_path", c_char_p), ("ort_lib", c_char_p),
        ("imgsz", c_int), ("max_cells", c_int), ("threads", c_int),
        ("warmup", c_int), ("max_coast", c_int), ("conf", c_float),
        ("pre_mode", c_int), ("min_hits", c_int), ("gate", c_float),
        ("app_w", c_float), ("diff_mode", c_int), ("bg_mode", c_int),
        ("flow_w", c_float),
    ]


class SflyTrack(Structure):
    _fields_ = [("track_id", c_int), ("x1", c_float), ("y1", c_float), ("x2", c_float),
                ("y2", c_float), ("px_per_frame", c_float), ("speed_ready", c_int),
                ("kmh", c_float)]


class SflyOrigin(Structure):
    _fields_ = [("x", c_int), ("y", c_int)]


class SflyFrame(Structure):
    _fields_ = [("tiles", c_int), ("n_dets", c_int), ("n_tracks", c_int),
                ("tracks", POINTER(SflyTrack)), ("origins", POINTER(SflyOrigin))]


class SflyStats(Structure):
    _fields_ = [("motion_ms", c_double), ("infer_ms", c_double), ("track_ms", c_double),
                ("tiles_total", c_int64), ("dets_total", c_int64), ("frames", c_int)]


def library_path() -> Path:
    """在 native/prebuilt 下找加速库。"""
    here = Path(__file__).resolve().parent
    names = ["satstream.dll"] if sys.platform == "win32" else ["libsatstream.so", "satstream.so"]
    search = [here / "prebuilt", here / "native" / "prebuilt",
              here.parent / "prebuilt", here.parent / "native" / "prebuilt"]
    for d in search:
        for n in names:
            p = d / n
            if p.exists():
                return p
    raise FileNotFoundError(
        "找不到加速库，请先编译：cd native && mingw32-make（Windows）或 make（Linux）。"
        "也支持 --native-lib 显式指定路径。")


def _ort_dll() -> str:
    """定位 onnxruntime 动态库（加速库内部会 dlopen 它）。"""
    try:
        import onnxruntime as ort
    except ImportError:
        return ""
    capi = Path(ort.__file__).resolve().parent / "capi"
    for n in ("onnxruntime.dll", "libonnxruntime.so", "libonnxruntime.dylib"):
        p = capi / n
        if p.exists():
            return str(p)
    hits = sorted(capi.glob("libonnxruntime.so*"))
    return str(hits[-1]) if hits else ""


class NativePipeline:
    """一次 create，然后逐帧 step。"""

    def __init__(self, model: str | Path, *, imgsz: int = 512, max_cells: int = 6,
                 threads: int = 4, conf: float = 0.2, flow_w: float = 1.0,
                 pre_mode: int = 1, min_hits: int = 5, gate: float = 0.7,
                 max_coast: int = 8, warmup: int = 0, app_w: float = 0.0,
                 lib_path: str | Path | None = None) -> None:
        lib = ctypes.CDLL(str(lib_path) if lib_path else str(library_path()))
        self._lib = lib
        lib.sfly_create.restype = c_void_p
        lib.sfly_create.argtypes = [POINTER(SflyConfig), c_char_p, c_int]
        lib.sfly_destroy.argtypes = [c_void_p]
        lib.sfly_step.restype = c_int
        lib.sfly_step.argtypes = [c_void_p, POINTER(ctypes.c_uint8), c_int, c_int,
                                  POINTER(SflyFrame), c_char_p, c_int]
        lib.sfly_get_stats.argtypes = [c_void_p, POINTER(SflyStats)]

        cfg = SflyConfig()
        cfg.model_path = str(model).encode()
        cfg.ort_lib = _ort_dll().encode() or None
        cfg.imgsz = int(imgsz)
        cfg.max_cells = int(max_cells)
        cfg.threads = int(threads)
        cfg.warmup = int(warmup)
        cfg.max_coast = int(max_coast)
        cfg.conf = float(conf)
        cfg.pre_mode = int(pre_mode)
        cfg.min_hits = int(min_hits)
        cfg.gate = float(gate)
        cfg.app_w = float(app_w)
        cfg.diff_mode = 0
        cfg.bg_mode = 0
        cfg.flow_w = float(flow_w)

        err = ctypes.create_string_buffer(512)
        self._ctx = lib.sfly_create(byref(cfg), err, 512)
        if not self._ctx:
            raise RuntimeError("加速库初始化失败: %s" % err.value.decode(errors="ignore"))
        self.tiles = 0
        self.dets = 0

    def step(self, frame) -> list[dict]:
        """frame: HxWx3 BGR uint8（numpy）。返回轨迹列表。"""
        import numpy as np
        f = np.ascontiguousarray(frame)
        h, w = f.shape[:2]
        fr = SflyFrame()
        err = ctypes.create_string_buffer(512)
        ptr = f.ctypes.data_as(POINTER(ctypes.c_uint8))
        if self._lib.sfly_step(self._ctx, ptr, w, h, byref(fr), err, 512) != 0:
            raise RuntimeError("step 失败: %s" % err.value.decode(errors="ignore"))
        self.tiles = fr.tiles
        self.dets = fr.n_dets
        out = []
        for i in range(fr.n_tracks):
            t = fr.tracks[i]
            out.append({"track_id": t.track_id,
                        "box": (t.x1, t.y1, t.x2, t.y2),
                        "px": t.px_per_frame,
                        "kmh": t.kmh,
                        "ready": bool(t.speed_ready)})
        return out

    def stats(self) -> dict:
        st = SflyStats()
        self._lib.sfly_get_stats(self._ctx, byref(st))
        return {"motion_ms": st.motion_ms, "infer_ms": st.infer_ms, "track_ms": st.track_ms,
                "tiles": st.tiles_total, "dets": st.dets_total, "frames": st.frames}

    def close(self) -> None:
        if getattr(self, "_ctx", None):
            self._lib.sfly_destroy(self._ctx)
            self._ctx = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass