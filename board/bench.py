#!/usr/bin/env python3
"""飞腾派板端基准：INT8 vs FP32、线程数扫描、端到端帧率。

用法: python3 bench.py [--video samples/xxx.mp4]
"""
import os
import subprocess
import sys
import time
import platform

import numpy as np
import cv2
import onnxruntime as ort

MODELS = {"int8": "model/a_p2_nwd_v3_raw_int8.onnx",
          "fp32": "model/a_p2_nwd_v3_raw_fp32.onnx"}


def bench_models():
    x = np.random.rand(1, 3, 512, 512).astype(np.float32)
    print(f"平台 {platform.machine()}  核数 {os.cpu_count()}")
    print(f"{'模型':>5} {'线程':>4} {'ms/块':>9} {'4块/帧':>9}")
    for name, path in MODELS.items():
        if not os.path.exists(path):
            continue
        for th in (2, 3, 4):
            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            so.intra_op_num_threads = th
            so.inter_op_num_threads = 1
            so.add_session_config_entry("session.intra_op.allow_spinning", "0")
            s = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
            s.run(None, {"batch": x})
            t0 = time.perf_counter()
            for _ in range(2):
                s.run(None, {"batch": x})
            ms = (time.perf_counter() - t0) / 2 * 1000
            print(f"{name:>5} {th:>4} {ms:>9.1f} {ms*4:>9.1f}")


def bench_end2end(video: str):
    """同步 vs 异步两种模式的端到端显示帧率。"""
    if not os.path.exists(video):
        print(f"缺少视频 {video}")
        return
    for label, extra in (("同步（优化前）", []),
                         ("异步（优化后）", ["--async-detect", "--play-fps", "10",
                                            "--max-coast", "15"])):
        cmd = [sys.executable, "stream_board.py", "--input", video, "--no-show",
               "--max-frames", "40", "--backend", "onnx"] + extra
        # 同步模式必须显式关掉板端默认的异步
        if not extra:
            cmd += ["--no-async-detect", "--play-fps", "0"]
        out = subprocess.run(cmd, capture_output=True, text=True)
        tail = [l for l in out.stdout.splitlines() if "完成" in l]
        print(f"{label}: {tail[-1] if tail else '失败'}")


if __name__ == "__main__":
    bench_models()
    video = "samples/sample_01_highway_002.mp4"
    if len(sys.argv) > 1:
        video = sys.argv[1]
    bench_end2end(video)