"""satdet 统一命令行入口。

命令总览
--------
doctor            环境自检（新同学第一条命令）
data index        扫描数据集，缓存序列索引
data split        生成 train/val/test/holdout 划分并校验
data verify       标注质检
data stats        数据集统计 + 报告图表（含 tiling 方案对比）
data tiling       只做图块方案对比（改 tile 参数时快速迭代）

设计约定
--------
* 子命令只做"取参数 -> 调模块 -> 打印/落盘"，业务逻辑留在 satdet/* 里。
* 所有输出都同时落盘到 artifacts/ 或 experiments/，便于写报告时引用。
"""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path

from satdet.core.logging import get_logger, setup_logging
from satdet.core.paths import artifacts_dir, ensure_dir, figures_dir, relpath, results_dir
from satdet.core.table import format_table
from satdet.data.split import load_split, role_totals, save_split, split_report, validate_split
from satdet.data.viso_index import discover_sequences

__all__ = ["main", "build_parser"]

log = get_logger("cli")

ROLES = ("train", "val", "test", "holdout")


# =====================================================================
# 公共辅助
# =====================================================================
def _seq_list(raw: str | None) -> list[str] | None:
    if not raw:
        return None
    return [s.strip() for s in raw.replace(";", ",").split(",") if s.strip()]


def _index_path() -> Path:
    return artifacts_dir() / "index" / "viso_index.json"


def _split_path() -> Path:
    return artifacts_dir() / "splits" / "viso_split.json"


def _get_index(args: argparse.Namespace, use_cache: bool = False):
    """取得索引：优先读缓存，缺失或被要求刷新时重新扫描。"""
    path = _index_path()
    if use_cache and path.is_file():
        from satdet.data.viso_index import VisoIndex

        log.info("读取索引缓存 %s", relpath(path))
        return VisoIndex.load(path)

    root = Path(args.dataset_root) if getattr(args, "dataset_root", None) else None
    index = discover_sequences(root, _seq_list(getattr(args, "seqs", None)))
    index.save(path)
    return index


def _load_split_cfg():
    from satdet.core.config import load_config

    return load_split(load_config("dataset/viso.yaml"))


def _parse_tiles(raw: str) -> list[int]:
    tiles = sorted({int(t) for t in raw.replace(";", ",").split(",") if t.strip()})
    if not tiles:
        raise ValueError("--tiles 不能为空")
    return tiles


def _compute_tiling(index, tiles: list[int], sample_frames: int) -> tuple[list[dict], list[dict]]:
    """逐序列算 tiling 曲线，再按**图块边长**分组求平均。

    要点
    ----
    1. 必须按 tile 分组，否则不同边长的结果会被混在一起（历史 bug）。
    2. 默认只统计 train/val/test：048/049 是 400x400 与 401x601 的 holdout，
       分辨率不同会让"均匀网格"的像素占比失真（边缘补边导致 >100%）。

    Returns
    -------
    (aggregated, per_sequence)
    """
    from satdet.data.stats import boxes_per_frame
    from satdet.data.tiling import tiling_curve

    buckets: dict[float, list[dict]] = {}
    per_seq: list[dict] = []
    for info in index:
        stride = max(1, info.num_frames // max(1, sample_frames))
        bpf = boxes_per_frame(info, stride=stride)
        if not bpf:
            continue
        for row in tiling_curve(bpf, info.width, info.height, tile_sizes=tuple(tiles)):
            buckets.setdefault(row["tile"], []).append(row)
            per_seq.append({"seq": info.seq_id, "resolution": f"{info.width}x{info.height}", **row})

    out: list[dict] = []
    for tile in sorted(buckets):
        group = buckets[tile]
        keys = [k for k in group[0] if k != "tile"]
        out.append({"tile": tile,
                    **{k: sum(g[k] for g in group) / len(group) for k in keys}})
    return out, per_seq


def _design_index(index):
    """只保留参与设计与调参的序列（train/val/test），排除 holdout。"""
    try:
        split = _load_split_cfg()
    except Exception:  # noqa: BLE001 - 没配置就不过滤
        return index
    keep = [s for role in ("train", "val", "test") for s in getattr(split, role)]
    keep = [s for s in keep if s in index.sequences]
    return index.filter(keep) if keep else index


def _print_tiling(rows: list[dict]) -> None:
    if not rows:
        print("（没有可用于评估的序列）")
        return
    print()
    print("图块方案对比（全数据集平均；RoI 默认网格对齐 snap=tile）")
    print(format_table(
        [{
            "tile": int(r["tile"]),
            "roi_px": f'{100 * r["roi_pixel_ratio"]:.1f}%',
            "roi_px2": f'{100 * r["roi_pixel_ratio_vs_upscaled"]:.1f}%',
            "roi_cov": f'{100 * r["roi_coverage"]:.2f}%',
            "roi_n": f'{r["roi_tiles_per_frame"]:.1f}',
            "grid_px": f'{100 * r["grid_pixel_ratio"]:.1f}%',
            "grid_n": f'{r["grid_tiles_per_frame"]:.1f}',
        } for r in rows],
        columns=[("tile", "图块边长"), ("roi_px", "像素/整帧"), ("roi_px2", "像素/2048整帧"),
                 ("roi_cov", "RoI覆盖率"), ("roi_n", "块数/帧"),
                 ("grid_px", "网格像素"), ("grid_n", "网格块数")],
        right_align=("tile",),
    ))
    print("说明: 像素/整帧 对比 1024x1024 单次全图推理；像素/2048整帧 对比"
          "“要达到同等分辨率需 2048x2048 整图推理”，即 4 倍像素。")


# =====================================================================
# doctor
# =====================================================================
def cmd_doctor(args: argparse.Namespace) -> int:
    from satdet.core.doctor import run_doctor

    report = run_doctor(save=not args.no_save, with_dataset=not args.no_dataset)
    torch = report.get("torch", {})
    problems = []
    if not torch.get("installed"):
        problems.append("PyTorch 未安装")
    elif not torch.get("cuda_available"):
        problems.append("CUDA 不可用：训练会退化为 CPU，请检查显卡驱动与 torch 版本")
    # 注意：这里的键必须与 doctor._PROBE_PACKAGES 里的"显示名"一致，
    # 例如 opencv 的显示名是 "opencv-python" 而不是 "cv2"。
    for label in ("opencv-python", "onnx", "onnxruntime"):
        if report.get("packages", {}).get(label) is None:
            problems.append(f"缺少依赖 {label}")
    if report.get("dataset", {}).get("exists") is False:
        problems.append("数据集目录不存在")

    print()
    if problems:
        print("需要注意：")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("环境自检通过。")
    return 0


# =====================================================================
# data index / split
# =====================================================================
def cmd_data_index(args: argparse.Namespace) -> int:
    index = _get_index(args, use_cache=False)
    print()
    print(format_table(
        index.summary_rows(),
        columns=[("seq", "序列"), ("frames", "帧数"), ("width", "宽"), ("height", "高"),
                 ("instances", "实例"), ("tracks", "轨迹"), ("gt_source", "标注源"),
                 ("has_ids", "含ID")],
        right_align=("frames", "width", "height", "instances", "tracks"),
    ))
    issues = index.diagnostics()
    if issues:
        print()
        print("数据隐患：")
        for i in issues:
            print(f"  - {i}")
    print()
    log.info("索引: %s", relpath(_index_path()))
    return 0


def cmd_data_split(args: argparse.Namespace) -> int:
    index = _get_index(args, use_cache=not args.refresh)
    split = _load_split_cfg()
    problems = validate_split(index, split)

    rows = split_report(index, split)
    print()
    print(format_table(
        rows,
        columns=[("role", "角色"), ("seq", "序列"), ("frames", "帧数"), ("instances", "实例"),
                 ("tracks", "轨迹"), ("resolution", "分辨率"), ("gt", "标注源")],
        right_align=("frames", "instances", "tracks"),
    ))
    print()
    for role, agg in role_totals(rows).items():
        print(f"  {role:<8} 序列 {agg['seqs']:>2} 个 / 帧 {agg['frames']:>6} / 实例 {agg['instances']:>7}")

    if problems:
        print()
        print("划分问题：")
        for p in problems:
            print(f"  - {p}")
        return 1

    save_split(split, _split_path())
    print()
    print(f"划分合法，已写入 {relpath(_split_path())}")
    return 0


# =====================================================================
# data verify / stats / tiling
# =====================================================================
def cmd_data_verify(args: argparse.Namespace) -> int:
    from satdet.data.stats import write_csv
    from satdet.data.verify import run_checks

    index = _get_index(args, use_cache=True)
    rows = run_checks(index, _seq_list(args.seqs))
    print()
    print(format_table(
        rows,
        columns=[("seq", "序列"), ("check", "检查项"), ("status", "状态"),
                 ("count", "数量"), ("detail", "说明")],
        right_align=("count",),
    ))
    path = write_csv(rows, results_dir() / "dataset_verify.csv")
    bad = [r for r in rows if r["status"] == "warn"]
    print()
    log.info("质检报告: %s", relpath(path))
    print(f"共 {len(rows)} 项检查，" + (f"{len(bad)} 项告警。" if bad else "无告警。"))
    return 0


def cmd_data_stats(args: argparse.Namespace) -> int:
    from satdet.data.stats import compute_stats, make_figures, write_csv

    index = _get_index(args, use_cache=True)
    split = _load_split_cfg()
    role_map = {s: role for role in ROLES for s in getattr(split, role)}

    seq_ids = _seq_list(args.seqs)
    if seq_ids:
        index = index.filter(seq_ids)

    rows = compute_stats(index, role_map)
    csv_path = write_csv(rows, results_dir() / "dataset_stats.csv")
    print()
    print(format_table(
        rows,
        columns=[("seq", "序列"), ("role", "角色"), ("frames", "帧"), ("instances", "实例"),
                 ("obj_per_frame", "密度"), ("side_med", "边长中位"), ("w_med", "w"),
                 ("h_med", "h"), ("track_len_med", "轨迹长"), ("speed_med", "速度px/帧")],
        right_align=("frames", "instances", "obj_per_frame", "side_med", "w_med", "h_med",
                     "track_len_med", "speed_med"),
    ))

    tiling_rows: list[dict] = []
    if not args.no_tiling:
        log.info("评估 tiling 方案（每序列采样约 %d 帧）...", args.sample_frames)
        tiling_rows, tiling_per_seq = _compute_tiling(
            _design_index(index), _parse_tiles(args.tiles), args.sample_frames)
        if tiling_rows:
            write_csv(tiling_rows, results_dir() / "tiling_curve.csv")
            write_csv(tiling_per_seq, results_dir() / "tiling_curve_per_seq.csv")
            _print_tiling(tiling_rows)

    if not args.no_figures:
        make_figures(index, rows, ensure_dir(figures_dir()),
                     tiling_rows=tiling_rows, sample_frames=args.sample_frames)

    print()
    log.info("统计表: %s", relpath(csv_path))
    return 0


def cmd_data_tiling(args: argparse.Namespace) -> int:
    from satdet.data.stats import make_figures, write_csv

    index = _design_index(_get_index(args, use_cache=True))
    rows, per_seq = _compute_tiling(index, _parse_tiles(args.tiles), args.sample_frames)
    if not rows:
        log.error("没有可用于评估的序列（缺标注？）")
        return 1
    write_csv(rows, results_dir() / "tiling_curve.csv")
    write_csv(per_seq, results_dir() / "tiling_curve_per_seq.csv")
    _print_tiling(rows)
    make_figures(index, [], ensure_dir(figures_dir()), tiling_rows=rows)
    return 0


# =====================================================================
# 解析器
# =====================================================================
# =====================================================================
# train / eval det / bench
# =====================================================================
def cmd_train(args: argparse.Namespace) -> int:
    from satdet.engine.train import train

    result = train(
        config_path=args.config,
        exp_id=args.exp_id,
        limit_frames=args.limit_frames,
        epochs=args.epochs,
        val_limit=args.val_limit,
        dataset_root=args.dataset_root,
    )
    print()
    print(f"运行目录     : {relpath(result.run_dir)}")
    print(f"训练样本     : {result.n_train_samples}")
    print(f"最优 mAP30   : {result.best_map30:.4f} (epoch {result.best_epoch})")
    return 0


def cmd_eval_det(args: argparse.Namespace) -> int:
    from satdet.core.config import load_config
    from satdet.data.stats import write_csv
    from satdet.eval.det_metrics import evaluate_detections
    from satdet.infer.detector import Detector
    from satdet.models.weight_loader import load_coco_weights
    from satdet.models.yolov8 import build_yolov8n

    cfg = load_config(args.config)
    index = _get_index(args, use_cache=True)
    seq_ids = _seq_list(args.seqs) or [str(s) for s in cfg["data"]["val_seqs"]]

    model = build_yolov8n(nc=int(cfg["model"].get("nc", 1)),
                          p2=bool(cfg["model"].get("p2", True)))
    wpath = args.weights or cfg.get("weights")
    if wpath and Path(wpath).is_file():
        load_coco_weights(model, wpath, strict=False)
    else:
        log.warning("未指定/未找到权重，使用随机初始化（只用于跑通流程）")

    frames: list[tuple[str, int]] = []
    for sid in seq_ids:
        info = index[sid]
        n = info.eval_frame_limit or info.num_frames
        frames.extend((sid, f) for f in range(1, n + 1))
    if args.limit:
        frames = frames[:args.limit]

    det = Detector(model=model, device=args.device,
                   nc=int(cfg["model"].get("nc", 1)),
                   conf_thr=float(cfg.get("val", {}).get("conf_floor", 0.05)),
                   iou_thr=float(cfg.get("val", {}).get("nms_iou", 0.5)),
                   use_amp=False,
                   tta=bool(getattr(args, "tta", False)))
    proposal = args.proposal or cfg.get("val", {}).get("proposal", "full")
    tile = int(args.tile or cfg["data"].get("tile", 256))
    imgsz = int(args.imgsz or cfg["data"].get("imgsz", 512))
    log.info("评测 %d 帧 | proposal=%s tile=%d imgsz=%d", len(frames), proposal, tile, imgsz)

    preds = det.evaluate(index, frames, proposal=proposal, tile=tile, imgsz=imgsz)
    gts = {(s, f): index[s].load_gt().frame_rows(f).boxes for s, f in frames}
    res = evaluate_detections(preds, gts)
    row = res.to_row(exp=args.exp_id or Path(args.config).stem, proposal=proposal,
                     tile=tile, imgsz=imgsz, seqs=",".join(seq_ids))
    print()
    print(format_table([{k: (round(v, 4) if isinstance(v, float) else v)
                         for k, v in row.items()}],
                       columns=[("exp", "实验"), ("mAP30", "mAP30"), ("mAP50", "mAP50"),
                                ("F1", "F1"), ("Precision", "P"), ("Recall", "R"),
                                ("recall_center2px", "中心距<=2px"), ("conf", "conf"),
                                ("n_gt", "GT"), ("n_pred", "预测")],
                       right_align=("mAP30", "mAP50", "F1", "Precision", "Recall",
                                    "recall_center2px", "conf", "n_gt", "n_pred")))
    out = Path(args.out) if args.out else (results_dir() / "det_eval.csv")
    _append_csv(row, out)
    log.info("结果已追加到 %s", relpath(out))
    return 0


def _append_csv(row: dict, path: Path) -> None:
    """追加一行（保留历史实验，便于横向对比）。"""
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file()
    with path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def cmd_bench(args: argparse.Namespace) -> int:
    from satdet.core.config import load_config
    from satdet.eval.bench_harness import benchmark
    from satdet.infer.detector import Detector
    from satdet.models.weight_loader import load_coco_weights
    from satdet.models.yolov8 import build_yolov8n

    cfg = load_config(args.config)
    index = _get_index(args, use_cache=True)
    seq_ids = _seq_list(args.seqs) or [str(s) for s in cfg["data"]["val_seqs"]]

    model = build_yolov8n(nc=int(cfg["model"].get("nc", 1)),
                          p2=bool(cfg["model"].get("p2", True)))
    wpath = args.weights or cfg.get("weights")
    if wpath and Path(wpath).is_file():
        load_coco_weights(model, wpath, strict=False)

    frames: list[tuple[str, int]] = []
    for sid in seq_ids:
        info = index[sid]
        n = info.eval_frame_limit or info.num_frames
        frames.extend((sid, f) for f in range(1, n + 1))
    frames = frames[:args.frames + args.warmup]

    det = Detector(model=model, device=args.device,
                   nc=int(cfg["model"].get("nc", 1)),
                   conf_thr=float(args.conf), iou_thr=float(args.nms_iou),
                   use_amp=bool(args.amp))
    proposal = args.proposal or cfg.get("val", {}).get("proposal", "full")
    tile = int(args.tile or cfg["data"].get("tile", 256))
    imgsz = int(args.imgsz or cfg["data"].get("imgsz", 512))

    res = benchmark(det, index, frames, proposal=proposal, tile=tile, imgsz=imgsz,
                    warmup=args.warmup, exp=args.exp_id or Path(args.config).stem)
    print()
    print(format_table([{k: (round(v, 3) if isinstance(v, float) else v)
                         for k, v in res.to_row().items()}],
                       columns=[("exp", "实验"), ("device", "设备"), ("fps", "FPS"),
                                ("lat_p50_ms", "p50ms"), ("lat_p95_ms", "p95ms"),
                                ("cpu_util_mean_pct", "CPU%"),
                                ("decode_ms", "解码"), ("preprocess_ms", "预处理"),
                                ("infer_ms", "推理"), ("postprocess_ms", "后处理"),
                                ("tiles_per_frame", "块数"), ("pixel_ratio", "像素比")],
                       right_align=("fps", "lat_p50_ms", "lat_p95_ms", "cpu_util_mean_pct",
                                    "decode_ms", "preprocess_ms", "infer_ms",
                                    "postprocess_ms", "tiles_per_frame", "pixel_ratio")))
    _append_csv(res.to_row(), results_dir() / "perf_bench.csv")
    return 0

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="satdet",
        description="遥感卫星视频流小目标连续检测系统（芯火计划·软件赛道）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="快速开始:\n"
               "  python -m satdet doctor\n"
               "  python -m satdet data index\n"
               "  python -m satdet data split\n"
               "  python -m satdet data verify\n"
               "  python -m satdet data stats\n",
    )
    p.add_argument("--dataset-root", default=None, help="数据集根目录（默认取 SATDET_DATA_ROOT）")
    p.add_argument("--seqs", default=None, help="只处理指定序列，逗号分隔，如 002,003")
    p.add_argument("--level", default="INFO", help="日志级别 (DEBUG/INFO/WARNING)")
    p.add_argument("--quiet", action="store_true", help="只输出错误")

    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("doctor", help="环境自检")
    d.add_argument("--no-save", action="store_true", help="不写 JSON 报告")
    d.add_argument("--no-dataset", action="store_true", help="跳过数据集扫描")
    d.set_defaults(func=cmd_doctor)

    data = sub.add_parser("data", help="数据层命令")
    dsub = data.add_subparsers(dest="data_command", required=True)

    dsub.add_parser("index", help="扫描并缓存序列索引").set_defaults(func=cmd_data_index)

    ps = dsub.add_parser("split", help="生成并校验划分")
    ps.add_argument("--refresh", action="store_true", help="重新扫描数据集")
    ps.set_defaults(func=cmd_data_split)

    dsub.add_parser("verify", help="标注质检").set_defaults(func=cmd_data_verify)

    pst = dsub.add_parser("stats", help="数据集统计与作图")
    pst.add_argument("--no-figures", action="store_true", help="不出图")
    pst.add_argument("--no-tiling", action="store_true", help="跳过 tiling 评估")
    pst.add_argument("--tiles", default="128,192,256,320,384,512", help="待评估的图块边长")
    pst.add_argument("--sample-frames", type=int, default=120, help="每序列采样帧数")
    pst.set_defaults(func=cmd_data_stats)

    pt = dsub.add_parser("tiling", help="仅做图块方案对比")
    pt.add_argument("--tiles", default="128,192,256,320,384,512")
    pt.add_argument("--sample-frames", type=int, default=120)
    pt.set_defaults(func=cmd_data_tiling)

    ptr = sub.add_parser("train", help="训练（自研循环）")
    ptr.add_argument("--config", required=True, help="configs/train/*.yaml")
    ptr.add_argument("--exp-id", default=None, help="实验 id（默认取配置里的 exp_id）")
    ptr.add_argument("--epochs", type=int, default=None, help="覆盖配置里的 epochs")
    ptr.add_argument("--limit-frames", type=int, default=None, help="只用前 N 帧训练（调试）")
    ptr.add_argument("--val-limit", type=int, default=None, help="验证帧上限")
    ptr.set_defaults(func=cmd_train)

    pev = sub.add_parser("eval", help="评测")
    evsub = pev.add_subparsers(dest="eval_command", required=True)
    pdet = evsub.add_parser("det", help="检测指标")
    pdet.add_argument("--config", required=True, help="configs/train/*.yaml（读取模型与数据设置）")
    pdet.add_argument("--weights", default=None, help="权重路径（默认取配置里的 weights）")
    pdet.add_argument("--proposal", default=None, choices=["full", "oracle", "grid"])
    pdet.add_argument("--tile", type=int, default=None)
    pdet.add_argument("--imgsz", type=int, default=None)
    pdet.add_argument("--limit", type=int, default=None, help="只评前 N 帧")
    pdet.add_argument("--device", default="cuda")
    pdet.add_argument("--tta", action="store_true", help="水平翻转 TTA（离线评测，003 上 +5 mAP30，代价 2×）")
    pdet.add_argument("--exp-id", default=None)
    pdet.add_argument("--out", default=None)
    pdet.set_defaults(func=cmd_eval_det)

    pb = sub.add_parser("bench", help="性能基准（FPS/延迟/CPU/阶段拆解）")
    pb.add_argument("--config", required=True)
    pb.add_argument("--weights", default=None)
    pb.add_argument("--proposal", default=None, choices=["full", "oracle", "grid"])
    pb.add_argument("--tile", type=int, default=None)
    pb.add_argument("--imgsz", type=int, default=None)
    pb.add_argument("--frames", type=int, default=120, help="正式测量的帧数")
    pb.add_argument("--warmup", type=int, default=20)
    pb.add_argument("--conf", type=float, default=0.25)
    pb.add_argument("--nms-iou", type=float, default=0.5)
    pb.add_argument("--device", default="cuda")
    pb.add_argument("--amp", action="store_true", default=True)
    pb.add_argument("--exp-id", default=None)
    pb.set_defaults(func=cmd_bench)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    setup_logging("ERROR" if getattr(args, "quiet", False) else args.level)
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        log.warning("已中断")
        return 130
    except Exception as exc:  # noqa: BLE001 - CLI 兜底，打堆栈便于排错
        traceback.print_exc()
        log.error("命令执行失败: %s", exc)
        return 2
