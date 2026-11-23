"""训练循环（自研，不依赖 ultralytics trainer）。

设计原则
--------
* **只训 Neck/Head**：Backbone 冻结。既省算力又减少过拟合，还让骨干可以放心
  做 INT8 量化（权重不再更新）。
* **AMP + 余弦退火**：RTX 5060 8 GB 上，512 输入 batch 16 可以开 AMP。
* **每次实验都完整落盘**：配置快照 + 逐 epoch 指标 + 权重 + 日志，
  对应 ``experiments/runs/<exp_id>/``。报告里的数字必须能追溯到这里。
* **不过度封装**：训练循环写在一处、看得懂，新人 10 分钟能改明白。
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from satdet.core.config import dump_yaml, load_config
from satdet.core.logging import get_logger
from satdet.core.paths import experiments_dir, weights_dir
from satdet.data.dataset import DetectionDataset, collate_fn
from satdet.eval.det_metrics import evaluate_detections
from satdet.infer.detector import Detector
from satdet.models.detection_loss import DetectionLoss
from satdet.models.weight_loader import count_parameters, freeze_backbone, load_coco_weights
from satdet.models.yolov8 import build_yolov8n

__all__ = ["TrainResult", "train"]

log = get_logger("train")


@dataclass
class TrainResult:
    run_dir: Path
    best_map30: float = float("nan")
    best_epoch: int = -1
    history: list[dict] = field(default_factory=list)
    n_train_samples: int = 0
    n_val_frames: int = 0

    def to_json(self) -> dict:
        return {
            "run_dir": str(self.run_dir),
            "best_map30": self.best_map30,
            "best_epoch": self.best_epoch,
            "n_train_samples": self.n_train_samples,
            "n_val_frames": self.n_val_frames,
            "history": self.history,
        }


def _lr_at(step: int, total: int, base_lr: float, warmup: int) -> float:
    """线性 warmup + 余弦退火。"""
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * min(1.0, progress)))


def _val_frames(index, seq_ids: list[str], limit: int | None) -> list[tuple[str, int]]:
    """验证帧清单（按序列轮转取样，避免只测第一段）。"""
    frames: list[tuple[str, int]] = []
    per_seq = None if limit is None else max(1, limit // max(1, len(seq_ids)))
    for sid in seq_ids:
        info = index[sid]
        n = info.eval_frame_limit or info.num_frames
        ids = list(range(1, n + 1))
        if per_seq is not None and len(ids) > per_seq:
            step = max(1, len(ids) // per_seq)
            ids = ids[::step][:per_seq]
        frames.extend((sid, f) for f in ids)
    return frames


def train(config_path: str, exp_id: str | None = None,
          limit_frames: int | None = None, epochs: int | None = None,
          val_limit: int | None = None, dataset_root: str | Path | None = None,
          overrides: dict | None = None) -> TrainResult:
    """跑一次训练并返回结果。配置来自 ``configs/train/*.yaml``。"""
    from satdet.data.viso_index import discover_sequences

    cfg = load_config(config_path)
    if overrides:
        cfg.update(overrides)

    exp_id = exp_id or str(cfg.get("exp_id") or Path(config_path).stem)
    run_dir = experiments_dir() / "runs" / exp_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # ---- 数据 ----
    index = discover_sequences(dataset_root) if dataset_root else _load_or_discover()
    data_cfg = cfg.get("data", {})
    train_seqs = [str(s) for s in data_cfg.get("train_seqs", [])]
    val_seqs = [str(s) for s in data_cfg.get("val_seqs", [])]
    if not train_seqs:
        raise ValueError("配置缺少 data.train_seqs")

    ds_kwargs = dict(
        mode=str(data_cfg.get("mode", "tile")),
        tile=int(data_cfg.get("tile", 256)),
        imgsz=int(data_cfg.get("imgsz", 512)),
        neg_per_frame=float(data_cfg.get("neg_per_frame", 1.0)),
        degrade=data_cfg.get("degrade", {}) or {},
        min_visible=float(data_cfg.get("min_visible", 0.5)),
        seed=int(cfg.get("seed", 0)),
    )
    train_ds = DetectionDataset(index, train_seqs, augment=True,
                               limit_frames=limit_frames, **ds_kwargs)
    if len(train_ds) == 0:
        raise RuntimeError("训练集为空，检查 tiling.min_visible 与序列选择")

    val_frames = _val_frames(index, val_seqs, val_limit)

    # ---- 模型 ----
    nc = int(cfg.get("model", {}).get("nc", 1))
    p2 = bool(cfg.get("model", {}).get("p2", True))
    model = build_yolov8n(nc=nc, p2=p2, use_eca=bool(cfg.get("model", {}).get("eca", True)))
    wpath = Path(cfg.get("pretrained", weights_dir() / "yolov8n_coco.pt"))
    if wpath.is_file():
        load_coco_weights(model, wpath)
    else:
        log.warning("未找到预训练权重 %s，将从头训练（不推荐）", wpath)
    if bool(cfg.get("freeze_backbone", True)):
        freeze_backbone(model, upto=int(cfg.get("freeze_upto", 9)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    tcfg = cfg.get("train", {})
    n_epochs = int(epochs or tcfg.get("epochs", 10))
    batch_size = int(tcfg.get("batch_size", 16))
    base_lr = float(tcfg.get("lr", 0.002))
    warmup = int(tcfg.get("warmup_iters", 200))
    workers = int(tcfg.get("workers", 4))
    eval_every = int(tcfg.get("eval_every", 2))
    amp = bool(tcfg.get("amp", True)) and device.type == "cuda"

    loss_cfg = cfg.get("loss", {})
    criterion = DetectionLoss(
        nc=nc, box_loss=str(loss_cfg.get("box", "nwd")),
        nwd_c=float(loss_cfg.get("nwd_c", 12.0)),
        w_box=float(loss_cfg.get("w_box", 7.5)),
        w_cls=float(loss_cfg.get("w_cls", 0.5)),
        w_dfl=float(loss_cfg.get("w_dfl", 1.5)),
        topk=int(loss_cfg.get("topk", 13)),
        center_radius=float(loss_cfg.get("center_radius", 1.5)),
    ).to(device)

    # ---- 知识蒸馏（可选）：大模型(512) 教 小模型(320) ----
    distiller = None
    dcfg = cfg.get("distill", {}) or {}
    if bool(dcfg.get("enabled", False)):
        from satdet.models.distill import FeatureDistiller

        tpath = Path(dcfg.get("teacher", weights_dir() / "a_nwd_p2_v3_best.pt"))
        if not tpath.is_file():
            raise FileNotFoundError(f"蒸馏需要教师权重: {tpath}")
        teacher = build_yolov8n(nc=nc, p2=p2, use_eca=bool(cfg.get("model", {}).get("eca", True)))
        ck = torch.load(tpath, map_location="cpu")
        teacher.load_state_dict(ck["model"] if "model" in ck else ck)
        teacher.to(device).eval()
        distiller = FeatureDistiller(teacher, teacher_imgsz=int(dcfg.get("teacher_imgsz", 512)),
                                     weight=float(dcfg.get("weight", 0.5)),
                                     proj=bool(dcfg.get("proj", True))).to(device)
        log.info("蒸馏已启用：教师 %s（%d 输入），权重 %.2f", tpath.name,
                 int(dcfg.get("teacher_imgsz", 512)), float(dcfg.get("weight", 0.5)))

    # ---- MobileOne 式重参数化（可选）：训练多分支、推理单分支 ----
    if bool(cfg.get("model", {}).get("rep", False)):
        from satdet.models.repconv import convert_model_to_repconv
        n_rep = convert_model_to_repconv(model)
        model.eval()          # 新建的 RepConv 默认 train 模式，先切回，保持初始化等价
        log.info("已启用重参数化：%d 个 Conv -> RepConv（推理前需 fuse）", n_rep)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=base_lr,
                                  weight_decay=float(tcfg.get("weight_decay", 5e-4)))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                        num_workers=workers, collate_fn=collate_fn,
                        pin_memory=(device.type == "cuda"), drop_last=False,
                        persistent_workers=workers > 0)

    total_iters = max(1, n_epochs * len(loader))
    step = 0
    result = TrainResult(run_dir=run_dir, n_train_samples=len(train_ds),
                         n_val_frames=len(val_frames))
    best_path = weights_dir() / f"{exp_id}_best.pt"
    best_map = -1.0
    t_start = time.time()

    log.info("实验 %s | 训练样本 %d | 验证帧 %d | 可训练参数 %.2fM | device=%s | AMP=%s",
             exp_id, len(train_ds), len(val_frames),
             count_parameters(model, trainable_only=True) / 1e6, device, amp)

    lr = base_lr
    for epoch in range(1, n_epochs + 1):
        model.train()
        ep_loss = 0.0
        ep_parts = {"box": 0.0, "cls": 0.0, "dfl": 0.0, "kd": 0.0}
        n_batches = 0
        t_ep = time.time()

        for imgs, targets in loader:
            lr = _lr_at(step, total_iters, base_lr, warmup)
            for g in optimizer.param_groups:
                g["lr"] = lr

            imgs = imgs.to(device, non_blocking=True)
            targets = [{k: (v.to(device) if torch.is_tensor(v) else v)
                        for k, v in t.items()} for t in targets]

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=amp):
                if distiller is not None:
                    feats = model.forward_neck(imgs)
                    preds = model.detect(feats)
                else:
                    feats = None
                    preds = model(imgs)
                losses = criterion(preds, targets, model.stride)
                if distiller is not None:
                    kd = distiller(feats, imgs)
                    losses["kd"] = kd
                    losses["total"] = losses["total"] + kd
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(params, max_norm=float(tcfg.get("grad_clip", 10.0)))
            scaler.step(optimizer)
            scaler.update()

            ep_loss += float(losses["total"])
            for k in ep_parts:
                ep_parts[k] += float(losses[k]) if k in losses else 0.0
            n_batches += 1
            step += 1

            if step % int(tcfg.get("log_every", 50)) == 0:
                log.info("  epoch %d/%d step %d/%d loss=%.3f (box %.3f cls %.3f dfl %.3f kd %.3f) lr=%.2e",
                         epoch, n_epochs, step, total_iters, float(losses["total"]),
                         float(losses["box"]), float(losses["cls"]), float(losses["dfl"]),
                         float(losses.get("kd", 0.0)), lr)

        rec = {
            "epoch": epoch,
            "loss": ep_loss / max(1, n_batches),
            "loss_box": ep_parts["box"] / max(1, n_batches),
            "loss_cls": ep_parts["cls"] / max(1, n_batches),
            "loss_dfl": ep_parts["dfl"] / max(1, n_batches),
            "loss_kd": round(ep_parts["kd"] / max(1, n_batches), 4),
            "lr": lr,
            "epoch_sec": round(time.time() - t_ep, 1),
        }

        if (epoch % eval_every == 0) or epoch == n_epochs:
            metrics = _validate(model, index, val_frames, cfg, device)
            rec.update(metrics)
            m = metrics.get("mAP30")
            if m is not None and m == m and m > best_map:
                best_map = m
                result.best_map30 = m
                result.best_epoch = epoch
                torch.save({"model": model.state_dict(), "epoch": epoch, "mAP30": m}, best_path)
                log.info("  * 新最优 mAP30=%.4f（epoch %d），已保存 %s", m, epoch, best_path.name)

        result.history.append(rec)
        log.info("epoch %d 完成: loss=%.3f mAP30=%s %.0fs", epoch, rec["loss"],
                 (f"{rec['mAP30']:.4f}" if "mAP30" in rec else "-"), rec["epoch_sec"])

    dump_yaml(cfg, run_dir / "config.yaml")
    (run_dir / "metrics.json").write_text(
        json.dumps(result.to_json(), ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("训练完成，耗时 %.1f 分钟；最优 mAP30=%.4f (epoch %d)；产物 %s",
             (time.time() - t_start) / 60, result.best_map30, result.best_epoch, run_dir)
    return result


def _load_or_discover():
    """优先读 artifacts 里的索引缓存，避免每次训练都扫盘。"""
    from satdet.core.paths import artifacts_dir
    from satdet.data.viso_index import VisoIndex, discover_sequences

    p = artifacts_dir() / "index" / "viso_index.json"
    if p.is_file():
        return VisoIndex.load(p)
    return discover_sequences()


def _validate(model, index, frames: list[tuple[str, int]], cfg: dict,
              device: torch.device) -> dict:
    """在验证帧上跑一遍图块/整帧推理并算指标。"""
    if not frames:
        return {}
    data_cfg = cfg.get("data", {})
    val_cfg = cfg.get("val", {})
    proposal = str(val_cfg.get("proposal",
                               "oracle" if data_cfg.get("mode") == "tile" else "full"))
    tile = int(data_cfg.get("tile", 256))
    imgsz = int(data_cfg.get("imgsz", 512))
    batch_size = int(val_cfg.get("batch_size", 8))

    det = Detector(model=model, device=device, nc=int(cfg.get("model", {}).get("nc", 1)),
                   conf_thr=float(val_cfg.get("conf_floor", 0.05)),
                   iou_thr=float(val_cfg.get("nms_iou", 0.5)),
                   use_amp=False)
    preds = det.evaluate(index, frames, proposal=proposal, tile=tile,
                         imgsz=imgsz, batch_size=batch_size)
    gts = {(seq_id, frame): index[seq_id].load_gt().frame_rows(frame).boxes
           for seq_id, frame in frames}
    res = evaluate_detections(preds, gts)
    return {
        "mAP30": res.map30, "mAP50": res.map50, "mAP50_95": res.map50_95,
        "Precision": res.precision, "Recall": res.recall, "F1": res.f1,
        "recall_center2px": res.recall_center2px,
        **{f"recall_{k}": v for k, v in res.recall_by_size.items()},
    }
