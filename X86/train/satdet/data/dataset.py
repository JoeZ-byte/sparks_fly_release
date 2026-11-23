"""训练/评测用数据集：支持"整帧"与"自适应图块"两种取样模式。

两种模式对应 A/B 组的两条路线
-----------------------------
``mode="full"``  整帧 letterbox 到 ``imgsz``。基线用。
                 1024 -> 512 等于把 6 px 目标缩成 3 px，是"原始痛点"的直接体现。
``mode="tile"``  按 GT 生成网格对齐图块（``tiling.roi_origins``），
                 裁 256 再放大到 512，目标从 6 px 变成 12 px，才可能被学到。

样本组织
--------
初始化时一次性把样本清单展开成 ``(frame_index, x0, y0)`` 三元组：
* 正样本：``roi_origins(该帧 GT)`` 的每个原点一个样本；
* 负样本：随机原点、块内无 GT（教模型"这是背景"），比例由 ``neg_per_frame`` 控制。

这样 ``__getitem__`` 只做「读图 -> 裁 -> resize -> 增强」，没有搜索逻辑。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from satdet.core.logging import get_logger
from satdet.core.paths import artifacts_dir
from satdet.data.copy_paste import PatchBank
from satdet.data.tiling import boxes_in_tile, roi_origins

__all__ = ["DetSample", "DetectionDataset", "build_dataset", "collate_fn"]

log = get_logger("dataset")


@dataclass(frozen=True)
class DetSample:
    """一个训练样本：某帧 + 某个图块原点（原点为 None 表示整帧模式）。"""

    seq_id: str
    frame: int                       # 1-based
    x0: int
    y0: int
    n_pos: int                       # 块内 GT 数（用于统计正负样本比例）


def _letterbox(img: np.ndarray, imgsz: int) -> tuple[np.ndarray, float, tuple[int, int]]:
    """等比缩放 + 右下补边，返回 (图像, 缩放比, 左上 padding)。"""
    h, w = img.shape[:2]
    scale = min(imgsz / h, imgsz / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    if (nh, nw) != (h, w):
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    out = np.full((imgsz, imgsz, 3), 114, np.uint8)
    out[:nh, :nw] = img
    return out, scale, (0, 0)


class DetectionDataset(Dataset):
    """检测数据集。

    Parameters
    ----------
    index : VisoIndex
    seq_ids : 参与本划分的序列
    mode : {"full", "tile"}
    tile : 图块边长（mode="tile" 时生效）
    imgsz : 网络输入边长
    augment : 是否做数据增强（离线增强，只用廉价变换）
    neg_per_frame : 每帧额外采几个纯背景图块
    min_visible : 框在图块内至少可见多少才算正样本
    limit_frames : 调试用，只取前 N 帧
    """

    def __init__(
        self,
        index,
        seq_ids: Sequence[str],
        mode: str = "tile",
        tile: int = 256,
        imgsz: int = 512,
        augment: bool = False,
        patch_bank: PatchBank | None = None,
        cp_prob: float = 0.0,
        cp_n_max: int = 3,
        cp_roads: dict | None = None,
        degrade: dict | None = None,
        neg_per_frame: float = 1.0,
        min_visible: float = 0.5,
        limit_frames: int | None = None,
        seed: int = 0,
    ) -> None:
        if mode not in ("full", "tile"):
            raise ValueError(f"mode 只能是 full/tile，收到 {mode}")
        if imgsz % 32 != 0:
            raise ValueError(f"imgsz 必须是 32 的倍数，收到 {imgsz}")

        self.index = index
        self.seq_ids = list(seq_ids)
        self.mode = mode
        self.tile = int(tile)
        self.imgsz = int(imgsz)
        self.augment = bool(augment)
        self._patch_bank = patch_bank
        self._cp_prob = float(cp_prob)
        self._cp_n_max = int(cp_n_max)
        self._cp_roads = cp_roads
        # JPEG 退化增强（对应"训练/推理分布对齐"：推理端本来就读压缩视频流）
        dg = degrade or {}
        self._jpeg_prob = float(dg.get("jpeg_prob", 0.0))
        self._jpeg_q = tuple(dg.get("jpeg_q", [30, 70]))
        self._noise_prob = float(dg.get("noise_prob", 0.0))
        self._noise_sigma = float(dg.get("noise_sigma", 3.0))
        self.neg_per_frame = int(neg_per_frame)
        self.min_visible = float(min_visible)

        self.samples: list[DetSample] = []
        self._gt_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        rng = np.random.default_rng(seed)
        self._build_samples(rng, limit_frames)

    # ------------------------------------------------------------------
    def _gt_of(self, seq_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """返回 (frame, boxes, labels)，同一次调用内按帧号排序好。"""
        info = self.index[seq_id]
        cache = getattr(self, "_gt_cache", {})
        if seq_id in cache:
            return cache[seq_id]
        gt = info.load_gt()
        order = np.argsort(gt.frame, kind="stable")
        payload = (gt.frame[order], gt.boxes[order], np.zeros(len(gt), dtype=np.int64))
        cache[seq_id] = payload
        return payload

    def _build_samples(self, rng: np.random.Generator, limit_frames: int | None) -> None:
        for seq_id in self.seq_ids:
            info = self.index[seq_id]
            frame_ids, boxes, _ = self._gt_of(seq_id)
            n_frames = info.eval_frame_limit or info.num_frames
            if limit_frames:
                n_frames = min(n_frames, limit_frames)

            # 按帧切分 GT（一次 searchsorted，避免逐帧过滤）
            starts = np.searchsorted(frame_ids, np.arange(1, n_frames + 2))

            for f in range(1, n_frames + 1):
                gt_boxes = boxes[starts[f - 1]:starts[f]]
                if self.mode == "full":
                    self.samples.append(DetSample(seq_id, f, 0, 0, len(gt_boxes)))
                    continue

                origins = roi_origins(gt_boxes, self.tile, info.width, info.height)
                for x0, y0 in origins:
                    keep, _ = boxes_in_tile(gt_boxes, int(x0), int(y0), self.tile,
                                            self.min_visible)
                    n_pos = int(keep.sum())
                    if n_pos == 0:
                        continue
                    self.samples.append(DetSample(seq_id, f, int(x0), int(y0), n_pos))

                # 负样本：随机原点，确保块内一个 GT 都没有
                n_neg = int(self.neg_per_frame) + (
                    1 if np.random.random() < (self.neg_per_frame % 1.0) else 0)
                for _ in range(n_neg):
                    x0 = int(rng.integers(0, max(1, info.width - self.tile + 1)))
                    y0 = int(rng.integers(0, max(1, info.height - self.tile + 1)))
                    if len(gt_boxes):
                        keep, _ = boxes_in_tile(gt_boxes, x0, y0, self.tile, self.min_visible)
                        if keep.any():
                            continue
                    self.samples.append(DetSample(seq_id, f, x0, y0, 0))

        log.info("数据集样本数: %d (mode=%s, tile=%d, imgsz=%d, 序列=%s)",
                 len(self.samples), self.mode, self.tile, self.imgsz, ",".join(self.seq_ids))

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, dict]:
        s = self.samples[i]
        info = self.index[s.seq_id]
        img = cv2.imread(str(info.image_path(s.frame)), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"读图失败: {info.image_path(s.frame)}")

        frame_ids, boxes, _ = self._gt_of(s.seq_id)
        lo = int(np.searchsorted(frame_ids, s.frame, side="left"))
        hi = int(np.searchsorted(frame_ids, s.frame, side="right"))
        gt = boxes[lo:hi]

        if self.mode == "full":
            img_lb, scale, _ = _letterbox(img, self.imgsz)
            out = img_lb
            local = gt * scale
        else:
            x0, y0 = s.x0, s.y0
            crop = img[y0:y0 + self.tile, x0:x0 + self.tile]
            if crop.shape[0] != self.tile or crop.shape[1] != self.tile:
                pad = np.full((self.tile, self.tile, 3), 114, np.uint8)
                pad[:crop.shape[0], :crop.shape[1]] = crop
                crop = pad
            scale = self.imgsz / float(self.tile)
            out = cv2.resize(crop, (self.imgsz, self.imgsz),
                             interpolation=cv2.INTER_LINEAR)
            keep, local_raw = boxes_in_tile(gt, x0, y0, self.tile, self.min_visible)
            local = local_raw.astype(np.float32) * scale

        out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

        if self.augment:
            road_test = None
            if self._cp_roads is not None:
                road = self._cp_roads.get(s.seq_id)
                if road is not None:
                    def road_test(fx: float, fy: float, _r=road) -> bool:
                        x = int(np.clip(fx, 0, _r.shape[1] - 1))
                        y = int(np.clip(fy, 0, _r.shape[0] - 1))
                        return bool(_r[y, x] > 0)
            out, local = self._augment(out, local, s.x0, s.y0, road_test)

        # 注意：labels 必须放在增强之后生成——copy-paste 会追加新框，
        # 提前生成会导致 labels 与 boxes 长度错位（贴进去的框没有标签）
        labels = np.zeros(len(local), dtype=np.int64)

        # 翻转/rot90 会产生负 stride 的视图，torch.from_numpy 不接受，必须物化
        out = np.ascontiguousarray(out)
        tensor = torch.from_numpy(out.transpose(2, 0, 1)).float().div_(255.0).contiguous()
        local = np.clip(local, 0, self.imgsz)
        target = {
            "boxes": torch.from_numpy(local.astype(np.float32)),
            "labels": torch.from_numpy(labels),
            "seq_id": s.seq_id,
            "frame": s.frame,
            "origin": torch.tensor([s.x0, s.y0], dtype=torch.int32),
        }
        return tensor, target

    # ------------------------------------------------------------------
    def _augment(self, img: np.ndarray, boxes: np.ndarray, x0: int = 0, y0: int = 0,
                 road_test=None) -> tuple[np.ndarray, np.ndarray]:
        """廉价增强：翻转 / rot90 / 亮度对比度。

        卫星图像的绝对朝向无意义，翻转与 90° 旋转是**物理合法**的；
        不做 mosaic（小目标会被切碎）与随机缩放（破坏 2 倍上采样设计）。
        """
        h, w = img.shape[:2]
        if np.random.rand() < 0.5:
            img = img[:, ::-1]
            if len(boxes):
                x1 = boxes[:, 0].copy()
                boxes[:, 0] = w - boxes[:, 2]
                boxes[:, 2] = w - x1
        if np.random.rand() < 0.5:
            img = img[::-1]
            if len(boxes):
                y1 = boxes[:, 1].copy()
                boxes[:, 1] = h - boxes[:, 3]
                boxes[:, 3] = h - y1
        k = int(np.random.randint(0, 4))
        if k:
            img = np.ascontiguousarray(np.rot90(img, k))
            for _ in range(k):
                if len(boxes):
                    x1 = boxes[:, 0].copy()
                    y1 = boxes[:, 1].copy()
                    boxes[:, 0] = y1
                    boxes[:, 1] = w - boxes[:, 2]
                    boxes[:, 2] = boxes[:, 3]
                    boxes[:, 3] = w - x1

        if np.random.rand() < 0.3:
            alpha = 1.0 + np.random.uniform(-0.2, 0.2)
            beta = np.random.uniform(-12, 12)
            img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

        # ---- 退化增强：模拟推理端的 JPEG 压缩与传感器噪声 ----
        # 推理端是压缩视频流，6px 目标的对比度会被压缩吃掉一部分；训练时不模拟，
        # 模型在真实数据上就"水土不服"。这里随机重压缩（q=30~70）+ 轻微高斯噪声。
        if self._jpeg_prob > 0 and np.random.rand() < self._jpeg_prob:
            q = int(np.random.randint(int(self._jpeg_q[0]), int(self._jpeg_q[1]) + 1))
            ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
            if ok:
                img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if self._noise_prob > 0 and np.random.rand() < self._noise_prob:
            noise = np.random.normal(0.0, self._noise_sigma, img.shape).astype(np.float32)
            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        # ---- Copy-Paste：真实小目标贴到任意位置（增强在 512 图上，贴片按 2 倍放大）----
        if self._patch_bank is not None and len(self._patch_bank)                 and np.random.rand() < self._cp_prob:
            rng = np.random.default_rng()
            for _ in range(int(rng.integers(1, self._cp_n_max + 1))):
                got = self._patch_bank.sample(rng)
                if got is None:
                    break
                patch, pbox = got
                ph, pw = patch.shape[:2]
                scale = float(self.imgsz) / float(self.tile)
                sp = cv2.resize(patch, (max(1, int(round(pw * scale))),
                                        max(1, int(round(ph * scale)))),
                                interpolation=cv2.INTER_LINEAR)
                sb = pbox * scale
                sh, sw = sp.shape[:2]
                if sw >= w - 8 or sh >= h - 8:
                    continue
                # 道路约束：贴片中心必须落在该序列的道路先验上（最多试 8 个位置）
                placed = False
                for _try in range(8):
                    x = int(rng.integers(0, w - sw))
                    y = int(rng.integers(0, h - sh))
                    if road_test is None:
                        placed = True
                        break
                    fx = x0 + (x + sw / 2.0) / 2.0      # 512 坐标 -> 原图坐标
                    fy = y0 + (y + sh / 2.0) / 2.0
                    if road_test(fx, fy):
                        placed = True
                        break
                if not placed:
                    continue
                nb = np.array([[x + sb[0, 0], y + sb[0, 1],
                                x + sb[0, 2], y + sb[0, 3]]], np.float32)
                if len(boxes):
                    bx1 = np.maximum(nb[0, 0], boxes[:, 0])
                    by1 = np.maximum(nb[0, 1], boxes[:, 1])
                    bx2 = np.minimum(nb[0, 2], boxes[:, 2])
                    by2 = np.minimum(nb[0, 3], boxes[:, 3])
                    iw = np.maximum(0.0, bx2 - bx1)
                    ih = np.maximum(0.0, by2 - by1)
                    inter = iw * ih
                    area_n = max(1.0, (nb[0, 2] - nb[0, 0]) * (nb[0, 3] - nb[0, 1]))
                    area_b = np.maximum(1.0, (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))
                    iou = inter / np.maximum(1.0, area_n + area_b - inter)
                    if float(iou.max()) > 0.3:
                        continue          # 别在同一处叠"分身"
                # 亮度 ±10% 抖动，防重复亮度指纹
                if rng.random() < 0.7:
                    jit = float(rng.uniform(0.9, 1.1))
                    sp = np.clip(sp.astype(np.float32) * jit, 0, 255).astype(np.uint8)
                img[y:y + sh, x:x + sw] = sp
                boxes = np.concatenate([boxes, nb], axis=0)
        return img, boxes


def build_dataset(index, seq_ids: Sequence[str], cfg: dict, split: str,
                  limit_frames: int | None = None) -> DetectionDataset:
    """按配置文件构造数据集。

    cfg 结构见 ``configs/train/*.yaml`` 的 ``data`` 段。
    """
    tile_cfg = cfg.get("tiling", {})
    mode = cfg.get("mode", "tile")
    cp_cfg = cfg.get("copy_paste") or {}
    patch_bank = None
    cp_prob = 0.0
    cp_n_max = 3
    degrade_cfg = cfg.get("degrade") or {}
    cp_roads = None
    if split == "train" and cp_cfg.get("prob", 0.0) > 0:
        bank_path = Path(str(cp_cfg.get("bank", PatchBank.default_path())))
        patch_bank = PatchBank()
        if bank_path.is_file():
            patch_bank.load(bank_path)
        else:
            patch_bank.build(index, seq_ids,
                             max_patches=int(cp_cfg.get("max_patches", 3000)))
            patch_bank.save(bank_path)
        cp_prob = float(cp_cfg["prob"])
        cp_n_max = int(cp_cfg.get("n_max", 3))
        if cp_cfg.get("roads", False):
            from satdet.motion.road_mask import load_road_mask

            cp_roads = {}
            for sid in seq_ids:
                m = load_road_mask(artifacts_dir() / "road" / f"{sid}_roadmask.npz")
                if m is not None:
                    cp_roads[sid] = m
    return DetectionDataset(
        index=index,
        seq_ids=seq_ids,
        mode=mode,
        tile=int(tile_cfg.get("tile", 256)),
        imgsz=int(cfg.get("imgsz", 512)),
        augment=(split == "train") and bool(cfg.get("augment", True)),
        patch_bank=patch_bank,
        cp_prob=cp_prob,
        cp_n_max=cp_n_max,
        cp_roads=cp_roads,
        degrade=degrade_cfg,
        neg_per_frame=float(cfg.get("neg_per_frame", 1.0)),
        min_visible=float(tile_cfg.get("min_visible", 0.5)),
        limit_frames=limit_frames,
        seed=int(cfg.get("seed", 0)),
    )


def collate_fn(batch: list[tuple[torch.Tensor, dict]]) -> tuple[torch.Tensor, list[dict]]:
    """把 batch 拼成 (B,3,H,W) + list[dict]，与 ``DetectionLoss`` 的入参一致。"""
    imgs = torch.stack([b[0] for b in batch], dim=0)
    targets = [b[1] for b in batch]
    return imgs, targets
