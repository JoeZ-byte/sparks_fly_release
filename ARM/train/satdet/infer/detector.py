"""推理封装：把"图块/整帧 -> 检测框（全图坐标）"这条路走通。

三种提案方式（proposal）
-----------------------
``full``    整帧 letterbox 推理 —— 基线
``oracle``  用 GT 生成的图块 —— A 组的**精度上界**（把提案漏检排除掉，只看检测器）
``grid``    均匀网格图块 —— 只做图块化的对照

M4 之后加 ``motion``（帧差 + 背景模型驱动），用它测真实提案的端到端指标。
接口先定死，后面的提案模块只需实现 ``propose(seq, frame) -> origins``。
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch

from satdet.core.logging import get_logger
from satdet.data.tiling import grid_origins, roi_origins
from satdet.infer.nms import nms_numpy
from satdet.models.yolov8 import decode_predictions

__all__ = ["Detector", "decode_batch"]

log = get_logger("detector")


@dataclass
class Detector:
    """把模型包成"输入图像 -> 检测框"的调用。"""

    model: torch.nn.Module
    device: torch.device | str = "cuda"
    nc: int = 1
    conf_thr: float = 0.25
    iou_thr: float = 0.5
    max_det: int = 300
    max_nms: int = 1000
    use_amp: bool = True
    tta: bool = False            # 推理期水平翻转 TTA（离线评测用，代价 2×）

    def __post_init__(self) -> None:
        self.device = torch.device(self.device)
        self.model.to(self.device).eval()
        self.reg_max = int(getattr(self.model.detect, "reg_max", 16))

    @torch.no_grad()
    def predict_tiles(self, batch: torch.Tensor) -> list[tuple[np.ndarray, np.ndarray]]:
        """对一个 batch 的图块（已归一化 ``(B,3,H,W)``）出框，坐标为图块局部坐标。"""
        b = batch.to(self.device, non_blocking=True)
        with torch.autocast("cuda", enabled=(self.use_amp and self.device.type == "cuda")):
            preds = self.model(b)
        preds = [p.float() for p in preds]
        boxes, scores = decode_predictions(preds, self.model.stride, self.nc, self.reg_max)
        cls_scores = scores.amax(dim=-1).float().cpu().numpy()
        boxes = boxes.float().cpu().numpy()

        out: list[tuple[np.ndarray, np.ndarray]] = []
        for i in range(boxes.shape[0]):
            s = cls_scores[i]
            keep = s >= self.conf_thr
            bx, sc = boxes[i][keep], s[keep]
            if len(bx) > self.max_nms:
                # 先按分数截断再做 NMS：否则密集 anchor 会让 O(n^2) 的 NMS 拖垮帧率
                top = np.argpartition(-sc, self.max_nms)[: self.max_nms]
                bx, sc = bx[top], sc[top]
            if len(bx):
                idx = nms_numpy(bx, sc, self.iou_thr, self.max_det)
                bx, sc = bx[idx], sc[idx]
            out.append((bx.astype(np.float32), sc.astype(np.float32)))
        return out

    def infer_frame(self, index, seq_id: str, frame: int, proposal: str = "full",
                    tile: int = 256, imgsz: int = 512, batch_size: int = 8,
                    pre: str = "none"
                    ) -> tuple[np.ndarray, np.ndarray]:
        """对一帧出框（全图坐标）。"""
        info = index[seq_id]
        img = cv2.imread(str(info.image_path(frame)), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"读图失败: {info.image_path(frame)}")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if proposal == "full":
            canvas, scale = _letterbox(rgb, imgsz)
            results = self.predict_tiles(_to_tensor(canvas))
            boxes, scores = results[0]
            if len(boxes):
                boxes = boxes / scale
            return boxes, scores

        if proposal == "oracle":
            gt_boxes = info.load_gt().frame_rows(frame).boxes
            origins = roi_origins(gt_boxes, tile, info.width, info.height)
        elif proposal == "grid":
            origins = grid_origins(info.width, info.height, tile)
        else:
            raise ValueError(f"未知 proposal: {proposal}")

        all_boxes, all_scores = [], []
        scale = imgsz / float(tile)
        for flip in ((False, True) if self.tta else (False,)):
            for start in range(0, len(origins), batch_size):
                chunk = origins[start:start + batch_size]
                tensors = []
                for x0, y0 in chunk:
                    crop = rgb[y0:y0 + tile, x0:x0 + tile]
                    if crop.shape[0] != tile or crop.shape[1] != tile:
                        pad = np.full((tile, tile, 3), 114, np.uint8)
                        pad[:crop.shape[0], :crop.shape[1]] = crop
                        crop = pad
                    # 图块级增强（docs/14 E3/E4）：只在进网前做，不改标定/不改坐标
                    if pre != "none":
                        crop = apply_tile_pre(crop, pre)
                    if flip:
                        crop = crop[:, ::-1]
                    resized = cv2.resize(crop, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
                    tensors.append(_to_tensor(resized))
                results = self.predict_tiles(torch.cat(tensors, 0))
                for (x0, y0), (bx, sc) in zip(chunk, results):
                    if len(bx) == 0:
                        continue
                    if flip:
                        # 翻转回原坐标：x' = tile - x（框左右互换）
                        bx = bx.copy()
                        x1 = imgsz - bx[:, 2]
                        x2 = imgsz - bx[:, 0]
                        bx[:, 0], bx[:, 2] = x1, x2
                    bx = bx / scale + np.asarray([x0, y0, x0, y0], dtype=np.float32)
                    all_boxes.append(bx)
                    all_scores.append(sc)

        if not all_boxes:
            return np.zeros((0, 4), np.float32), np.zeros(0, np.float32)

        boxes = np.concatenate(all_boxes, 0)
        scores = np.concatenate(all_scores, 0)
        keep = nms_numpy(boxes, scores, self.iou_thr, self.max_det)
        return boxes[keep], scores[keep]

    def evaluate(self, index, frames: list[tuple[str, int]], proposal: str = "full",
                 tile: int = 256, imgsz: int = 512, batch_size: int = 8,
                 pre: str = "none") -> list[dict]:
        """对若干 (seq, frame) 出预测，返回评测所需的列表。"""
        preds: list[dict] = []
        for k, (seq_id, frame) in enumerate(frames):
            boxes, scores = self.infer_frame(index, seq_id, frame, proposal,
                                             tile, imgsz, batch_size, pre)
            preds.append({"seq": seq_id, "frame": frame,
                          "boxes": boxes, "scores": scores})
            if (k + 1) % 200 == 0:
                log.info("  已推理 %d/%d 帧", k + 1, len(frames))
        return preds


def _letterbox(img: np.ndarray, imgsz: int) -> tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    scale = min(imgsz / h, imgsz / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    if (nh, nw) != (h, w):
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    out = np.full((imgsz, imgsz, 3), 114, np.uint8)
    out[:nh, :nw] = img
    return out, scale


def _to_tensor(img_rgb: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float().div_(255.0)
    return t.unsqueeze(0).contiguous()


def decode_batch(model, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """辅助函数：直接返回 (boxes, scores)，供导出/单测使用。"""
    preds = model(x)
    return decode_predictions(preds, model.stride, model.detect.nc, model.detect.reg_max)


def _deblock_jpeg(img, thr=25):
    """弱去块：只对 8x8 块边界做一次跨边界平滑，且只在边界两侧差不大时生效。

    强边缘（真实目标/道路）不会被抹平：差值超过 thr 就跳过。
    """
    out = img.astype(np.int16)
    h, w = out.shape[:2]
    for x in range(8, w - 1, 8):
        p0, p1 = out[:, x - 1], out[:, x]
        mask = (np.abs(p1 - p0).sum(axis=1) < thr * 3)[:, None]
        mid = (p0 * 2 + p1 * 2 + out[:, x - 2] + out[:, x + 1]) // 6
        out[:, x - 1] = np.where(mask, mid, p0)
        out[:, x] = np.where(mask, mid, p1)
    for y in range(8, h - 1, 8):
        p0, p1 = out[y - 1, :], out[y, :]
        mask = (np.abs(p1 - p0).sum(axis=1) < thr * 3)[:, None]
        mid = (p0 * 2 + p1 * 2 + out[y - 2, :] + out[y + 1, :]) // 6
        out[y - 1, :] = np.where(mask, mid, p0)
        out[y, :] = np.where(mask, mid, p1)
    return np.clip(out, 0, 255).astype(np.uint8)


_CLAHE = {}
_CLAHE_CFG = {"clahe": (2.0, 4, 4), "clahe_soft": (1.0, 8, 8), "clahe_hard": (4.0, 4, 4)}


def _clahe_lab(img, mode="clahe"):
    """CLAHE 作用在 LAB 的 L 通道；三档强度用于看趋势。"""
    key = mode if mode in _CLAHE_CFG else "clahe"
    clip, gy, gx = _CLAHE_CFG[key]
    if key not in _CLAHE:
        _CLAHE[key] = cv2.createCLAHE(clipLimit=clip, tileGridSize=(gy, gx))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = _CLAHE[key].apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def apply_tile_pre(img, mode):
    """图块级预处理：none / clahe / deblock / clahe+deblock。"""
    if mode == "none":
        return img
    out = img
    if "clahe" in mode:
        out = _clahe_lab(out, mode)
    if "deblock" in mode:
        out = _deblock_jpeg(out)
    if mode == "up2":
        # 纯重采样对照：先 2x 上采样（bicubic），再降回原尺寸
        h, w = out.shape[:2]
        up = cv2.resize(out, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
        out = cv2.resize(up, (w, h), interpolation=cv2.INTER_AREA)
    return out