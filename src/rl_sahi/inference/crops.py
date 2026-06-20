from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from rl_sahi.common.device import DeviceLike, configure_ultralytics_for_device, resolve_torch_device


def crop_roi(image_path: Path, roi: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    x1, y1, x2, y2 = [int(round(v)) for v in roi]
    x1 = max(x1, 0)
    y1 = max(y1, 0)
    x2 = min(x2, image.shape[1])
    y2 = min(y2, image.shape[0])
    return image[y1:y2, x1:x2].copy(), (x1, y1)


def run_yolo_on_crop(
    model: YOLO,
    image_path: Path,
    roi: np.ndarray,
    imgsz: int,
    conf: float,
    iou: float,
    max_det: int,
    device: DeviceLike,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    crop, offset = crop_roi(image_path, roi)
    if crop.size == 0:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )
    resolved_device = resolve_torch_device(device)
    configure_ultralytics_for_device(resolved_device)
    results = model.predict(
        crop,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        max_det=max_det,
        device=resolved_device,
        verbose=False,
    )
    result = results[0]
    if result.boxes is None or len(result.boxes) == 0:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )
    boxes = result.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
    boxes[:, [0, 2]] += offset[0]
    boxes[:, [1, 3]] += offset[1]
    scores = result.boxes.conf.detach().cpu().numpy().astype(np.float32)
    classes = result.boxes.cls.detach().cpu().numpy().astype(np.float32)
    return boxes, scores, classes
