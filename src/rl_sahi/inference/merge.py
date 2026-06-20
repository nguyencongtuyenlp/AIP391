from __future__ import annotations

from pathlib import Path

import numpy as np

from rl_sahi.common.boxes import nms_numpy


def save_prediction_txt(
    path: Path,
    boxes: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    sources: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for cls, score, box, source in zip(classes, scores, boxes, sources):
            x1, y1, x2, y2 = [float(v) for v in box]
            f.write(f"{int(cls)} {float(score):.6f} {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} {int(source)}\n")


def class_aware_nms(boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray, iou_threshold: float) -> np.ndarray:
    if len(boxes) == 0:
        return np.zeros((0,), dtype=np.int64)
    keep_parts: list[np.ndarray] = []
    for cls in np.unique(classes.astype(np.int64)):
        idx = np.flatnonzero(classes.astype(np.int64) == cls)
        keep_local = nms_numpy(boxes[idx], scores[idx], iou_threshold)
        keep_parts.append(idx[keep_local])
    keep = np.concatenate(keep_parts, axis=0) if keep_parts else np.zeros((0,), dtype=np.int64)
    return keep[np.argsort(scores[keep])[::-1]].astype(np.int64)
