from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

from rl_sahi.common.boxes import area, clip_boxes, iou_matrix
from rl_sahi.common.cache import DetectionCache
from rl_sahi.common.class_mapping import ClassMapping
from rl_sahi.common.data import image_to_label_path, iter_images, read_yolo_labels
from rl_sahi.common.device import resolve_torch_device
from rl_sahi.detection.yolo import detect_one_image, load_yolo
from rl_sahi.inference.config import InferenceConfig
from rl_sahi.inference.crops import run_yolo_on_crop
from rl_sahi.inference.merge import class_aware_nms
from rl_sahi.inference.pipeline import _attempt_overlap, _filter_classes, _merged_source_counts, get_initial_detection
from rl_sahi.inference.rollout import rollout_one_slice
from rl_sahi.rl.checkpoint import load_policy
from rl_sahi.rl.slice_env import SliceEnv
from rl_sahi.rl.state_config import StateConfig
from rl_sahi.rl.state_maps import build_detection_map, build_ranking_density
from rl_sahi.rl.hotspot_env import HotspotEnv
from rl_sahi.rl.yield_env import YieldAwareHotspotEnv
from rl_sahi.rl.multiscale_env import MultiScaleYieldEnv
from rl_sahi.rl.adaptiveconf_env import AdaptiveConfEnv


@dataclass(slots=True)
class BenchmarkConfig:
    iou_threshold: float = 0.5
    fixed_slice_fraction: float = 0.35
    fixed_overlap: float = 0.2
    small_area_percentile: float = 40.0
    target_classes: tuple[int, ...] = (0, 2, 3, 5, 8, 9)
    class_mapping: ClassMapping = field(default_factory=ClassMapping)


def _empty_preds() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.zeros((0, 4), dtype=np.float32),
        np.zeros((0,), dtype=np.float32),
        np.zeros((0,), dtype=np.float32),
    )


def _read_gt(
    image_path: Path,
    image_root: Path,
    label_root: Path,
    target_classes: tuple[int, ...],
    class_mapping: ClassMapping,
):
    classes, boxes = read_yolo_labels(image_to_label_path(image_path, image_root, label_root), _image_shape(image_path))
    classes = class_mapping.map_label_classes(classes)
    if target_classes:
        mask = np.isin(classes.astype(np.int64), np.asarray(target_classes, dtype=np.int64))
        classes, boxes = classes[mask], boxes[mask]
    return boxes.astype(np.float32), classes.astype(np.float32)


def _image_shape(image_path: Path) -> tuple[int, int]:
    import cv2

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    h, w = image.shape[:2]
    return int(h), int(w)


def _small_area_threshold(
    images: list[Path],
    image_root: Path,
    label_root: Path,
    target_classes: tuple[int, ...],
    percentile: float,
    class_mapping: ClassMapping,
) -> float:
    ratios: list[np.ndarray] = []
    for image_path in images:
        boxes, _classes = _read_gt(image_path, image_root, label_root, target_classes, class_mapping)
        if len(boxes) == 0:
            continue
        h, w = _image_shape(image_path)
        ratios.append(area(boxes) / max(float(h * w), 1.0))
    if not ratios:
        return 0.0
    return float(np.percentile(np.concatenate(ratios, axis=0), percentile))


def _merge_predictions(
    image_shape: tuple[int, int],
    merge_iou: float,
    boxes_parts: list[np.ndarray],
    scores_parts: list[np.ndarray],
    classes_parts: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    boxes = np.concatenate(boxes_parts, axis=0) if boxes_parts else np.zeros((0, 4), dtype=np.float32)
    scores = np.concatenate(scores_parts, axis=0) if scores_parts else np.zeros((0,), dtype=np.float32)
    classes = np.concatenate(classes_parts, axis=0) if classes_parts else np.zeros((0,), dtype=np.float32)
    if len(boxes) == 0:
        return _empty_preds()
    boxes = clip_boxes(boxes, image_shape)
    keep = class_aware_nms(boxes, scores, classes, merge_iou)
    return boxes[keep], scores[keep], classes[keep]


def _full_predictions(det: DetectionCache, cfg: InferenceConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = det.scores >= cfg.output_conf
    boxes, scores = det.boxes[mask], det.scores[mask]
    classes = cfg.class_mapping.map_model_classes(det.classes[mask])
    return _filter_classes(boxes, scores, classes, cfg.target_classes)


def _fixed_grid_rois(image_shape: tuple[int, int], fraction: float, overlap: float) -> list[np.ndarray]:
    h, w = image_shape
    side = max(1.0, min(h, w) * float(fraction))
    stride = max(1.0, side * (1.0 - float(overlap)))
    xs = list(np.arange(0.0, max(w - side, 0.0) + 1.0, stride))
    ys = list(np.arange(0.0, max(h - side, 0.0) + 1.0, stride))
    if not xs or xs[-1] < w - side:
        xs.append(max(w - side, 0.0))
    if not ys or ys[-1] < h - side:
        ys.append(max(h - side, 0.0))
    rois: list[np.ndarray] = []
    for y in ys:
        for x in xs:
            rois.append(np.asarray([x, y, min(x + side, w), min(y + side, h)], dtype=np.float32))
    return rois


def _predict_fixed_sahi(
    model: YOLO,
    image_path: Path,
    det: DetectionCache,
    cfg: InferenceConfig,
    bench_cfg: BenchmarkConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    full_boxes, full_scores, full_classes = _full_predictions(det, cfg)
    boxes_parts = [full_boxes]
    scores_parts = [full_scores]
    classes_parts = [full_classes]
    rois = _fixed_grid_rois(det.image_shape, bench_cfg.fixed_slice_fraction, bench_cfg.fixed_overlap)
    for roi in rois:
        boxes_i, scores_i, classes_i = run_yolo_on_crop(
            model,
            image_path,
            roi,
            imgsz=cfg.slice_imgsz,
            conf=cfg.output_conf,
            iou=cfg.iou,
            max_det=cfg.max_det,
            device=cfg.device,
        )
        classes_i = cfg.class_mapping.map_model_classes(classes_i)
        boxes_i, scores_i, classes_i = _filter_classes(boxes_i, scores_i, classes_i, cfg.target_classes)
        boxes_parts.append(boxes_i)
        scores_parts.append(scores_i)
        classes_parts.append(classes_i)
    boxes, scores, classes = _merge_predictions(det.image_shape, cfg.merge_iou, boxes_parts, scores_parts, classes_parts)
    return boxes, scores, classes, len(rois)


def _predict_rl_sahi(
    model: YOLO,
    policy,
    device_t: torch.device,
    image_path: Path,
    det: DetectionCache,
    cfg: InferenceConfig,
    env_cfg,
    state_cfg: StateConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    full_boxes, full_scores, full_classes = _full_predictions(det, cfg)
    slice_boxes_all: list[np.ndarray] = []
    slice_scores_all: list[np.ndarray] = []
    slice_classes_all: list[np.ndarray] = []
    accepted_rois: list[np.ndarray] = []
    attempted_rois: list[np.ndarray] = []
    _base_count, accepted_new_count = _merged_source_counts(
        full_boxes, full_scores, full_classes, [], [], [], det.image_shape, cfg.merge_iou
    )

    max_attempts = int(cfg.max_slice_attempts) if cfg.max_slice_attempts > 0 else int(env_cfg.max_slices * 2)
    for _attempt_idx in range(1, max_attempts + 1):
        if len(accepted_rois) >= env_cfg.max_slices:
            break
        history_arr = np.stack(attempted_rois).astype(np.float32) if attempted_rois else np.zeros((0, 4), dtype=np.float32)
        overlap_arr = np.stack(accepted_rois).astype(np.float32) if accepted_rois else np.zeros((0, 4), dtype=np.float32)
        env = SliceEnv(
            det,
            None,
            env_cfg=env_cfg,
            state_cfg=state_cfg,
            previous_rois=history_arr,
            overlap_rois=overlap_arr,
            target_classes=cfg.target_classes,
            class_mapping=cfg.class_mapping,
        )
        roi, _actions, info = rollout_one_slice(policy, env, device_t)
        if info.get("stop_due_to_old_overlap", False):
            repeat_attempt_overlap = _attempt_overlap(roi, attempted_rois)
            attempted_rois.append(roi)
            if repeat_attempt_overlap >= 0.95:
                break
            continue
        if cfg.require_stop_for_acceptance and info.get("stop_due_to_max_steps", False):
            repeat_attempt_overlap = _attempt_overlap(roi, attempted_rois)
            attempted_rois.append(roi)
            if repeat_attempt_overlap >= 0.95:
                break
            continue
        if cfg.require_stop_for_acceptance and info.get("stop_due_to_stalled_roi", False):
            repeat_attempt_overlap = _attempt_overlap(roi, attempted_rois)
            attempted_rois.append(roi)
            if repeat_attempt_overlap >= 0.95:
                break
            continue
        boxes_i, scores_i, classes_i = run_yolo_on_crop(
            model,
            image_path,
            roi,
            imgsz=cfg.slice_imgsz,
            conf=cfg.output_conf,
            iou=cfg.iou,
            max_det=cfg.max_det,
            device=cfg.device,
        )
        attempted_rois.append(roi)
        classes_i = cfg.class_mapping.map_model_classes(classes_i)
        boxes_i, scores_i, classes_i = _filter_classes(boxes_i, scores_i, classes_i, cfg.target_classes)
        _full_after, new_count_after = _merged_source_counts(
            full_boxes,
            full_scores,
            full_classes,
            [*slice_boxes_all, boxes_i],
            [*slice_scores_all, scores_i],
            [*slice_classes_all, classes_i],
            det.image_shape,
            cfg.merge_iou,
        )
        if new_count_after - accepted_new_count < int(cfg.min_slice_detections):
            continue
        accepted_rois.append(roi)
        slice_boxes_all.append(boxes_i)
        slice_scores_all.append(scores_i)
        slice_classes_all.append(classes_i)
        accepted_new_count = new_count_after

    boxes, scores, classes = _merge_predictions(
        det.image_shape,
        cfg.merge_iou,
        [full_boxes, *slice_boxes_all],
        [full_scores, *slice_scores_all],
        [full_classes, *slice_classes_all],
    )
    return boxes, scores, classes, len(accepted_rois)


def _density_guided_rois(det: DetectionCache, state_cfg: StateConfig, bench_cfg: BenchmarkConfig, k: int, residual: bool = False, output_conf: float = 0.25) -> list[np.ndarray]:
    if residual:
        dens = build_ranking_density(det.boxes, det.scores, det.image_shape, state_cfg, use_residual=True, output_conf=output_conf)
    else:
        dens = build_detection_map(det.boxes, det.scores, det.image_shape, state_cfg)[2]
    grid = int(state_cfg.grid_size)
    floor = (2.0 - 0.5) / max(float(state_cfg.count_norm), 1.0)
    h, w = det.image_shape
    side = max(1.0, min(h, w) * float(bench_cfg.fixed_slice_fraction))
    flat = dens.reshape(-1)
    order = np.argsort(flat)[::-1]
    rois: list[np.ndarray] = []
    centers: list[tuple[float, float]] = []
    for flat_idx in order:
        if float(flat[flat_idx]) < floor or len(rois) >= int(k):
            break
        gy, gx = divmod(int(flat_idx), grid)
        cx = (gx + 0.5) * w / grid
        cy = (gy + 0.5) * h / grid
        if any(abs(cx - ux) < side * 0.5 and abs(cy - uy) < side * 0.5 for ux, uy in centers):
            continue
        x1 = float(np.clip(cx - side / 2.0, 0.0, max(w - side, 0.0)))
        y1 = float(np.clip(cy - side / 2.0, 0.0, max(h - side, 0.0)))
        rois.append(np.asarray([x1, y1, min(x1 + side, w), min(y1 + side, h)], dtype=np.float32))
        centers.append((cx, cy))
    return rois


def _predict_density_guided(
    model: YOLO,
    image_path: Path,
    det: DetectionCache,
    cfg: InferenceConfig,
    bench_cfg: BenchmarkConfig,
    state_cfg: StateConfig,
    k: int,
    residual: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    full_boxes, full_scores, full_classes = _full_predictions(det, cfg)
    boxes_parts = [full_boxes]
    scores_parts = [full_scores]
    classes_parts = [full_classes]
    rois = _density_guided_rois(det, state_cfg, bench_cfg, k, residual=residual, output_conf=cfg.output_conf)
    for roi in rois:
        boxes_i, scores_i, classes_i = run_yolo_on_crop(
            model, image_path, roi, imgsz=cfg.slice_imgsz, conf=cfg.output_conf, iou=cfg.iou, max_det=cfg.max_det, device=cfg.device
        )
        classes_i = cfg.class_mapping.map_model_classes(classes_i)
        boxes_i, scores_i, classes_i = _filter_classes(boxes_i, scores_i, classes_i, cfg.target_classes)
        boxes_parts.append(boxes_i)
        scores_parts.append(scores_i)
        classes_parts.append(classes_i)
    boxes, scores, classes = _merge_predictions(det.image_shape, cfg.merge_iou, boxes_parts, scores_parts, classes_parts)
    return boxes, scores, classes, len(rois)


def _predict_hotspot_rl(
    model: YOLO,
    policy,
    device_t: torch.device,
    image_path: Path,
    det: DetectionCache,
    cfg: InferenceConfig,
    env_cfg,
    state_cfg: StateConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    full_boxes, full_scores, full_classes = _full_predictions(det, cfg)
    boxes_parts = [full_boxes]
    scores_parts = [full_scores]
    classes_parts = [full_classes]
    env = HotspotEnv(det, None, env_cfg=env_cfg, state_cfg=state_cfg, target_classes=cfg.target_classes, class_mapping=cfg.class_mapping)
    rois = env.rollout_rois(policy, device_t)
    for roi in rois:
        boxes_i, scores_i, classes_i = run_yolo_on_crop(
            model, image_path, roi, imgsz=cfg.slice_imgsz, conf=cfg.output_conf, iou=cfg.iou, max_det=cfg.max_det, device=cfg.device
        )
        classes_i = cfg.class_mapping.map_model_classes(classes_i)
        boxes_i, scores_i, classes_i = _filter_classes(boxes_i, scores_i, classes_i, cfg.target_classes)
        boxes_parts.append(boxes_i)
        scores_parts.append(scores_i)
        classes_parts.append(classes_i)
    boxes, scores, classes = _merge_predictions(det.image_shape, cfg.merge_iou, boxes_parts, scores_parts, classes_parts)
    return boxes, scores, classes, len(rois)


def _predict_yield_rl(model, policy, device_t, image_path, det, cfg, env_cfg, state_cfg):
    """Yield-aware agent (CROP/SKIP, observe live yield) — chay CUNG pipeline GPU nhu cac method khac."""
    env = YieldAwareHotspotEnv(
        det, None, raw_yields=np.zeros(64, dtype=np.float32), real_yields=None,
        env_cfg=env_cfg, state_cfg=state_cfg, target_classes=cfg.target_classes, class_mapping=cfg.class_mapping,
    )
    full_boxes, full_scores, full_classes = _full_predictions(det, cfg)
    boxes_parts = [full_boxes]; scores_parts = [full_scores]; classes_parts = [full_classes]
    state = env.reset()
    for _ in range(len(env.rois) + 1):
        with torch.no_grad():
            q = policy(torch.from_numpy(state).float().unsqueeze(0).to(device_t))
            action = int(q.argmax(dim=1).item())
        if action == 0 and env.i < len(env.rois):
            roi = env.rois[env.i]
            boxes_i, scores_i, classes_i = run_yolo_on_crop(model, image_path, roi, imgsz=cfg.slice_imgsz, conf=cfg.output_conf, iou=cfg.iou, max_det=cfg.max_det, device=cfg.device)
            classes_i = cfg.class_mapping.map_model_classes(classes_i)
            boxes_i, scores_i, classes_i = _filter_classes(boxes_i, scores_i, classes_i, cfg.target_classes)
            env.raw_yields[env.i] = int((iou_matrix(boxes_i, full_boxes).max(1) < 0.5).sum()) if len(boxes_i) and len(full_boxes) else len(boxes_i)
            boxes_parts.append(boxes_i); scores_parts.append(scores_i); classes_parts.append(classes_i)
        result = env.step(action)
        state = result.state
        if result.done:
            break
    boxes, scores, classes = _merge_predictions(det.image_shape, cfg.merge_iou, boxes_parts, scores_parts, classes_parts)
    return boxes, scores, classes, len(env.placed)


def _random_k_rois(det, state_cfg, bench_cfg, k, rng):
    """Baseline DOC LAP: cat K hotspot NGAU NHIEN (trong cell vuot floor) — pha tautology 'agent ⊆ density top-K'."""
    dens = build_detection_map(det.boxes, det.scores, det.image_shape, state_cfg)[2]
    floor = (2.0 - 0.5) / max(float(state_cfg.count_norm), 1.0)
    grid = int(state_cfg.grid_size); h, w = det.image_shape
    side = max(1.0, min(h, w) * float(bench_cfg.fixed_slice_fraction))
    cand = np.where(dens.reshape(-1) >= floor)[0]
    rng.shuffle(cand)
    rois: list[np.ndarray] = []
    centers: list[tuple[float, float]] = []
    for flat_idx in cand:
        if len(rois) >= int(k):
            break
        gy, gx = divmod(int(flat_idx), grid)
        cx = (gx + 0.5) * w / grid; cy = (gy + 0.5) * h / grid
        if any(abs(cx - ux) < side * 0.5 and abs(cy - uy) < side * 0.5 for ux, uy in centers):
            continue
        x1 = float(np.clip(cx - side / 2.0, 0.0, max(w - side, 0.0)))
        y1 = float(np.clip(cy - side / 2.0, 0.0, max(h - side, 0.0)))
        rois.append(np.asarray([x1, y1, min(x1 + side, w), min(y1 + side, h)], dtype=np.float32))
        centers.append((cx, cy))
    return rois


def _predict_random_k(model, image_path, det, cfg, bench_cfg, state_cfg, k, rng):
    full_boxes, full_scores, full_classes = _full_predictions(det, cfg)
    boxes_parts = [full_boxes]; scores_parts = [full_scores]; classes_parts = [full_classes]
    rois = _random_k_rois(det, state_cfg, bench_cfg, k, rng)
    for roi in rois:
        boxes_i, scores_i, classes_i = run_yolo_on_crop(model, image_path, roi, imgsz=cfg.slice_imgsz, conf=cfg.output_conf, iou=cfg.iou, max_det=cfg.max_det, device=cfg.device)
        classes_i = cfg.class_mapping.map_model_classes(classes_i)
        boxes_i, scores_i, classes_i = _filter_classes(boxes_i, scores_i, classes_i, cfg.target_classes)
        boxes_parts.append(boxes_i); scores_parts.append(scores_i); classes_parts.append(classes_i)
    boxes, scores, classes = _merge_predictions(det.image_shape, cfg.merge_iou, boxes_parts, scores_parts, classes_parts)
    return boxes, scores, classes, len(rois)


def _multiscale_candidates(det, state_cfg, k_max, scales, dedup_frac):
    dens = build_detection_map(det.boxes, det.scores, det.image_shape, state_cfg)[2]
    grid = int(state_cfg.grid_size); floor = (2.0 - 0.5) / max(float(state_cfg.count_norm), 1.0)
    h, w = det.image_shape
    dedup_side = min(h, w) * float(dedup_frac)
    flat = dens.reshape(-1); order = np.argsort(flat)[::-1]
    cells: list[int] = []; used: list[tuple[float, float]] = []
    for flat_idx in order:
        if float(flat[flat_idx]) < floor or len(cells) >= int(k_max):
            break
        gy, gx = divmod(int(flat_idx), grid); cx = (gx + 0.5) * w / grid; cy = (gy + 0.5) * h / grid
        if any(abs(cx - ux) < dedup_side * 0.5 and abs(cy - uy) < dedup_side * 0.5 for ux, uy in used):
            continue
        cells.append(int(flat_idx)); used.append((cx, cy))
    K = len(cells); S = len(scales); rois = np.zeros((K, S, 4), dtype=np.float32)
    for i, c in enumerate(cells):
        gy, gx = divmod(c, grid); cx = (gx + 0.5) * w / grid; cy = (gy + 0.5) * h / grid
        for j, frac in enumerate(scales):
            side = max(1.0, min(h, w) * float(frac))
            x1 = float(np.clip(cx - side / 2.0, 0.0, max(w - side, 0.0))); y1 = float(np.clip(cy - side / 2.0, 0.0, max(h - side, 0.0)))
            rois[i, j] = [x1, y1, min(x1 + side, w), min(y1 + side, h)]
    return cells, rois


def _predict_multiscale_rl(model, policy, device_t, image_path, det, cfg, env_cfg, state_cfg, scales=(0.25, 0.35, 0.45)):
    """Multi-scale agent (A): chon SKIP/CROP@scale per-hotspot, chay live YOLO o scale da chon."""
    scales = list(scales)
    cells, rois = _multiscale_candidates(det, state_cfg, int(env_cfg.k_max), scales, dedup_frac=min(0.35, max(scales)))
    K, S = len(cells), len(scales)
    env = MultiScaleYieldEnv(det, cells, rois, np.zeros((K, S), np.float32), None, None, np.asarray(scales, np.float32), env_cfg=env_cfg, state_cfg=state_cfg)
    full_boxes, full_scores, full_classes = _full_predictions(det, cfg)
    boxes_parts = [full_boxes]; scores_parts = [full_scores]; classes_parts = [full_classes]
    state = env.reset()
    for _ in range(K + 1):
        with torch.no_grad():
            action = int(policy(torch.from_numpy(state).float().unsqueeze(0).to(device_t)).argmax(dim=1).item())
        if 1 <= action <= S and env.i < K:
            j = action - 1
            roi = env.rois[env.i, j]
            boxes_i, scores_i, classes_i = run_yolo_on_crop(model, image_path, roi, imgsz=cfg.slice_imgsz, conf=cfg.output_conf, iou=cfg.iou, max_det=cfg.max_det, device=cfg.device)
            classes_i = cfg.class_mapping.map_model_classes(classes_i)
            boxes_i, scores_i, classes_i = _filter_classes(boxes_i, scores_i, classes_i, cfg.target_classes)
            env.raw_yields[env.i, j] = int((iou_matrix(boxes_i, full_boxes).max(1) < 0.5).sum()) if len(boxes_i) and len(full_boxes) else len(boxes_i)
            boxes_parts.append(boxes_i); scores_parts.append(scores_i); classes_parts.append(classes_i)
        result = env.step(action)
        state = result.state
        if result.done:
            break
    boxes, scores, classes = _merge_predictions(det.image_shape, cfg.merge_iou, boxes_parts, scores_parts, classes_parts)
    return boxes, scores, classes, len(env.placed)


def _adaptiveconf_candidates(det, state_cfg, k_max, frac, dedup_frac):
    dens = build_detection_map(det.boxes, det.scores, det.image_shape, state_cfg)[2]
    grid = int(state_cfg.grid_size); floor = (2.0 - 0.5) / max(float(state_cfg.count_norm), 1.0)
    h, w = det.image_shape
    dedup_side = min(h, w) * float(dedup_frac)
    flat = dens.reshape(-1); order = np.argsort(flat)[::-1]
    cells: list[int] = []; used: list[tuple[float, float]] = []
    for flat_idx in order:
        if float(flat[flat_idx]) < floor or len(cells) >= int(k_max):
            break
        gy, gx = divmod(int(flat_idx), grid); cx = (gx + 0.5) * w / grid; cy = (gy + 0.5) * h / grid
        if any(abs(cx - ux) < dedup_side * 0.5 and abs(cy - uy) < dedup_side * 0.5 for ux, uy in used):
            continue
        cells.append(int(flat_idx)); used.append((cx, cy))
    K = len(cells); rois = np.zeros((K, 4), dtype=np.float32)
    for i, c in enumerate(cells):
        gy, gx = divmod(c, grid); cx = (gx + 0.5) * w / grid; cy = (gy + 0.5) * h / grid
        side = max(1.0, min(h, w) * float(frac))
        x1 = float(np.clip(cx - side / 2.0, 0.0, max(w - side, 0.0))); y1 = float(np.clip(cy - side / 2.0, 0.0, max(h - side, 0.0)))
        rois[i] = [x1, y1, min(x1 + side, w), min(y1 + side, h)]
    return cells, rois


def _predict_adaptiveconf_rl(model, policy, device_t, image_path, det, cfg, env_cfg, state_cfg, confs=(0.25, 0.10, 0.05), frac=0.35):
    """Adaptive-conf agent: per-hotspot chon SKIP/CROP@conf, chay live YOLO o conf da chon (ha conf cuu vat-bo-lo)."""
    confs = list(confs)
    cells, rois = _adaptiveconf_candidates(det, state_cfg, int(env_cfg.k_max), frac, dedup_frac=frac)
    K, C = len(cells), len(confs)
    env = AdaptiveConfEnv(det, cells, rois, np.zeros((K, C), np.float32), None, None, np.asarray(confs, np.float32), env_cfg=env_cfg, state_cfg=state_cfg)
    full_boxes, full_scores, full_classes = _full_predictions(det, cfg)
    boxes_parts = [full_boxes]; scores_parts = [full_scores]; classes_parts = [full_classes]
    state = env.reset()
    for _ in range(K + 1):
        with torch.no_grad():
            action = int(policy(torch.from_numpy(state).float().unsqueeze(0).to(device_t)).argmax(dim=1).item())
        if 1 <= action <= C and env.i < K:
            j = action - 1
            roi = env.rois[env.i]
            boxes_i, scores_i, classes_i = run_yolo_on_crop(model, image_path, roi, imgsz=cfg.slice_imgsz, conf=float(confs[j]), iou=cfg.iou, max_det=cfg.max_det, device=cfg.device)
            classes_i = cfg.class_mapping.map_model_classes(classes_i)
            boxes_i, scores_i, classes_i = _filter_classes(boxes_i, scores_i, classes_i, cfg.target_classes)
            env.raw_yields[env.i, j] = int((iou_matrix(boxes_i, full_boxes).max(1) < 0.5).sum()) if len(boxes_i) and len(full_boxes) else len(boxes_i)
            boxes_parts.append(boxes_i); scores_parts.append(scores_i); classes_parts.append(classes_i)
        result = env.step(action)
        state = result.state
        if result.done:
            break
    boxes, scores, classes = _merge_predictions(det.image_shape, cfg.merge_iou, boxes_parts, scores_parts, classes_parts)
    return boxes, scores, classes, len(env.placed)


def _ap_from_pr(tp: np.ndarray, fp: np.ndarray, total_gt: int) -> float:
    if total_gt == 0 or len(tp) == 0:
        return 0.0
    recall = np.cumsum(tp) / max(float(total_gt), 1.0)
    precision = np.cumsum(tp) / np.maximum(np.cumsum(tp) + np.cumsum(fp), 1e-9)
    recall = np.concatenate([[0.0], recall, [1.0]])
    precision = np.concatenate([[1.0], precision, [0.0]])
    for i in range(len(precision) - 1, 0, -1):
        precision[i - 1] = max(precision[i - 1], precision[i])
    changed = np.flatnonzero(recall[1:] != recall[:-1])
    return float(np.sum((recall[changed + 1] - recall[changed]) * precision[changed + 1]))


def _evaluate_method(
    predictions: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    ground_truth: dict[str, tuple[np.ndarray, np.ndarray, tuple[int, int]]],
    target_classes: tuple[int, ...],
    iou_threshold: float,
    small_area_threshold: float,
) -> dict[str, float]:
    aps: list[float] = []
    total_fp = 0
    small_total = 0
    small_hit = 0
    for cls in target_classes:
        pred_rows: list[tuple[str, float, np.ndarray]] = []
        gt_by_image: dict[str, np.ndarray] = {}
        matched_by_image: dict[str, np.ndarray] = {}
        for image_id, (gt_boxes, gt_classes, _shape) in ground_truth.items():
            gt_cls_boxes = gt_boxes[gt_classes.astype(np.int64) == int(cls)]
            gt_by_image[image_id] = gt_cls_boxes
            matched_by_image[image_id] = np.zeros((len(gt_cls_boxes),), dtype=bool)
        for image_id, (boxes, scores, classes) in predictions.items():
            mask = classes.astype(np.int64) == int(cls)
            for box, score in zip(boxes[mask], scores[mask]):
                pred_rows.append((image_id, float(score), box))
        pred_rows.sort(key=lambda row: row[1], reverse=True)
        tp = np.zeros((len(pred_rows),), dtype=np.float32)
        fp = np.zeros((len(pred_rows),), dtype=np.float32)
        for index, (image_id, _score, box) in enumerate(pred_rows):
            gt_boxes = gt_by_image[image_id]
            if len(gt_boxes) == 0:
                fp[index] = 1.0
                continue
            ious = iou_matrix(box.reshape(1, 4), gt_boxes)[0]
            best = int(ious.argmax())
            if float(ious[best]) >= iou_threshold and not matched_by_image[image_id][best]:
                tp[index] = 1.0
                matched_by_image[image_id][best] = True
            else:
                fp[index] = 1.0
        total_gt = sum(len(x) for x in gt_by_image.values())
        if total_gt > 0:
            aps.append(_ap_from_pr(tp, fp, total_gt))
        total_fp += int(fp.sum())

    for image_id, (gt_boxes, gt_classes, shape) in ground_truth.items():
        if len(gt_boxes) == 0:
            continue
        h, w = shape
        small_mask = (area(gt_boxes) / max(float(h * w), 1.0)) <= small_area_threshold
        small_total += int(small_mask.sum())
        if not small_mask.any():
            continue
        boxes, _scores, classes = predictions[image_id]
        for gt_box, gt_cls in zip(gt_boxes[small_mask], gt_classes[small_mask]):
            pred_mask = classes.astype(np.int64) == int(gt_cls)
            if pred_mask.any() and float(iou_matrix(gt_box.reshape(1, 4), boxes[pred_mask]).max()) >= iou_threshold:
                small_hit += 1

    return {
        "mAP50": float(np.mean(aps)) if aps else 0.0,
        "small_recall": float(small_hit / max(small_total, 1)),
        "fp_per_image": float(total_fp / max(len(ground_truth), 1)),
    }


def evaluate_rl_sahi_policy(
    model: YOLO,
    policy,
    device_t: torch.device,
    weights: Path,
    images: list[Path],
    image_root: Path,
    label_root: Path,
    cache_root: Path,
    split: str,
    infer_cfg: InferenceConfig,
    bench_cfg: BenchmarkConfig,
    env_cfg,
    state_cfg: StateConfig,
    use_cache: bool = True,
) -> dict[str, float]:
    if not images:
        raise FileNotFoundError(f"No images provided for split '{split}'")

    infer_cfg.target_classes = bench_cfg.target_classes
    infer_cfg.class_mapping = bench_cfg.class_mapping
    small_threshold = _small_area_threshold(
        images,
        image_root,
        label_root,
        bench_cfg.target_classes,
        bench_cfg.small_area_percentile,
        bench_cfg.class_mapping,
    )
    ground_truth: dict[str, tuple[np.ndarray, np.ndarray, tuple[int, int]]] = {}
    predictions: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    crops: list[int] = []
    latency: list[float] = []

    for image_path in images:
        image_id = image_path.stem
        gt_boxes, gt_classes = _read_gt(
            image_path,
            image_root,
            label_root,
            bench_cfg.target_classes,
            bench_cfg.class_mapping,
        )
        shape = _image_shape(image_path)
        ground_truth[image_id] = (gt_boxes, gt_classes, shape)

        det = get_initial_detection(
            model=model,
            weights=weights,
            image_path=image_path,
            weights_imgsz=infer_cfg.full_imgsz,
            full_conf=infer_cfg.full_conf,
            full_iou=infer_cfg.iou,
            max_det=infer_cfg.max_det,
            device=infer_cfg.device,
            feature_layers=infer_cfg.feature_layers,
            aux_grid_size=state_cfg.grid_size,
            spatial_feature_channels=state_cfg.spatial_feature_channels,
            cache_root=cache_root,
            split=split,
            use_cache=use_cache,
        )

        start = time.perf_counter()
        boxes, scores, classes, crop_count = _predict_rl_sahi(
            model,
            policy,
            device_t,
            image_path,
            det,
            infer_cfg,
            env_cfg,
            state_cfg,
        )
        predictions[image_id] = (boxes, scores, classes)
        latency.append(time.perf_counter() - start)
        crops.append(crop_count)

    metrics = _evaluate_method(
        predictions,
        ground_truth,
        bench_cfg.target_classes,
        bench_cfg.iou_threshold,
        small_threshold,
    )
    return {
        **metrics,
        "crops_per_image": float(np.mean(crops)),
        "latency_ms_per_image": float(np.mean(latency) * 1000.0),
        "images": float(len(images)),
        "small_area_threshold": small_threshold,
    }


def benchmark_split(
    weights: Path,
    checkpoint: Path,
    image_root: Path,
    label_root: Path,
    cache_root: Path,
    split: str,
    infer_cfg: InferenceConfig,
    bench_cfg: BenchmarkConfig,
    out_dir: Path,
    limit: int | None = None,
    use_cache: bool = True,
    density_k: tuple[int, ...] = (),
    hotspot: bool = False,
    yield_rl: bool = False,
    multiscale: bool = False,
    adaptive_conf: bool = False,
    random_k: tuple[int, ...] = (),
    residual_density_k: tuple[int, ...] = (),
    seed: int = 42,
) -> list[dict[str, float | str]]:
    images = iter_images(image_root, split=split, limit=limit)
    if not images:
        raise FileNotFoundError(f"No images found for split '{split}'")

    infer_cfg.target_classes = bench_cfg.target_classes
    infer_cfg.class_mapping = bench_cfg.class_mapping
    small_threshold = _small_area_threshold(
        images,
        image_root,
        label_root,
        bench_cfg.target_classes,
        bench_cfg.small_area_percentile,
        bench_cfg.class_mapping,
    )
    model = load_yolo(weights, device=infer_cfg.device)
    device_t = resolve_torch_device(infer_cfg.device)
    policy, checkpoint_data = load_policy(checkpoint, device_t)
    env_cfg = checkpoint_data["env_cfg_obj"]
    state_cfg = checkpoint_data.get("state_cfg_obj", StateConfig())

    ground_truth: dict[str, tuple[np.ndarray, np.ndarray, tuple[int, int]]] = {}
    rl_key = "rl_adaptiveconf" if adaptive_conf else ("rl_multiscale" if multiscale else ("rl_yield" if yield_rl else ("rl_hotspot" if hotspot else "rl_sahi")))
    density_methods = [f"density_k{int(k)}" for k in density_k]
    random_methods = [f"random_k{int(k)}" for k in random_k]
    rdensity_methods = [f"rdensity_k{int(k)}" for k in residual_density_k]
    predictions = {"yolo_full": {}, "fixed_grid_sahi": {}, rl_key: {}, **{m: {} for m in density_methods}, **{m: {} for m in random_methods}, **{m: {} for m in rdensity_methods}}
    crops = {key: [] for key in predictions}
    latency = {key: [] for key in predictions}
    rng = np.random.default_rng(seed)

    print(f"[benchmark] bat dau {len(images)} anh, methods: {list(predictions.keys())}", flush=True)
    for img_idx, image_path in enumerate(images):
        if img_idx % 25 == 0:
            print(f"[benchmark] ...{img_idx}/{len(images)} anh", flush=True)
        image_id = image_path.stem
        gt_boxes, gt_classes = _read_gt(
            image_path,
            image_root,
            label_root,
            bench_cfg.target_classes,
            bench_cfg.class_mapping,
        )
        shape = _image_shape(image_path)
        ground_truth[image_id] = (gt_boxes, gt_classes, shape)

        det = get_initial_detection(
            model=model,
            weights=weights,
            image_path=image_path,
            weights_imgsz=infer_cfg.full_imgsz,
            full_conf=infer_cfg.full_conf,
            full_iou=infer_cfg.iou,
            max_det=infer_cfg.max_det,
            device=infer_cfg.device,
            feature_layers=infer_cfg.feature_layers,
            aux_grid_size=state_cfg.grid_size,
            spatial_feature_channels=state_cfg.spatial_feature_channels,
            cache_root=cache_root,
            split=split,
            use_cache=use_cache,
        )

        start = time.perf_counter()
        predictions["yolo_full"][image_id] = _full_predictions(det, infer_cfg)
        latency["yolo_full"].append(time.perf_counter() - start)
        crops["yolo_full"].append(0)

        start = time.perf_counter()
        boxes, scores, classes, crop_count = _predict_fixed_sahi(model, image_path, det, infer_cfg, bench_cfg)
        predictions["fixed_grid_sahi"][image_id] = (boxes, scores, classes)
        latency["fixed_grid_sahi"].append(time.perf_counter() - start)
        crops["fixed_grid_sahi"].append(crop_count)

        start = time.perf_counter()
        if adaptive_conf:
            boxes, scores, classes, crop_count = _predict_adaptiveconf_rl(model, policy, device_t, image_path, det, infer_cfg, env_cfg, state_cfg)
        elif multiscale:
            boxes, scores, classes, crop_count = _predict_multiscale_rl(model, policy, device_t, image_path, det, infer_cfg, env_cfg, state_cfg)
        elif yield_rl:
            boxes, scores, classes, crop_count = _predict_yield_rl(model, policy, device_t, image_path, det, infer_cfg, env_cfg, state_cfg)
        elif hotspot:
            boxes, scores, classes, crop_count = _predict_hotspot_rl(model, policy, device_t, image_path, det, infer_cfg, env_cfg, state_cfg)
        else:
            boxes, scores, classes, crop_count = _predict_rl_sahi(model, policy, device_t, image_path, det, infer_cfg, env_cfg, state_cfg)
        predictions[rl_key][image_id] = (boxes, scores, classes)
        latency[rl_key].append(time.perf_counter() - start)
        crops[rl_key].append(crop_count)

        for k in density_k:
            mkey = f"density_k{int(k)}"
            start = time.perf_counter()
            boxes, scores, classes, crop_count = _predict_density_guided(
                model, image_path, det, infer_cfg, bench_cfg, state_cfg, int(k)
            )
            predictions[mkey][image_id] = (boxes, scores, classes)
            latency[mkey].append(time.perf_counter() - start)
            crops[mkey].append(crop_count)

        for k in random_k:
            mkey = f"random_k{int(k)}"
            start = time.perf_counter()
            boxes, scores, classes, crop_count = _predict_random_k(model, image_path, det, infer_cfg, bench_cfg, state_cfg, int(k), rng)
            predictions[mkey][image_id] = (boxes, scores, classes)
            latency[mkey].append(time.perf_counter() - start)
            crops[mkey].append(crop_count)

        for k in residual_density_k:
            mkey = f"rdensity_k{int(k)}"
            start = time.perf_counter()
            boxes, scores, classes, crop_count = _predict_density_guided(model, image_path, det, infer_cfg, bench_cfg, state_cfg, int(k), residual=True)
            predictions[mkey][image_id] = (boxes, scores, classes)
            latency[mkey].append(time.perf_counter() - start)
            crops[mkey].append(crop_count)

    rows: list[dict[str, float | str]] = []
    for method, method_predictions in predictions.items():
        metrics = _evaluate_method(
            method_predictions,
            ground_truth,
            bench_cfg.target_classes,
            bench_cfg.iou_threshold,
            small_threshold,
        )
        rows.append(
            {
                "method": method,
                **metrics,
                "crops_per_image": float(np.mean(crops[method])),
                "latency_ms_per_image": float(np.mean(latency[method]) * 1000.0),
                "images": float(len(images)),
                "small_area_threshold": small_threshold,
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "benchmark.json").write_text(
        json.dumps({"config": asdict(bench_cfg), "results": rows}, indent=2),
        encoding="utf-8",
    )
    with (out_dir / "benchmark.csv").open("w", encoding="utf-8") as f:
        header = list(rows[0].keys())
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(str(row[key]) for key in header) + "\n")
    return rows
