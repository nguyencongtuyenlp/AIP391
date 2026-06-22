from __future__ import annotations

import numpy as np

from rl_sahi.common.actions import NUM_ACTIONS, Action
from rl_sahi.common.boxes import (
    area,
    as_boxes,
    box_from_center,
    centers,
    intersection_matrix,
    ioa_matrix,
    rasterize_boxes,
    translate_box,
    zoom_box,
)
from rl_sahi.common.cache import DetectionCache, HardRegionCache
from rl_sahi.common.class_mapping import ClassMapping
from rl_sahi.rl.env_config import EnvConfig, StepResult
from rl_sahi.rl.state_config import StateConfig
from rl_sahi.rl.state_maps import build_detection_map, mark_history, proposal_mask, proposal_quality
from rl_sahi.rl.state_summary import detection_summary
from rl_sahi.rl.state_vector import build_state_vector


class SliceEnv:
    def __init__(
        self,
        detection: DetectionCache,
        hard_regions: HardRegionCache | None,
        env_cfg: EnvConfig | None = None,
        state_cfg: StateConfig | None = None,
        previous_rois: np.ndarray | None = None,
        overlap_rois: np.ndarray | None = None,
        previous_covered: np.ndarray | None = None,
        target_classes: tuple[int, ...] = (),
        class_mapping: ClassMapping | None = None,
    ) -> None:
        self.detection = detection
        self.hard_regions = hard_regions
        self.env_cfg = env_cfg or EnvConfig()
        self.state_cfg = state_cfg or StateConfig()
        self.target_classes = tuple(int(x) for x in target_classes)
        self.class_mapping = class_mapping or ClassMapping()
        self.image_shape = detection.image_shape
        self.det_boxes, self.det_scores, self.det_classes = self._filtered_detections()
        self.detection_map = build_detection_map(self.det_boxes, self.det_scores, self.image_shape, self.state_cfg)
        self.hard_boxes = as_boxes(hard_regions.hard_boxes if hard_regions is not None else np.zeros((0, 4)))
        self.previous_rois = as_boxes(previous_rois if previous_rois is not None else np.zeros((0, 4), dtype=np.float32))
        self.overlap_rois = as_boxes(overlap_rois if overlap_rois is not None else self.previous_rois)
        self.previous_slice_map = self._build_previous_slice_map()
        self.previous_covered = self._init_previous_covered(previous_covered)
        self.history = np.zeros((self.state_cfg.grid_size, self.state_cfg.grid_size), dtype=np.float32)
        self.covered = self.previous_covered.copy()
        self.roi = self._initial_roi()
        self.step_index = 0

    def _filtered_detections(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        boxes = as_boxes(self.detection.boxes)
        scores = np.asarray(self.detection.scores, dtype=np.float32).reshape(-1)
        classes = self.class_mapping.map_model_classes(self.detection.classes)
        if not self.target_classes:
            return boxes, scores, classes
        target = np.asarray(self.target_classes, dtype=np.int64)
        mask = np.isin(classes.astype(np.int64), target)
        return boxes[mask], scores[mask], classes[mask]

    def reset(self) -> np.ndarray:
        self.history.fill(0.0)
        self.covered = self.previous_covered.copy()
        self.roi = self._initial_roi()
        self.step_index = 0
        self.history = mark_history(self.history, self.roi, self.image_shape, self.state_cfg.grid_size)
        return self._state()

    def step(self, action: int | Action) -> StepResult:
        action = Action(int(action))
        done = action == Action.STOP
        stalled_roi = False
        previous_roi = self.roi.copy()
        seen_before = np.clip(self.history + self.previous_slice_map, 0.0, 1.0)
        if action != Action.STOP:
            self.roi = self._apply_action(action)
            stalled_roi = bool(np.allclose(previous_roi, self.roi, atol=1e-3))
            self.step_index += 1
            self.history = mark_history(self.history, self.roi, self.image_shape, self.state_cfg.grid_size)

        reward, info = self._reward(action, previous_roi, seen_before)
        if stalled_roi:
            if self.env_cfg.use_boundary_fix:
                # ROI bi kep o bien (action hop le nhung khong di chuyen duoc) -> chi phat NHE, KHONG ket thuc
                reward -= self.env_cfg.empty_slice_penalty
            else:
                done = True
                reward -= self.env_cfg.stalled_without_stop_penalty
        if info["old_slice_overlap"] >= self.env_cfg.old_slice_overlap_threshold:
            done = True
            info["stop_due_to_old_overlap"] = True
        else:
            info["stop_due_to_old_overlap"] = False
        if self.step_index >= self.env_cfg.max_steps:
            done = True
            info["stop_due_to_max_steps"] = action != Action.STOP
            if info["stop_due_to_max_steps"]:
                reward -= self.env_cfg.max_steps_without_stop_penalty
        else:
            info["stop_due_to_max_steps"] = False
        info["stop_due_to_stalled_roi"] = stalled_roi
        info["roi"] = self.roi.copy()
        info["covered"] = int(self.covered.sum())
        info["hard_total"] = int(len(self.hard_boxes))
        return StepResult(self._state(), reward, done, info)

    def valid_actions(self) -> np.ndarray:
        valid = np.ones((NUM_ACTIONS,), dtype=bool)
        for action in Action:
            if action == Action.STOP:
                continue
            next_roi = self._apply_action(action)
            if np.allclose(next_roi, self.roi, atol=1e-3):
                valid[int(action)] = False
        valid[int(Action.STOP)] = True
        return valid

    def guided_action(self) -> Action:
        heatmap_target = self._heatmap_target()
        boxes = self.det_boxes
        scores = self.det_scores
        valid_mask = scores >= self.state_cfg.proposal_min_conf
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]
        if len(boxes) == 0:
            return self._action_toward_target(heatmap_target[0]) if heatmap_target is not None else Action.STOP

        image_area = max(float(self.image_shape[0] * self.image_shape[1]), 1.0)
        det_area_ratio = area(boxes) / image_area
        prop_mask = proposal_mask(scores, self.state_cfg)
        small_mask = det_area_ratio <= self.state_cfg.small_area_ratio
        target_mask = prop_mask | (small_mask & (scores < self.env_cfg.high_conf_threshold))
        if not target_mask.any():
            return self._action_toward_target(heatmap_target[0]) if heatmap_target is not None else Action.STOP

        candidate_boxes = boxes[target_mask]
        candidate_scores = scores[target_mask]
        candidate_centers = centers(candidate_boxes)
        if len(self.previous_rois) > 0:
            old_seen = self._points_in_previous_rois(candidate_centers)
        else:
            old_seen = np.zeros((len(candidate_boxes),), dtype=bool)

        roi_center = centers(self.roi.reshape(1, 4))[0]
        distances = np.linalg.norm(candidate_centers - roi_center[None, :], axis=1)
        quality = proposal_quality(candidate_scores, self.state_cfg)
        heat_support = self._objectness_values_at_points(candidate_centers)
        density_support = self._proposal_density_values_at_points(candidate_centers)
        high_seen = self._points_in_boxes(
            candidate_centers,
            boxes[scores >= self.env_cfg.high_conf_threshold],
        )
        priority = quality
        priority += small_mask[target_mask].astype(np.float32) * 0.5
        priority += heat_support * 0.5
        priority += density_support * 0.75
        priority -= distances / max(min(self.image_shape), 1)
        priority -= old_seen.astype(np.float32) * 2.0
        priority -= high_seen.astype(np.float32) * 1.0
        target_idx = int(priority.argmax())
        if heatmap_target is not None:
            heat_point, heat_score = heatmap_target
            heat_distance = float(np.linalg.norm(heat_point - roi_center) / max(min(self.image_shape), 1))
            heat_priority = float(heat_score - heat_distance)
            if heat_priority > float(priority[target_idx]):
                return self._action_toward_target(heat_point)
        if priority[target_idx] < -1.5:
            return Action.STOP
        return self._action_toward_target(candidate_centers[target_idx], candidate_boxes[[target_idx]])

    def _heatmap_target(self) -> tuple[np.ndarray, float] | None:
        obj = np.asarray(self.detection.objectness_map, dtype=np.float32)
        if obj.size == 0:
            return None
        grid_size = self.state_cfg.grid_size
        obj = np.nan_to_num(obj.reshape(-1, grid_size, grid_size), nan=0.0, posinf=0.0, neginf=0.0)
        heat = obj.max(axis=0)
        if self.detection_map.shape[0] > 2:
            density = np.clip(self.detection_map[2] * self.state_cfg.count_norm / 10.0, 0.0, 1.0)
            heat = np.maximum(heat * 0.7, density)
        if heat.size == 0:
            return None
        priority = heat.copy()
        priority -= 0.6 * np.asarray(self.previous_slice_map, dtype=np.float32)
        priority -= 0.2 * np.asarray(self.history, dtype=np.float32)
        y, x = np.unravel_index(int(priority.argmax()), priority.shape)
        score = float(priority[y, x])
        if score <= 0.02:
            return None
        h, w = self.image_shape
        target = np.array([(x + 0.5) * w / grid_size, (y + 0.5) * h / grid_size], dtype=np.float32)
        return target, score

    def _objectness_values_at_points(self, points: np.ndarray) -> np.ndarray:
        obj = np.asarray(self.detection.objectness_map, dtype=np.float32)
        if obj.size == 0:
            return np.zeros((len(points),), dtype=np.float32)
        grid = obj.reshape(-1, self.state_cfg.grid_size, self.state_cfg.grid_size).max(axis=0)
        return self._grid_values_at_points(grid, points)

    def _proposal_density_values_at_points(self, points: np.ndarray) -> np.ndarray:
        if self.detection_map.shape[0] <= 2:
            return np.zeros((len(points),), dtype=np.float32)
        density = np.clip(self.detection_map[2] * self.state_cfg.count_norm / 10.0, 0.0, 1.0)
        return self._grid_values_at_points(density, points)

    def _grid_values_at_points(self, grid: np.ndarray, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        if len(points) == 0:
            return np.zeros((0,), dtype=np.float32)
        grid = np.asarray(grid, dtype=np.float32).reshape(self.state_cfg.grid_size, self.state_cfg.grid_size)
        h, w = self.image_shape
        xs = np.clip((points[:, 0] / max(w, 1)) * self.state_cfg.grid_size, 0, self.state_cfg.grid_size - 1).astype(int)
        ys = np.clip((points[:, 1] / max(h, 1)) * self.state_cfg.grid_size, 0, self.state_cfg.grid_size - 1).astype(int)
        return grid[ys, xs].astype(np.float32)

    def _points_in_boxes(self, points: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        boxes = as_boxes(boxes)
        points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        if len(points) == 0 or len(boxes) == 0:
            return np.zeros((len(points),), dtype=bool)
        mask = np.zeros((len(points),), dtype=bool)
        for box in boxes:
            mask |= (
                (points[:, 0] >= box[0])
                & (points[:, 0] <= box[2])
                & (points[:, 1] >= box[1])
                & (points[:, 1] <= box[3])
            )
        return mask

    def _action_toward_target(self, target: np.ndarray, target_box: np.ndarray | None = None) -> Action:
        target = np.asarray(target, dtype=np.float32).reshape(2)
        roi_center = centers(self.roi.reshape(1, 4))[0]
        x1, y1, x2, y2 = self.roi
        inside = x1 <= target[0] <= x2 and y1 <= target[1] <= y2
        if inside:
            if self._roi_area_ratio() > self.env_cfg.max_roi_area_ratio or self._scale_gain() < self.env_cfg.min_scale_gain:
                return Action.ZOOM_IN
            if target_box is not None:
                projected_size = self._projected_sizes(target_box)[0]
                if projected_size < self.env_cfg.target_projected_size:
                    return Action.ZOOM_IN
                if projected_size > self.env_cfg.max_projected_size:
                    return Action.ZOOM_OUT
            min_side, _max_side = self._side_limits()
            if self._roi_side() > min_side * 1.25:
                return Action.ZOOM_IN
            return Action.STOP

        dx = target[0] - roi_center[0]
        dy = target[1] - roi_center[1]
        if abs(dx) >= abs(dy):
            return Action.RIGHT if dx > 0 else Action.LEFT
        return Action.DOWN if dy > 0 else Action.UP

    def _initial_roi(self) -> np.ndarray:
        h, w = self.image_shape
        _min_side, max_side = self._side_limits()
        side = min(min(h, w) * self.env_cfg.initial_slice_fraction, max_side)
        return box_from_center(w / 2.0, h / 2.0, side, self.image_shape)

    def _apply_action(self, action: Action) -> np.ndarray:
        side = self._roi_side()
        step = side * self.env_cfg.move_fraction
        if action == Action.LEFT:
            return translate_box(self.roi, -step, 0.0, self.image_shape)
        if action == Action.RIGHT:
            return translate_box(self.roi, step, 0.0, self.image_shape)
        if action == Action.UP:
            return translate_box(self.roi, 0.0, -step, self.image_shape)
        if action == Action.DOWN:
            return translate_box(self.roi, 0.0, step, self.image_shape)
        min_side, max_side = self._side_limits()
        if action == Action.ZOOM_IN:
            return zoom_box(self.roi, self.env_cfg.zoom_factor, self.image_shape, min_side, max_side)
        if action == Action.ZOOM_OUT:
            return zoom_box(self.roi, 1.0 / self.env_cfg.zoom_factor, self.image_shape, min_side, max_side)
        return self.roi

    def _side_limits(self) -> tuple[float, float]:
        h, w = self.image_shape
        min_side = min(h, w) * self.env_cfg.min_slice_fraction
        max_side_by_fraction = min(h, w) * self.env_cfg.max_slice_fraction
        max_side_by_area = np.sqrt(max(float(h * w) * self.env_cfg.max_roi_area_ratio, 1.0))
        max_side = max(min(max_side_by_fraction, max_side_by_area), min_side)
        return float(min_side), float(max_side)

    def _build_previous_slice_map(self) -> np.ndarray:
        if len(self.previous_rois) == 0:
            return np.zeros((self.state_cfg.grid_size, self.state_cfg.grid_size), dtype=np.float32)
        return rasterize_boxes(self.previous_rois, self.image_shape, self.state_cfg.grid_size)

    def _init_previous_covered(self, previous_covered: np.ndarray | None) -> np.ndarray:
        if len(self.hard_boxes) == 0:
            return np.zeros((0,), dtype=bool)
        if previous_covered is not None:
            arr = np.asarray(previous_covered, dtype=bool).reshape(-1)
            if len(arr) != len(self.hard_boxes):
                raise ValueError("previous_covered length must match hard region count")
            return arr.copy()
        covered = np.zeros((len(self.hard_boxes),), dtype=bool)
        for roi in self.previous_rois:
            _scores, hit_mask = self._hard_target_scores(roi)
            covered |= hit_mask
        return covered

    def _points_in_previous_rois(self, points: np.ndarray) -> np.ndarray:
        mask = np.zeros((len(points),), dtype=bool)
        for roi in self.previous_rois:
            mask |= (
                (points[:, 0] >= roi[0])
                & (points[:, 0] <= roi[2])
                & (points[:, 1] >= roi[1])
                & (points[:, 1] <= roi[3])
            )
        return mask

    def _old_slice_overlap(self, roi: np.ndarray | None = None) -> float:
        if len(self.overlap_rois) == 0:
            return 0.0
        roi = self.roi if roi is None else np.asarray(roi, dtype=np.float32).reshape(4)
        inter = intersection_matrix(roi.reshape(1, 4), self.overlap_rois)[0]
        current_area = max(float(area(roi.reshape(1, 4))[0]), 1.0)
        return float(np.clip(inter.max() / current_area, 0.0, 1.0))

    def _roi_side(self, roi: np.ndarray | None = None) -> float:
        roi = self.roi if roi is None else np.asarray(roi, dtype=np.float32).reshape(4)
        return max(float(roi[2] - roi[0]), float(roi[3] - roi[1]), 1.0)

    def _roi_area_ratio(self, roi: np.ndarray | None = None) -> float:
        roi = self.roi if roi is None else np.asarray(roi, dtype=np.float32).reshape(4)
        image_area = max(float(self.image_shape[0] * self.image_shape[1]), 1.0)
        return float(area(roi.reshape(1, 4))[0] / image_area)

    def _scale_gain(self, roi: np.ndarray | None = None) -> float:
        return float(min(self.image_shape) / self._roi_side(roi))

    def _projected_sizes(self, boxes: np.ndarray, roi: np.ndarray | None = None) -> np.ndarray:
        boxes = as_boxes(boxes)
        if len(boxes) == 0:
            return np.zeros((0,), dtype=np.float32)
        widths = np.maximum(boxes[:, 2] - boxes[:, 0], 1.0)
        heights = np.maximum(boxes[:, 3] - boxes[:, 1], 1.0)
        return (np.maximum(widths, heights) * float(self.env_cfg.reward_imgsz) / self._roi_side(roi)).astype(np.float32)

    def _projected_size_scores(self, boxes: np.ndarray, roi: np.ndarray | None = None) -> np.ndarray:
        projected = self._projected_sizes(boxes, roi)
        if len(projected) == 0:
            return projected
        cfg = self.env_cfg
        below_target = (projected - cfg.min_projected_size) / max(cfg.target_projected_size - cfg.min_projected_size, 1e-6)
        above_target = (cfg.max_projected_size - projected) / max(cfg.max_projected_size - cfg.target_projected_size, 1e-6)
        return np.clip(np.minimum(below_target, above_target), 0.0, 1.0).astype(np.float32)

    def _center_context_mask(self, boxes: np.ndarray, roi: np.ndarray | None = None) -> np.ndarray:
        boxes = as_boxes(boxes)
        if len(boxes) == 0:
            return np.zeros((0,), dtype=bool)
        x1, y1, x2, y2 = self.roi if roi is None else np.asarray(roi, dtype=np.float32).reshape(4)
        width = max(float(x2 - x1), 1.0)
        height = max(float(y2 - y1), 1.0)
        margin_x = width * self.env_cfg.context_margin
        margin_y = height * self.env_cfg.context_margin
        pts = centers(boxes)
        return (
            (pts[:, 0] >= x1 + margin_x)
            & (pts[:, 0] <= x2 - margin_x)
            & (pts[:, 1] >= y1 + margin_y)
            & (pts[:, 1] <= y2 - margin_y)
        )

    def _hard_target_scores(self, roi: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        if len(self.hard_boxes) == 0:
            return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=bool)
        if self._roi_area_ratio(roi) > self.env_cfg.max_roi_area_ratio or self._scale_gain(roi) < self.env_cfg.min_scale_gain:
            return np.zeros((len(self.hard_boxes),), dtype=np.float32), np.zeros((len(self.hard_boxes),), dtype=bool)
        context_mask = self._center_context_mask(self.hard_boxes, roi)
        size_scores = self._projected_size_scores(self.hard_boxes, roi)
        target_scores = np.where(context_mask, size_scores, 0.0).astype(np.float32)
        return target_scores, target_scores > 0.0

    def _roi_grid_window(self, roi: np.ndarray | None = None) -> tuple[int, int, int, int]:
        roi = self.roi if roi is None else np.asarray(roi, dtype=np.float32).reshape(4)
        h, w = self.image_shape
        grid = self.state_cfg.grid_size
        x1 = int(np.floor(np.clip(roi[0] / max(w, 1), 0.0, 1.0) * grid))
        y1 = int(np.floor(np.clip(roi[1] / max(h, 1), 0.0, 1.0) * grid))
        x2 = int(np.ceil(np.clip(roi[2] / max(w, 1), 0.0, 1.0) * grid))
        y2 = int(np.ceil(np.clip(roi[3] / max(h, 1), 0.0, 1.0) * grid))
        x1 = int(np.clip(x1, 0, grid - 1))
        y1 = int(np.clip(y1, 0, grid - 1))
        x2 = int(np.clip(max(x2, x1 + 1), 1, grid))
        y2 = int(np.clip(max(y2, y1 + 1), 1, grid))
        return y1, y2, x1, x2

    def _objectness_roi_score(self, roi: np.ndarray | None = None) -> float:
        obj = np.asarray(self.detection.objectness_map, dtype=np.float32)
        if obj.size == 0:
            return 0.0
        grid = np.nan_to_num(
            obj.reshape(-1, self.state_cfg.grid_size, self.state_cfg.grid_size).max(axis=0),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        y1, y2, x1, x2 = self._roi_grid_window(roi)
        window = grid[y1:y2, x1:x2]
        return float(np.clip(window.max() if window.size else 0.0, 0.0, 1.0))

    def _observable_target_score(self, roi: np.ndarray | None = None) -> float:
        roi = self.roi if roi is None else np.asarray(roi, dtype=np.float32).reshape(4)
        scores = np.asarray(self.det_scores, dtype=np.float32).reshape(-1)
        boxes = as_boxes(self.det_boxes)
        proposal_score = 0.0
        if len(boxes) > 0:
            image_area = max(float(self.image_shape[0] * self.image_shape[1]), 1.0)
            det_area_ratio = area(boxes) / image_area
            prop_mask = proposal_mask(scores, self.state_cfg)
            small_uncertain = (det_area_ratio <= self.state_cfg.small_area_ratio) & (
                scores < self.env_cfg.high_conf_threshold
            )
            target_mask = prop_mask | small_uncertain
            if target_mask.any():
                candidate_boxes = boxes[target_mask]
                candidate_scores = scores[target_mask]
                center_mask = self._center_context_mask(candidate_boxes, roi)
                if center_mask.any():
                    values = proposal_quality(candidate_scores, self.state_cfg)
                    values += small_uncertain[target_mask].astype(np.float32) * 0.5
                    proposal_score = float(values[center_mask].sum())
                    roi_area_ratio = max(self._roi_area_ratio(roi), 1e-6)
                    density_gain = self.env_cfg.max_roi_area_ratio / roi_area_ratio
                    proposal_score *= float(np.clip(density_gain, 0.25, 2.0))
        objectness_score = self._objectness_roi_score(roi)
        return float(np.clip(proposal_score + objectness_score, 0.0, 4.0))

    def _density_potential(self, roi: np.ndarray | None, seen: np.ndarray) -> float:
        if self.detection_map.shape[0] <= 2:
            base = 0.0
        else:
            density = np.clip(self.detection_map[2] * self.state_cfg.count_norm / 10.0, 0.0, 1.0)
            if self.env_cfg.density_potential_min_count > 1:
                floor = (self.env_cfg.density_potential_min_count - 0.5) / max(self.state_cfg.count_norm, 1.0)
                density = np.where(self.detection_map[2] >= floor, density, 0.0)
            unseen = density * (1.0 - np.asarray(seen, dtype=np.float32))
            y1, y2, x1, x2 = self._roi_grid_window(roi)
            window = unseen[y1:y2, x1:x2]
            base = float(window.mean()) if window.size else 0.0
        obj = self._objectness_roi_score(roi)
        return float(np.clip(base + self.env_cfg.density_potential_obj_weight * obj, 0.0, 1.0))

    def _state(self) -> np.ndarray:
        summary = detection_summary(
            boxes=self.det_boxes,
            scores=self.det_scores,
            roi=self.roi,
            history=self.history,
            previous_slice_map=self.previous_slice_map,
            image_shape=self.image_shape,
            step_index=self.step_index,
            max_steps=self.env_cfg.max_steps,
            old_slice_overlap=self._old_slice_overlap(),
            scale_gain=self._scale_gain(),
            previous_slice_count=len(self.previous_rois),
            cfg=self.state_cfg,
        )
        return build_state_vector(
            self.detection.feature,
            self.history,
            self.previous_slice_map,
            self.detection_map,
            self.detection.objectness_map,
            self.detection.spatial_feature_map,
            summary,
        )

    def _update_covered(self) -> tuple[np.ndarray, np.ndarray]:
        target_scores, hit_mask = self._hard_target_scores()
        self.covered |= hit_mask
        return target_scores, hit_mask

    def _reward(self, action: Action, previous_roi: np.ndarray | None = None, seen_before: np.ndarray | None = None) -> tuple[float, dict]:
        if self.env_cfg.use_gtfree_reward:
            return self._gtfree_reward(action, previous_roi, seen_before)
        if self.env_cfg.use_simplified_reward:
            return self._simplified_reward(action, previous_roi, seen_before)
        return self._legacy_reward(action, previous_roi)

    def _gtfree_reward(self, action: Action, previous_roi: np.ndarray | None = None, seen_before: np.ndarray | None = None) -> tuple[float, dict]:
        """Reward GT-FREE: thuong cho ROI phu tin hieu QUAN SAT (density+objectness) CHUA tung thay.
        Khong dung hard_boxes (GT) -> MDP luc train == luc infer (bo distribution shift)."""
        cfg = self.env_cfg
        seen = seen_before if seen_before is not None else np.zeros((self.state_cfg.grid_size, self.state_cfg.grid_size), dtype=np.float32)
        gain = self._density_potential(self.roi, seen)        # [0,1] GT-free: density+obj CHUA thay trong ROI
        observable = self._observable_target_score()          # GT-free: tong proposal-quality + objectness trong ROI
        roi_area_ratio = self._roi_area_ratio()
        scale_gain = self._scale_gain()
        old_slice_overlap = self._old_slice_overlap()
        reward = 0.0
        info = {
            "new_hits": 0, "hit_count": 0, "target_score": 0.0, "total_target_score": 0.0,
            "retained_hits": 0, "compactness_score": 0.0, "compactness_delta": 0.0,
            "observable_score": observable, "observable_delta": float(gain),
            "roi_area_ratio": roi_area_ratio, "scale_gain": scale_gain,
            "old_slice_overlap": old_slice_overlap, "detected_overlap": 0.0,
        }
        if action != Action.STOP:
            reward += cfg.target_reward * cfg.gtfree_gain_weight * float(gain)
        step_cost = 0.05 + roi_area_ratio * 0.5
        reward -= cfg.efficiency_weight * step_cost
        constraint_penalty = 0.0
        if roi_area_ratio > cfg.max_roi_area_ratio:
            constraint_penalty += roi_area_ratio / max(cfg.max_roi_area_ratio, 1e-6) - 1.0
        if scale_gain < cfg.min_scale_gain:
            constraint_penalty += cfg.min_scale_gain / max(scale_gain, 1e-6) - 1.0
        if old_slice_overlap >= cfg.old_slice_overlap_threshold:
            constraint_penalty += 1.0
        reward -= cfg.constraint_weight * constraint_penalty
        if action == Action.STOP and observable < 0.3:
            # CHI phat khi dung o cho TRONG (chong dung som vo nghia); KHONG thuong dung -> agent
            # tiep tuc cat chung nao con gain (chua thay) -> chong under-crop.
            reward -= cfg.stop_bonus_weight * 0.5
        return float(reward), info

    def _simplified_reward(self, action: Action, previous_roi: np.ndarray | None = None, seen_before: np.ndarray | None = None) -> tuple[float, dict]:
        prev_covered = self.covered.copy()
        target_scores, hit_mask = self._update_covered()
        new_mask = hit_mask & ~prev_covered
        new_hits = int(new_mask.sum())
        target_score = float(target_scores[new_mask].sum()) if len(target_scores) else 0.0
        hit_count = int(hit_mask.sum()) if len(hit_mask) else 0
        total_target_score = float(target_scores[hit_mask].sum()) if len(target_scores) else 0.0
        roi_area_ratio = self._roi_area_ratio()
        scale_gain = self._scale_gain()
        old_slice_overlap = self._old_slice_overlap()
        observable_score = self._observable_target_score()

        cfg = self.env_cfg
        reward = 0.0
        info = {
            "new_hits": new_hits,
            "hit_count": hit_count,
            "target_score": target_score,
            "total_target_score": total_target_score,
            "retained_hits": int((hit_mask & prev_covered).sum()) if len(hit_mask) else 0,
            "compactness_score": 0.0,
            "compactness_delta": 0.0,
            "observable_score": observable_score,
            "observable_delta": 0.0,
            "roi_area_ratio": roi_area_ratio,
            "scale_gain": scale_gain,
            "old_slice_overlap": old_slice_overlap,
            "detected_overlap": 0.0,
        }

        if new_hits > 0:
            reward += cfg.target_reward * float(new_hits)
            density = target_score * cfg.max_roi_area_ratio / max(roi_area_ratio, 1e-6)
            reward += cfg.target_reward * 0.3 * float(np.clip(density, 0.0, 3.0))

        step_cost = 0.05 + roi_area_ratio * 0.5
        reward -= cfg.efficiency_weight * step_cost

        if cfg.nav_shaping_weight > 0.0 and seen_before is not None and action != Action.STOP:
            phi_now = self._density_potential(self.roi, seen_before)
            phi_prev = self._density_potential(previous_roi, seen_before) if previous_roi is not None else phi_now
            nav_shaping = float(np.clip(cfg.shaping_gamma * phi_now - phi_prev, -0.5, 0.5))
            reward += cfg.nav_shaping_weight * nav_shaping
            info["observable_delta"] = nav_shaping
            info["density_potential"] = phi_now

        constraint_penalty = 0.0
        if roi_area_ratio > cfg.max_roi_area_ratio:
            overflow = roi_area_ratio / max(cfg.max_roi_area_ratio, 1e-6) - 1.0
            constraint_penalty += overflow
        if scale_gain < cfg.min_scale_gain:
            under = cfg.min_scale_gain / max(scale_gain, 1e-6) - 1.0
            constraint_penalty += under
        if old_slice_overlap >= cfg.old_slice_overlap_threshold:
            constraint_penalty += 1.0
        reward -= cfg.constraint_weight * constraint_penalty

        if action == Action.STOP:
            if total_target_score > 0.0 and old_slice_overlap < cfg.old_slice_overlap_threshold:
                quality = min(total_target_score, 4.0)
                reward += cfg.stop_bonus_weight * quality
            elif observable_score > 0.3:
                reward += cfg.stop_bonus_weight * 0.3 * min(observable_score, 2.0)
            else:
                reward -= cfg.stop_bonus_weight * 0.5

        return float(reward), info

    def _legacy_reward(self, action: Action, previous_roi: np.ndarray | None = None) -> tuple[float, dict]:
        prev_covered = self.covered.copy()
        target_scores, hit_mask = self._update_covered()
        new_mask = hit_mask & ~prev_covered
        new_hits = int(new_mask.sum())
        target_score = float(target_scores[new_mask].sum()) if len(target_scores) else 0.0
        hit_count = int(hit_mask.sum()) if len(hit_mask) else 0
        total_target_score = float(target_scores[hit_mask].sum()) if len(target_scores) else 0.0
        retained_hits = int((hit_mask & prev_covered).sum()) if len(hit_mask) else 0
        roi_area_ratio = self._roi_area_ratio()
        compactness = 1.0 - min(roi_area_ratio / max(self.env_cfg.max_roi_area_ratio, 1e-6), 1.0)
        compactness_score = total_target_score * compactness
        previous_compactness_score = 0.0
        if previous_roi is not None and len(self.hard_boxes) > 0:
            previous_scores, previous_hit_mask = self._hard_target_scores(previous_roi)
            previous_total_score = (
                float(previous_scores[previous_hit_mask].sum()) if len(previous_scores) else 0.0
            )
            previous_area_ratio = self._roi_area_ratio(previous_roi)
            previous_compactness = 1.0 - min(
                previous_area_ratio / max(self.env_cfg.max_roi_area_ratio, 1e-6),
                1.0,
            )
            previous_compactness_score = previous_total_score * previous_compactness
        compactness_delta = compactness_score - previous_compactness_score
        observable_score = self._observable_target_score()
        previous_observable_score = (
            self._observable_target_score(previous_roi) if previous_roi is not None else observable_score
        )
        observable_delta = observable_score - previous_observable_score
        scale_gain = self._scale_gain()
        old_slice_overlap = self._old_slice_overlap()

        reward = -self.env_cfg.step_penalty
        info = {
            "new_hits": new_hits,
            "hit_count": hit_count,
            "target_score": target_score,
            "total_target_score": total_target_score,
            "retained_hits": retained_hits,
            "compactness_score": compactness_score,
            "compactness_delta": compactness_delta,
            "observable_score": observable_score,
            "observable_delta": observable_delta,
            "roi_area_ratio": roi_area_ratio,
            "scale_gain": scale_gain,
            "old_slice_overlap": old_slice_overlap,
            "detected_overlap": 0.0,
        }

        if len(self.hard_boxes) > 0:
            if new_hits > 0:
                reward += self.env_cfg.new_hard_reward * new_hits
                density = target_score * self.env_cfg.max_roi_area_ratio / max(roi_area_ratio, 1e-6)
                reward += self.env_cfg.hard_density_reward * float(np.clip(density, 0.0, 4.0))
            if hit_count > 0:
                reward += self.env_cfg.compactness_reward * float(np.clip(compactness_delta, -4.0, 4.0))
                if action != Action.STOP and new_hits == 0 and compactness_delta <= 0.0:
                    reward -= self.env_cfg.continue_target_penalty * min(total_target_score, 4.0)
            else:
                reward -= self.env_cfg.empty_slice_penalty
        elif action != Action.STOP:
            reward -= self.env_cfg.empty_slice_penalty

        if self.env_cfg.observable_target_reward > 0.0:
            reward += self.env_cfg.observable_target_reward * float(np.clip(observable_delta, -2.0, 2.0))

        det_mask = self.det_scores >= self.env_cfg.high_conf_threshold
        det_boxes = self.det_boxes[det_mask]
        if len(det_boxes) > 0:
            det_cover = ioa_matrix(self.roi.reshape(1, 4), det_boxes)[0]
            detected_overlap = float(np.clip(det_cover.sum() / max(len(det_boxes), 1), 0.0, 1.0))
            reward -= self.env_cfg.detected_overlap_penalty * detected_overlap
            info["detected_overlap"] = detected_overlap

        reward -= self.env_cfg.area_penalty * roi_area_ratio
        if roi_area_ratio > self.env_cfg.max_roi_area_ratio:
            overflow = roi_area_ratio / max(self.env_cfg.max_roi_area_ratio, 1e-6) - 1.0
            reward -= self.env_cfg.large_roi_penalty * overflow
        if scale_gain < self.env_cfg.min_scale_gain:
            under_scale = self.env_cfg.min_scale_gain / max(scale_gain, 1e-6) - 1.0
            reward -= self.env_cfg.low_scale_penalty * under_scale
        if old_slice_overlap >= self.env_cfg.old_slice_overlap_threshold:
            overflow = old_slice_overlap / max(self.env_cfg.old_slice_overlap_threshold, 1e-6) - 1.0
            reward -= self.env_cfg.old_slice_overlap_penalty * (1.0 + overflow)

        if action == Action.STOP:
            if total_target_score > 0.0 and old_slice_overlap < self.env_cfg.old_slice_overlap_threshold:
                stop_quality = min(total_target_score, 4.0)
                stop_quality += min(total_target_score / max(hit_count, 1), 1.0)
                reward += self.env_cfg.stop_target_reward * stop_quality
            elif observable_score > 0.25 and old_slice_overlap < self.env_cfg.old_slice_overlap_threshold:
                reward += self.env_cfg.stop_observable_target_reward * min(observable_score, 2.0)
            else:
                reward -= self.env_cfg.stop_early_penalty
        return float(reward), info
