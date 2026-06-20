from __future__ import annotations

import numpy as np

from rl_sahi.common.boxes import rasterize_boxes
from rl_sahi.common.cache import DetectionCache
from rl_sahi.common.class_mapping import ClassMapping
from rl_sahi.rl.env_config import EnvConfig, StepResult
from rl_sahi.rl.state_config import StateConfig
from rl_sahi.rl.state_maps import build_detection_map

_YN = 10.0  # yield normaliser
_MAX_SCALES = 3  # so scale mac dinh (state dim co dinh theo day)
# State INFER-VALID: KHONG dung raw_yield cell hien tai (luc infer chua cat nen chua biet).
# Chi dung density/objness/pos + OVERLAP tung scale voi vung da cat + yield cac o DA CAT.
MULTISCALE_STATE_DIM = 2 + 2 + 4 + _MAX_SCALES + 1  # static(2)+pos(2)+progress/obs(4)+overlap(S)+max_obs(1) = 12


def num_multiscale_actions(n_scales: int) -> int:
    return 1 + int(n_scales)  # SKIP + CROP@scale_j


class MultiScaleYieldEnv:
    """Free-placement (A): moi hotspot density, agent chon SKIP hoac CROP o 1 trong N SCALE.

    STATE GT-FREE: dac trung cell + raw_yield tung scale (so vat-moi YOLO, GT-free) + OVERLAP
    cua tung scale voi vung da cat (de hoc chon scale it trung). Reward dedup (vat-bo-lo MOI)
    dung small_gt_caught -> chi vao reward (train), khong vao state.
    """

    def __init__(
        self,
        detection: DetectionCache,
        cells: list[int],
        rois: np.ndarray,            # (K, S, 4)
        raw_yields: np.ndarray,      # (K, S)
        real_yields: np.ndarray | None,   # (K, S) — khong dung neu co caught
        small_gt_caught: np.ndarray | None,  # (K, S, N) bool — dedup reward
        scales: np.ndarray,
        env_cfg: EnvConfig | None = None,
        state_cfg: StateConfig | None = None,
    ) -> None:
        self.env_cfg = env_cfg or EnvConfig()
        self.state_cfg = state_cfg or StateConfig()
        self.image_shape = detection.image_shape
        self.grid = int(self.state_cfg.grid_size)
        self.cells = list(cells)
        self.rois = np.asarray(rois, dtype=np.float32).reshape(len(self.cells), -1, 4) if len(self.cells) else np.zeros((0, _MAX_SCALES, 4), np.float32)
        self.raw_yields = np.asarray(raw_yields, dtype=np.float32).reshape(len(self.cells), -1) if len(self.cells) else np.zeros((0, _MAX_SCALES), np.float32)
        self.scales = np.asarray(scales, dtype=np.float32)
        self.S = int(self.rois.shape[1]) if len(self.cells) else _MAX_SCALES
        self._has_gt = small_gt_caught is not None
        if self._has_gt:
            self.caught = np.asarray(small_gt_caught, dtype=bool).reshape(len(self.cells), self.S, -1) if len(self.cells) else np.zeros((0, self.S, 0), dtype=bool)
            self.N = int(self.caught.shape[2])
        else:
            self.caught = None
            self.N = 0
        # dac trung tinh GT-free
        dens = build_detection_map(detection.boxes, detection.scores, self.image_shape, self.state_cfg)[2]
        obj = detection.objectness_map[0] if detection.objectness_map.ndim == 3 else detection.objectness_map
        h, w = self.image_shape
        self.density = np.array([dens[c // self.grid, c % self.grid] / max(self.state_cfg.count_norm, 1.0) for c in self.cells], dtype=np.float32) if self.cells else np.zeros(0, np.float32)
        self.objness = np.array([obj[c // self.grid, c % self.grid] for c in self.cells], dtype=np.float32) if self.cells else np.zeros(0, np.float32)
        self.pos = np.array([[((c % self.grid) + 0.5) / self.grid, ((c // self.grid) + 0.5) / self.grid] for c in self.cells], dtype=np.float32) if self.cells else np.zeros((0, 2), np.float32)

    def reset(self) -> np.ndarray:
        self.i = 0
        self.cropped: list[tuple[int, int]] = []
        self.observed_raw: list[float] = []
        self.covered = np.zeros(self.N, dtype=bool)
        self.coverage_grid = np.zeros((self.grid, self.grid), dtype=np.float32)
        self.placed: list[np.ndarray] = []
        return self._state()

    def _scale_overlap(self, i: int) -> np.ndarray:
        """Phan dien tich moi scale-ROI da bi vung-da-cat phu (GT-free) — proxy do trung."""
        ov = np.zeros(_MAX_SCALES, dtype=np.float32)
        if i >= len(self.cells):
            return ov
        for j in range(self.S):
            rmap = rasterize_boxes(self.rois[i, j].reshape(1, 4), self.image_shape, self.grid)
            cells_in = rmap > 0.0
            n_in = float(cells_in.sum())
            ov[j] = float((self.coverage_grid[cells_in] > 0).sum()) / max(n_in, 1.0)
        return ov

    def _state(self) -> np.ndarray:
        if self.i >= len(self.cells):
            base = np.zeros(2, np.float32); pos = np.zeros(2, np.float32)
        else:
            base = np.array([self.density[self.i], self.objness[self.i]], np.float32)
            pos = self.pos[self.i]
        ov = self._scale_overlap(self.i)  # infer-valid: tu vung da cat
        n = max(len(self.cells), 1)
        mean_y = float(np.mean(self.observed_raw)) / _YN if self.observed_raw else 0.0
        max_y = float(np.max(self.observed_raw)) / _YN if self.observed_raw else 0.0
        prog = np.array([len(self.cropped) / n, self.i / n, (len(self.cells) - self.i) / n, mean_y], np.float32)
        s = np.concatenate([base, pos, prog, ov, np.array([max_y], np.float32)])
        s = s[:MULTISCALE_STATE_DIM] if len(s) >= MULTISCALE_STATE_DIM else np.concatenate([s, np.zeros(MULTISCALE_STATE_DIM - len(s), np.float32)])
        return np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)

    def valid_actions(self) -> np.ndarray:
        v = np.zeros(num_multiscale_actions(_MAX_SCALES), dtype=bool)
        v[0] = True  # SKIP luon hop le
        if self.i < len(self.cells):
            for j in range(self.S):
                v[1 + j] = True
        return v

    def step(self, action: int) -> StepResult:
        a = int(action)
        reward = 0.0
        if self.i < len(self.cells) and 1 <= a <= self.S:
            j = a - 1
            if self._has_gt:
                new = self.caught[self.i, j] & ~self.covered
                reward = self.env_cfg.w_cov * float(new.sum()) - self.env_cfg.crop_cost
                self.covered |= self.caught[self.i, j]
            else:
                reward = -self.env_cfg.crop_cost
            roi = self.rois[self.i, j]
            self.coverage_grid = np.maximum(self.coverage_grid, rasterize_boxes(roi.reshape(1, 4), self.image_shape, self.grid))
            self.observed_raw.append(float(self.raw_yields[self.i, j]))
            self.cropped.append((self.i, j))
            self.placed.append(roi.copy())
        self.i += 1
        done = self.i >= len(self.cells)
        info = {"n_crops": len(self.placed), "covered": int(self.covered.sum()) if self._has_gt else 0, "small_total": self.N}
        return StepResult(self._state(), float(reward), bool(done), info)
