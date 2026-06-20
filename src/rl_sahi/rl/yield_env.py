from __future__ import annotations

import numpy as np

from rl_sahi.common.boxes import as_boxes, centers
from rl_sahi.common.cache import DetectionCache, HardRegionCache
from rl_sahi.common.class_mapping import ClassMapping
from rl_sahi.rl.env_config import EnvConfig, StepResult
from rl_sahi.rl.hotspot_env import rank_density_hotspots
from rl_sahi.rl.state_config import StateConfig
from rl_sahi.rl.state_maps import build_detection_map

NUM_YIELD_ACTIONS = 2  # 0 = CROP, 1 = SKIP
YIELD_STATE_DIM = 15
_YN = 10.0  # yield normaliser


def compute_static_features(
    detection: DetectionCache, cells: list[int], state_cfg: StateConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Per-hotspot GT-FREE static features (k, 8) + centers (k, 2)."""
    grid = int(state_cfg.grid_size)
    h, w = detection.image_shape
    density = build_detection_map(detection.boxes, detection.scores, detection.image_shape, state_cfg)[2]
    obj = detection.objectness_map[0] if detection.objectness_map.ndim == 3 else detection.objectness_map
    prop_c = centers(detection.boxes) if len(detection.boxes) else np.zeros((0, 2), dtype=np.float32)
    feats = np.zeros((len(cells), 8), dtype=np.float32)
    cents = np.zeros((len(cells), 2), dtype=np.float32)
    for i, cell in enumerate(cells):
        gy, gx = cell // grid, cell % grid
        cx = (gx + 0.5) * w / grid
        cy = (gy + 0.5) * h / grid
        if len(prop_c):
            inc = (prop_c[:, 0] >= gx * w / grid) & (prop_c[:, 0] < (gx + 1) * w / grid) & \
                  (prop_c[:, 1] >= gy * h / grid) & (prop_c[:, 1] < (gy + 1) * h / grid)
            ncell = int(inc.sum())
            mc = float(detection.scores[inc].mean()) if inc.any() else 0.0
            xc = float(detection.scores[inc].max()) if inc.any() else 0.0
        else:
            ncell, mc, xc = 0, 0.0, 0.0
        feats[i] = [
            float(density[gy, gx] / max(state_cfg.count_norm, 1.0)),
            float(obj[gy, gx]),
            cx / w,
            cy / h,
            float(np.hypot(cx / w - 0.5, cy / h - 0.5)),
            ncell / 10.0,
            mc,
            xc,
        ]
        cents[i] = [cx, cy]
    return feats, cents


def yield_state(
    static: np.ndarray, cents: np.ndarray, i: int, k_max: int,
    cropped_idx: list[int], skipped: int, observed_yield: list[float],
) -> np.ndarray:
    """GT-FREE state for candidate i: static feats + OBSERVED yields of already-cropped hotspots."""
    if i >= len(static):
        base = np.zeros(8, dtype=np.float32)
        pos = np.array([0.0, 0.0], dtype=np.float32)
    else:
        base = static[i]
        pos = cents[i]
    if cropped_idx:
        ys = np.asarray(observed_yield, dtype=np.float32)
        d = np.hypot(cents[cropped_idx, 0] - pos[0], cents[cropped_idx, 1] - pos[1])
        nearest = float(ys[int(np.argmin(d))])
        mean_y, max_y = float(ys.mean()), float(ys.max())
    else:
        nearest = mean_y = max_y = 0.0
    n = max(k_max, 1)
    dyn = np.array(
        [len(cropped_idx) / n, mean_y / _YN, nearest / _YN, max_y / _YN, i / n, (len(static) - i) / n, skipped / n],
        dtype=np.float32,
    )
    return np.nan_to_num(np.concatenate([base, dyn]), nan=0.0, posinf=0.0, neginf=0.0)


class YieldAwareHotspotEnv:
    """Yield-aware placement: walk density-ranked hotspots, action {CROP, SKIP}.

    STATE is GT-FREE: static hotspot feats + raw_yield (so detection moi YOLO cho ra) cua cac
    hotspot DA CAT -> co o train va infer. GT (real_yield) vao DUY NHAT reward (train).
    """

    def __init__(
        self,
        detection: DetectionCache,
        hard_regions: HardRegionCache | None,
        raw_yields: np.ndarray | None = None,
        real_yields: np.ndarray | None = None,
        rois: np.ndarray | None = None,
        cells: list[int] | None = None,
        env_cfg: EnvConfig | None = None,
        state_cfg: StateConfig | None = None,
        target_classes: tuple[int, ...] = (),
        class_mapping: ClassMapping | None = None,
    ) -> None:
        self.env_cfg = env_cfg or EnvConfig()
        self.state_cfg = state_cfg or StateConfig()
        self.image_shape = detection.image_shape
        self.k_max = int(self.env_cfg.k_max)
        if rois is None:
            grid = int(self.state_cfg.grid_size)
            density = build_detection_map(detection.boxes, detection.scores, self.image_shape, self.state_cfg)[2]
            floor = (self.env_cfg.density_potential_min_count - 0.5) / max(self.state_cfg.count_norm, 1.0)
            h, w = self.image_shape
            side = max(1.0, min(h, w) * self.env_cfg.hotspot_slice_fraction)
            roi_list, cell_list = rank_density_hotspots(density, self.image_shape, grid, floor, side, self.k_max)
            self.rois = np.asarray(roi_list, dtype=np.float32).reshape(-1, 4)
            self.cells = cell_list
        else:
            self.rois = np.asarray(rois, dtype=np.float32).reshape(-1, 4)
            self.cells = list(cells) if cells is not None else list(range(len(self.rois)))
        self.static, self.cents = compute_static_features(detection, self.cells, self.state_cfg)
        k = len(self.rois)
        self.raw_yields = np.asarray(raw_yields, dtype=np.float32) if raw_yields is not None else np.zeros(k, dtype=np.float32)
        self.real_yields = np.asarray(real_yields, dtype=np.float32) if real_yields is not None else np.zeros(k, dtype=np.float32)
        self._has_gt = real_yields is not None

    def reset(self) -> np.ndarray:
        self.i = 0
        self.cropped_idx: list[int] = []
        self.observed_yield: list[float] = []
        self.skipped = 0
        self.placed: list[np.ndarray] = []
        return self._state()

    def _state(self) -> np.ndarray:
        return yield_state(self.static, self.cents, self.i, self.k_max, self.cropped_idx, self.skipped, self.observed_yield)

    def valid_actions(self) -> np.ndarray:
        return np.ones((NUM_YIELD_ACTIONS,), dtype=bool)

    def set_observed_yield(self, value: float) -> None:
        """Infer-time hook: feed the live YOLO raw-yield of the crop just taken."""
        if self.cropped_idx:
            self.observed_yield[-1] = float(value)

    def step(self, action: int) -> StepResult:
        a = int(action)
        done = False
        reward = 0.0
        if self.i >= len(self.rois):
            return StepResult(self._state(), 0.0, True, self._info(stopped=True))
        if a == 0:  # CROP
            reward = self.env_cfg.w_cov * float(self.real_yields[self.i]) - self.env_cfg.crop_cost
            self.cropped_idx.append(self.i)
            self.observed_yield.append(float(self.raw_yields[self.i]))
            self.placed.append(self.rois[self.i].copy())
        else:  # SKIP
            self.skipped += 1
        self.i += 1
        if self.i >= len(self.rois):
            done = True
        return StepResult(self._state(), float(reward), bool(done), self._info(stopped=False))

    def _info(self, stopped: bool) -> dict:
        return {
            "n_crops": len(self.placed),
            "captured_yield": float(sum(self.observed_yield)),
            "real_captured": float(self.real_yields[self.cropped_idx].sum()) if self._has_gt and self.cropped_idx else 0.0,
            "stopped": stopped,
        }

    def rollout_rois(self, policy, device) -> list[np.ndarray]:
        """Train-time / cached-yield rollout (greedy argmax)."""
        import torch

        state = self.reset()
        for _ in range(len(self.rois) + 1):
            with torch.no_grad():
                q = policy(torch.from_numpy(state).float().unsqueeze(0).to(device))
                action = int(q.argmax(dim=1).item())
            result = self.step(action)
            state = result.state
            if result.done:
                break
        return [roi.copy() for roi in self.placed]
