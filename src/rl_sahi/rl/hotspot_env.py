from __future__ import annotations

import numpy as np

from rl_sahi.common.boxes import as_boxes, centers, rasterize_boxes
from rl_sahi.common.cache import DetectionCache, HardRegionCache
from rl_sahi.common.class_mapping import ClassMapping
from rl_sahi.rl.env_config import EnvConfig, StepResult
from rl_sahi.rl.state_config import StateConfig
from rl_sahi.rl.state_maps import build_detection_map

NUM_HOTSPOT_ACTIONS = 2  # 0 = CROP_NEXT, 1 = STOP
HOTSPOT_STATE_DIM = 12


def rank_density_hotspots(
    density: np.ndarray, image_shape: tuple[int, int], grid: int, floor: float, side: float, k_max: int
) -> tuple[list[np.ndarray], list[int]]:
    flat = density.reshape(-1)
    order = np.argsort(flat)[::-1]
    h, w = image_shape
    rois: list[np.ndarray] = []
    cells: list[int] = []
    used: list[tuple[float, float]] = []
    for idx in order:
        if float(flat[idx]) < floor or len(rois) >= k_max:
            break
        gy, gx = divmod(int(idx), grid)
        cx = (gx + 0.5) * w / grid
        cy = (gy + 0.5) * h / grid
        if any(abs(cx - ux) < side * 0.5 and abs(cy - uy) < side * 0.5 for ux, uy in used):
            continue
        x1 = float(np.clip(cx - side / 2.0, 0.0, max(w - side, 0.0)))
        y1 = float(np.clip(cy - side / 2.0, 0.0, max(h - side, 0.0)))
        rois.append(np.asarray([x1, y1, min(x1 + side, w), min(y1 + side, h)], dtype=np.float32))
        cells.append(int(idx))
        used.append((cx, cy))
    return rois, cells


class HotspotEnv:
    """Optimal-stopping over density-ranked hotspots: action {CROP_NEXT, STOP}.

    State is a pure function of proposal_density (GT-FREE) -> identical at train and infer.
    GT (small_gt_boxes) enters ONLY the reward (train); at infer hard_regions=None -> reward unused.
    """

    def __init__(
        self,
        detection: DetectionCache,
        hard_regions: HardRegionCache | None,
        env_cfg: EnvConfig | None = None,
        state_cfg: StateConfig | None = None,
        target_classes: tuple[int, ...] = (),
        class_mapping: ClassMapping | None = None,
    ) -> None:
        self.env_cfg = env_cfg or EnvConfig()
        self.state_cfg = state_cfg or StateConfig()
        self.class_mapping = class_mapping or ClassMapping()
        self.image_shape = detection.image_shape
        self.grid = int(self.state_cfg.grid_size)
        self.density = build_detection_map(detection.boxes, detection.scores, self.image_shape, self.state_cfg)[2]
        self.floor = (self.env_cfg.density_potential_min_count - 0.5) / max(self.state_cfg.count_norm, 1.0)
        h, w = self.image_shape
        self.side = max(1.0, min(h, w) * self.env_cfg.hotspot_slice_fraction)
        self.k_max = int(self.env_cfg.k_max)
        self.rois, self.cells = rank_density_hotspots(self.density, self.image_shape, self.grid, self.floor, self.side, self.k_max)
        if hard_regions is not None:
            gt = as_boxes(hard_regions.small_gt_boxes)
            self.small_gt_centers = centers(gt) if len(gt) > 0 else np.zeros((0, 2), dtype=np.float32)
            self._has_gt = True
        else:
            self.small_gt_centers = np.zeros((0, 2), dtype=np.float32)
            self._has_gt = False

    def reset(self) -> np.ndarray:
        self.i = 0
        self.residual = self.density.copy()
        self.cropped_density = 0.0
        self.last_yield = float(self.density.reshape(-1)[self.cells[0]]) if self.cells else 0.0
        self.first_yield = self.last_yield
        self.covered = np.zeros((len(self.small_gt_centers),), dtype=bool)
        self.placed: list[np.ndarray] = []
        return self._state()

    def _next_yield(self) -> float:
        if self.i >= len(self.cells):
            return 0.0
        cell = self.cells[self.i]
        return float(self.residual[cell // self.grid, cell % self.grid])

    def _can_crop(self) -> bool:
        return self.i < len(self.cells) and self._next_yield() >= self.floor

    def _state(self) -> np.ndarray:
        # GT-FREE: derives ONLY from density / residual / ranked cells.
        ny = self._next_yield()
        rem = np.array(
            [self.residual[c // self.grid, c % self.grid] for c in self.cells[self.i:]], dtype=np.float32
        )
        useful = rem[rem >= self.floor] if len(rem) else rem
        total = float(self.density.sum()) + 1e-6
        s = np.array(
            [
                ny,
                float(useful.sum()),
                self.i / max(self.k_max, 1),
                len(useful) / max(self.k_max, 1),
                ny / max(self.first_yield, 1e-6),
                ny / max(self.last_yield, 1e-6),
                float(useful.mean()) if len(useful) else 0.0,
                float(useful.std()) if len(useful) else 0.0,
                float(self.density.sum() / max(self.state_cfg.count_norm, 1.0)),
                self.cropped_density / total,
                self.floor,
                ny - self.floor,
            ],
            dtype=np.float32,
        )
        return np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)

    def valid_actions(self) -> np.ndarray:
        valid = np.ones((NUM_HOTSPOT_ACTIONS,), dtype=bool)
        valid[0] = self._can_crop()
        valid[1] = True
        return valid

    def _count_new_small(self, roi: np.ndarray) -> int:
        if not self._has_gt or len(self.small_gt_centers) == 0:
            return 0
        cx, cy = self.small_gt_centers[:, 0], self.small_gt_centers[:, 1]
        new = (cx >= roi[0]) & (cx <= roi[2]) & (cy >= roi[1]) & (cy <= roi[3]) & ~self.covered
        n = int(new.sum())
        if n:
            self.covered |= new
        return n

    def step(self, action: int) -> StepResult:
        a = int(action)
        if a == 0 and self._can_crop():
            roi = self.rois[self.i]
            cell = self.cells[self.i]
            n_new = self._count_new_small(roi)
            reward = self.env_cfg.w_cov * float(n_new) - self.env_cfg.crop_cost
            roi_map = rasterize_boxes(roi.reshape(1, 4), self.image_shape, self.grid)
            self.residual = (self.residual * (roi_map <= 0.0)).astype(np.float32)
            self.cropped_density += float(self.density.reshape(-1)[cell])
            self.last_yield = float(self.density.reshape(-1)[cell])
            self.placed.append(roi)
            self.i += 1
            done = not self._can_crop()
        else:
            reward = 0.0
            done = True
        info = {
            "n_crops": len(self.placed),
            "covered": int(self.covered.sum()) if self._has_gt else 0,
            "small_total": int(len(self.small_gt_centers)),
            "stopped": bool(a == 1),
        }
        return StepResult(self._state(), float(reward), bool(done), info)

    def rollout_rois(self, policy, device) -> list[np.ndarray]:
        import torch

        state = self.reset()
        for _ in range(self.k_max + 1):
            with torch.no_grad():
                q = policy(torch.from_numpy(state).float().unsqueeze(0).to(device))
                valid = torch.from_numpy(self.valid_actions()).bool().to(device)
                q[:, ~valid] = -torch.inf
                action = int(q.argmax(dim=1).item())
            result = self.step(action)
            state = result.state
            if result.done:
                break
        return [roi.copy() for roi in self.placed]
