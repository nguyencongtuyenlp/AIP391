from __future__ import annotations

import numpy as np

from rl_sahi.common.boxes import rasterize_boxes
from rl_sahi.common.cache import DetectionCache
from rl_sahi.rl.env_config import EnvConfig, StepResult
from rl_sahi.rl.state_config import StateConfig
from rl_sahi.rl.state_maps import build_detection_map

_YN = 10.0  # yield normaliser
_MAX_CONFS = 3  # so muc conf mac dinh (state dim co dinh theo day)
# Adaptive-conf (lever moi): moi hotspot density, agent chon SKIP hoac CROP o 1 trong C MUC CONF.
# Ha conf -> cuu vat-bo-lo o tin hieu thap (YOLO da sinh box, chi bi nguong vut). Reward tru FP
# de agent chi ha conf khi box that. KHONG fine-tune, KHONG SR — chi RL dieu khien nguong.
# State GT-FREE + INFER-VALID: density/objness/pos + overlap voi vung da cat + raw-count DA CAT.
# (raw-count = tong box-moi, GT-free; KHONG dua fp/real vao state vi can GT.)
ADAPTIVECONF_STATE_DIM = 2 + 2 + 3 + 2 + 1  # base(2)+pos(2)+prog(3)+obs(2)+overlap(1) = 10


def num_adaptiveconf_actions(n_confs: int) -> int:
    return 1 + int(n_confs)  # SKIP + CROP@conf_j


class AdaptiveConfEnv:
    """Moi hotspot density, agent chon SKIP hoac CROP o 1 muc conf (cao->thap).

    Reward = w_cov * (vat-nho-bo-lo MOI bat duoc, dedup) - fp_weight * (FP moi) - crop_cost.
    Ha conf bat them vat that NHUNG cong them FP -> agent hoc ha conf DUNG VUNG (density/objness cao).
    """

    def __init__(
        self,
        detection: DetectionCache,
        cells: list[int],
        rois: np.ndarray,            # (K, 4) — single scale, dung chung moi conf
        raw_yields: np.ndarray,      # (K, C) — GT-free observed signal
        small_gt_caught: np.ndarray | None,  # (K, C, N) bool — dedup reward
        fp: np.ndarray | None,       # (K, C) — so FP moi tung conf (reward, train)
        confs: np.ndarray,
        env_cfg: EnvConfig | None = None,
        state_cfg: StateConfig | None = None,
    ) -> None:
        self.env_cfg = env_cfg or EnvConfig()
        self.state_cfg = state_cfg or StateConfig()
        self.image_shape = detection.image_shape
        self.grid = int(self.state_cfg.grid_size)
        self.cells = list(cells)
        self.rois = np.asarray(rois, dtype=np.float32).reshape(len(self.cells), 4) if len(self.cells) else np.zeros((0, 4), np.float32)
        self.raw_yields = np.asarray(raw_yields, dtype=np.float32).reshape(len(self.cells), -1) if len(self.cells) else np.zeros((0, _MAX_CONFS), np.float32)
        self.confs = np.asarray(confs, dtype=np.float32)
        self.C = int(self.raw_yields.shape[1]) if len(self.cells) else _MAX_CONFS
        self._has_gt = small_gt_caught is not None
        if self._has_gt:
            self.caught = np.asarray(small_gt_caught, dtype=bool).reshape(len(self.cells), self.C, -1) if len(self.cells) else np.zeros((0, self.C, 0), dtype=bool)
            self.N = int(self.caught.shape[2])
        else:
            self.caught = None
            self.N = 0
        self.fp = np.asarray(fp, dtype=np.float32).reshape(len(self.cells), self.C) if (fp is not None and len(self.cells)) else np.zeros((len(self.cells), self.C), np.float32)
        # dac trung tinh GT-free
        dens = build_detection_map(detection.boxes, detection.scores, self.image_shape, self.state_cfg)[2]
        obj = detection.objectness_map[0] if detection.objectness_map.ndim == 3 else detection.objectness_map
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

    def _cell_overlap(self, i: int) -> float:
        """Phan dien tich ROI cell i da bi vung-da-cat phu (GT-free) — proxy do trung."""
        if i >= len(self.cells):
            return 0.0
        rmap = rasterize_boxes(self.rois[i].reshape(1, 4), self.image_shape, self.grid)
        cells_in = rmap > 0.0
        n_in = float(cells_in.sum())
        return float((self.coverage_grid[cells_in] > 0).sum()) / max(n_in, 1.0)

    def _state(self) -> np.ndarray:
        if self.i >= len(self.cells):
            base = np.zeros(2, np.float32); pos = np.zeros(2, np.float32)
        else:
            base = np.array([self.density[self.i], self.objness[self.i]], np.float32)
            pos = self.pos[self.i]
        n = max(len(self.cells), 1)
        mean_y = float(np.mean(self.observed_raw)) / _YN if self.observed_raw else 0.0
        max_y = float(np.max(self.observed_raw)) / _YN if self.observed_raw else 0.0
        prog = np.array([len(self.cropped) / n, self.i / n, (len(self.cells) - self.i) / n], np.float32)
        obs = np.array([mean_y, max_y], np.float32)
        ov = np.array([self._cell_overlap(self.i)], np.float32)
        s = np.concatenate([base, pos, prog, obs, ov])
        s = s[:ADAPTIVECONF_STATE_DIM] if len(s) >= ADAPTIVECONF_STATE_DIM else np.concatenate([s, np.zeros(ADAPTIVECONF_STATE_DIM - len(s), np.float32)])
        return np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)

    def valid_actions(self) -> np.ndarray:
        v = np.zeros(num_adaptiveconf_actions(_MAX_CONFS), dtype=bool)
        v[0] = True  # SKIP luon hop le
        if self.i < len(self.cells):
            for j in range(self.C):
                v[1 + j] = True
        return v

    def step(self, action: int) -> StepResult:
        a = int(action)
        reward = 0.0
        if self.i < len(self.cells) and 1 <= a <= self.C:
            j = a - 1
            if self._has_gt:
                new = self.caught[self.i, j] & ~self.covered
                reward = self.env_cfg.w_cov * float(new.sum()) - self.env_cfg.fp_weight * float(self.fp[self.i, j]) - self.env_cfg.crop_cost
                self.covered |= self.caught[self.i, j]
            else:
                reward = -self.env_cfg.crop_cost
            roi = self.rois[self.i]
            self.coverage_grid = np.maximum(self.coverage_grid, rasterize_boxes(roi.reshape(1, 4), self.image_shape, self.grid))
            self.observed_raw.append(float(self.raw_yields[self.i, j]))
            self.cropped.append((self.i, j))
            self.placed.append(roi.copy())
        self.i += 1
        done = self.i >= len(self.cells)
        info = {"n_crops": len(self.placed), "covered": int(self.covered.sum()) if self._has_gt else 0, "small_total": self.N}
        return StepResult(self._state(), float(reward), bool(done), info)
