from __future__ import annotations

import glob
import unittest
from pathlib import Path

import numpy as np

from rl_sahi.common.cache import load_detection_cache
from rl_sahi.rl.adaptiveconf_env import ADAPTIVECONF_STATE_DIM, AdaptiveConfEnv
from rl_sahi.rl.env_config import EnvConfig
from rl_sahi.rl.state_config import StateConfig


def _first_detection_cache() -> Path | None:
    for split in ("val", "test", "train"):
        hits = sorted(glob.glob(f"data/cache/detections/{split}/*.npz"))
        if hits:
            return Path(hits[0])
    return None


class AdaptiveConfEnvStateGTFreeTest(unittest.TestCase):
    """State PHAI GT-free: small_gt_caught + fp (=GT) chi vao REWARD, KHONG vao state."""

    def setUp(self) -> None:
        path = _first_detection_cache()
        if path is None:
            self.skipTest("khong co detection cache")
        self.det = load_detection_cache(path)
        self.cfg = EnvConfig()
        self.sc = StateConfig()
        rng = np.random.default_rng(0)
        self.K, self.C, self.N = 12, 3, 30
        self.cells = list(range(self.K))
        self.rois = rng.uniform(0, 200, size=(self.K, 4)).astype(np.float32)
        self.rois[:, 2:] += 50.0
        self.raw = rng.integers(0, 20, size=(self.K, self.C)).astype(np.float32)
        self.caught = rng.random((self.K, self.C, self.N)) > 0.7
        self.fp = rng.integers(0, 8, size=(self.K, self.C)).astype(np.float32)
        self.confs = np.array([0.25, 0.10, 0.05], dtype=np.float32)

    def _make(self, with_gt: bool) -> AdaptiveConfEnv:
        return AdaptiveConfEnv(
            self.det, self.cells, self.rois, self.raw,
            self.caught if with_gt else None, self.fp if with_gt else None,
            self.confs, env_cfg=self.cfg, state_cfg=self.sc,
        )

    def test_state_bit_identical_with_and_without_gt(self) -> None:
        env_gt, env_no = self._make(True), self._make(False)
        np.testing.assert_array_equal(env_gt.reset(), env_no.reset())
        actions = [1, 2, 0, 3, 1, 0, 2, 3, 1, 1, 0, 2, 3, 1]
        for t in range(self.K):
            a = actions[t % len(actions)]
            r_gt, r_no = env_gt.step(a), env_no.step(a)
            np.testing.assert_array_equal(r_gt.state, r_no.state, err_msg=f"GT leak o buoc {t}")
            if r_gt.done:
                break

    def test_state_dim(self) -> None:
        env = self._make(True)
        self.assertEqual(env.reset().shape, (ADAPTIVECONF_STATE_DIM,))


if __name__ == "__main__":
    unittest.main()
