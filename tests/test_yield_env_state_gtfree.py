from __future__ import annotations

import glob
import unittest
from pathlib import Path

import numpy as np

from rl_sahi.common.cache import load_detection_cache
from rl_sahi.rl.env_config import EnvConfig
from rl_sahi.rl.state_config import StateConfig
from rl_sahi.rl.yield_env import YieldAwareHotspotEnv


def _first_detection_cache() -> Path | None:
    for split in ("val", "test", "train"):
        hits = sorted(glob.glob(f"data/cache/detections/{split}/*.npz"))
        if hits:
            return Path(hits[0])
    return None


class YieldEnvStateGTFreeTest(unittest.TestCase):
    """State cua yield-aware env PHAI GT-free: real_yield (=GT) chi vao reward, KHONG vao state.

    Neu state lo real_yield -> tai tao train<->infer gap (sai ca de tai). Test khoa cua nay.
    """

    def setUp(self) -> None:
        path = _first_detection_cache()
        if path is None:
            self.skipTest("khong co detection cache de test")
        self.det = load_detection_cache(path)
        self.env_cfg = EnvConfig()
        self.state_cfg = StateConfig()

    def _make(self, with_gt: bool, raw: np.ndarray, real: np.ndarray) -> YieldAwareHotspotEnv:
        return YieldAwareHotspotEnv(
            self.det, None,
            raw_yields=raw,
            real_yields=real if with_gt else None,
            env_cfg=self.env_cfg, state_cfg=self.state_cfg,
        )

    def test_state_bit_identical_with_and_without_gt(self) -> None:
        probe = self._make(True, np.zeros(0), np.zeros(0))
        k = len(probe.rois)
        if k == 0:
            self.skipTest("anh nay khong co hotspot")
        rng = np.random.default_rng(0)
        raw = rng.integers(0, 20, size=k).astype(np.float32)
        real = rng.integers(0, 10, size=k).astype(np.float32)  # "GT" yields

        env_gt = self._make(True, raw, real)
        env_no = self._make(False, raw, real)
        s_gt, s_no = env_gt.reset(), env_no.reset()
        np.testing.assert_array_equal(s_gt, s_no)

        actions = [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]
        for t in range(k):
            a = actions[t % len(actions)]
            r_gt = env_gt.step(a)
            r_no = env_no.step(a)
            np.testing.assert_array_equal(
                r_gt.state, r_no.state, err_msg=f"state khac nhau o buoc {t} (action {a}) -> GT leak!"
            )
            if r_gt.done:
                break

    def test_state_dim_constant(self) -> None:
        from rl_sahi.rl.yield_env import YIELD_STATE_DIM

        probe = self._make(True, np.zeros(0), np.zeros(0))
        if len(probe.rois) == 0:
            self.skipTest("anh nay khong co hotspot")
        k = len(probe.rois)
        env = self._make(True, np.zeros(k, np.float32), np.zeros(k, np.float32))
        s = env.reset()
        self.assertEqual(s.shape, (YIELD_STATE_DIM,))


if __name__ == "__main__":
    unittest.main()
