from __future__ import annotations

import unittest
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl_sahi.common.cache import DetectionCache, HardRegionCache
from rl_sahi.rl.env_config import EnvConfig
from rl_sahi.rl.hotspot_env import HotspotEnv, HOTSPOT_STATE_DIM, NUM_HOTSPOT_ACTIONS


def _det() -> DetectionCache:
    # proposal boxes (low-conf) tạo density ở vài ô khác nhau
    boxes = np.array(
        [[40, 40, 70, 70], [42, 44, 68, 72], [200, 60, 230, 90], [205, 62, 235, 92], [120, 200, 150, 230]],
        dtype=np.float32,
    )
    scores = np.array([0.1, 0.12, 0.2, 0.18, 0.08], dtype=np.float32)
    classes = np.zeros((5,), dtype=np.float32)
    return DetectionCache(
        "x.jpg", (300, 400), boxes, scores, classes,
        np.zeros((4,), np.float32), (10,), np.zeros((1, 16, 16), np.float32), np.zeros((4, 16, 16), np.float32),
    )


def _hard() -> HardRegionCache:
    hb = np.array([[50, 50, 60, 60], [210, 70, 220, 80]], dtype=np.float32)
    return HardRegionCache(
        "x.jpg", (300, 400), hb, hb.copy(), hb.copy(),
        np.zeros((2,), np.float32), np.zeros((2,), np.float32),
    )


class HotspotEnvGtFreeTest(unittest.TestCase):
    """State PHẢI bit-identical dù có GT hay không -> chứng minh không rò GT vào observation (chống train↔infer gap)."""

    def test_state_dim_and_actions(self) -> None:
        env = HotspotEnv(_det(), _hard(), env_cfg=EnvConfig(use_hotspot_env=True))
        s = env.reset()
        self.assertEqual(s.shape[0], HOTSPOT_STATE_DIM)
        self.assertEqual(NUM_HOTSPOT_ACTIONS, 2)
        self.assertEqual(env.valid_actions().shape[0], 2)

    def test_state_bit_identical_with_and_without_gt(self) -> None:
        cfg = EnvConfig(use_hotspot_env=True)
        env_gt = HotspotEnv(_det(), _hard(), env_cfg=cfg)
        env_none = HotspotEnv(_det(), None, env_cfg=cfg)
        np.testing.assert_array_equal(env_gt.reset(), env_none.reset())
        # qua nhiều bước CROP: state vẫn phải trùng tuyệt đối (GT chỉ vào reward, không vào state)
        for _ in range(4):
            if not env_gt.valid_actions()[0]:
                break
            r_gt = env_gt.step(0)
            r_none = env_none.step(0)
            np.testing.assert_array_equal(r_gt.state, r_none.state)

    def test_reward_uses_gt_only_when_present(self) -> None:
        cfg = EnvConfig(use_hotspot_env=True, w_cov=1.0, crop_cost=0.15)
        env_gt = HotspotEnv(_det(), _hard(), env_cfg=cfg)
        env_none = HotspotEnv(_det(), None, env_cfg=cfg)
        env_gt.reset(); env_none.reset()
        # infer (no GT): reward chỉ là -crop_cost (n_new=0). train (GT): có thể dương khi crop trúng small-GT.
        r_none = env_gt_none = env_none.step(0)
        self.assertAlmostEqual(r_none.reward, -cfg.crop_cost, places=5)
        # với GT, tổng reward qua cả episode >= reward không-GT (vì có thể +w_cov*n_new)
        total_gt = 0.0
        e = HotspotEnv(_det(), _hard(), env_cfg=cfg); e.reset()
        for _ in range(e.k_max + 1):
            if not e.valid_actions()[0]:
                break
            total_gt += e.step(0).reward
        self.assertGreaterEqual(total_gt, -cfg.crop_cost * e.k_max)

    def test_stop_ends_episode(self) -> None:
        env = HotspotEnv(_det(), _hard(), env_cfg=EnvConfig(use_hotspot_env=True))
        env.reset()
        result = env.step(1)  # STOP
        self.assertTrue(result.done)


if __name__ == "__main__":
    unittest.main()
