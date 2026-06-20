from __future__ import annotations

import unittest
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl_sahi.common.actions import Action, NUM_ACTIONS, ACTION_NAMES
from rl_sahi.common.cache import DetectionCache
from rl_sahi.rl.env_config import EnvConfig
from rl_sahi.rl.slice_env import SliceEnv


EXPECTED_ACTIONS = {"left", "right", "up", "down", "zoom_in", "zoom_out", "stop"}
DIAGONAL_NAMES = {"up_left", "up_right", "down_left", "down_right"}


def _detection_cache() -> DetectionCache:
    return DetectionCache(
        image_path="synthetic.jpg",
        image_shape=(100, 100),
        boxes=np.zeros((0, 4), dtype=np.float32),
        scores=np.zeros((0,), dtype=np.float32),
        classes=np.zeros((0,), dtype=np.float32),
        feature=np.zeros((4,), dtype=np.float32),
        feature_layers=(10,),
        objectness_map=np.zeros((1, 16, 16), dtype=np.float32),
        spatial_feature_map=np.zeros((4, 16, 16), dtype=np.float32),
    )


class ActionSetTest(unittest.TestCase):
    """Chặn doc↔code drift về action set (lesson L6: doc từng tuyên bố 'diagonal actions' không có trong code)."""

    def test_num_actions_is_seven(self) -> None:
        self.assertEqual(NUM_ACTIONS, 7)
        self.assertEqual(len(Action), 7)

    def test_action_names_match_expected(self) -> None:
        self.assertEqual(set(ACTION_NAMES.values()), EXPECTED_ACTIONS)

    def test_no_diagonal_actions(self) -> None:
        names = set(ACTION_NAMES.values())
        for diag in DIAGONAL_NAMES:
            self.assertNotIn(diag, names)

    def test_apply_action_handles_every_enum_member(self) -> None:
        env = SliceEnv(_detection_cache(), None, env_cfg=EnvConfig())
        env.reset()
        for action in Action:
            roi = env._apply_action(action)
            self.assertEqual(tuple(roi.shape), (4,))

    def test_stop_does_not_move_roi(self) -> None:
        env = SliceEnv(_detection_cache(), None, env_cfg=EnvConfig())
        env.reset()
        before = env.roi.copy()
        self.assertTrue(np.allclose(env._apply_action(Action.STOP), before))


if __name__ == "__main__":
    unittest.main()
