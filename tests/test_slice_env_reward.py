from __future__ import annotations

import unittest
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl_sahi.common.actions import Action
from rl_sahi.common.cache import DetectionCache, HardRegionCache
from rl_sahi.rl.env_config import EnvConfig
from rl_sahi.rl.slice_env import SliceEnv


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


def _hard_region_cache() -> HardRegionCache:
    hard_box = np.array([[49.0, 49.0, 51.0, 51.0]], dtype=np.float32)
    return HardRegionCache(
        image_path="synthetic.jpg",
        image_shape=(100, 100),
        hard_boxes=hard_box,
        small_gt_boxes=hard_box.copy(),
        gt_boxes=hard_box.copy(),
        matched_iou=np.zeros((1,), dtype=np.float32),
        matched_score=np.zeros((1,), dtype=np.float32),
    )


def _high_conf_detection_cache(cls: float) -> DetectionCache:
    return DetectionCache(
        image_path="synthetic.jpg",
        image_shape=(100, 100),
        boxes=np.array([[40.0, 40.0, 60.0, 60.0]], dtype=np.float32),
        scores=np.array([0.9], dtype=np.float32),
        classes=np.array([cls], dtype=np.float32),
        feature=np.zeros((4,), dtype=np.float32),
        feature_layers=(10,),
        objectness_map=np.zeros((1, 16, 16), dtype=np.float32),
        spatial_feature_map=np.zeros((4, 16, 16), dtype=np.float32),
    )


class SimplifiedRewardTest(unittest.TestCase):
    """Production path: use_simplified_reward=True (mặc định trong EnvConfig)."""

    def make_env(self) -> SliceEnv:
        return SliceEnv(_detection_cache(), _hard_region_cache(), env_cfg=EnvConfig())

    def test_new_hard_hit_gives_positive_reward(self) -> None:
        env = self.make_env()
        env.reset()
        first = env.step(Action.ZOOM_IN)
        self.assertEqual(first.info["new_hits"], 1)
        self.assertGreater(first.reward, 0.0)

    def test_retained_hit_without_stop_costs_efficiency(self) -> None:
        # Simplified KHÔNG thưởng lại hit cũ ở bước non-STOP: chỉ trừ efficiency (đúng thiết kế chống reward farming).
        env = self.make_env()
        env.reset()
        env.step(Action.ZOOM_IN)
        second = env.step(Action.ZOOM_IN)
        self.assertEqual(second.info["new_hits"], 0)
        self.assertEqual(second.info["retained_hits"], 1)
        self.assertLess(second.reward, 0.0)
        self.assertGreater(second.reward, -1.0)

    def test_stop_on_covered_target_is_rewarded(self) -> None:
        env = self.make_env()
        env.reset()
        env.step(Action.ZOOM_IN)
        stop = env.step(Action.STOP)
        self.assertTrue(stop.done)
        self.assertEqual(stop.info["new_hits"], 0)
        self.assertEqual(stop.info["retained_hits"], 1)
        self.assertGreater(stop.info["total_target_score"], 0.0)
        self.assertGreater(stop.reward, 0.0)

    def test_max_steps_without_stop_applies_terminal_penalty(self) -> None:
        # Delta-based: chênh reward giữa penalty=3 và penalty=0 đúng bằng terminal penalty,
        # bất kể target reward nền là bao nhiêu (robust hơn ngưỡng tuyệt đối).
        def run(penalty: float):
            env = SliceEnv(
                _detection_cache(),
                _hard_region_cache(),
                env_cfg=EnvConfig(max_steps=1, max_steps_without_stop_penalty=penalty),
            )
            env.reset()
            return env.step(Action.RIGHT)

        with_penalty = run(3.0)
        without_penalty = run(0.0)
        self.assertTrue(with_penalty.done)
        self.assertTrue(with_penalty.info["stop_due_to_max_steps"])
        self.assertAlmostEqual(without_penalty.reward - with_penalty.reward, 3.0, places=4)


class LegacyRewardTest(unittest.TestCase):
    """Legacy shaping path: use_simplified_reward=False (giữ làm đường lui + ablation)."""

    def make_env(self) -> SliceEnv:
        return SliceEnv(
            _detection_cache(),
            _hard_region_cache(),
            env_cfg=EnvConfig(use_simplified_reward=False),
        )

    def test_zoom_keeps_previously_covered_target_rewarded(self) -> None:
        env = self.make_env()
        env.reset()
        first = env.step(Action.ZOOM_IN)
        self.assertEqual(first.info["new_hits"], 1)

        second = env.step(Action.ZOOM_IN)
        self.assertEqual(second.info["new_hits"], 0)
        self.assertEqual(second.info["retained_hits"], 1)
        self.assertGreater(second.info["total_target_score"], 0.0)
        self.assertGreater(second.reward, 0.0)

    def test_zoom_out_after_compact_zoom_is_not_reward_farming(self) -> None:
        env = self.make_env()
        env.reset()
        env.step(Action.ZOOM_IN)
        zoom_in = env.step(Action.ZOOM_IN)
        zoom_out = env.step(Action.ZOOM_OUT)

        self.assertGreater(zoom_in.info["compactness_delta"], 0.0)
        self.assertLess(zoom_out.info["compactness_delta"], 0.0)
        self.assertLess(zoom_out.reward, zoom_in.reward)

    def test_target_class_filter_excludes_non_target_detection_penalty(self) -> None:
        cfg = EnvConfig(
            use_simplified_reward=False,
            step_penalty=0.0,
            area_penalty=0.0,
            detected_overlap_penalty=1.0,
        )
        env_all = SliceEnv(_high_conf_detection_cache(99), None, env_cfg=cfg)
        env_all.reset()
        all_result = env_all.step(Action.STOP)

        env_target = SliceEnv(_high_conf_detection_cache(99), None, env_cfg=cfg, target_classes=(0,))
        env_target.reset()
        target_result = env_target.step(Action.STOP)

        self.assertGreater(all_result.info["detected_overlap"], 0.0)
        self.assertEqual(target_result.info["detected_overlap"], 0.0)
        self.assertGreater(target_result.reward, all_result.reward)


class EnvMechanicsTest(unittest.TestCase):
    """Cơ chế env độc lập reward mode (terminal khi stalled, action mask)."""

    def test_stalled_without_stop_gets_terminal_penalty(self) -> None:
        env = SliceEnv(
            _detection_cache(),
            None,
            env_cfg=EnvConfig(
                initial_slice_fraction=0.35,
                min_slice_fraction=0.35,
                stalled_without_stop_penalty=2.5,
            ),
        )
        env.reset()

        result = env.step(Action.ZOOM_IN)

        self.assertTrue(result.done)
        self.assertTrue(result.info["stop_due_to_stalled_roi"])
        self.assertLess(result.reward, -2.5)

    def test_valid_actions_mask_stalled_zoom(self) -> None:
        env = SliceEnv(
            _detection_cache(),
            None,
            env_cfg=EnvConfig(initial_slice_fraction=0.35, min_slice_fraction=0.35),
        )
        env.reset()

        valid = env.valid_actions()

        self.assertFalse(valid[int(Action.ZOOM_IN)])
        self.assertTrue(valid[int(Action.STOP)])


if __name__ == "__main__":
    unittest.main()
