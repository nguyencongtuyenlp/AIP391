from __future__ import annotations

import unittest
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl_sahi.common.device import resolve_torch_device, device_description


class DeviceSmokeTest(unittest.TestCase):
    """Backend-agnostic: device phân giải đúng (CUDA / DirectML / CPU) và một op tensor chạy được."""

    def test_resolve_and_run_op(self) -> None:
        device = resolve_torch_device("")  # config device:'' -> auto
        x = torch.tensor([1.0, 2.0]).to(device)
        y = (x * 2).cpu()
        self.assertTrue(torch.allclose(y, torch.tensor([2.0, 4.0])))

    def test_device_description_nonempty(self) -> None:
        self.assertGreater(len(device_description("")), 0)


if __name__ == "__main__":
    unittest.main()
