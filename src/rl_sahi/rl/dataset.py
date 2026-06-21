from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rl_sahi.common.cache import (
    DetectionCache,
    HardRegionCache,
    detection_cache_is_current,
    detection_cache_path,
    hard_region_cache_path,
    load_detection_cache,
    load_hard_region_cache,
)
from rl_sahi.common.data import iter_images


@dataclass(slots=True)
class CachedSample:
    image_path: Path
    detection_path: Path
    hard_region_path: Path
    detection: DetectionCache | None = None
    hard_region: HardRegionCache | None = None


class CachedEpisodeDataset:
    def __init__(
        self,
        image_root: Path,
        cache_root: Path,
        split: str,
        limit: int | None = None,
        preload: bool = False,
        detection_metadata: dict[str, Any] | None = None,
        require_hard_region: bool = True,
    ) -> None:
        self.image_root = Path(image_root)
        self.cache_root = Path(cache_root)
        self.split = split
        self.samples = []
        for image_path in iter_images(self.image_root, split=split, limit=limit):
            det_path = detection_cache_path(self.cache_root, split, image_path)
            hard_path = hard_region_cache_path(self.cache_root, split, image_path)
            has_hard = hard_path.exists()
            # GT-free reward khong can hard-region (vung-kho tu GT) -> chi can detection cache
            if detection_cache_is_current(det_path, detection_metadata) and (has_hard or not require_hard_region):
                detection = load_detection_cache(det_path) if preload else None
                hard_region = load_hard_region_cache(hard_path) if (preload and has_hard) else None
                self.samples.append(CachedSample(image_path, det_path, hard_path, detection, hard_region))
        if not self.samples:
            extra = " and scripts/hard_region.py" if require_hard_region else ""
            raise FileNotFoundError(
                f"No current {'detection/hard-region' if require_hard_region else 'detection'} caches found for split '{split}'. "
                f"Run scripts/detect.py{extra} first."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def random_episode(self):
        sample = random.choice(self.samples)
        det = sample.detection if sample.detection is not None else load_detection_cache(sample.detection_path)
        hard = sample.hard_region
        if hard is None and sample.hard_region_path.exists():
            hard = load_hard_region_cache(sample.hard_region_path)
        return det, hard  # hard = None khi GT-free (khong co hard-region cache)

    def first_detection(self):
        sample = self.samples[0]
        if sample.detection is not None:
            return sample.detection
        return load_detection_cache(sample.detection_path)
