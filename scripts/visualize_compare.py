from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2
import numpy as np

from rl_sahi.common.class_mapping import ClassMapping
from rl_sahi.common.config import load_default_config
from rl_sahi.common.data import read_image
from rl_sahi.detection.yolo import load_yolo
from rl_sahi.eval.benchmark import BenchmarkConfig, _full_predictions, _predict_density_guided
from rl_sahi.inference.config import InferenceConfig
from rl_sahi.inference.pipeline import get_initial_detection
from rl_sahi.rl.state_config import StateConfig


def _int_tuple(value) -> tuple[int, ...]:
    if isinstance(value, str):
        return tuple(int(x.strip()) for x in value.split(",") if x.strip())
    return tuple(int(x) for x in value)


def _draw(image: np.ndarray, boxes: np.ndarray, color: tuple[int, int, int] = (0, 210, 0), thickness: int = 2) -> np.ndarray:
    for box in boxes:
        cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, thickness)
    return image


def main() -> None:
    parser = argparse.ArgumentParser(description="Xuat 2 anh: YOLO goc vs RL-SAHI (density-guided) cho 1 anh.")
    parser.add_argument("image", type=Path, help="Duong dan anh (hoac chi ten file trong split).")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--k", type=int, default=8, help="So crop density-guided (mac dinh 8).")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "runs" / "report")
    args = parser.parse_args()

    cfg = load_default_config(args.config, ROOT)
    inf = cfg.section("infer")
    device = cfg.optional_str("infer", "device")
    cm = ClassMapping.from_config(cfg.section("classes"))
    sc = cfg.dataclass_instance("state", StateConfig)
    tc = _int_tuple(inf.get("target_classes", [0, 2, 3, 5, 8, 9]))

    image_path = args.image if args.image.is_absolute() else cfg.path_value("image_root") / args.split / args.image.name
    if not image_path.exists():
        raise FileNotFoundError(image_path)

    icfg = InferenceConfig(
        full_imgsz=int(inf["full_imgsz"]), slice_imgsz=int(inf["slice_imgsz"]), full_conf=float(inf["full_conf"]),
        output_conf=float(inf["output_conf"]), iou=float(inf["iou"]), merge_iou=float(inf["merge_iou"]),
        max_det=int(inf["max_det"]), device=device, feature_layers=cfg.feature_layers("infer"),
        min_slice_detections=1, max_slice_attempts=0, target_classes=tc,
        require_stop_for_acceptance=True, class_mapping=cm,
    )
    bcfg = BenchmarkConfig(
        iou_threshold=float(cfg.section("benchmark").get("iou_threshold", 0.5)),
        fixed_slice_fraction=float(cfg.section("benchmark").get("fixed_slice_fraction", 0.35)),
        fixed_overlap=float(cfg.section("benchmark").get("fixed_overlap", 0.2)),
        small_area_percentile=float(cfg.section("benchmark").get("small_area_percentile", 40.0)),
        target_classes=tc, class_mapping=cm,
    )

    model = load_yolo(cfg.path_value("weights"), device=device)
    det = get_initial_detection(
        model=model, weights=cfg.path_value("weights"), image_path=image_path, weights_imgsz=icfg.full_imgsz,
        full_conf=icfg.full_conf, full_iou=icfg.iou, max_det=icfg.max_det, device=device,
        feature_layers=icfg.feature_layers, aux_grid_size=sc.grid_size,
        spatial_feature_channels=sc.spatial_feature_channels, cache_root=cfg.path_value("cache_root"),
        split=args.split, use_cache=True,
    )

    full_boxes, _, _ = _full_predictions(det, icfg)
    rl_boxes, _, _, k = _predict_density_guided(model, image_path, det, icfg, bcfg, sc, args.k)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem
    yolo_path = args.out_dir / f"{stem}_yolo.jpg"
    rlsahi_path = args.out_dir / f"{stem}_rlsahi.jpg"
    cv2.imwrite(str(yolo_path), _draw(read_image(image_path), full_boxes))
    cv2.imwrite(str(rlsahi_path), _draw(read_image(image_path), rl_boxes))

    print(f"[viz] YOLO goc : {len(full_boxes):3d} box -> {yolo_path}")
    print(f"[viz] RL-SAHI  : {len(rl_boxes):3d} box ({k} crop) -> {rlsahi_path}")
    print(f"[viz] RL-SAHI bat them {len(rl_boxes) - len(full_boxes)} box so voi YOLO.")


if __name__ == "__main__":
    main()
