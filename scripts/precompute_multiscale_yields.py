from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from rl_sahi.common.boxes import area, centers
from rl_sahi.common.cache import detection_cache_path, load_detection_cache
from rl_sahi.common.class_mapping import ClassMapping
from rl_sahi.common.config import load_default_config
from rl_sahi.common.data import iter_images
from rl_sahi.detection.yolo import load_yolo
from rl_sahi.eval.benchmark import _filter_classes, _full_predictions, _read_gt
from rl_sahi.inference.config import InferenceConfig
from rl_sahi.inference.crops import run_yolo_on_crop
from rl_sahi.rl.state_config import StateConfig
from rl_sahi.rl.state_maps import build_detection_map

# Multi-scale yield cache (A = free-placement). Moi hotspot density top-K -> cat o NHIEU SCALE,
# cache yield tung scale -> agent hoc chon scale toi uu (GT-free state, real_yield chi vao reward).


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, 0:1], b[:, 0]); y1 = np.maximum(a[:, 1:2], b[:, 1])
    x2 = np.minimum(a[:, 2:3], b[:, 2]); y2 = np.minimum(a[:, 3:4], b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = ((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]))[:, None]
    ab = ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))[None, :]
    return inter / np.maximum(aa + ab - inter, 1e-6)


def _rank_cells(det, state_cfg: StateConfig, k_max: int, dedup_side: float) -> list[int]:
    dens = build_detection_map(det.boxes, det.scores, det.image_shape, state_cfg)[2]
    grid = int(state_cfg.grid_size)
    floor = (2.0 - 0.5) / max(float(state_cfg.count_norm), 1.0)
    h, w = det.image_shape
    order = np.argsort(dens.reshape(-1))[::-1]
    cells: list[int] = []
    used: list[tuple[float, float]] = []
    for flat_idx in order:
        if float(dens.reshape(-1)[flat_idx]) < floor or len(cells) >= k_max:
            break
        gy, gx = divmod(int(flat_idx), grid)
        cx = (gx + 0.5) * w / grid; cy = (gy + 0.5) * h / grid
        if any(abs(cx - ux) < dedup_side * 0.5 and abs(cy - uy) < dedup_side * 0.5 for ux, uy in used):
            continue
        cells.append(int(flat_idx)); used.append((cx, cy))
    return cells


def _roi_at(cell: int, grid: int, h: int, w: int, frac: float) -> np.ndarray:
    gy, gx = divmod(int(cell), grid)
    cx = (gx + 0.5) * w / grid; cy = (gy + 0.5) * h / grid
    side = max(1.0, min(h, w) * frac)
    x1 = float(np.clip(cx - side / 2.0, 0.0, max(w - side, 0.0)))
    y1 = float(np.clip(cy - side / 2.0, 0.0, max(h - side, 0.0)))
    return np.asarray([x1, y1, min(x1 + side, w), min(y1 + side, h)], dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-compute yield multi-scale moi hotspot cho free-placement agent (A).")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--k-max", type=int, default=16)
    parser.add_argument("--scales", type=str, default="0.25,0.35,0.45")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--small-area", type=float, default=0.0022)
    args = parser.parse_args()

    cfg = load_default_config(args.config, ROOT)
    inf = cfg.section("infer")
    device = cfg.optional_str("infer", "device")
    cm = ClassMapping.from_config(cfg.section("classes"))
    sc = cfg.dataclass_instance("state", StateConfig)
    tc = tuple(int(x) for x in inf.get("target_classes", [0, 2, 3, 5, 8, 9]))
    scales = [float(x) for x in args.scales.split(",") if x.strip()]
    icfg = InferenceConfig(
        full_imgsz=int(inf["full_imgsz"]), slice_imgsz=int(inf["slice_imgsz"]), full_conf=float(inf["full_conf"]),
        output_conf=float(inf["output_conf"]), iou=float(inf["iou"]), merge_iou=float(inf["merge_iou"]),
        max_det=int(inf["max_det"]), device=device, feature_layers=cfg.feature_layers("infer"),
        min_slice_detections=1, max_slice_attempts=0, target_classes=tc, require_stop_for_acceptance=True, class_mapping=cm,
    )
    dedup_side = min(0.35, max(scales)) * 1.0  # dedup theo scale lon nhat de o khong trung
    model = load_yolo(cfg.path_value("weights"), device=device)
    ir = cfg.path_value("image_root"); crt = cfg.path_value("cache_root"); lr = cfg.path_value("label_root")
    grid = int(sc.grid_size)
    out_root = crt / "multiscale_yields" / args.split
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"[multiscale] scales={scales} k_max={args.k_max} -> {out_root}", flush=True)

    n = 0
    for ip in iter_images(ir, split=args.split, limit=args.limit):
        dp = detection_cache_path(crt, args.split, ip)
        if not dp.exists():
            continue
        det = load_detection_cache(dp)
        h, w = det.image_shape
        side_dedup = min(h, w) * dedup_side
        cells = _rank_cells(det, sc, args.k_max, side_dedup)
        S = len(scales); K = len(cells)
        rois = np.zeros((K, S, 4), dtype=np.float32)
        raw_yield = np.zeros((K, S), dtype=np.int32)
        real_yield = np.zeros((K, S), dtype=np.int32)
        N = 0
        small_gt_caught = np.zeros((K, S, 0), dtype=bool)  # (cell, scale, small-GT) -> vat-bo-lo nao bat duoc (dedup reward)
        if K > 0:
            full_boxes, _, _ = _full_predictions(det, icfg)
            gt, _ = _read_gt(ip, ir, lr, tc, cm)
            small_gt = gt[(area(gt) / max(h * w, 1)) <= args.small_area] if len(gt) else gt
            N = len(small_gt)
            small_gt_caught = np.zeros((K, S, N), dtype=bool)
            for i, cell in enumerate(cells):
                for j, frac in enumerate(scales):
                    roi = _roi_at(cell, grid, h, w, frac)
                    rois[i, j] = roi
                    bi, si, ci = run_yolo_on_crop(model, ip, roi, imgsz=icfg.slice_imgsz, conf=icfg.output_conf, iou=icfg.iou, max_det=icfg.max_det, device=device)
                    ci = cm.map_model_classes(ci); bi, si, ci = _filter_classes(bi, si, ci, tc)
                    if len(bi) == 0:
                        continue
                    is_new = _iou(bi, full_boxes).max(1) < 0.5 if len(full_boxes) else np.ones(len(bi), bool)
                    raw_yield[i, j] = int(is_new.sum())
                    if N:
                        new_dets = bi[is_new]
                        caught = (_iou(small_gt, new_dets).max(1) >= 0.5) if len(new_dets) else np.zeros(N, bool)
                        small_gt_caught[i, j] = caught
                        real_yield[i, j] = int(caught.sum())
        np.savez(out_root / f"{ip.stem}.npz", rois=rois, cells=np.asarray(cells, dtype=np.int32), scales=np.asarray(scales, dtype=np.float32), raw_yield=raw_yield, real_yield=real_yield, small_gt_caught=small_gt_caught)
        n += 1
        if n % 100 == 0:
            print(f"[multiscale] {args.split}: {n} anh", flush=True)
    print(f"[multiscale] DONE {args.split}: {n} anh -> {out_root}", flush=True)


if __name__ == "__main__":
    main()
