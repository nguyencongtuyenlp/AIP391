from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from rl_sahi.common.boxes import area
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

# Adaptive-conf yield cache. Moi hotspot density: chay YOLO 1 LAN o conf THAP NHAT roi
# cat nguong post-hoc cho NHIEU muc conf -> moi (cell, conf): caught small-GT (dedup) + so FP moi.
# Agent hoc HA NGUONG dung vung -> cuu vat-bo-lo conf-thap ma khong no FP. KHONG fine-tune, KHONG SR.


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, 0:1], b[:, 0]); y1 = np.maximum(a[:, 1:2], b[:, 1])
    x2 = np.minimum(a[:, 2:3], b[:, 2]); y2 = np.minimum(a[:, 3:4], b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = ((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]))[:, None]
    ab = ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))[None, :]
    return inter / np.maximum(aa + ab - inter, 1e-6)


def _rank_cells(det, state_cfg, k_max, dedup_side):
    dens = build_detection_map(det.boxes, det.scores, det.image_shape, state_cfg)[2]
    grid = int(state_cfg.grid_size); floor = (2.0 - 0.5) / max(float(state_cfg.count_norm), 1.0)
    h, w = det.image_shape; order = np.argsort(dens.reshape(-1))[::-1]
    cells: list[int] = []; used: list[tuple[float, float]] = []
    for flat_idx in order:
        if float(dens.reshape(-1)[flat_idx]) < floor or len(cells) >= k_max:
            break
        gy, gx = divmod(int(flat_idx), grid); cx = (gx + 0.5) * w / grid; cy = (gy + 0.5) * h / grid
        if any(abs(cx - ux) < dedup_side * 0.5 and abs(cy - uy) < dedup_side * 0.5 for ux, uy in used):
            continue
        cells.append(int(flat_idx)); used.append((cx, cy))
    return cells


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-compute adaptive-conf yield: moi hotspot, caught+FP o nhieu muc conf.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--k-max", type=int, default=16)
    parser.add_argument("--confs", type=str, default="0.25,0.10,0.05")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--small-area", type=float, default=0.0022)
    args = parser.parse_args()

    cfg = load_default_config(args.config, ROOT)
    inf = cfg.section("infer"); device = cfg.optional_str("infer", "device")
    cm = ClassMapping.from_config(cfg.section("classes")); sc = cfg.dataclass_instance("state", StateConfig)
    tc = tuple(int(x) for x in inf.get("target_classes", [0, 2, 3, 5, 8, 9]))
    confs = sorted([float(x) for x in args.confs.split(",") if x.strip()], reverse=True)  # cao -> thap
    conf_min = min(confs)
    icfg = InferenceConfig(full_imgsz=int(inf["full_imgsz"]), slice_imgsz=int(inf["slice_imgsz"]), full_conf=float(inf["full_conf"]), output_conf=0.25, iou=float(inf["iou"]), merge_iou=float(inf["merge_iou"]), max_det=int(inf["max_det"]), device=device, feature_layers=cfg.feature_layers("infer"), min_slice_detections=1, max_slice_attempts=0, target_classes=tc, require_stop_for_acceptance=True, class_mapping=cm)
    frac = float(cfg.section("benchmark").get("fixed_slice_fraction", 0.35))
    model = load_yolo(cfg.path_value("weights"), device=device)
    ir = cfg.path_value("image_root"); crt = cfg.path_value("cache_root"); lr = cfg.path_value("label_root")
    grid = int(sc.grid_size); C = len(confs)
    out_root = crt / "adaptiveconf_yields" / args.split
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"[adaptiveconf] confs={confs} k_max={args.k_max} -> {out_root}", flush=True)

    n = 0
    for ip in iter_images(ir, split=args.split, limit=args.limit):
        dp = detection_cache_path(crt, args.split, ip)
        if not dp.exists():
            continue
        det = load_detection_cache(dp); h, w = det.image_shape
        cells = _rank_cells(det, sc, args.k_max, min(h, w) * frac)
        K = len(cells)
        rois = np.zeros((K, 4), dtype=np.float32)
        real_yield = np.zeros((K, C), dtype=np.int32)
        raw_yield = np.zeros((K, C), dtype=np.int32)  # GT-free: tong box-moi (real+nhieu) -> observed signal infer-valid
        fp = np.zeros((K, C), dtype=np.int32)
        N = 0
        caught = np.zeros((K, C, 0), dtype=bool)
        if K > 0:
            full_boxes, _, _ = _full_predictions(det, icfg)
            gt, _ = _read_gt(ip, ir, lr, tc, cm)
            small_gt = gt[(area(gt) / max(h * w, 1)) <= args.small_area] if len(gt) else gt
            N = len(small_gt)
            caught = np.zeros((K, C, N), dtype=bool)
            for i, cell in enumerate(cells):
                gy, gx = divmod(cell, grid); cx = (gx + 0.5) * w / grid; cy = (gy + 0.5) * h / grid
                side = max(1.0, min(h, w) * frac)
                x1 = float(np.clip(cx - side / 2.0, 0.0, max(w - side, 0.0))); y1 = float(np.clip(cy - side / 2.0, 0.0, max(h - side, 0.0)))
                roi = np.asarray([x1, y1, min(x1 + side, w), min(y1 + side, h)], dtype=np.float32)
                rois[i] = roi
                bi, si, ci = run_yolo_on_crop(model, ip, roi, imgsz=icfg.slice_imgsz, conf=conf_min, iou=icfg.iou, max_det=icfg.max_det, device=device)
                ci = cm.map_model_classes(ci); bi, si, ci = _filter_classes(bi, si, ci, tc)
                if len(bi) == 0:
                    continue
                is_new = _iou(bi, full_boxes).max(1) < 0.5 if len(full_boxes) else np.ones(len(bi), bool)
                for jc, conf in enumerate(confs):
                    keep = is_new & (si >= conf)
                    nb = bi[keep]
                    raw_yield[i, jc] = int(len(nb))
                    if len(nb) == 0:
                        continue
                    if N:
                        c_mask = _iou(small_gt, nb).max(1) >= 0.5
                        caught[i, jc] = c_mask
                        real_yield[i, jc] = int(c_mask.sum())
                    # FP = box moi khong khop GT nao
                    fp[i, jc] = int((_iou(nb, gt).max(1) < 0.5).sum()) if len(gt) else len(nb)
        np.savez(out_root / f"{ip.stem}.npz", rois=rois, cells=np.asarray(cells, dtype=np.int32), confs=np.asarray(confs, dtype=np.float32), real_yield=real_yield, raw_yield=raw_yield, fp=fp, small_gt_caught=caught)
        n += 1
        if n % 100 == 0:
            print(f"[adaptiveconf] {args.split}: {n} anh", flush=True)
    print(f"[adaptiveconf] DONE {args.split}: {n} anh -> {out_root}", flush=True)


if __name__ == "__main__":
    main()
