from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2
import numpy as np
import torch

from rl_sahi.common.cache import detection_cache_path, load_detection_cache
from rl_sahi.common.class_mapping import ClassMapping
from rl_sahi.common.config import load_default_config
from rl_sahi.common.data import read_image
from rl_sahi.detection.yolo import load_yolo
from rl_sahi.eval.benchmark import (
    BenchmarkConfig, _filter_classes, _fixed_grid_rois, _full_predictions, _merge_predictions, _predict_fixed_sahi,
    _predict_rl_sahi,
)
from rl_sahi.inference.config import InferenceConfig
from rl_sahi.inference.crops import run_yolo_on_crop
from rl_sahi.rl.checkpoint import load_policy
from rl_sahi.rl.state_config import StateConfig
from rl_sahi.rl.yield_env import YieldAwareHotspotEnv


def _iou(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, 0:1], b[:, 0]); y1 = np.maximum(a[:, 1:2], b[:, 1])
    x2 = np.minimum(a[:, 2:3], b[:, 2]); y2 = np.minimum(a[:, 3:4], b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = ((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]))[:, None]
    ab = ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))[None, :]
    return inter / np.maximum(aa + ab - inter, 1e-6)


def _yield_rl_sahi(model, policy, ip, det, icfg, env_cfg, sc, tc, cm):
    env = YieldAwareHotspotEnv(det, None, raw_yields=np.zeros(64, np.float32), real_yields=None, env_cfg=env_cfg, state_cfg=sc)
    full_boxes, full_scores, full_classes = _full_predictions(det, icfg)
    bp = [full_boxes]; spp = [full_scores]; cpp = [full_classes]
    placed: list = []  # (roi, raw_yield)
    state = env.reset()
    for _ in range(len(env.rois) + 1):
        with torch.no_grad():
            action = int(policy(torch.from_numpy(state).float().unsqueeze(0)).argmax(dim=1).item())
        if action == 0 and env.i < len(env.rois):
            roi = env.rois[env.i]
            bi, si, ci = run_yolo_on_crop(model, ip, roi, imgsz=icfg.slice_imgsz, conf=icfg.output_conf, iou=icfg.iou, max_det=icfg.max_det, device=icfg.device)
            ci = cm.map_model_classes(ci); bi, si, ci = _filter_classes(bi, si, ci, tc)
            ny = int((_iou(bi, full_boxes).max(1) < 0.5).sum()) if len(bi) and len(full_boxes) else len(bi)
            env.raw_yields[env.i] = ny
            bp.append(bi); spp.append(si); cpp.append(ci)
            placed.append((roi.copy(), ny))
        result = env.step(action)
        state = result.state
        if result.done:
            break
    boxes, _, _ = _merge_predictions(det.image_shape, icfg.merge_iou, bp, spp, cpp)
    productive = [roi for roi, y in placed if y > 0]  # bo o +0 (thua)
    return boxes, productive, len(placed)


def _panel(image: np.ndarray, boxes: np.ndarray, caption: str, rois=None, roi_color=(0, 0, 255), roi_t: int = 3) -> np.ndarray:
    img = image.copy()
    for b in boxes:
        cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 220, 0), 2)
    if rois is not None:
        for r in rois:
            cv2.rectangle(img, (int(r[0]), int(r[1])), (int(r[2]), int(r[3])), roi_color, roi_t)
    h, w = img.shape[:2]
    strip_h = max(46, h // 14)
    strip = np.full((strip_h, w, 3), 28, np.uint8)
    scale = strip_h / 46.0
    cv2.putText(strip, caption, (14, int(strip_h * 0.7)), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), max(1, int(2 * scale)))
    return np.vstack([strip, img])


def main() -> None:
    ap = argparse.ArgumentParser(description="Xuat 3 anh ngang: YOLO | SAHI | RL-SAHI (cung 1 anh, co chu thich).")
    ap.add_argument("image", type=Path)
    ap.add_argument("--checkpoint", type=Path, default=ROOT / "runs" / "dqn_yield_c2" / "best.pt")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "runs" / "report")
    ap.add_argument("--gap", type=int, default=12, help="Khoang trang giua 3 panel (px).")
    ap.add_argument("--slice", dest="slice_agent", action="store_true", help="Panel 3 dung agent DI-CHUYEN (SliceEnv) thay yield. Checkpoint phai la slice agent.")
    args = ap.parse_args()

    cfg = load_default_config(None, ROOT)
    inf = cfg.section("infer"); dev = cfg.optional_str("infer", "device")
    cm = ClassMapping.from_config(cfg.section("classes")); sc = cfg.dataclass_instance("state", StateConfig)
    tc = tuple(int(x) for x in inf.get("target_classes", [0, 2, 3, 5, 8, 9]))
    icfg = InferenceConfig(full_imgsz=int(inf["full_imgsz"]), slice_imgsz=int(inf["slice_imgsz"]), full_conf=float(inf["full_conf"]), output_conf=float(inf["output_conf"]), iou=float(inf["iou"]), merge_iou=float(inf["merge_iou"]), max_det=int(inf["max_det"]), device=dev, feature_layers=cfg.feature_layers("infer"), min_slice_detections=1, max_slice_attempts=0, target_classes=tc, require_stop_for_acceptance=True, class_mapping=cm)
    bcfg = BenchmarkConfig(iou_threshold=0.5, fixed_slice_fraction=float(cfg.section("benchmark").get("fixed_slice_fraction", 0.35)), fixed_overlap=float(cfg.section("benchmark").get("fixed_overlap", 0.2)), small_area_percentile=40.0, target_classes=tc, class_mapping=cm)

    policy, ck = load_policy(args.checkpoint, torch.device("cpu"))
    policy = policy.to("cpu")
    env_cfg = ck["env_cfg_obj"]
    model = load_yolo(cfg.path_value("weights"), device=dev)

    ir = cfg.path_value("image_root"); crt = cfg.path_value("cache_root")
    ip = args.image if args.image.is_absolute() else ir / args.split / args.image.name
    det = load_detection_cache(detection_cache_path(crt, args.split, ip))
    image = read_image(ip)

    yb, _, _ = _full_predictions(det, icfg)
    sb, _, _, sn = _predict_fixed_sahi(model, ip, det, icfg, bcfg)
    sahi_rois = _fixed_grid_rois(det.image_shape, bcfg.fixed_slice_fraction, bcfg.fixed_overlap)

    if args.slice_agent:  # agent DI-CHUYEN (SliceEnv)
        rb, _, _, rn, rprod = _predict_rl_sahi(model, policy, torch.device("cpu"), ip, det, icfg, env_cfg, sc, return_rois=True)
        p3_caption = f"3. RL-SAHI (di-chuyen): {len(rb)} vat ({rn} o)"
    else:  # agent CHON-O (yield, mac dinh)
        rb, rprod, rn = _yield_rl_sahi(model, policy, ip, det, icfg, env_cfg, sc, tc, cm)
        p3_caption = f"3. RL-SAHI: {len(rb)} vat ({len(rprod)} o trung / {rn} quet)"

    p1 = _panel(image, yb, f"1. YOLO goc: {len(yb)} vat (1 o)")
    p2 = _panel(image, sb, f"2. SAHI (fixed grid): {len(sb)} vat ({sn} o)", rois=sahi_rois, roi_color=(0, 0, 255), roi_t=1)
    p3 = _panel(image, rb, p3_caption, rois=rprod, roi_color=(0, 0, 255), roi_t=3)
    gap = np.full((p1.shape[0], args.gap, 3), 255, np.uint8)
    combo = np.hstack([p1, gap, p2, gap, p3])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"{ip.stem}_compare3.jpg"
    cv2.imwrite(str(out), combo)
    print(f"[compare3] YOLO {len(yb)} | SAHI {len(sb)} ({sn} o) | RL-SAHI {len(rb)} ({rn} o) -> {out}")


if __name__ == "__main__":
    main()
