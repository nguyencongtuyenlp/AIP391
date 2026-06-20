from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from rl_sahi.common.boxes import area
from rl_sahi.common.cache import detection_cache_path, load_detection_cache
from rl_sahi.common.class_mapping import ClassMapping
from rl_sahi.common.config import load_default_config
from rl_sahi.common.data import iter_images
from rl_sahi.common.device import resolve_torch_device
from rl_sahi.eval.benchmark import (
    BenchmarkConfig, _density_guided_rois, _filter_classes, _full_predictions, _merge_predictions, _read_gt,
)
from rl_sahi.inference.config import InferenceConfig
from rl_sahi.inference.crops import run_yolo_on_crop
from rl_sahi.detection.yolo import load_yolo
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


def _run_crops(model, ip, rois, icfg, cm, tc, full_boxes):
    """Chay YOLO tren danh sach ROI, tra (parts, raw_yields)."""
    parts = []
    yields = []
    for roi in rois:
        bi, si, ci = run_yolo_on_crop(model, ip, roi, imgsz=icfg.slice_imgsz, conf=icfg.output_conf, iou=icfg.iou, max_det=icfg.max_det, device=icfg.device)
        ci = cm.map_model_classes(ci); bi, si, ci = _filter_classes(bi, si, ci, tc)
        ny = 0
        if len(bi):
            new = _iou(bi, full_boxes).max(1) < 0.5 if len(full_boxes) else np.ones(len(bi), bool)
            ny = int(new.sum())
        parts.append((bi, si, ci)); yields.append(ny)
    return parts, yields


def yield_agent_predict(model, policy, device, ip, det, icfg, env_cfg, state_cfg, tc, cm):
    env = YieldAwareHotspotEnv(det, None, raw_yields=np.zeros(64, np.float32), real_yields=None, env_cfg=env_cfg, state_cfg=state_cfg)
    full_boxes, full_scores, full_classes = _full_predictions(det, icfg)
    bp = [full_boxes]; sp = [full_scores]; cp = [full_classes]
    state = env.reset()
    for _ in range(len(env.rois) + 1):
        with torch.no_grad():
            q = policy(torch.from_numpy(state).float().unsqueeze(0).to(device))
            action = int(q.argmax(dim=1).item())
        if action == 0 and env.i < len(env.rois):
            roi = env.rois[env.i]
            bi, si, ci = run_yolo_on_crop(model, ip, roi, imgsz=icfg.slice_imgsz, conf=icfg.output_conf, iou=icfg.iou, max_det=icfg.max_det, device=icfg.device)
            ci = cm.map_model_classes(ci); bi, si, ci = _filter_classes(bi, si, ci, tc)
            ny = int((_iou(bi, full_boxes).max(1) < 0.5).sum()) if len(bi) and len(full_boxes) else len(bi)
            env.raw_yields[env.i] = ny
            bp.append(bi); sp.append(si); cp.append(ci)
        result = env.step(action)
        state = result.state
        if result.done:
            break
    boxes, scores, classes = _merge_predictions(det.image_shape, icfg.merge_iou, bp, sp, cp)
    return boxes, len(env.placed)


def _recall(boxes, small_gt):
    if len(small_gt) == 0:
        return 0, 0
    if len(boxes) == 0:
        return 0, len(small_gt)
    hit = int((_iou(small_gt, boxes).max(1) >= 0.5).sum())
    return hit, len(small_gt)


def main() -> None:
    ap = argparse.ArgumentParser(description="Eval yield-aware agent vs density-guided (cung ngan sach o, full-set).")
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cfg = load_default_config(None, ROOT)
    inf = cfg.section("infer"); dev = cfg.optional_str("infer", "device")
    cm = ClassMapping.from_config(cfg.section("classes")); sc = cfg.dataclass_instance("state", StateConfig)
    tc = tuple(int(x) for x in inf.get("target_classes", [0, 2, 3, 5, 8, 9]))
    icfg = InferenceConfig(full_imgsz=int(inf["full_imgsz"]), slice_imgsz=int(inf["slice_imgsz"]), full_conf=float(inf["full_conf"]), output_conf=float(inf["output_conf"]), iou=float(inf["iou"]), merge_iou=float(inf["merge_iou"]), max_det=int(inf["max_det"]), device=dev, feature_layers=cfg.feature_layers("infer"), min_slice_detections=1, max_slice_attempts=0, target_classes=tc, require_stop_for_acceptance=True, class_mapping=cm)
    bcfg = BenchmarkConfig(iou_threshold=0.5, fixed_slice_fraction=0.35, fixed_overlap=0.2, small_area_percentile=40.0, target_classes=tc, class_mapping=cm)
    # Load policy len CPU TRUOC khi YOLO/ultralytics dung den CUDA (no remap CUDA_VISIBLE_DEVICES).
    policy, ck = load_policy(args.checkpoint, torch.device("cpu"))
    policy = policy.to("cpu")
    dt = torch.device("cpu")
    env_cfg = ck["env_cfg_obj"]
    model = load_yolo(cfg.path_value("weights"), device=dev)
    ir = cfg.path_value("image_root"); crt = cfg.path_value("cache_root"); lr = cfg.path_value("label_root")

    yh = yt = yc = dh = dtot = dc = 0
    n = 0
    for ip in iter_images(ir, split=args.split, limit=args.limit):
        dp = detection_cache_path(crt, args.split, ip)
        if not dp.exists():
            continue
        det = load_detection_cache(dp); gt, _ = _read_gt(ip, ir, lr, tc, cm); h, w = det.image_shape
        small_gt = gt[(area(gt) / max(h * w, 1)) <= 0.0022] if len(gt) else gt
        yb, ncrop = yield_agent_predict(model, policy, dt, ip, det, icfg, env_cfg, sc, tc, cm)
        hh, tt = _recall(yb, small_gt); yh += hh; yt += tt; yc += ncrop
        # density-guided CUNG ngan sach o
        drois = _density_guided_rois(det, sc, bcfg, max(ncrop, 0))
        fbx, fsx, fcx = _full_predictions(det, icfg)
        bp = [fbx]; sp = [fsx]; cp = [fcx]
        parts, _ = _run_crops(model, ip, drois, icfg, cm, tc, fbx)
        for b, s, c in parts:
            bp.append(b); sp.append(s); cp.append(c)
        db, dsc, dcl = _merge_predictions(det.image_shape, icfg.merge_iou, bp, sp, cp)
        hh, tt = _recall(db, small_gt); dh += hh; dtot += tt; dc += len(drois)
        n += 1
        if n % 100 == 0:
            print(f"  ...{n} anh", flush=True)
    print(f"\n=== {args.split} ({n} anh) — yield-agent vs density-guided CUNG ngan sach o ===")
    print(f"  YIELD-AGENT : small_recall={yh/max(yt,1):.4f}  crops/anh={yc/max(n,1):.2f}")
    print(f"  density-k=N : small_recall={dh/max(dtot,1):.4f}  crops/anh={dc/max(n,1):.2f}")
    print(f"  -> chenh recall = {(yh/max(yt,1))-(dh/max(dtot,1)):+.4f} (cung ~{yc/max(n,1):.1f} o)")


if __name__ == "__main__":
    main()
