from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2
import numpy as np
import torch

from rl_sahi.common.boxes import area
from rl_sahi.common.cache import detection_cache_path, load_detection_cache
from rl_sahi.common.class_mapping import ClassMapping
from rl_sahi.common.config import load_default_config
from rl_sahi.common.data import read_image
from rl_sahi.detection.yolo import load_yolo
from rl_sahi.eval.benchmark import _filter_classes, _full_predictions, _read_gt
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


def main() -> None:
    ap = argparse.ArgumentParser(description="Ve quyet dinh CROP/SKIP cua yield-agent tren 1 anh.")
    ap.add_argument("image", type=Path, help="Ten file anh trong split.")
    ap.add_argument("--checkpoint", type=Path, default=ROOT / "runs" / "dqn_yield_full" / "best.pt")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "runs" / "report")
    ap.add_argument("--clean", action="store_true", help="Chi ve ROI THAT (CROP, do) + vat moi; BO het o skip. Anh production cho bao cao.")
    ap.add_argument("--productive-only", dest="productive_only", action="store_true", help="Chi ve ROI BAT DUOC >=1 vat moi (bo ROI +0). Implies --clean. Anh 'dream': ROI chi o vung co vat bo lo.")
    args = ap.parse_args()
    if args.productive_only:
        args.clean = True

    cfg = load_default_config(None, ROOT)
    inf = cfg.section("infer"); dev = cfg.optional_str("infer", "device")
    cm = ClassMapping.from_config(cfg.section("classes")); sc = cfg.dataclass_instance("state", StateConfig)
    tc = tuple(int(x) for x in inf.get("target_classes", [0, 2, 3, 5, 8, 9]))
    icfg = InferenceConfig(full_imgsz=int(inf["full_imgsz"]), slice_imgsz=int(inf["slice_imgsz"]), full_conf=float(inf["full_conf"]), output_conf=float(inf["output_conf"]), iou=float(inf["iou"]), merge_iou=float(inf["merge_iou"]), max_det=int(inf["max_det"]), device=dev, feature_layers=cfg.feature_layers("infer"), min_slice_detections=1, max_slice_attempts=0, target_classes=tc, require_stop_for_acceptance=True, class_mapping=cm)

    # policy len CPU TRUOC khi YOLO dung CUDA
    policy, ck = load_policy(args.checkpoint, torch.device("cpu"))
    policy = policy.to("cpu")
    env_cfg = ck["env_cfg_obj"]
    model = load_yolo(cfg.path_value("weights"), device=dev)

    ir = cfg.path_value("image_root"); crt = cfg.path_value("cache_root"); lr = cfg.path_value("label_root")
    ip = args.image if args.image.is_absolute() else ir / args.split / args.image.name
    det = load_detection_cache(detection_cache_path(crt, args.split, ip))
    gt, _ = _read_gt(ip, ir, lr, tc, cm); h, w = det.image_shape
    small_gt = gt[(area(gt) / max(h * w, 1)) <= 0.0022] if len(gt) else gt

    env = YieldAwareHotspotEnv(det, None, raw_yields=np.zeros(64, np.float32), real_yields=None, env_cfg=env_cfg, state_cfg=sc)
    full_boxes, _, _ = _full_predictions(det, icfg)
    decisions = []
    state = env.reset()
    for _ in range(len(env.rois) + 1):
        with torch.no_grad():
            action = int(policy(torch.from_numpy(state).float().unsqueeze(0)).argmax(dim=1).item())
        i = env.i
        if i < len(env.rois):
            roi = env.rois[i]
            if action == 0:
                bi, si, ci = run_yolo_on_crop(model, ip, roi, imgsz=icfg.slice_imgsz, conf=icfg.output_conf, iou=icfg.iou, max_det=icfg.max_det, device=dev)
                ci = cm.map_model_classes(ci); bi, si, ci = _filter_classes(bi, si, ci, tc)
                new = _iou(bi, full_boxes).max(1) < 0.5 if len(bi) and len(full_boxes) else np.ones(len(bi), bool)
                env.raw_yields[i] = int(new.sum()) if len(bi) else 0
                real = (_iou(bi, small_gt).max(1) >= 0.5) if len(small_gt) and len(bi) else np.zeros(len(bi), bool)
                nb = bi[new & real] if len(bi) else np.zeros((0, 4))
                decisions.append((roi, "crop", int((new & real).sum()), nb))
            else:
                decisions.append((roi, "skip", 0, None))
        result = env.step(action)
        state = result.state
        if result.done:
            break

    img = read_image(ip)
    ctx_color = (0, 200, 0)  # YOLO da thay = XANH LA (de nhin)
    for b in full_boxes:
        cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), ctx_color, 1)
    nc = ns = 0
    for roi, act, yld, nb in decisions:
        if act == "crop":
            if args.productive_only and yld == 0:
                continue  # bo ROI khong bat duoc vat moi
            nc += 1
            if nb is not None:
                for b in nb:
                    cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 255), 2)  # vang = vat MOI that bat duoc
            # ROI THAT = do dam (giac mo "roi do")
            cv2.rectangle(img, (int(roi[0]), int(roi[1])), (int(roi[2]), int(roi[3])), (0, 0, 255), 3)
            cv2.putText(img, f"ROI +{yld}", (int(roi[0]) + 5, int(roi[1]) + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        else:
            ns += 1
            if not args.clean:  # che do clean: KHONG ve skip (skip khong phai output)
                cv2.rectangle(img, (int(roi[0]), int(roi[1])), (int(roi[2]), int(roi[3])), (80, 80, 255), 1)
                cv2.putText(img, "skip", (int(roi[0]) + 5, int(roi[1]) + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 255), 1)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_yield_clean" if args.clean else "_yield"
    out = args.out_dir / f"{ip.stem}{suffix}.jpg"
    cv2.imwrite(str(out), img)
    print(f"[viz] {ip.stem}: {nc} ROI (CROP) / {ns} skip | YOLO goc {len(full_boxes)} box -> {out}")
    if args.clean:
        print("[viz] CLEAN: Do dam = ROI THAT cua RL-SAHI (chi o vung bo lo) | Vang = vat moi bat duoc | Xam mo = YOLO da thay")
    else:
        print("[viz] Do dam = ROI (CROP) | Do nhat = candidate bi SKIP (KHONG phai output) | Vang = vat moi")


if __name__ == "__main__":
    main()
