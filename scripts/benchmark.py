from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl_sahi.common.class_mapping import ClassMapping
from rl_sahi.common.config import load_default_config
from rl_sahi.common.device import print_device_info
from rl_sahi.eval.benchmark import BenchmarkConfig, benchmark_split
from rl_sahi.inference.config import InferenceConfig


def _int_tuple(value) -> tuple[int, ...]:
    if isinstance(value, str):
        return tuple(int(x.strip()) for x in value.split(",") if x.strip())
    return tuple(int(x) for x in value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark YOLO full, fixed-grid SAHI, and RL-SAHI.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--max-slice-attempts", type=int, default=None, help="Ablation: override số lần thử slice tối đa.")
    parser.add_argument("--min-slice-detections", type=int, default=None, help="Ablation: override ngưỡng detection mới để accept slice.")
    parser.add_argument("--no-require-stop", action="store_true", help="Ablation: accept slice kể cả khi agent không STOP sạch.")
    parser.add_argument("--density-k", type=str, default=None, help="Density-guided crops: danh sach K, vd '2,4,8'. Bo trong = tat.")
    parser.add_argument("--hotspot", action="store_true", help="Checkpoint la HotspotStopAgent (RL adaptive-K) -> method 'rl_hotspot'.")
    parser.add_argument("--yield-rl", dest="yield_rl", action="store_true", help="Checkpoint la Yield-aware agent -> method 'rl_yield' (chay cung pipeline, do FP+mAP).")
    parser.add_argument("--multiscale", action="store_true", help="Checkpoint la MultiScale agent (A) -> method 'rl_multiscale'.")
    parser.add_argument("--adaptive-conf", dest="adaptive_conf", action="store_true", help="Checkpoint la AdaptiveConf agent (lever conf) -> method 'rl_adaptiveconf'.")
    parser.add_argument("--random-k", type=str, default=None, help="Baseline doc lap: cat K hotspot ngau nhien, vd '4,8'. Pha tautology agent-subset-density.")
    parser.add_argument("--residual-density-k", type=str, default=None, help="RESIDUAL-density: tru vung YOLO da detect khoi density -> crop vung BO LO, vd '8,12'. method 'rdensity_kN'.")
    parser.add_argument("--seed", type=int, default=42, help="Seed cho random-K + reproducibility (chay nhieu seed de bao mean±std).")
    args = parser.parse_args()

    cfg = load_default_config(args.config, ROOT)
    infer_cfg = cfg.section("infer")
    device = cfg.optional_str("infer", "device")
    print_device_info("benchmark", device)
    benchmark_cfg = cfg.section("benchmark")
    target_classes = _int_tuple(infer_cfg.get("target_classes", [0, 2, 3, 5, 8, 9]))
    class_mapping = ClassMapping.from_config(cfg.section("classes"))
    checkpoint = cfg.path_value("checkpoint") if args.checkpoint is None else args.checkpoint
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    out_dir = args.out_dir if args.out_dir is not None else ROOT / "runs" / "benchmark" / args.split
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir

    max_attempts = args.max_slice_attempts if args.max_slice_attempts is not None else int(infer_cfg.get("max_slice_attempts", 0))
    min_slice_det = args.min_slice_detections if args.min_slice_detections is not None else int(infer_cfg.get("min_slice_detections", 1))
    require_stop = False if args.no_require_stop else bool(infer_cfg.get("require_stop_for_acceptance", True))
    print(f"[benchmark] gate: max_slice_attempts={max_attempts} min_slice_detections={min_slice_det} require_stop={require_stop}")

    rows = benchmark_split(
        weights=cfg.path_value("weights"),
        checkpoint=checkpoint,
        image_root=cfg.path_value("image_root"),
        label_root=cfg.path_value("label_root"),
        cache_root=cfg.path_value("cache_root"),
        split=args.split,
        infer_cfg=InferenceConfig(
            full_imgsz=int(infer_cfg["full_imgsz"]),
            slice_imgsz=int(infer_cfg["slice_imgsz"]),
            full_conf=float(infer_cfg["full_conf"]),
            output_conf=float(infer_cfg["output_conf"]),
            iou=float(infer_cfg["iou"]),
            merge_iou=float(infer_cfg["merge_iou"]),
            max_det=int(infer_cfg["max_det"]),
            device=device,
            feature_layers=cfg.feature_layers("infer"),
            min_slice_detections=min_slice_det,
            max_slice_attempts=max_attempts,
            target_classes=target_classes,
            require_stop_for_acceptance=require_stop,
            class_mapping=class_mapping,
        ),
        bench_cfg=BenchmarkConfig(
            iou_threshold=float(benchmark_cfg.get("iou_threshold", 0.5)),
            fixed_slice_fraction=float(benchmark_cfg.get("fixed_slice_fraction", 0.35)),
            fixed_overlap=float(benchmark_cfg.get("fixed_overlap", 0.2)),
            small_area_percentile=float(benchmark_cfg.get("small_area_percentile", 40.0)),
            target_classes=target_classes,
            class_mapping=class_mapping,
        ),
        out_dir=out_dir,
        limit=args.limit,
        use_cache=bool(infer_cfg.get("use_cache", True)) and not args.no_cache,
        density_k=tuple(int(x) for x in args.density_k.split(",") if x.strip()) if args.density_k else (),
        hotspot=args.hotspot,
        yield_rl=args.yield_rl,
        multiscale=args.multiscale,
        adaptive_conf=args.adaptive_conf,
        random_k=tuple(int(x) for x in args.random_k.split(",") if x.strip()) if args.random_k else (),
        residual_density_k=tuple(int(x) for x in args.residual_density_k.split(",") if x.strip()) if args.residual_density_k else (),
        seed=args.seed,
    )
    for row in rows:
        print(
            f"[benchmark] {row['method']}: mAP50={row['mAP50']:.4f} "
            f"small_recall={row['small_recall']:.4f} fp/image={row['fp_per_image']:.2f} "
            f"crops/image={row['crops_per_image']:.2f} latency={row['latency_ms_per_image']:.1f}ms"
        )
    print(f"[benchmark] wrote {out_dir / 'benchmark.csv'}")


if __name__ == "__main__":
    main()
