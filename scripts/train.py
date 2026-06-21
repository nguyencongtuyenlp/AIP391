from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl_sahi.common.cache import detection_cache_metadata
from rl_sahi.common.class_mapping import ClassMapping
from rl_sahi.common.config import load_default_config
from rl_sahi.common.device import print_device_info
from rl_sahi.eval.benchmark import BenchmarkConfig
from rl_sahi.inference.config import InferenceConfig
from rl_sahi.rl.env_config import EnvConfig
from rl_sahi.rl.state_config import StateConfig
from rl_sahi.rl.trainer import TrainConfig, train_dqn
from rl_sahi.rl.batched_trainer import batched_train_dqn
from rl_sahi.rl.hotspot_trainer import train_hotspot_dqn
from rl_sahi.rl.yield_trainer import train_yield_dqn
from rl_sahi.rl.multiscale_trainer import train_multiscale_dqn
from rl_sahi.rl.adaptiveconf_trainer import train_adaptiveconf_dqn

def _int_tuple(value) -> tuple[int, ...]:
    if isinstance(value, str):
        return tuple(int(x.strip()) for x in value.split(",") if x.strip())
    return tuple(int(x) for x in value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DQN to choose one adaptive slice from cached YOLO state.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate (adaptive-conf can lr=1e-4 qua thap -> Q ko fit -> collapse; dung 5e-4).")
    parser.add_argument("--overfit", action="store_true", help="Overfit 1 ảnh (sanity Karpathy): limit=1, tắt val+benchmark eval.")
    parser.add_argument("--trust-cache", action="store_true", help="Bỏ qua kiểm tra metadata cache (dùng cache có sẵn dù file weights đã mất/đổi).")
    parser.add_argument("--eval-benchmark-images", type=int, default=None, help="Số ảnh val cho benchmark eval định kỳ (0 = tắt).")
    parser.add_argument("--eval-interval", type=int, default=None, help="Eval mỗi N episodes.")
    parser.add_argument("--out-dir", type=Path, default=None, help="Thư mục lưu checkpoint/log (mặc định runs/dqn). Dùng để tách thí nghiệm.")
    parser.add_argument("--hotspot", action="store_true", help="Train HotspotStopAgent (RL optimal-stopping trên density hotspot, GT-free state).")
    parser.add_argument("--crop-cost", type=float, default=None, help="crop_cost cho hotspot agent (quét điểm Pareto: 0.1/0.15/0.2/0.3).")
    parser.add_argument("--w-cov", type=float, default=None, help="Trọng số coverage cho hotspot agent (cao = cắt bạo hơn).")
    parser.add_argument("--yield-aware", dest="yield_aware", action="store_true", help="Train YieldAwareHotspotEnv (Cửa 2: action CROP/SKIP, state có yield quan sát; cần pre-compute yield cache).")
    parser.add_argument("--residual", action="store_true", help="Xếp hạng hotspot bằng RESIDUAL density (loại vùng đã-detect) -> ROI chỉ nhắm vùng bỏ lỡ. Cần yield cache build với --residual.")
    parser.add_argument("--multiscale", action="store_true", help="Train MultiScaleYieldEnv (A=free-placement: agent chọn SKIP/CROP@scale). Cần multiscale cache.")
    parser.add_argument("--adaptive-conf", dest="adaptive_conf", action="store_true", help="Train AdaptiveConfEnv (lever conf: agent chọn SKIP/CROP@conf cao-vừa-thấp để cứu vật conf-thấp). Cần adaptiveconf cache.")
    parser.add_argument("--fp-weight", type=float, default=None, help="Trọng số phạt FP cho adaptive-conf (cao = giữ conf cao, ít FP; thấp = bạo hơn).")
    parser.add_argument("--fp-dedup", dest="fp_dedup", action="store_true", help="Phạt FP dedup qua grid (khớp FP per-image, sửa thổi-phồng x1.58). Cần cache có fp_grid.")
    parser.add_argument("--gtfree-reward", dest="gtfree_reward", action="store_true", help="SliceEnv (agent di-chuyển): reward GT-free (density+objectness) thay hard_boxes(GT) -> bỏ train/infer gap.")
    parser.add_argument("--boundary-fix", dest="boundary_fix", action="store_true", help="SliceEnv: ROI kẹp biên thì không kết thúc + phạt oan (chống xoay chong chóng).")
    args = parser.parse_args()

    cfg = load_default_config(args.config, ROOT)
    train_cfg = cfg.dataclass_instance("train", TrainConfig)
    env_cfg = cfg.dataclass_instance("env", EnvConfig)
    state_cfg = cfg.dataclass_instance("state", StateConfig)
    detect_cfg = cfg.section("detect")
    hard_cfg = cfg.section("hard_region")
    infer_cfg = cfg.section("infer")
    benchmark_cfg = cfg.section("benchmark")
    target_classes = _int_tuple(hard_cfg.get("target_classes", ()))
    class_mapping = ClassMapping.from_config(cfg.section("classes"))
    if args.episodes is not None:
        train_cfg.episodes = args.episodes
    if args.seed is not None:
        train_cfg.seed = args.seed
    if args.lr is not None:
        train_cfg.lr = args.lr
    if args.gtfree_reward:
        env_cfg.use_gtfree_reward = True
    if args.boundary_fix:
        env_cfg.use_boundary_fix = True
    if args.overfit:
        args.limit = args.limit or 1
        train_cfg.eval_benchmark_images = 0
        train_cfg.val_split = ""
        train_cfg.use_curriculum = False
        train_cfg.epsilon_decay_steps = 2000
        train_cfg.guide_decay_steps = 2000
        if args.episodes is None:
            train_cfg.episodes = 1500
        print(f"[train] OVERFIT: limit={args.limit} episodes={train_cfg.episodes} eps_decay=2000 (val+benchmark off, curriculum off)")
    if args.eval_benchmark_images is not None:
        train_cfg.eval_benchmark_images = args.eval_benchmark_images
    if args.eval_interval is not None:
        train_cfg.eval_interval = args.eval_interval
    device_name = args.device or cfg.optional_str("train", "device")
    print_device_info("train", device_name)

    detection_metadata = detection_cache_metadata(
        weights=cfg.path_value("weights"),
        imgsz=int(detect_cfg["imgsz"]),
        conf=float(detect_cfg["conf"]),
        iou=float(detect_cfg["iou"]),
        max_det=int(detect_cfg["max_det"]),
        feature_layers=cfg.feature_layers("detect"),
        aux_grid_size=int(state_cfg.grid_size),
        spatial_feature_channels=int(state_cfg.spatial_feature_channels),
    )
    if args.trust_cache:
        detection_metadata = None
        print("[train] trust-cache: bỏ qua kiểm tra metadata cache (dùng cache có sẵn).")

    if args.out_dir is None:
        out_dir = cfg.path_value("dqn_out_dir")
    else:
        out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    print(f"[train] out_dir: {out_dir}")

    if args.adaptive_conf:
        env_cfg.use_hotspot_env = True
        if args.crop_cost is not None:
            env_cfg.crop_cost = args.crop_cost
        if args.w_cov is not None:
            env_cfg.w_cov = args.w_cov
        if args.fp_weight is not None:
            env_cfg.fp_weight = args.fp_weight
        if args.fp_dedup:
            env_cfg.use_fp_dedup = True
        train_fn = train_adaptiveconf_dqn
        print(f"[train] ADAPTIVE-CONF agent (lever conf): w_cov={env_cfg.w_cov} crop_cost={env_cfg.crop_cost} fp_weight={env_cfg.fp_weight} fp_dedup={env_cfg.use_fp_dedup}")
    elif args.multiscale:
        env_cfg.use_hotspot_env = True
        if args.crop_cost is not None:
            env_cfg.crop_cost = args.crop_cost
        if args.w_cov is not None:
            env_cfg.w_cov = args.w_cov
        train_fn = train_multiscale_dqn
        print(f"[train] MULTI-SCALE agent (A=free-placement): w_cov={env_cfg.w_cov} crop_cost={env_cfg.crop_cost}")
    elif args.yield_aware:
        env_cfg.use_hotspot_env = True
        if args.crop_cost is not None:
            env_cfg.crop_cost = args.crop_cost
        if args.w_cov is not None:
            env_cfg.w_cov = args.w_cov
        if args.residual:
            env_cfg.use_residual_ranking = True
        train_fn = train_yield_dqn
        print(f"[train] YIELD-AWARE agent (Cua 2): w_cov={env_cfg.w_cov} crop_cost={env_cfg.crop_cost} residual={env_cfg.use_residual_ranking}")
    elif args.hotspot:
        env_cfg.use_hotspot_env = True
        if args.crop_cost is not None:
            env_cfg.crop_cost = args.crop_cost
        if args.w_cov is not None:
            env_cfg.w_cov = args.w_cov
        train_fn = train_hotspot_dqn
        print(f"[train] HOTSPOT agent: w_cov={env_cfg.w_cov} crop_cost={env_cfg.crop_cost} k_max={env_cfg.k_max}")
    else:
        train_fn = batched_train_dqn if train_cfg.num_envs > 1 else train_dqn
    checkpoint = train_fn(
        image_root=cfg.path_value("image_root"),
        cache_root=cfg.path_value("cache_root"),
        split=args.split,
        out_dir=out_dir,
        cfg=train_cfg,
        env_cfg=env_cfg,
        state_cfg=state_cfg,
        limit=args.limit,
        device_name=device_name,
        detection_metadata=detection_metadata,
        target_classes=target_classes,
        class_mapping=class_mapping,
        label_root=cfg.path_value("label_root"),
        eval_weights=cfg.path_value("weights"),
        infer_cfg=InferenceConfig(
            full_imgsz=int(infer_cfg["full_imgsz"]),
            slice_imgsz=int(infer_cfg["slice_imgsz"]),
            full_conf=float(infer_cfg["full_conf"]),
            output_conf=float(infer_cfg["output_conf"]),
            iou=float(infer_cfg["iou"]),
            merge_iou=float(infer_cfg["merge_iou"]),
            max_det=int(infer_cfg["max_det"]),
            device=device_name or cfg.optional_str("infer", "device"),
            feature_layers=cfg.feature_layers("infer"),
            min_slice_detections=int(infer_cfg.get("min_slice_detections", 1)),
            max_slice_attempts=int(infer_cfg.get("max_slice_attempts", 0)),
            target_classes=target_classes,
            require_stop_for_acceptance=bool(infer_cfg.get("require_stop_for_acceptance", True)),
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
        eval_use_cache=bool(infer_cfg.get("use_cache", True)),
    )
    print(f"[train] best checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()
