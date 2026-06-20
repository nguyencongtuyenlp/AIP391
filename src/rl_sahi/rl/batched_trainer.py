from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from rl_sahi.common.actions import Action, NUM_ACTIONS, ACTION_NAMES
from rl_sahi.common.boxes import as_boxes
from rl_sahi.common.class_mapping import ClassMapping
from rl_sahi.common.data import iter_images
from rl_sahi.common.device import resolve_torch_device
from rl_sahi.detection.yolo import load_yolo
from rl_sahi.eval.benchmark import BenchmarkConfig, evaluate_rl_sahi_policy
from rl_sahi.inference.config import InferenceConfig
from rl_sahi.rl.checkpoint import save_checkpoint
from rl_sahi.rl.dataset import CachedEpisodeDataset
from rl_sahi.rl.env_config import EnvConfig
from rl_sahi.rl.network import QNetwork
from rl_sahi.rl.replay import PrioritizedReplayBuffer, ReplayBuffer
from rl_sahi.rl.slice_env import SliceEnv
from rl_sahi.rl.state_config import StateConfig
from rl_sahi.rl.state_layout import state_layout_from_detection
from rl_sahi.rl.trainer import TrainConfig, epsilon_by_step, guide_prob_by_step, soft_update, optimize, evaluate_policy, benchmark_score

@dataclass
class EnvWorker:
    episode: int
    det: Any
    hard: Any
    previous_rois: list[np.ndarray]
    previous_covered: np.ndarray
    current_max_slices: int
    slice_idx: int
    env: SliceEnv
    state: np.ndarray
    n_step_buffer: list
    total_reward: float
    total_steps: int
    accepted_slices: int
    losses: list[float]
    info: dict
    done: bool

def batched_train_dqn(
    image_root: Path, cache_root: Path, split: str, out_dir: Path, cfg: TrainConfig, env_cfg: EnvConfig, state_cfg: StateConfig,
    limit: int | None = None, device_name: str | None = None, detection_metadata: dict[str, Any] | None = None,
    target_classes: tuple[int, ...] = (), class_mapping: ClassMapping | None = None, label_root: Path | None = None,
    eval_weights: Path | None = None, infer_cfg: InferenceConfig | None = None, bench_cfg: BenchmarkConfig | None = None,
    eval_use_cache: bool = True,
) -> Path:
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    dataset = CachedEpisodeDataset(image_root=image_root, cache_root=cache_root, split=split, limit=limit, preload=cfg.preload_cache, detection_metadata=detection_metadata)
    
    val_dataset = None
    if getattr(cfg, "val_split", ""):
        try:
            val_dataset = CachedEpisodeDataset(image_root=image_root, cache_root=cache_root, split=cfg.val_split, limit=limit, preload=cfg.preload_cache, detection_metadata=detection_metadata)
        except FileNotFoundError as exc:
            print(f"[batched_train] validation disabled: {exc}")

    benchmark_model = None
    benchmark_images: list[Path] = []
    if getattr(cfg, "eval_benchmark_images", 0) > 0:
        if eval_weights is None or label_root is None or infer_cfg is None or bench_cfg is None:
            pass
        else:
            benchmark_images = iter_images(image_root, split=cfg.val_split, limit=cfg.eval_benchmark_images)
            if benchmark_images:
                benchmark_model = load_yolo(eval_weights, device=infer_cfg.device)

    probe_det = dataset.first_detection()
    probe_env = SliceEnv(probe_det, None, env_cfg=env_cfg, state_cfg=state_cfg, target_classes=target_classes, class_mapping=class_mapping)
    state_dim = int(probe_env.reset().shape[0])
    layout = state_layout_from_detection(probe_det, state_cfg)

    device = resolve_torch_device(device_name)
    policy = QNetwork(state_dim, hidden_dim=cfg.hidden_dim, layout=layout, use_spatial_cnn=cfg.use_spatial_cnn, dueling=cfg.dueling).to(device)
    target_net = QNetwork(state_dim, hidden_dim=cfg.hidden_dim, layout=layout, use_spatial_cnn=cfg.use_spatial_cnn, dueling=cfg.dueling).to(device)
    target_net.load_state_dict(policy.state_dict())
    
    optimizer = torch.optim.AdamW(policy.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.episodes, eta_min=1e-6)

    if cfg.use_per:
        replay: ReplayBuffer | PrioritizedReplayBuffer = PrioritizedReplayBuffer(capacity=cfg.replay_size, alpha=cfg.per_alpha, beta_start=cfg.per_beta_start, beta_frames=cfg.per_beta_frames)
    else:
        replay = ReplayBuffer(cfg.replay_size)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.csv"
    best_path = out_dir / "best.pt"
    last_path = out_dir / "last.pt"

    best_score = -float("inf")
    best_reward = -float("inf")
    global_step = 0
    num_envs = getattr(cfg, "num_envs", 8)
    episodes_started = 0
    episodes_completed = 0

    print(f"[batched_train] num_envs={num_envs}, episodes={cfg.episodes}")

    action_counts = np.zeros(NUM_ACTIONS, dtype=np.int64)
    stop_reason_counts = {"old_overlap": 0, "max_steps": 0, "stalled": 0, "low_new_hits": 0, "covered_all": 0, "policy_stop": 0}
    q_recent: list[float] = []
    (out_dir / "run_meta.json").write_text(
        json.dumps(
            {
                "seed": cfg.seed, "device": str(device), "num_envs": num_envs, "episodes": cfg.episodes,
                "split": split, "started_at": datetime.now().isoformat(timespec="seconds"),
                "train_cfg": asdict(cfg), "env_cfg": asdict(env_cfg), "state_cfg": asdict(state_cfg),
            },
            indent=2, default=str,
        ),
        encoding="utf-8",
    )

    def reset_worker(episode: int) -> EnvWorker:
        det, hard = dataset.random_episode()
        current_max_slices = env_cfg.max_slices
        if cfg.use_curriculum:
            curriculum_frac = min(float(global_step) / max(cfg.curriculum_steps, 1), 1.0)
            current_max_slices = max(1, int(env_cfg.max_slices * curriculum_frac))
        previous_covered = np.zeros((len(as_boxes(hard.hard_boxes)),), dtype=bool)
        env = SliceEnv(det, hard, env_cfg=env_cfg, state_cfg=state_cfg, previous_rois=np.zeros((0, 4), dtype=np.float32), previous_covered=previous_covered, target_classes=target_classes, class_mapping=class_mapping)
        return EnvWorker(
            episode=episode, det=det, hard=hard, previous_rois=[], previous_covered=previous_covered, current_max_slices=current_max_slices, slice_idx=0,
            env=env, state=env.reset(), n_step_buffer=[], total_reward=0.0, total_steps=0, accepted_slices=0, losses=[], info={}, done=False
        )

    active_workers = []
    for _ in range(num_envs):
        episodes_started += 1
        if episodes_started <= cfg.episodes:
            active_workers.append(reset_worker(episodes_started))

    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["episode", "reward", "loss", "mean_q", "epsilon", "steps", "slices", "covered", "hard_total", "val_recall", "val_slices", "val_score", "val_mAP50", "val_small_recall", "val_fp_per_image", "val_crops", "val_benchmark_score"]
        )
        writer.writeheader()

        while active_workers:
            states = [w.state for w in active_workers]
            epsilons = [epsilon_by_step(global_step, cfg)] * len(active_workers)
            guide_probs = [guide_prob_by_step(global_step, cfg)] * len(active_workers)
            
            actions = [Action.STOP] * len(active_workers)
            nn_indices = []
            
            for i, w in enumerate(active_workers):
                valid_actions = np.flatnonzero(w.env.valid_actions())
                if len(valid_actions) == 0:
                    valid_actions = np.asarray([int(Action.STOP)], dtype=np.int64)
                if random.random() < guide_probs[i]:
                    action = w.env.guided_action()
                    if int(action) in set(int(x) for x in valid_actions):
                        actions[i] = action
                        continue
                if random.random() < epsilons[i]:
                    actions[i] = Action(int(random.choice(valid_actions.tolist())))
                    continue
                nn_indices.append(i)
                
            if nn_indices:
                batch_states = np.stack([states[i] for i in nn_indices])
                with torch.no_grad():
                    x = torch.from_numpy(batch_states).float().to(device)
                    q = policy(x)
                    for j, i in enumerate(nn_indices):
                        valid = torch.from_numpy(active_workers[i].env.valid_actions()).bool().to(device)
                        q[j, ~valid] = -torch.inf
                        actions[i] = Action(int(q[j].argmax().item()))
                    q_recent.append(float(q.max(dim=1).values.mean().item()))
                    if len(q_recent) > 1000:
                        del q_recent[0]
                        
            for i in range(len(active_workers)):
                w = active_workers[i]
                action = actions[i]
                action_counts[int(action)] += 1
                result = w.env.step(action)
                
                w.n_step_buffer.append((w.state, action, result.reward, result.state, result.done))
                if len(w.n_step_buffer) >= getattr(cfg, "n_step", 1):
                    ret = 0.0
                    for k, (_, _, r, _, _) in enumerate(w.n_step_buffer):
                        ret += r * (cfg.gamma ** k)
                    s0, a0, _, _, _ = w.n_step_buffer[0]
                    _, _, _, sn, dn = w.n_step_buffer[-1]
                    replay.push(s0, a0, ret, sn, dn)
                    w.n_step_buffer.pop(0)
                    
                w.state = result.state
                w.total_reward += result.reward
                w.total_steps += 1
                w.info = result.info
                global_step += 1
                
                optimize_every = max(int(cfg.optimize_every), 1)
                if len(replay) >= cfg.min_replay and global_step % optimize_every == 0:
                    loss = optimize(policy, target_net, optimizer, replay, cfg.batch_size, cfg.gamma ** getattr(cfg, "n_step", 1), device, double_dqn=cfg.double_dqn, reward_clip=cfg.reward_clip)
                    if loss is not None:
                        w.losses.append(loss)
                        
                if cfg.use_soft_update:
                    soft_update(policy, target_net, cfg.tau)
                elif global_step % cfg.target_update == 0:
                    target_net.load_state_dict(policy.state_dict())
                    
                if result.done:
                    while len(w.n_step_buffer) > 0:
                        ret = 0.0
                        for k, (_, _, r, _, d) in enumerate(w.n_step_buffer):
                            ret += r * (cfg.gamma ** k)
                            if d: break
                        s0, a0, _, _, _ = w.n_step_buffer[0]
                        _, _, _, sn, dn = w.n_step_buffer[-1]
                        replay.push(s0, a0, ret, sn, dn)
                        w.n_step_buffer.pop(0)
                        
                    new_hits = int((w.env.covered & ~w.previous_covered).sum())
                    w.previous_rois.append(w.env.roi.copy())
                    w.previous_covered = w.env.covered.copy()
                    w.accepted_slices += 1
                    
                    stop_slice = False
                    if w.info.get("stop_due_to_old_overlap", False) or \
                       w.info.get("stop_due_to_max_steps", False) or \
                       w.info.get("stop_due_to_stalled_roi", False) or \
                       new_hits < env_cfg.min_new_hits_to_accept or \
                       (w.previous_covered.all() and len(w.previous_covered) > 0):
                        stop_slice = True
                        
                    w.slice_idx += 1
                    if w.slice_idx >= w.current_max_slices or stop_slice:
                        if w.info.get("stop_due_to_old_overlap"):
                            stop_reason_counts["old_overlap"] += 1
                        elif w.info.get("stop_due_to_max_steps"):
                            stop_reason_counts["max_steps"] += 1
                        elif w.info.get("stop_due_to_stalled_roi"):
                            stop_reason_counts["stalled"] += 1
                        elif new_hits < env_cfg.min_new_hits_to_accept:
                            stop_reason_counts["low_new_hits"] += 1
                        elif len(w.previous_covered) > 0 and w.previous_covered.all():
                            stop_reason_counts["covered_all"] += 1
                        else:
                            stop_reason_counts["policy_stop"] += 1
                        scheduler.step()
                        mean_loss = float(np.mean(w.losses)) if w.losses else 0.0
                        mean_q = round(float(np.mean(q_recent[-100:])), 6) if q_recent else 0.0
                        row = {
                            "episode": w.episode, "reward": round(w.total_reward, 6), "loss": round(mean_loss, 6),
                            "mean_q": mean_q,
                            "epsilon": round(epsilon_by_step(global_step, cfg), 6), "steps": w.total_steps,
                            "slices": w.accepted_slices, "covered": int(w.previous_covered.sum()),
                            "hard_total": int(len(w.previous_covered)), "val_recall": "", "val_slices": "",
                            "val_score": "", "val_mAP50": "", "val_small_recall": "", "val_fp_per_image": "",
                            "val_crops": "", "val_benchmark_score": ""
                        }
                        
                        selected_score = None
                        if w.episode == 1 or w.episode % max(int(cfg.eval_interval), 1) == 0:
                            if val_dataset is not None:
                                metrics = evaluate_policy(policy, val_dataset, env_cfg, state_cfg, cfg, device, target_classes=target_classes, class_mapping=class_mapping)
                                row["val_recall"] = round(metrics["val_recall"], 6)
                                row["val_slices"] = round(metrics["val_slices"], 6)
                                row["val_score"] = round(metrics["val_score"], 6)
                                selected_score = metrics["val_score"]
                            if benchmark_model is not None and infer_cfg is not None and bench_cfg is not None and label_root is not None:
                                bench_metrics = evaluate_rl_sahi_policy(
                                    model=benchmark_model, policy=policy, device_t=device, weights=eval_weights,
                                    images=benchmark_images, image_root=image_root, label_root=label_root,
                                    cache_root=cache_root, split=cfg.val_split, infer_cfg=infer_cfg, bench_cfg=bench_cfg,
                                    env_cfg=env_cfg, state_cfg=state_cfg, use_cache=eval_use_cache,
                                )
                                selected_score = benchmark_score(bench_metrics, cfg, env_cfg)
                                row["val_mAP50"] = round(bench_metrics["mAP50"], 6)
                                row["val_small_recall"] = round(bench_metrics["small_recall"], 6)
                                row["val_fp_per_image"] = round(bench_metrics["fp_per_image"], 6)
                                row["val_crops"] = round(bench_metrics["crops_per_image"], 6)
                                row["val_benchmark_score"] = round(selected_score, 6)

                        if selected_score is not None and selected_score > best_score:
                            best_score = selected_score
                            save_checkpoint(best_path, policy, state_dim, cfg, env_cfg, state_cfg, layout, detection_metadata=detection_metadata)
                        elif val_dataset is None and benchmark_model is None and w.total_reward > best_reward:
                            best_reward = w.total_reward
                            save_checkpoint(best_path, policy, state_dim, cfg, env_cfg, state_cfg, layout, detection_metadata=detection_metadata)
                            
                        writer.writerow(row)
                        f.flush()
                        
                        if w.episode % cfg.log_interval == 0 or w.episode == 1:
                            val_msg = ""
                            if row["val_score"] != "":
                                val_msg = f" val_recall={row['val_recall']} val_slices={row['val_slices']} val_score={row['val_score']}"
                            if row["val_benchmark_score"] != "":
                                val_msg += f" mAP50={row['val_mAP50']} small_recall={row['val_small_recall']} bench={row['val_benchmark_score']}"
                            print(f"[batched_train] ep={w.episode}/{cfg.episodes} reward={w.total_reward:.3f} loss={mean_loss:.4f} meanQ={mean_q:.3f} eps={epsilon_by_step(global_step, cfg):.3f} slices={w.accepted_slices} covered={row['covered']}/{row['hard_total']}{val_msg}")
                        
                        w.done = True
                        episodes_completed += 1
                    else:
                        prev_arr = np.stack(w.previous_rois).astype(np.float32)
                        w.env = SliceEnv(w.det, w.hard, env_cfg=env_cfg, state_cfg=state_cfg, previous_rois=prev_arr, previous_covered=w.previous_covered, target_classes=target_classes, class_mapping=class_mapping)
                        w.state = w.env.reset()
                        w.n_step_buffer.clear()
            
            next_active_workers = []
            for w in active_workers:
                if w.done:
                    if episodes_started < cfg.episodes:
                        episodes_started += 1
                        next_active_workers.append(reset_worker(episodes_started))
                else:
                    next_active_workers.append(w)
            active_workers = next_active_workers
            
    save_checkpoint(last_path, policy, state_dim, cfg, env_cfg, state_cfg, layout, detection_metadata=detection_metadata)
    (out_dir / "train_summary.json").write_text(
        json.dumps(
            {
                "episodes_completed": episodes_completed, "global_steps": global_step,
                "best_score": None if best_score == -float("inf") else round(best_score, 6),
                "best_reward": None if best_reward == -float("inf") else round(best_reward, 6),
                "action_distribution": {ACTION_NAMES[Action(i)]: int(action_counts[i]) for i in range(NUM_ACTIONS)},
                "stop_reasons": stop_reason_counts,
                "mean_q_last": round(float(np.mean(q_recent[-100:])), 6) if q_recent else None,
                "finished_at": datetime.now().isoformat(timespec="seconds"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[batched_train] summary -> {out_dir / 'train_summary.json'}")
    return best_path
