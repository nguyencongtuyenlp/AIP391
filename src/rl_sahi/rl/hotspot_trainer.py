from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rl_sahi.common.class_mapping import ClassMapping
from rl_sahi.common.device import resolve_torch_device
from rl_sahi.rl.checkpoint import save_checkpoint
from rl_sahi.rl.dataset import CachedEpisodeDataset
from rl_sahi.rl.env_config import EnvConfig
from rl_sahi.rl.hotspot_env import HotspotEnv, NUM_HOTSPOT_ACTIONS
from rl_sahi.rl.network import QNetwork
from rl_sahi.rl.replay import PrioritizedReplayBuffer, ReplayBuffer
from rl_sahi.rl.state_config import StateConfig
from rl_sahi.rl.trainer import TrainConfig, epsilon_by_step, optimize, select_action, soft_update


def _greedy_eval(policy, dataset, env_cfg, state_cfg, device, target_classes, class_mapping, episodes):
    cov_total = small_total = crops_total = 0
    n = min(int(episodes), len(dataset))
    for _ in range(n):
        det, hard = dataset.random_episode()
        env = HotspotEnv(det, hard, env_cfg=env_cfg, state_cfg=state_cfg, target_classes=target_classes, class_mapping=class_mapping)
        state = env.reset()
        for _ in range(env.k_max + 1):
            with torch.no_grad():
                q = policy(torch.from_numpy(state).float().unsqueeze(0).to(device))
                valid = torch.from_numpy(env.valid_actions()).bool().to(device)
                q[:, ~valid] = -torch.inf
                action = int(q.argmax(dim=1).item())
            result = env.step(action)
            state = result.state
            if result.done:
                break
        cov_total += int(env.covered.sum())
        small_total += len(env.small_gt_centers)
        crops_total += len(env.placed)
    recall = cov_total / max(small_total, 1)
    avg_crops = crops_total / max(n, 1)
    score = recall - float(env_cfg.crop_cost) * avg_crops / max(env_cfg.k_max, 1)
    return {"val_recall": recall, "val_crops": avg_crops, "val_score": score}


def train_hotspot_dqn(
    image_root: Path, cache_root: Path, split: str, out_dir: Path, cfg: TrainConfig, env_cfg: EnvConfig, state_cfg: StateConfig,
    limit: int | None = None, device_name: str | None = None, detection_metadata: dict[str, Any] | None = None,
    target_classes: tuple[int, ...] = (), class_mapping: ClassMapping | None = None, **_kw,
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
            print(f"[hotspot] validation disabled: {exc}")

    probe = HotspotEnv(dataset.first_detection(), None, env_cfg=env_cfg, state_cfg=state_cfg, target_classes=target_classes, class_mapping=class_mapping)
    state_dim = int(probe.reset().shape[0])
    device = resolve_torch_device(device_name)
    policy = QNetwork(state_dim, hidden_dim=cfg.hidden_dim, num_actions=NUM_HOTSPOT_ACTIONS, layout=None, use_spatial_cnn=False, dueling=cfg.dueling).to(device)
    target_net = QNetwork(state_dim, hidden_dim=cfg.hidden_dim, num_actions=NUM_HOTSPOT_ACTIONS, layout=None, use_spatial_cnn=False, dueling=cfg.dueling).to(device)
    target_net.load_state_dict(policy.state_dict())
    optimizer = torch.optim.AdamW(policy.parameters(), lr=cfg.lr)
    if cfg.use_per:
        replay: ReplayBuffer | PrioritizedReplayBuffer = PrioritizedReplayBuffer(capacity=cfg.replay_size, alpha=cfg.per_alpha, beta_start=cfg.per_beta_start, beta_frames=cfg.per_beta_frames)
    else:
        replay = ReplayBuffer(cfg.replay_size)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path, best_path, last_path = out_dir / "train_log.csv", out_dir / "best.pt", out_dir / "last.pt"
    best_score = -float("inf")
    global_step = 0
    print(f"[hotspot] state_dim={state_dim} num_actions={NUM_HOTSPOT_ACTIONS} crop_cost={env_cfg.crop_cost} k_max={env_cfg.k_max}")

    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["episode", "reward", "loss", "epsilon", "crops", "covered", "small_total", "val_recall", "val_crops", "val_score"])
        writer.writeheader()
        for episode in range(1, cfg.episodes + 1):
            det, hard = dataset.random_episode()
            env = HotspotEnv(det, hard, env_cfg=env_cfg, state_cfg=state_cfg, target_classes=target_classes, class_mapping=class_mapping)
            state = env.reset()
            total_reward = 0.0
            losses: list[float] = []
            n_step_buffer: list = []
            for _ in range(env.k_max + 1):
                epsilon = epsilon_by_step(global_step, cfg)
                action = select_action(policy, state, epsilon, 0.0, env, device)
                result = env.step(action)
                n_step_buffer.append((state, action, result.reward, result.state, result.done))
                if len(n_step_buffer) >= cfg.n_step:
                    ret = sum(r * (cfg.gamma ** i) for i, (_, _, r, _, _) in enumerate(n_step_buffer))
                    s0, a0, _, _, _ = n_step_buffer[0]
                    _, _, _, sn, dn = n_step_buffer[-1]
                    replay.push(s0, a0, ret, sn, dn)
                    n_step_buffer.pop(0)
                state = result.state
                total_reward += result.reward
                global_step += 1
                if len(replay) >= cfg.min_replay and global_step % max(int(cfg.optimize_every), 1) == 0:
                    loss = optimize(policy, target_net, optimizer, replay, cfg.batch_size, cfg.gamma ** cfg.n_step, device, double_dqn=cfg.double_dqn, reward_clip=cfg.reward_clip)
                    if loss is not None:
                        losses.append(loss)
                if cfg.use_soft_update:
                    soft_update(policy, target_net, cfg.tau)
                if result.done:
                    while n_step_buffer:
                        ret = 0.0
                        for i, (_, _, r, _, d) in enumerate(n_step_buffer):
                            ret += r * (cfg.gamma ** i)
                            if d:
                                break
                        s0, a0, _, _, _ = n_step_buffer[0]
                        _, _, _, sn, dn = n_step_buffer[-1]
                        replay.push(s0, a0, ret, sn, dn)
                        n_step_buffer.pop(0)
                    break

            mean_loss = float(np.mean(losses)) if losses else 0.0
            row = {"episode": episode, "reward": round(total_reward, 4), "loss": round(mean_loss, 4), "epsilon": round(epsilon_by_step(global_step, cfg), 4),
                   "crops": len(env.placed), "covered": int(env.covered.sum()), "small_total": len(env.small_gt_centers), "val_recall": "", "val_crops": "", "val_score": ""}
            if (episode == 1 or episode % max(int(cfg.eval_interval), 1) == 0) and val_dataset is not None:
                m = _greedy_eval(policy, val_dataset, env_cfg, state_cfg, device, target_classes, class_mapping, cfg.eval_episodes)
                row["val_recall"], row["val_crops"], row["val_score"] = round(m["val_recall"], 4), round(m["val_crops"], 3), round(m["val_score"], 4)
                if m["val_score"] > best_score:
                    best_score = m["val_score"]
                    save_checkpoint(best_path, policy, state_dim, cfg, env_cfg, state_cfg, layout=None, detection_metadata=detection_metadata)
            writer.writerow(row)
            f.flush()
            if episode % cfg.log_interval == 0 or episode == 1:
                print(f"[hotspot] ep={episode}/{cfg.episodes} reward={total_reward:.2f} loss={mean_loss:.4f} eps={row['epsilon']} crops={len(env.placed)} cov={row['covered']}/{row['small_total']} valR={row['val_recall']} valC={row['val_crops']}")

    save_checkpoint(last_path, policy, state_dim, cfg, env_cfg, state_cfg, layout=None, detection_metadata=detection_metadata)
    return best_path
