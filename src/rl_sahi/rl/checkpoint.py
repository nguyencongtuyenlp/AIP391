from __future__ import annotations

from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

import torch

from rl_sahi.common.actions import ACTION_NAMES, NUM_ACTIONS
from rl_sahi.common.device import DeviceLike, resolve_torch_device
from rl_sahi.rl.env_config import EnvConfig
from rl_sahi.rl.network import QNetwork
from rl_sahi.rl.state_config import StateConfig, StateLayout


def save_checkpoint(
    path: Path,
    policy: QNetwork,
    state_dim: int,
    train_cfg: Any,
    env_cfg: EnvConfig,
    state_cfg: StateConfig,
    layout: StateLayout | None = None,
    detection_metadata: dict[str, Any] | None = None,
) -> None:
    torch.save(
        {
            "model": policy.state_dict(),
            "state_dim": state_dim,
            "network_type": "spatial_cnn" if policy.use_spatial_cnn else "mlp",
            "dueling": policy.dueling,
            "num_actions": int(policy.num_actions),
            "state_layout": asdict(layout) if layout is not None else None,
            "train_cfg": asdict(train_cfg),
            "env_cfg": asdict(env_cfg),
            "state_cfg": asdict(state_cfg),
            "detection_metadata": detection_metadata,
            "actions": {int(k): v for k, v in ACTION_NAMES.items()},
        },
        path,
    )


def load_policy(checkpoint_path: Path, device: DeviceLike = None) -> tuple[QNetwork, dict]:
    device = resolve_torch_device(device)
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except (TypeError, Exception):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    env_allowed = {field.name for field in fields(EnvConfig)}
    state_allowed = {field.name for field in fields(StateConfig)}
    env_cfg = EnvConfig(**{key: value for key, value in checkpoint.get("env_cfg", {}).items() if key in env_allowed})
    state_cfg = StateConfig(**{key: value for key, value in checkpoint.get("state_cfg", {}).items() if key in state_allowed})
    hidden_dim = checkpoint.get("train_cfg", {}).get("hidden_dim", 512)
    layout_data = checkpoint.get("state_layout")
    layout = StateLayout(**layout_data) if isinstance(layout_data, dict) else None
    use_spatial_cnn = checkpoint.get("network_type") == "spatial_cnn"
    dueling = checkpoint.get("dueling", checkpoint.get("train_cfg", {}).get("dueling", False))
    num_actions = int(checkpoint.get("num_actions", NUM_ACTIONS))
    policy = QNetwork(
        int(checkpoint["state_dim"]),
        hidden_dim=hidden_dim,
        num_actions=num_actions,
        layout=layout,
        use_spatial_cnn=use_spatial_cnn,
        dueling=dueling,
    )
    policy.load_state_dict(checkpoint["model"])
    policy.to(device)
    policy.eval()
    checkpoint["env_cfg_obj"] = env_cfg
    checkpoint["state_cfg_obj"] = state_cfg
    return policy, checkpoint
