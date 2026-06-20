# 🏗️ Current Construction — RL-SAHI Project

> Cập nhật: 2026-06-13

## 1. Tổng Quan Phương Pháp

**RL-SAHI** (Reinforcement Learning — Slicing Aided Hyper Inference) sử dụng **Dueling Double DQN + PER** để điều khiển adaptive image slicing cho object detection, thay thế fixed-grid SAHI truyền thống.

## 2. Pipeline Hiện Tại

```
Input Image
    │
    ▼
YOLO11s Full Detection (640×640, conf=0.01)
    │
    ├── DetectionCache: boxes, scores, classes
    ├── Feature Vector: backbone layer 16 (mean+std)
    ├── Objectness Heatmap: max class logit → 16×16 grid
    └── Spatial Feature Maps: 3 levels × 4 channels → 16×16
    │
    ▼
Hard Region Analysis (training only)
    │ GT matching → small undetected objects → HardRegionCache
    │
    ▼
┌─────────────────────────────────────────┐
│  DQN Agent (Dueling Double DQN + PER)   │
│                                          │
│  State (~5000 dim):                      │
│    • Feature vector (256–512)            │
│    • History map (16×16)                 │
│    • Previous slice map (16×16)          │
│    • Detection map (4×16×16)             │
│    • Objectness map (1×16×16)            │
│    • Spatial features (~12×16×16)        │
│    • Summary vector (28 scalars)         │
│                                          │
│  Actions: LEFT, RIGHT, UP, DOWN,         │
│           ZOOM_IN, ZOOM_OUT, STOP        │
│                                          │
│  Network: 2-stream (Spatial CNN + MLP)   │
│    → Dueling heads: V(s) + A(s,a)        │
│                                          │
│  Reward: Simplified 4-component system   │
│    1. Target reward (+3.0/hit)           │
│    2. Efficiency penalty (0.5×cost)      │
│    3. Constraint penalty (3.0×violation) │
│    4. Stop bonus/penalty (1.5×quality)   │
└─────────────────────────────────────────┘
    │
    ▼ (max 8 slices × max 16 attempts)
    │
YOLO11s Crop Detection (640×640, conf=0.25)
    │
    ▼
class_aware_nms merge (IoU=0.5)
    │
    ▼
Final Detections
```

## 3. Thuật Toán RL Hiện Tại

| Component | Method |
|-----------|--------|
| **Algorithm** | Dueling Double DQN + N-step returns (n=3) |
| **Replay** | Prioritized Experience Replay (α=0.6, β: 0.4→1.0) |
| **Target Update** | Soft (Polyak averaging, τ=0.005) |
| **Exploration** | ε-greedy (1.0→0.05 / 8k steps) + Guided (25%) |
| **Loss** | IS-weighted Smooth L1 (Huber) |
| **Reward** | Simplified 4-component (target + efficiency + constraint + stop) |

## 4. Tham Số Chính

### 4.1 Environment
| Param | Value | Mô tả |
|-------|-------|-------|
| `max_steps` | 20 | Steps tối đa mỗi slice |
| `max_slices` | 8 | Slices tối đa mỗi ảnh |
| `move_fraction` | 0.30 | Di chuyển 30% side/step |
| `zoom_factor` | 0.75 | Zoom ×0.75 hoặc ×1.33 |
| `min_slice_fraction` | 0.12 | Slice nhỏ nhất 12% |
| `max_roi_area_ratio` | 0.20 | ROI ≤ 20% diện tích ảnh |

### 4.2 Training
| Param | Value | Mô tả |
|-------|-------|-------|
| `episodes` | 30,000 | Tổng training episodes |
| `batch_size` | 128 | Batch size cho DQN |
| `replay_size` | 50,000 | Capacity replay buffer |
| `gamma` | 0.95 | Discount factor |
| `lr` | 1e-4 | Learning rate (AdamW) |
| `tau` | 0.005 | Soft update coefficient |
| `hidden_dim` | 512 | Network hidden dimension |

### 4.3 Reward (Simplified)
| Component | Weight | Logic |
|-----------|--------|-------|
| `target_reward` | 3.0 | +R per new hard hit + density bonus |
| `efficiency_weight` | 0.5 | Step cost (0.05) + area ratio (0.5×ratio) |
| `constraint_weight` | 3.0 | Overflow ROI + low scale + old overlap |
| `stop_bonus_weight` | 1.5 | Good stop (+quality) / Bad stop (-0.5) |

## 5. Network Architecture

```
┌─ Spatial Branch ────────────────────┐
│ Conv2d(N, 32, k=3) → ReLU          │
│ Conv2d(32, 64, k=3) → ReLU         │
│ AdaptiveAvgPool(4×4) → Flatten     │
│ Output: 1024 dim                    │
└─────────────────────────────────────┘
         ↓
         ├── Concat ──→ Trunk(Linear→512→ReLU)
         ↑                    ↓
┌─ Vector Branch ─────────┐   ├─→ Value Head  → V(s)     [1]
│ Linear(vec, 512) → ReLU │   └─→ Advantage Head → A(s,a) [7]
│ Input: feature + summary│
└─────────────────────────┘   Q(s,a) = V + (A - mean(A))
```

## 6. Cấu Trúc Modules

```
src/rl_sahi/
├── common/          # Boxes, caching, config, device, actions
├── detection/       # YOLO wrapper, feature extraction
├── hard_region/     # GT-based hard region analysis
├── inference/       # Pipeline, crops, merge, rollout, visualize
├── rl/              # DQN agent, env, training
│   ├── network.py       ← Dueling DQN
│   ├── replay.py        ← PER + uniform buffer
│   ├── trainer.py       ← Training loop (soft update, PER)
│   ├── slice_env.py     ← RL environment (simplified reward)
│   ├── env_config.py    ← Environment + reward params
│   ├── state_*.py       ← State construction
│   └── checkpoint.py    ← Model save/load
└── eval/            # Benchmark (mAP50, small_recall, FP)
```

## 7. Backward Compatibility

Tất cả cải tiến có toggle flags:

| Flag | Default | Tắt = code cũ |
|------|---------|---------------|
| `dueling` | `true` | Standard DQN head |
| `use_per` | `true` | Uniform replay buffer |
| `use_simplified_reward` | `true` | Legacy 15-component reward |
| `use_soft_update` | `true` | Hard target copy every 200 |

## 8. TODO (Đã hoàn thành)

- [x] Giảm `move_fraction` 0.45 → 0.30
- [x] Epsilon decay chậm hơn: 8000 → 15000 steps
- [x] Guide_prob decay theo training progress (0.25 → 0.05)
- [x] Learning rate scheduler (CosineAnnealingLR)
- [x] Thêm diagonal actions (UP_LEFT, UP_RIGHT, DOWN_LEFT, DOWN_RIGHT)
- [x] Curriculum learning (Tăng dần max_slices theo steps)
- [x] Multi-step returns (N-step DQN với n=3)
- [x] Batch rollout (Vectorized Environments xử lý song song nhiều slices)
