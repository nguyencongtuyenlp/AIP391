# RL-SAHI — Adaptive Slicing cho Small-Object Detection (VisDrone)

Đồ án AIP391: dùng **Reinforcement Learning điều khiển việc cắt ô (slicing)** cho SAHI, để phát hiện
vật thể nhỏ trên ảnh drone mà YOLO ảnh-gốc bỏ lỡ (~77%).

## Ý tưởng
- Detector **YOLO11s đóng băng** sinh proposal (conf 0.01) → bản đồ **proposal-density** (GT-free).
- **density-guided slicing**: đặt K crop lên top-K hotspot density → phóng to + detect lại + merge.
  Dominate fixed-grid SAHI (~98% recall @ 38% chi phí ô).
- **Yield-aware RL agent** (`rl/yield_env.py`): đi qua hotspot, action **{CROP, SKIP}**, state có
  **yield quan sát được của ô đã cắt** (GT-free) → học **chỉ ROI vùng vật-nhỏ bỏ lỡ, skip vùng đã detect**.

## Cấu trúc
```
src/rl_sahi/   detection · rl (env/agent/trainer) · eval · inference · common
scripts/       detect.py · precompute_hotspot_yields.py · train.py · eval_yield_agent.py · visualize_compare.py
configs/       *.yaml (paths, detect, infer, rl, state)
tests/         GT-free guards, reward, NMS...  ->  pytest tests/ -q
```

## Train
Local subset hoặc **full trên Lightning.ai (T4)** — xem **[LIGHTNING_TRAINING.md](LIGHTNING_TRAINING.md)**.

## Luật làm việc / context
Xem **[CLAUDE.md](CLAUDE.md)** (code style Karpathy + kỷ luật RL) và `.agents/context/` (long/mid/short-term + lessons).
