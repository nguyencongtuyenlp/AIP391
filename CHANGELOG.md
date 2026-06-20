# Changelog — RL-SAHI

Mọi thay đổi đáng kể của đồ án được ghi tại đây.
Định dạng theo [Keep a Changelog](https://keepachangelog.com/vi/1.1.0/).
Phân loại: **Added** (thêm) · **Changed** (đổi) · **Fixed** (sửa lỗi) · **Removed** (xóa) ·
**Experiment** (kết quả thí nghiệm) · **Decision** (quyết định nền tảng).

> Quy ước: mỗi mục ghi đủ để truy nguồn — **file ảnh hưởng**, **giá trị cũ → mới** với hyperparameter,
> và với Experiment thì kèm **config + seed + đường dẫn `runs/...`**. Mục mới nằm ở TRÊN cùng.

---

## [Unreleased]

### Added
- **Hệ thống context 4 tầng** `.agents/context/`: `long_term.md`, `mid_term.md`, `short_term.md`,
  `session.md`, `README.md` — chuẩn hóa ngữ cảnh ngắn/trung/dài hạn + nhật ký phiên.
- **`CLAUDE.md`** (project root) — quy tắc đọc context + code style Karpathy + kỷ luật RL + Definition of Done.
- **`CHANGELOG.md`** (file này) — theo dõi mọi thay đổi & tiến trình đồ án.
- **Skill `research-council`** (Hội đồng Giáo sư AI) — review đa góc (RL / CV / senior eng / paper / advisor),
  chấm điểm, chỉ lỗ hổng, định hướng và chuẩn bị bảo vệ.
- **Vòng Lặp Đúc Kết** (cơ chế tự học kiểu Reflexion, *docs-only, không hook*) — file mới `.agents/context/lessons.md`
  (replay buffer bài học, cap ~12, 5 seed thật: train-infer / benchmark-fairness / reward-hacking / state-layout / directml)
  + móc vào `CLAUDE.md` (§0 bước 6 đọc lessons · §3 cổng quét-theo-tag · §8 vòng lặp 4 bước + guardrail + landing zone luật)
  + `session.md` (dòng `🔁 Lessons` bắt buộc + box checklist thứ 5) + `README.md` (tầng Lessons + mục Vòng Lặp Đúc Kết + RL-mapping)
  + `short_term.md` (tag cho task để retrieve).

### Decision
- Coi `.agents/agent.md` và `.agents/current_contruction.md` cũ là nguồn hợp lệ; hệ thống context mới **bổ sung**, không thay thế.
- Nút thắt hiện tại của đồ án được xác định là **bằng chứng số liệu (baseline 3 chiều + ablation)**, không phải thêm thuật toán RL.
- **Lesson sống DƯỚI hiến pháp:** `lessons.md` low-trust (agent ghi tự do); nâng thành luật `CLAUDE.md` cần **token người duyệt**;
  denylist (`long_term.md §3/§4/§5/§6` + DoD) không bao giờ bị làm yếu. Thiết kế qua workflow 4 chuyên gia + 3 phản biện
  (cắt over-engineering: bỏ Priority/decay, n-step field, 8-bước→4-bước, rollback ceremony; giữ lõi chống "file chết").

### Changed
- **Phần cứng: iGPU/DirectML → NVIDIA GTX 1650 4GB + CUDA** (torch `2.5.1+cu121`) — **user duyệt đổi ràng buộc cứng/denylist 2026-06-19**.
  Cập nhật `long_term.md §5` + `requirements.txt` (bỏ `torch-directml`) + lesson `L5` + `mid_term.md §5` + memory.
  **KHÔNG sửa code**: `common/device.py` đã backend-agnostic (`resolve_torch_device('')` ưu tiên CUDA; patch DirectML là no-op trên CUDA).
  Smoke-test OK: `device_description('')=cuda/GPU (GTX 1650)`, QNetwork forward `(8,7)` trên `cuda:0`, VRAM 8.6MB/4.29GB.
  Không ảnh hưởng bài toán/method/metrics (mAP/recall/crops độc lập phần cứng) — chỉ nhanh hơn. Số cũ (16 ảnh, split test) vốn đã vô hiệu.

### Fixed (Bước 0 — repo trustworthiness)
- **Reward test suite test đúng path** (`tests/test_slice_env_reward.py`): trước **4/7 fail** vì assert legacy trên path
  production (`use_simplified_reward=true`). Tách 3 class: `SimplifiedRewardTest` (production — thêm test: new-hit dương,
  retained-hit non-STOP âm-nhỏ, STOP-on-target dương, max-steps **delta-based**), `LegacyRewardTest` (`use_simplified_reward=False`),
  `EnvMechanicsTest` (stalled / valid-actions). → **9/9 pass**.
- **`tests/test_actions.py` (mới)** — chặn doc↔code drift action set: `NUM_ACTIONS==7`, tên action khớp, **KHÔNG có action chéo**,
  `_apply_action` xử lý mọi enum, STOP không đổi ROI (lesson `L6`). → 5/5 pass.
- **`tests/test_gpu.py` backend-agnostic** — bỏ hard-import `torch_directml` (vỡ collection trên CUDA); test qua `resolve_torch_device('')`.
- ✅ **`pytest tests/` XANH: 18 passed** trên CUDA/GTX 1650.

### Added (Bước 1 — provenance + logging + sanity)
- **CUDA cleanup cuối:** `inference.yaml` `device: "directml" → ""` (auto→cuda) — chỗ directml hardcode cuối cùng (sẽ cần cho benchmark Bước 2).
- **`scripts/train.py`** cờ mới: `--overfit` (limit=1, tắt val/benchmark, ε-decay=2000, curriculum off), `--trust-cache`
  (dùng cache có sẵn dù `yolo11s.pt` đã mất → `detection_metadata=None`), `--seed`.
- **Logging `batched_trainer.py`**: cột `mean_q` (CSV + print), `run_meta.json` (seed/device/config — provenance),
  `train_summary.json` (action-histogram + stop-reason + mean-Q cuối). Không đụng vòng học.
- ✅ **Sanity overfit-1-ảnh PASS**: agent phủ **20–22/22 hard region** (random 2/22; đạt 22/22 ở ep1300), reward 8→~100,
  meanQ ổn định 10→14 (không phân kỳ) → **reward/logic ĐÚNG**, loại trừ lỗi reward. `runs/dqn/` khôi phục provenance
  (`best.pt`/`last.pt`/`train_log.csv`/`run_meta.json`/`train_summary.json`).
- 📌 **Quan sát theo dõi ở train đầy đủ:** `policy_stop=0` — agent ít STOP "sạch", dựa vào low_new_hits/old_overlap để kết thúc slice (liên quan lo ngại STOP/reward của hội đồng).

### Changed (Bước 5 partial — fix metric chọn checkpoint)
- **Port benchmark eval vào `batched_trainer.py`** (vá lỗ hổng #4 hội đồng): khi `benchmark_model` có → chạy `evaluate_rl_sahi_policy`
  → `benchmark_score` (mAP50 + small_recall − crop_cost − fp_cost) làm tiêu chí chọn `best.pt` + điền cột val_mAP50/small_recall/fp/crops.
  Trước: batched chỉ chọn theo `val_score` (recall hard-region = tín hiệu GT của train). Smoke xác nhận: best.pt chọn theo bench_score.
- **`train.py`** thêm cờ `--eval-benchmark-images`, `--eval-interval` (override nhịp eval).
- **`yolo11s.pt`** tải về repo root (18.4MB, ultralytics v8.4.0) — cần cho benchmark eval + benchmark Bước 2.
- ▶️ **Train official ĐANG CHẠY (nền):** `scripts/train.py --trust-cache --eval-benchmark-images 150 --eval-interval 1000`
  — 20k episodes, num_envs=8, seed=42, best.pt theo `benchmark_score`. ETA ~3h trên GTX 1650.

### Experiment (Bước 1 train + Bước 2 baseline)
- **Train official** chạy nền: dừng ở **11974/20000 ep** (bị kill từ ngoài — RAM/sleep, KHÔNG lỗi code). `benchmark_score` đạt đỉnh **~0.206 ở ep5000** rồi giảm → **hội tụ sớm**, `best.pt` = policy tốt nhất (~ep5k); 20k là thừa. best.pt valid (state_dim=5148, dueling, spatial_cnn, seed=42).
- **Baseline 3-method (548 ảnh val, seed=42, `runs/benchmark/val/`)** — bộ số liệu hợp lệ ĐẦU TIÊN của đồ án:

  | method | mAP50 | small_recall | fp/img | crops/img |
  |--------|-------|--------------|--------|-----------|
  | yolo_full | 0.133 | 0.023 | 5.55 | 0 |
  | fixed_grid_sahi | **0.219** | **0.258** | 28.7 | 28 |
  | rl_sahi | 0.165 | 0.114 | 10.3 | **1.82** |

  → RL-SAHI **>** no-slice, **thua SAHI về recall** (0.114 vs 0.258) nhưng **1.82 vs 28 ô** (recall/ô gấp ~6.8×). Chẩn đoán: **agent quá dè dặt**.
- **Thí nghiệm #1 — reward rebalance** (`efficiency_weight 0.5→0.2`, `target_reward 3→4`, `stop_bonus_weight 1.5→0.5`; train 5k ep → `runs/dqn_exp1/`) — **KẾT QUẢ ÂM TÍNH:**
  hành vi train đổi mạnh (zoom ×4.4, `policy_stop` 0→207, covered_all ×5.4, bench_score train 0.206→0.220) NHƯNG benchmark 548 val
  **gần như không đổi** (rl_sahi: mAP 0.165→0.164, small_recall 0.114→0.113, crops 1.82→1.96, fp 10.3→10.6).
  → **Nút thắt KHÔNG phải reward weight** mà ở phía inference: **train↔infer gap + accept-gate** (slice không sinh detection mới thì dừng).
  Đã **revert** reward về baseline. (Negative result: loại trừ reward tuning, định hướng đúng bước tiếp.)
- **Ablation accept-gate** (KHÔNG retrain; `benchmark.py --max-slice-attempts 32 --no-require-stop`, baseline best.pt, 548 val → `runs/benchmark/val_gate/`) — **CHỐT:**
  mở gate hết cỡ → crops **1.82→1.84** (Δ+0.02), small_recall Δ+0.0001, mAP Δ0 → **gate KHÔNG phải nút thắt.**
  Kết hợp exp1 (reward không đổi inference) → **đóng đinh nút thắt = train↔infer gap** (agent tự dừng ~1.84 ô vì không tìm ra vùng mới có vật nhỏ khi không có GT). **Hai ablation có kiểm soát = kết quả phân tích cốt lõi của đồ án.**
- **`benchmark.py`** thêm cờ ablation `--max-slice-attempts` / `--min-slice-detections` / `--no-require-stop`.

### Added (Thí nghiệm #2 — vá train↔infer: density nav-shaping)
- **Panel 9-agent** (analyze→design→synthesize→critique) thiết kế. Phát hiện cốt tử: `objectness_map` = proxy detector-CONFIDENCE
  (mạnh ở vật gần, YẾU ở vật xa = cùng pha bug) → KHÔNG dùng điều hướng. Dùng `detection_map[2]` = proposal_density (đếm box conf 0.01-0.5).
  **Sanity tự đo:** 54.7% density-mass ở nửa-trên (vùng xa), 0/80 ảnh trống → tín hiệu GT-free TỒN TẠI ở vùng xa.
- **Implement** (`slice_env.py`, `env_config.py`, `rl.yaml`): potential-based shaping `F=γ·Φ(s')−Φ(s)` density-led + unseen-masked
  vào `_simplified_reward` (helper `_density_potential`). **Vá lỗi timing** (critic 3): truyền `seen_before` snapshot TRƯỚC mark_history
  để Φ_now không bị mask seen → shaping không phạt nhầm. 4 knob `EnvConfig` mặc định TẮT (backward-compat, pytest **18/18**).
- **A1 (density nav-shaping, `nav_shaping_weight=0.3`, 5k ep → `runs/dqn_exp2`) — KẾT QUẢ:** train-level tốt nhất mọi run
  (best_score **0.221**, small_recall train 0.104→0.119, policy_stop=234, crops train giảm 1.85→1.67 = cắt thông minh hơn), NHƯNG
  benchmark **548 val GẦN NHƯ PHẲNG:** small_recall 0.1144→0.1155 (**Δ+0.0011**, thu hẹp gap tới SAHI chỉ 0.7%), crops 1.82→1.83, fp ~bằng.
  KHÔNG đạt success (+0.03). Đã **revert** nav_shaping về 0.0.
  → **3 thí nghiệm (reward / gate / nav-shaping) ĐỀU phẳng full-val → nút thắt là COVERAGE, không phải navigation** (xem lesson L11):
  1.82 ô không phủ đủ vật nhỏ rải rác; SAHI 28 ô chứng minh vật xa detect được KHI có coverage → efficiency của RL-SAHI vốn cap recall.

### Experiment (Bước 3 — density-guided slicing: 🏆 WIN)
- **Phát hiện then chốt (sanity tĩnh):** grid 16→32 BÁC BỎ (61% vs 77% hard-on-hotspot — đếm box nên lưới mịn làm trải mỏng).
  Quan trọng: **76% vật nhỏ bị bỏ sót ĐÃ có tín hiệu density ở grid 16** → tín hiệu không thiếu, nút thắt là **COVERAGE**.
- **Implement** `density-guided` vào `eval/benchmark.py` (`_density_guided_rois`/`_predict_density_guided`, cờ `--density-k`):
  đặt K crop lên top-K proposal-density hotspot (cùng crop-size 0.35 như SAHI, chỉ khác CHỖ ĐẶT). pytest 18/18.
- **Kết quả 548 val (seed 42, `runs/benchmark/val_density/`)** — Pareto-dominate fixed-grid SAHI:

  | method | mAP50 | small_recall | crops |
  |--------|-------|--------------|-------|
  | rl_sahi (DQN) | 0.165 | 0.114 | 1.82 |
  | density_k4 | 0.205 | 0.177 | 3.96 |
  | density_k8 | 0.229 | 0.233 | 7.70 |
  | **density_k12** | **0.237** | **0.254** | 10.64 |
  | fixed_grid SAHI | 0.219 | 0.258 | 28.0 |

  → **density_k12: ≈98% recall SAHI + mAP 0.237 > SAHI 0.219, ở 38% chi phí.** THẮNG trục hiệu quả.
- **Ý nghĩa:** WIN là **density heuristic**, KHÔNG phải DQN (rl_sahi 0.114@1.82 ≈ density_k2 0.117@1.99 → DQN không thêm giá trị ở cùng budget). Xem L12.

### Added (Test split + HotspotStopAgent RL)
- **Số TEST split (1000 ảnh unseen, `runs/benchmark/test_density/`)** — win GENERALIZE: density_k12 mAP **0.119 = SAHI 0.119** + small_recall 0.086 (91% SAHI) @ **10.5 vs 27.8 ô**; density_k8 mAP 0.115 @ 7.6 ô. Pareto dominance giữ trên test.
- **HotspotStopAgent (B) — RL agent chính thức cho rubric.** Panel 7-agent thiết kế: tách PLACEMENT (density ranking, đã thắng) khỏi STOPPING (RL học) → **optimal-stopping**, action {CROP_NEXT, STOP}, **state GT-FREE** (12-dim từ density) → KHÔNG train↔infer gap.
  - File mới: `rl/hotspot_env.py` (env optimal-stopping), `rl/hotspot_trainer.py` (DQN loop, KHÔNG chạy YOLO khi train), `tests/test_hotspot_env_state_gtfree.py` (**test bit-identical state có/không GT — guard chống GT-leak**, pass).
  - `env_config.py` (use_hotspot_env/k_max/crop_cost/w_cov/stop_bonus/hotspot_slice_fraction), `checkpoint.py` (lưu/nạp `num_actions` — B dùng 2), `scripts/train.py` (`--hotspot`/`--crop-cost`), `benchmark.py` (`_predict_hotspot_rl` + method `rl_hotspot`, cờ `--hotspot`). pytest **22/22**.
  - Smoke + end-to-end OK (rl_hotspot 0.103@2.30 ô đã > DQN cũ 0.094@1.82). **▶️ Train đầy đủ ĐANG CHẠY** (crop_cost 0.15, 8k ep → `runs/dqn_hotspot/`). Kỳ vọng (panel): HÒA/nhỉnh fixed-K density-guided.

### Added (B benchmark + denoise negative result — 2026-06-20)
- **HotspotStopAgent B benchmark (548 val, `runs/benchmark/val_hotspot/`):** rl_hotspot **0.134 small_recall @ 2.68 ô, mAP 0.186** — nằm ĐÚNG trên đường density-guided (nội suy @2.68 ô = 0.138, Δ−0.004 = hòa) và **vượt DQN cũ** (0.114@1.82, mAP 0.165). RL agent sạch (no train↔infer gap), đủ rubric.
- **B KHÔNG span được Pareto:** sweep crop_cost {0.05,0.10,0.15} + w_cov×5 đều cho **~2.7 ô** (val-proxy phẳng từ ep1→ep8000). → optimal-stopping DQN hội tụ cứng điểm bảo thủ; tuning reward không dịch được (xác nhận L9 retired ở tầng agent). `runs/dqn_hotspot{,_c05,_c10,_w5}/`.
- **L13 — Denoise density BẤT KHẢ THI (negative result vàng):** viz `runs/report/density_noise_map.jpg` xác nhận clutter công trường tạo hotspot density GIẢ (user phát hiện). Thử bóc noise: lọc lớp-xe −8% ô; lọc conf≥0.2 giảm 71% ô (8.91→2.60) NHƯNG **chỉ phủ 39-42% vật-bỏ-lỡ** (raw 70-88% @K≤4/8). **77% small-GT bị YOLO bỏ lỡ, và vật-bỏ-lỡ = tín hiệu conf thấp** → không tách khỏi clutter. ⇒ RAW density nhiều ô near-optimal; ít-ô/RL không thể lên recall cao (giới hạn signal, không phải lỗi agent). Test 120 val. Viz: `runs/report/rl_hotspot_example.jpg`, `density_noise_map.jpg`.

### Added (Viz tooling 2-ảnh + merge_iou sweep — 2026-06-20)
- **`scripts/visualize_compare.py` (mới):** 1 ảnh → xuất 2 file `<stem>_yolo.jpg` (YOLO gốc) + `<stem>_rlsahi.jpg` (density-guided k=8), dùng `get_initial_detection` (cache-or-build). Format before/after sạch cho báo cáo. VD: 0000006 (29→59 box), 0000127 (49→108 box). Output `runs/report/cmp2_*` + `<stem>_{yolo,rlsahi}.jpg`.
- **merge_iou sweep (120 val, density k=8):** 0.50→0.3765 recall/13.4 FP/4.9 dup · 0.45→0.3754/13.1/4.2 · 0.35→0.3717/12.3/3.5. Hạ ngưỡng diệt trùng CHẬM (−29% dup nhưng −1.3% recall) vì phần lớn "trùng" là xe đỗ SÁT NHAU thật. **Giữ merge_iou=0.50** (recall cao nhất + nhất quán benchmark). → tune param chỉ ±1.5%, method đã near-optimal ở mức heuristic.

### Pending (việc tiếp theo)
- [ ] Chạy baseline 3 chiều: full-image / fixed-grid SAHI / RL-SAHI (xem `.agents/context/mid_term.md §3`).
- [ ] Điền bảng số liệu mục tiêu `long_term.md §3.B` và bảng số liệu sống `mid_term.md §4`.

---

## [2026-06-13] — Tối ưu hóa toàn diện RL pipeline

> Nguồn: `.agents/memory.md`. Toàn bộ TODO High/Medium/Low Priority hoàn thành; pipeline sẵn sàng scale training.

### Added
- **Dueling DQN** (`rl/network.py`) — tách Q thành Value stream `V(s)` và Advantage stream `A(s,a)`; hội tụ nhanh hơn khi action ít ảnh hưởng.
- **Prioritized Experience Replay** (`rl/replay.py`) — `PrioritizedReplayBuffer` thay uniform sampling theo TD-error (`per_alpha=0.6`, `per_beta_start=0.4`).
- **N-step returns (n=3)** (`rl/trainer.py`, `rl/batched_trainer.py`) — target = `r_t + γ·r_{t+1} + γ²·r_{t+2}`; cải thiện credit assignment.
- **Batch Rollout / Vectorized Envs** (`rl/batched_trainer.py`) — `num_envs=8` chạy song song, gom state thành 1 batch gọi `policy(states)` một lần.
- ⚠️ **Diagonal actions — KHÔNG có trong code** (đã xác minh `common/actions.py`: chỉ 7 action `LEFT/RIGHT/UP/DOWN/ZOOM_IN/ZOOM_OUT/STOP`, `NUM_ACTIONS=7`; `_apply_action` không có nhánh chéo). `memory.md` (2026-06-13) tuyên bố đã thêm 4 action chéo nhưng **code không hề có** → doc nói về feature chưa tồn tại. Mô tả method phải dùng đúng **7 action**; muốn chéo = *future work*. (Phát hiện bởi council review `w9msfkh87`, 2026-06-19.)
- **Curriculum learning** (`rl/trainer.py`) — `max_slices` tăng tuyến tính 1 → 8 qua `curriculum_steps=15000`.

### Changed
- **Reward đơn giản hóa** (`rl/slice_env.py`, `rl/env_config.py`) — từ 15+ thành phần xuống 4 chính
  (`target_reward`, `efficiency_penalty`, `constraint_penalty`, `stop_bonus`); chống reward hacking. Cờ `use_simplified_reward`.
- **Soft target update** (`rl/trainer.py`) — Polyak averaging `tau=0.005` thay hard update mỗi 200 steps. Cờ `use_soft_update`.
- **Hyperparameters** (`configs/rl.yaml`):
  - `move_fraction`: `0.45 → 0.30` (di chuyển ROI mượt, chính xác hơn).
  - `epsilon_decay_steps`: `8000 → 15000` (khám phá lâu hơn).
  - `guide_prob`: `0.25 → 0.05` giảm dần qua 15000 steps (agent tự lập dần).
  - Thêm: `num_envs: 8`, `n_step: 3`, `use_per`, `use_curriculum`, scheduler (CosineAnnealingLR).
- **`scripts/train.py`** — định tuyến tự động: gọi `batched_train_dqn` nếu `num_envs > 1`, giữ đường tuần tự cũ.

### Removed
- Dọn sạch docstrings/comments thừa ở `network.py`, `replay.py`, `env_config.py`, `slice_env.py` (chuẩn clean-code dự án).

### Decision
- Mọi cải tiến đặt sau **toggle flag** (`dueling`, `use_per`, `use_simplified_reward`, `use_soft_update`,
  `use_curriculum`) để giữ backward-compat và phục vụ ablation.

---

## Cách thêm mục mới (hướng dẫn cho phiên sau)

```markdown
## [YYYY-MM-DD] — <tiêu đề>
### Changed
- **<thành phần>** (`đường/dẫn/file.py`) — mô tả; `giá_trị_cũ → giá_trị_mới` nếu là hyperparameter.
### Experiment
- **<tên thí nghiệm>** — config `configs/...`, seed `42`, kết quả: mAP@50=…, small_recall=…, slices/ảnh=… → `runs/...`
```
