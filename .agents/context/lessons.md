# 📓 Lessons — RL-SAHI (Vòng Lặp Đúc Kết)

> **Replay buffer** của agent: mỗi bài học = một "kinh nghiệm" đã trả giá để biết (lỗi/insight).
> Đọc file này ở **bước 6** của nghi thức session-start (`CLAUDE.md §0`) — *lỗi cũ KHÔNG được lặp lại*.
> Đây là **giàn giáo tư duy theo RL** (scaffold), KHÔNG phải bộ máy chạy code. Vòng lặp: `README.md §Vòng Lặp Đúc Kết`.

## ⚙️ Luật của buffer (3 dòng)
1. **Cap = 12 bài active.** Muốn thêm dòng thứ 13 → phải *promote* hoặc *retire* 1 dòng trước.
2. **Promote:** `Hits ≥ 3` qua **≥ 2 phiên khác nhau**, có bằng chứng → ứng viên lên luật `CLAUDE.md` (**cần người duyệt bằng token**, xem `CLAUDE.md §8`).
3. **Forget:** `Hits = 0` sau ~5 phiên kể từ ngày tạo → chuyển `retired` (giữ dòng làm provenance, **KHÔNG xóa**).

**Tag — từ vựng đóng** (thêm tag mới phải kèm 1 dòng `CHANGELOG`):
`reward` · `state-layout` · `train-infer` · `directml` · `benchmark-fairness` · `eval-leak` · `curriculum` · `per` · `doc-drift` · `cache`

**Mỗi bài cần:** trigger → việc nên/không nên · cách làm đúng (fix) · **Bằng chứng (BẮT BUỘC, artifact mở được)** · Hits · Status.
Bài học **chỉ được THÊM kỷ luật** — không bao giờ làm yếu DoD / ràng buộc cứng (`long_term.md §3/§4/§5/§6`). Status: `provisional | active | promoted | retired`.

> Thêm bài mới ở **TRÊN CÙNG** của mục active, ID tăng dần (`L6`, `L7`…). Áp dụng bài nào trong phiên → ghi vết ở dòng `🔁 Lessons` của `session.md`.

---

## Bài học (active) — mới nhất trên cùng

### L13 — KHONG-denoise-duoc-vi-recall-opportunity-LA-low-conf (negative result vàng)
- **Tags:** `benchmark-fairness` · `train-infer`
- **Khi:** thấy proposal-density bị clutter (công trường/texture) tạo hotspot GIẢ (viz `density_noise_map.jpg`: vùng nóng nhất ở clutter trái, vật nhỏ thật ở ngã tư xa) → định "denoise" để cắt ít ô, nhắm đúng hơn. **ĐÃ THỬ & BÁC BỎ:** lọc lớp-xe chỉ −8% ô; lọc conf≥0.2 giảm 71% ô (8.91→2.60) NHƯNG **chỉ còn phủ 39-42% vật-BỎ-LỠ** (raw phủ 70-88% @ K≤4/8). Vì **77% small-GT bị YOLO bỏ lỡ, và vật-bỏ-lỡ CHÍNH LÀ tín hiệu conf thấp** → bỏ box conf thấp = vứt luôn signal cùng với noise. Noise (clutter low-conf) ⟂ signal (miss low-conf) **dính chặt** ở vùng conf thấp.
- **Cách làm đúng:** KHÔNG denoise bằng confidence/class. RAW density **nhiều ô** là near-optimal cho recall vật-bỏ-lỡ. Đây là LÝ DO ít-ô/RL (B ~2.7 ô phủ ~50%) **không thể** lên recall cao — *giới hạn signal, không phải lỗi agent*. Learned denoiser = future work, ngoài scope. Khẳng định L11 (coverage là tường) + L9 (tuning không đổi inference) ở tầng signal.
- **Bằng chứng:** test 120 val 2026-06-20 (useful-crops: raw 8.91 / vehicle 8.19 / conf≥0.1 4.85 / conf≥0.2 2.60; missed-coverage K≤8: raw 88% vs conf≥0.2 42%); `runs/report/density_noise_map.jpg`.
- **Hits:** 1 · **Last-hit:** 2026-06-20 · **Status:** active

### L12 — density-guided-slicing-LA-DAP-AN (WIN)
- **Tags:** `benchmark-fairness` · `train-infer`
- **Khi:** cần method tốt hơn fixed-grid SAHI + agent DQN. **density-guided** (đặt K crop lên top-K proposal-density hotspot, dùng chính YOLO@conf0.01) **DOMINATE**: density_k12 = 0.254 small_recall @ 10.6 ô (≈98% SAHI 0.258 @ 28 ô) + **mAP 0.237 > SAHI 0.219**; density_k8 = 90% recall @ 27% chi phí. DQN (0.114@1.82) ≈ density_k2 (0.117@1.99) → **DQN KHÔNG thêm giá trị** so với heuristic ở cùng budget.
- **Cách làm đúng:** density-guided = method chính (defensible, Pareto-dominant). RL reframe = "vì sao RL chưa đủ" (train↔infer + coverage), HOẶC train RL học chiến lược phủ-density (option B).
- **Bằng chứng:** `runs/benchmark/val_density/benchmark.json` (548 val, seed 42, 2026-06-20); `eval/benchmark.py` `_density_guided_rois`.
- **Hits:** 1 · **Last-hit:** 2026-06-20 · **Status:** active

### L11 — nut-that-la-COVERAGE-khong-phai-navigation
- **Tags:** `benchmark-fairness` · `reward`
- **Khi:** sau **3 thí nghiệm** (reward rebalance, gate ablation, density nav-shaping) ĐỀU phẳng full-val recall (~0.114): nút thắt KHÔNG phải reward/navigation/gate mà là **COVERAGE** — 1.82 ô không phủ đủ vật nhỏ RẢI RÁC khắp ảnh. SAHI (28 ô) chứng minh vật xa **DETECT ĐƯỢC khi có coverage** → efficiency (ít ô) của RL-SAHI VỐN cap recall. Recall ≈ hàm của số-ô-phủ, không phải chất-lượng-điều-hướng.
- **Cách làm đúng:** đừng tốn retrain để "navigate tốt hơn". Hoặc (a) chấp nhận RL-SAHI là điểm **Pareto hiệu-quả** + dựng **đường cong recall-vs-crops** chứng minh dominate fixed-grid ở ngân sách ô THẤP (win hiệu quả, defensible); hoặc (b) ép coverage cao (mất lợi thế efficiency, về regime SAHI).
- **Bằng chứng:** `runs/benchmark/{val,val_exp1,val_gate,val_exp2}/` (2026-06-20) — 3 negative result có kiểm soát.
- **Hits:** 1 · **Last-hit:** 2026-06-20 · **Status:** active

### L10 — objectness-la-confidence-khong-phai-presence
- **Tags:** `train-infer` · `state-layout`
- **Khi:** chọn tín hiệu GT-free để điều hướng agent tới vật xa khó. `objectness_map = sigmoid(max class logit)` = proxy detector-**CONFIDENCE** → mạnh ở vật gần/to (YOLO đã thấy), YẾU ở vật xa tí hon (YOLO bỏ sót) = **CÙNG PHA với bug** → dùng nó dạy agent NÉ xa. Tín hiệu đúng = `detection_map[2]` **proposal_density** (đếm box conf 0.01-0.5; YOLO conf=0.01 vẫn sinh box rác ở vật xa → density chỉ điểm vùng đông vật nhỏ).
- **Cách làm đúng:** điều hướng bằng **density (count-based)**, không phải objectness (confidence-based). Sanity TĨNH đo %density-mass vùng xa TRƯỚC train (đo được 54.7% nửa-trên → mới train).
- **Bằng chứng:** `src/rl_sahi/detection/features.py:155-161`, `src/rl_sahi/rl/state_maps.py:44-51`, sanity 80 val + panel `wjj94hfi6` (2026-06-20).
- **Hits:** 1 · **Last-hit:** 2026-06-20 · **Status:** active

### L8 — rl-sahi-too-conservative (baseline)
- **Tags:** `reward` · `benchmark-fairness`
- **Khi:** baseline 548 val (seed 42): RL-SAHI cắt chỉ **1.82 ô/ảnh**, small_recall **0.114** « SAHI **0.258** (28 ô). Hiệu quả recall/ô gấp ~6.8× nhưng recall tuyệt đối thấp → **CHƯA thắng SAHI**. Gốc rễ: reward phạt cắt nhiều quá nặng + STOP sớm (`policy_stop=0`).
- **Cách làm đúng:** để thắng → ép agent cắt bạo hơn: giảm `efficiency_weight`/step cost, sửa `stop_bonus` chống STOP sớm, nới accept-slice; chứng minh bằng đường cong **recall-vs-crops** (KHÔNG so 1.82 vs 28). Coi chừng tác dụng phụ: cắt nhiều → FP/cost tăng.
- **Bằng chứng:** `runs/benchmark/val/benchmark.json` (2026-06-20), `runs/dqn/train_summary.json` (policy_stop=0), council `w9msfkh87`.
- **Hits:** 1 · **Last-hit:** 2026-06-20 · **Status:** active

### L7 — train-from-cache-without-weights
- **Tags:** `cache` · `directml`
- **Khi:** muốn train/eval mà `yolo11s.pt` đã mất/đổi → cache bị `detection_cache_is_current` coi là **stale CHỈ vì fingerprint weights** (path/size/mtime) đổi, dù dữ liệu cache vẫn đúng → dataset rỗng (`FileNotFoundError`). Dùng `--trust-cache` (truyền `detection_metadata=None`) để train từ cache có sẵn, KHÔNG cần tải/rebuild YOLO.
- **Cách làm đúng:** sanity nhanh `python scripts/train.py --overfit --trust-cache`. Official run nên có weights thật + cache khớp. Cân nhắc nới `detection_cache_is_current` bỏ qua mtime weights (future fix).
- **Bằng chứng:** `src/rl_sahi/common/cache.py:118-135` + `:78-97`, `dataset.py:46`, overfit run `runs/dqn/run_meta.json` 2026-06-20.
- **Hits:** 1 · **Last-hit:** 2026-06-20 · **Status:** active

### L6 — doc-code-drift (action set)
- **Tags:** `doc-drift`
- **Khi:** trích mô tả method từ doc (`memory.md` / `CHANGELOG` / báo cáo) → **xác minh lại bằng code TRƯỚC khi tin/báo cáo**. Cụ thể: doc nói "4 action chéo (1/√2)" nhưng `actions.py` chỉ có **7 action, không có chéo**. Hội đồng mở `actions.py` 24 dòng là thấy ngay → mất uy tín cả phần method.
- **Cách làm đúng:** mô tả method trỏ `file:line` thật; thêm unit test `NUM_ACTIONS == số nhánh _apply_action` để chặn drift; sửa doc cho khớp code (đã sửa CHANGELOG).
- **Bằng chứng:** `src/rl_sahi/common/actions.py:6-13,26` (7 action), `.agents/memory.md` (claim chéo), council review `w9msfkh87` (2026-06-19).
- **Hits:** 1 · **Last-hit:** 2026-06-19 · **Status:** active

### L1 — train-infer-gap
- **Tags:** `train-infer` · `eval-leak`
- **Khi:** đánh giá / báo cáo số liệu RL-SAHI → nhớ **train có HardRegionCache (GT) còn infer KHÔNG**. Phải kiểm agent có generalize sang ảnh không-GT không; đừng báo số liệu mà lờ đi khoảng cách này.
- **Cách làm đúng:** chạy infer trên `val` (không dùng GT để chọn ROI), quan sát hành vi agent (action dist, số slice); chuẩn bị câu trả lời cho hội đồng về điểm này.
- **Bằng chứng:** `.agents/current_contruction.md` (Hard Region = *training only*), `mid_term.md §5`, `CLAUDE.md §4`.
- **Hits:** 0 · **Last-hit:** — · **Status:** active (seed)

### L2 — benchmark-fairness
- **Tags:** `benchmark-fairness` · `eval-leak`
- **Khi:** so sánh full-image / fixed-grid SAHI / RL-SAHI → **cùng detector, cùng NMS, cùng tập val, cùng seed**. Lệch 1 yếu tố = số liệu vô giá trị, hội đồng đánh sập.
- **Cách làm đúng:** cố định `seed=42`, dùng chung tập val đã chốt; ghi đầy đủ config vào `runs/`; một thí nghiệm = một biến.
- **Bằng chứng:** `mid_term.md §5`, `long_term.md §3.B`, `configs/rl.yaml` (seed).
- **Hits:** 0 · **Last-hit:** — · **Status:** active (seed)

### L3 — reward-hacking-watch
- **Tags:** `reward`
- **Khi:** sửa reward trong `slice_env.py` / `env_config.py` → **xem NGAY phân bố action + lý do STOP**, đừng tin mỗi reward curve (agent có thể đứng yên / STOP sớm ăn `stop_bonus` / overflow ROI).
- **Cách làm đúng:** log action distribution + stop reasons trong rollout; chạy overfit 1 ảnh để xác nhận agent thật sự "thắng" được mục tiêu.
- **Bằng chứng:** `src/rl_sahi/rl/slice_env.py:475` (`_simplified_reward`), `CLAUDE.md §4`, `tests/test_slice_env_reward.py`.
- **Hits:** 0 · **Last-hit:** — · **Status:** active (seed)

### L4 — state-layout-desync
- **Tags:** `state-layout`
- **Khi:** đổi state layout → **đồng bộ `rl/state_*.py` và `rl/network.py` qua `StateLayout`**. Lệch nhau là vỡ ngầm: không crash nhưng học sai.
- **Cách làm đúng:** sửa cả hai phía + chạy test; kiểm shape vào/ra của tensor (Karpathy: luôn biết shape).
- **Bằng chứng:** `src/rl_sahi/rl/network.py:80-90` (reshape theo `layout`), `long_term.md §4`.
- **Hits:** 0 · **Last-hit:** — · **Status:** active (seed)

### L5 — cuda-not-directml
- **Tags:** `directml` · `per`
- **Khi:** máy thật là **GTX 1650 4GB + CUDA** (torch cu121), KHÔNG phải iGPU/DirectML → dùng `device='cuda'` (auto qua `resolve_torch_device('')`). Đừng cài/giả định DirectML. VRAM 4GB: YOLO11s đóng băng vừa đủ; OOM → giảm `num_envs`/`slice_imgsz`; replay buffer ở RAM nên cân `replay_size`.
- **Cách làm đúng:** smoke-test `resolve_torch_device('')` ra `cuda` trước khi train; `device.py` đã backend-agnostic (patch DirectML là no-op trên CUDA) → KHÔNG cần sửa code.
- **Bằng chứng:** `nvidia-smi` (GTX 1650 4GB), `torch 2.5.1+cu121` (`cuda.is_available=True`), `src/rl_sahi/common/device.py:40-47`, smoke-test 2026-06-19, `long_term.md §5`.
- **Hits:** 1 · **Last-hit:** 2026-06-19 · **Status:** active

---

## Đã lên luật / Lưu trữ (promoted & retired)

> Dòng promoted/retired chuyển xuống đây (giữ provenance, không xóa). Promoted kèm back-link `→ CLAUDE.md §<n>` + token người duyệt.

### L9 — reward-tuning-khong-doi-inference (negative result) — **RETIRED 2026-06-20** (consolidate: bị [[L11]] coverage-wall + L13 signal-limit bao hàm)
- **Tags:** `reward` · `train-infer`
- **Khi:** chỉnh reward weight (efficiency/target/stop_bonus) để agent cắt bạo hơn → **hành vi TRAIN đổi mạnh** (zoom ×4.4, policy_stop 0→207) nhưng **benchmark inference GẦN NHƯ KHÔNG ĐỔI** (crops 1.82→1.96, recall ~bằng). Vì lúc infer: reward KHÔNG dùng (argmax) + accept-gate + không có GT chỉ chỗ. *(Sau xác nhận: cùng họ với hotspot agent B — crop_cost/w_cov sweep cũng không đổi ~2.7 ô; tuning reward không phá được tường coverage.)*
- **Bằng chứng:** `runs/benchmark/val_exp1/` vs `val/`, `runs/benchmark/val_gate/`, `runs/dqn_exp1/train_summary.json`; B sweep `runs/dqn_hotspot{,_c05,_c10,_w5}/` — 2026-06-20.
- **Hits:** 1 · **Status:** retired (provenance giữ lại)
