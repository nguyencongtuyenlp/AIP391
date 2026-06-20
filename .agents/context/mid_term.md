# 🌊 Mid-Term Context — RL-SAHI

> Tầng trung hạn = milestone hiện tại + giả thuyết đang test + bảng số liệu sống.
> Cập nhật mỗi 1–2 tuần hoặc khi milestone/số liệu đổi.
> Cập nhật gần nhất: 2026-06-19

---

## 1. Milestone hiện tại

- **Tên milestone:** _M? — đặt tên (vd: "Baseline số liệu + Scale training")_
- **Hạn:** _TBD (điền deadline đồ án / mốc bảo vệ)_
- **Mục tiêu milestone (1 câu):** Có **bộ số liệu so sánh 3 chiều** (full-image / fixed-grid SAHI / RL-SAHI)
  trên cùng tập val VisDrone, đủ để dựng bảng kết quả trong báo cáo.
- **Định nghĩa Done của milestone:**
  - [ ] Chạy benchmark đủ 3 cấu hình trên ≥ 500 ảnh val, lưu vào `runs/benchmark/`.
  - [ ] Điền bảng số liệu mục tiêu ở `long_term.md §3.B`.
  - [ ] Có ít nhất 1 ablation (bật/tắt 1 flag) cho thấy chênh lệch rõ.

> ⚠️ Trạng thái code theo `memory.md`: framework RL đã "hoàn thiện 100% TODO" (Dueling, PER, simplified reward,
> n-step, curriculum, batch rollout) và sẵn sàng **scale training**. Việc còn thiếu KHÔNG phải thêm tính năng RL —
> mà là **bằng chứng số liệu**. Đừng thêm thuật toán mới khi chưa có baseline để chứng minh cái đang có hoạt động.

## 2. Giả thuyết đang kiểm chứng (hypotheses)

> Mỗi hypothesis: phát biểu rõ + cách đo + trạng thái. Một hypothesis "thắng" chỉ khi số liệu xác nhận, không phải cảm giác.

| # | Giả thuyết | Đo bằng | Trạng thái |
|---|-----------|---------|-----------|
| H1 | RL-SAHI đạt recall vật nhỏ ≥ fixed-grid SAHI với **ít lát cắt hơn** | benchmark small_recall + slices/ảnh | ⏳ chưa có số |
| H2 | Simplified reward (4-component) hội tụ nhanh & ổn định hơn legacy 15-component | reward curve + eval mAP qua episodes, 2 lần seed | ⏳ chưa đo có kiểm soát |
| H3 | Batch rollout (num_envs=8) cho cùng chất lượng nhưng nhanh 5–10× | wall-clock + eval mAP so với tuần tự | 🟡 nhanh đã thấy; chất lượng chưa so chặt |
| H4 | Curriculum (max_slices 1→8) tránh agent "rối" giai đoạn đầu | eval mAP sớm có/không curriculum | ⏳ chưa ablation |

## 3. Experiment queue (ưu tiên từ trên xuống)

> Quy tắc: **1 thí nghiệm = 1 biến đổi**. Đổi nhiều thứ cùng lúc thì không biết cái nào gây ra kết quả.

1. **[P0] Baseline 3 chiều** — chạy `scripts/benchmark.py` cho: (a) full-image, (b) fixed-grid SAHI, (c) RL-SAHI checkpoint hiện tại. Cùng tập val, cùng seed. → điền bảng `long_term.md`.
2. **[P0] Scale training thật** — train tới hội tụ (episodes theo `rl.yaml`), lưu checkpoint + reward/eval curve vào `runs/dqn/`.
3. **[P1] Ablation simplified vs legacy reward** (H2) — chỉ đổi `use_simplified_reward`, giữ seed.
4. **[P1] Ablation curriculum on/off** (H4).
5. **[P2] Ablation PER vs uniform, Dueling vs standard** — gom thành 1 bảng ablation cho báo cáo.
6. **[P2] Sanity checks RL** (xem CLAUDE.md §"Kỷ luật RL"): overfit 1 ảnh, kiểm Q-value không phân kỳ.

## 4. Bảng số liệu sống (latest results)

> Mỗi dòng PHẢI có: ngày · config · seed · checkpoint · nguồn (đường dẫn `runs/...`). Số không truy vết được = bỏ.

| Ngày | Cấu hình | Seed | mAP@50 | small_recall | slices/ảnh | fp/ảnh | Nguồn |
|------|----------|------|--------|--------------|-----------|--------|-------|
| 2026-06-20 | yolo_full (no slice) | 42 | 0.133 | 0.023 | 0 | 5.55 | `runs/benchmark/val/` (548 val) |
| 2026-06-20 | fixed_grid_sahi | 42 | **0.219** | **0.258** | 28 | 28.7 | `runs/benchmark/val/` |
| 2026-06-20 | rl_sahi (best.pt ~ep5k, train dừng 12k) | 42 | 0.165 | 0.114 | **1.82** | 10.3 | `runs/benchmark/val/` |
| 2026-06-20 | rl_sahi **EXP1** (reward rebalance) | 42 | 0.164 | 0.113 | 1.96 | 10.6 | `runs/benchmark/val_exp1/` — **ÂM TÍNH, ≈ baseline** → nút thắt là train↔infer/accept-gate, KHÔNG phải reward |
| 2026-06-20 | rl_sahi **GATE-MỞ** (32 attempts, no require_stop) | 42 | 0.165 | 0.114 | 1.84 | 10.4 | `runs/benchmark/val_gate/` — **gate KHÔNG phải nút thắt** (Δcrops +0.02) → **đóng đinh train↔infer gap** |
| 2026-06-20 | rl_sahi **A1** (density nav-shaping, exp2) | 42 | 0.166 | 0.116 | 1.83 | 10.4 | `runs/benchmark/val_exp2/` — train tốt nhất nhưng val **PHẲNG** (Δrecall +0.0011) → **3/3 negative → nút thắt là COVERAGE** |
| 2026-06-20 | **density_k4** (guided slice) | 42 | 0.205 | 0.177 | 3.96 | — | `runs/benchmark/val_density/` |
| 2026-06-20 | **density_k8** | 42 | 0.229 | 0.233 | 7.70 | — | ↑ — **90% recall SAHI @ 27% chi phí, mAP > SAHI** |
| 2026-06-20 | **density_k12** ✅ | 42 | **0.237** | **0.254** | 10.64 | — | ↑ — **≈SAHI recall (98%) @ 38% chi phí + mAP 0.237 > SAHI 0.219 → THẮNG trục hiệu quả** |

**Đọc:** RL-SAHI **>** no-slice; **thua SAHI về recall** (0.114 vs 0.258) NHƯNG **1.82 vs 28 ô** → recall/ô gấp **~6.8×**. Chẩn đoán: agent **quá dè dặt, cắt quá ít**. Hướng cải tiến: ép cắt bạo hơn (rebalance reward, sửa STOP sớm).

## 5. Rủi ro & câu hỏi mở

- ❓ **Hội tụ trên GTX 1650/CUDA:** episodes trong `rl.yaml` (20k) có đủ hội tụ thật trong thời gian cho phép không? (CUDA nhanh hơn DirectML nhiều → khả thi hơn; cần đo wall-clock thật khi train.)
- ❓ **Reward hacking:** simplified reward có khiến agent "STOP sớm để ăn stop_bonus" hay "đứng yên"? → cần xem phân bố action + stop reasons trong rollout.
- ❓ **Khoảng cách train↔infer:** lúc train có HardRegionCache (GT); lúc infer thì không. Agent có generalize sang ảnh không-GT không? → đây là rủi ro học thuật lớn nhất, hội đồng sẽ hỏi.
- ⚠️ **So sánh công bằng:** RL-SAHI vs SAHI phải cùng detector, cùng NMS, cùng tập ảnh. Lệch 1 yếu tố là số liệu mất giá trị.

## 6. Quyết định đang chờ (cần chốt)

- [ ] Chốt tập val chuẩn (bao nhiêu ảnh) để mọi benchmark dùng chung.
- [ ] Chốt định nghĩa "small object" (ngưỡng diện tích) khớp chuẩn VisDrone để báo cáo nhất quán.
- [ ] Chốt 1 checkpoint "official" để mọi số liệu RL-SAHI quy về.
