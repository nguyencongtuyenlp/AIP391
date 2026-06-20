# ⚡ Short-Term Context — RL-SAHI

> Tầng ngắn hạn = task NGAY BÂY GIỜ. Sửa mỗi phiên làm việc. Giữ ngắn (< 1 màn hình).
> Cập nhật gần nhất: 2026-06-19

---

## 🎯 Đang làm gì (task hiện tại)

> _🏆 **WIN:** density-guided slicing **dominate SAHI** — `density_k12` ≈98% recall (0.254 vs 0.258) + **mAP 0.237 > SAHI 0.219** @ 38% chi phí (10.6 vs 28 ô). `runs/benchmark/val_density/`. DQN (0.114@1.82) ≈ density_k2 → DQN không hơn heuristic._
> _Đường đi: baseline → 3 ablation (reward/gate/nav-shaping đều phẳng) → đóng đinh COVERAGE → density-guided phủ hotspot = đáp án._
> **⚖️ Quyết định hướng hoàn thiện (chờ user):** (A) density-guided làm method chính + RL = phân tích; (B) train RL HỌC chiến lược phủ-density (adaptive K, prune, refine) để RL hơn cả heuristic.
> _Test split (1000 ảnh): win GENERALIZE (density_k12 = SAHI mAP @ 38% chi phí). Rubric CẦN RL agent → đang làm B._
> **▶️ ĐANG CHẠY: HotspotStopAgent (B)** — RL optimal-stopping trên density hotspot, state GT-FREE (test bit-identical pass, no gap). crop_cost=0.15, 8k ep → `runs/dqn_hotspot/`. Implement xong, end-to-end OK.
> 🏷️ **Tags:** `train-infer` · `benchmark-fairness`

## ⏭️ 3 bước kế tiếp (cụ thể)

1. [ ] (ĐANG CHẠY) Train B → benchmark `rl_hotspot` 548 val, overlay lên Pareto density. Xem hòa/nhỉnh fixed-K.
2. [ ] (tùy) Quét crop_cost {0.1,0.2,0.3} → đường cong adaptive của B cho báo cáo.
3. [ ] Đóng gói báo cáo: bảng val+test + biểu đồ recall-vs-crops + viz + 5-ablation + B agent. Dọn `lessons.md` (cap 12).

## 🚧 Đang kẹt / chờ (blockers)

- _Chưa có blocker. (Ghi vào đây nếu bí: thiếu checkpoint? lỗi DirectML? thiếu nhãn val?)_

## 🗒️ Scratchpad (quyết định đang cân nhắc, xóa khi chốt)

- Cần xác nhận checkpoint nào trong `runs/dqn/` là mới nhất / dùng được trước khi benchmark RL-SAHI.
- Khi chạy benchmark nhớ cố định `seed` (đang là 42 trong `rl.yaml`) để số liệu lặp lại được.

---

### Cách dùng file này
- Mỗi lần ngồi vào việc: đọc mục 🎯 + ⏭️, làm, rồi cập nhật.
- Task xong → ghi `CHANGELOG.md`, dời thành tựu sang `session.md`, kéo task mới từ `mid_term.md §3` lên đây.
- Đừng để file này phình to: nó chỉ phản ánh "hôm nay/đang làm", không phải lịch sử.
