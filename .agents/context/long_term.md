# 🪨 Long-Term Context — RL-SAHI

> Tầng dài hạn = "hiến pháp" của đồ án. Chỉ sửa khi đổi mục tiêu, ràng buộc hoặc quyết định nền tảng.
> Cập nhật gần nhất: 2026-06-19

---

## 1. Định danh đồ án

- **Tên:** RL-SAHI — *Reinforcement Learning–guided Slicing Aided Hyper Inference*
- **Môn / mã:** AIP391 (đồ án AI)
- **Một câu:** Dùng RL agent điều khiển **adaptive slicing** (cắt vùng ROI thông minh) thay cho
  fixed-grid SAHI để phát hiện vật thể nhỏ trên ảnh drone, **đắt tính toán hơn ít mà recall vật nhỏ cao hơn**.

## 2. Bài toán & động lực

- Ảnh drone (VisDrone) có **rất nhiều vật thể nhỏ** (người, xe nhìn từ trên cao). Detector chạy full-image
  ở 640×640 bỏ sót vật nhỏ vì chúng chỉ còn vài pixel.
- **SAHI truyền thống** cắt ảnh thành lưới cố định rồi detect từng ô → recall tốt hơn nhưng **đắt** (nhiều
  lát cắt đều nhau, phần lớn là nền trống → lãng phí inference).
- **Ý tưởng RL-SAHI:** một agent học cách *chọn vùng nào đáng cắt* (nơi có vật nhỏ khó), zoom/di chuyển ROI,
  và **biết dừng** → ít lát cắt hơn mà vẫn bắt được vật nhỏ.

## 3. Định nghĩa "THÀNH CÔNG" của đồ án

> Đây là thước đo để mọi quyết định kỹ thuật quy chiếu về. Một thay đổi chỉ "đáng làm" nếu nó đẩy ít nhất 1 mục dưới đây.

**A. Khoa học (bảo vệ được trước hội đồng)**
- [ ] Chứng minh được RL-SAHI **> baseline full-image** và **≥ fixed-grid SAHI về recall vật nhỏ**,
      với **chi phí (số lát cắt / FPS) thấp hơn** fixed-grid SAHI ở cùng mức recall.
- [ ] Có **ablation** chứng minh từng thành phần (Dueling, PER, simplified reward, n-step, curriculum)
      thực sự đóng góp — không phải "thêm cho oai".
- [ ] Câu chuyện đóng góp (contribution) rõ ràng, 1 câu, examiner gật đầu.

**B. Số liệu mục tiêu (điền khi có baseline thật — xem `mid_term.md`)**
| Metric | Baseline (full-image) | Fixed-grid SAHI | RL-SAHI hiện tại | RL-SAHI (mục tiêu) |
|--------|----------------------|-----------------|------------------|--------------------|
| mAP@50 | 0.133 | 0.219 | 0.165 | ≥ SAHI ở chi phí thấp hơn |
| small-object recall | 0.023 | 0.258 | 0.114 | tiệm cận SAHI |
| slices / ảnh | 0 | 28 | **1.82** | thấp hơn SAHI rõ rệt |

> Đo 2026-06-20, 548 ảnh val, seed 42 (`runs/benchmark/val/`). **Trạng thái: RL-SAHI hiệu quả/ô gấp ~6.8× SAHI nhưng recall tuyệt đối còn thấp (cắt quá ít) → CHƯA thắng. Tiêu chí thành công KHÔNG đổi; cần cải tiến để agent cắt bạo hơn.**

**C. Kỹ thuật (reproducible)**
- [ ] Chạy lại từ `configs/` ra đúng số liệu (seed cố định, không phụ thuộc máy).
- [ ] Code sạch, có test cho phần dễ sai (reward, NMS merge, state layout).

## 4. Kiến trúc bất biến (contract giữa các module)

> Chi tiết kỹ thuật đầy đủ ở [`../current_contruction.md`](../current_contruction.md). Đây chỉ ghi các *ranh giới* không được phá vỡ tùy tiện.

```
Ảnh → YOLO11s full (conf=0.01) → DetectionCache + Feature + Objectness heatmap + Spatial maps
                                        │
                      (train) Hard Region Analysis (GT) → HardRegionCache
                                        │
                          DQN Agent (state ~5000-dim) → chuỗi ROI
                                        │
              YOLO11s crop (conf=0.25) trên từng ROI → class_aware_nms merge → Detections cuối
```

- **State → Network:** layout state do `rl/state_*.py` định nghĩa; `network.py` đọc qua `StateLayout`.
  Đổi state layout **bắt buộc** đồng bộ cả hai, nếu không vỡ âm thầm.
- **Reward chỉ sống trong `slice_env.py` + `env_config.py`.** Mọi shaping reward nằm ở đây, không rải nơi khác.
- **Backward-compat bằng toggle flags** (`dueling`, `use_per`, `use_simplified_reward`, `use_soft_update`,
  `use_curriculum`...). Cải tiến mới phải có flag để bật/tắt → so sánh ablation và không xóa đường lui.
- **Train nhanh = `batched_trainer.py`** (num_envs > 1). Đường tuần tự (`trainer.py`) giữ lại làm reference.

## 5. Ràng buộc cứng (không thương lượng nếu không có lý do lớn)

| Ràng buộc | Giá trị | Hệ quả |
|-----------|---------|--------|
| **Phần cứng** | **NVIDIA GTX 1650 4GB + CUDA** (torch cu121); Windows; RAM 16GB; Acer Nitro 5 (i5-11400H) | Dùng `device='cuda'` (auto qua `resolve_torch_device('')`). VRAM 4GB: YOLO11s đóng băng @640 vừa đủ; OOM → giảm `num_envs`/`slice_imgsz`. Replay buffer ở RAM (~5GB trống → cân `replay_size`). *(Đổi từ iGPU/DirectML — user duyệt 2026-06-19, xem CHANGELOG.)* |
| **Dataset** | VisDrone (drone, vật nhỏ) | Mọi tuning quy về vật nhỏ. Không tối ưu cho vật lớn. |
| **Detector** | YOLO11s (Ultralytics) | Backbone cố định; RL điều khiển *cắt ảnh*, không sửa detector. |
| **Phụ thuộc** | numpy, opencv, PyYAML, torch (cu121), ultralytics | Giữ tối thiểu. **Bỏ `torch-directml`** (máy có CUDA). Thêm dependency mới phải có lý do. |
| **Cấu hình** | `configs/*.yaml` | Không hardcode hyperparameter trong code. |

## 6. Non-goals (đừng đi lạc vào đây)

- ❌ Train lại / sửa YOLO11s. RL-SAHI là *meta-controller*, không phải detector mới.
- ❌ Đổi sang thuật toán RL khác (PPO/SAC...) khi DQN chưa được khai thác hết — trừ khi có lý do số liệu.
- ❌ Tổng quát hóa cho mọi dataset/phần cứng. Phạm vi = VisDrone + iGPU.
- ❌ Tối ưu hóa sớm (micro-optimize) khi chưa có baseline số liệu để so.

## 7. Glossary (thuật ngữ dùng xuyên suốt)

- **SAHI** — Slicing Aided Hyper Inference: cắt ảnh thành lát nhỏ, detect rồi merge để bắt vật nhỏ.
- **ROI / slice** — vùng chữ nhật agent chọn để crop và detect lại ở độ phân giải cao hơn.
- **Hard region** — vùng chứa vật nhỏ mà full-image detector **bỏ sót** (xác định bằng GT lúc train). Đây là "mục tiêu" agent phải tới.
- **Projected size** — kích thước vật khi ROI được resize về `reward_imgsz` (320). Quá nhỏ → khó detect; quá lớn → ROI phí. Reward thưởng vùng "vừa tầm".
- **Objectness heatmap** — max class logit của YOLO → lưới 16×16, gợi ý nơi "có thể có vật".
- **Dueling DQN** — tách Q thành V(s) + A(s,a); ổn định khi action ít ảnh hưởng.
- **PER** — Prioritized Experience Replay: sample theo TD-error, hiệu quả mẫu cao hơn uniform.
- **N-step return** — target dùng tổng chiết khấu n=3 reward → credit assignment tốt hơn.
- **Curriculum** — tăng dần `max_slices` (1 → 8) để agent học crop 1 vật trước.
- **Guided action** — heuristic chỉ đường (`guided_action()`); ε-greedy pha trộn để khởi động học.

## 8. Tài liệu & nguồn gốc

- Kiến trúc kỹ thuật chi tiết: `.agents/current_contruction.md`
- Lịch sử thay đổi: `CHANGELOG.md`
- Đánh giá phương pháp (nếu có): `rl_sahi_method_evaluation.md` (tham chiếu trong memory.md)
- Hội đồng review tự động: skill `research-council` (xem CLAUDE.md §"Khi cần đánh giá lớn").
