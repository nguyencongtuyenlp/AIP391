<a id="top"></a>
# 🔧 METHOD — Phương pháp RL-SAHI (chi tiết kỹ thuật)

> Tài liệu này mô tả **chính xác** cách method hoạt động, lấy từ **code thật** trong `src/rl_sahi/`. Dùng cho phần "Phương pháp" của báo cáo và để team hiểu hệ thống.
> Method chính = **rl_yield** (Yield-aware Hotspot Agent). File code: `src/rl_sahi/rl/yield_env.py`, `yield_trainer.py`.

---

## 🧭 Mục lục
1. [Tổng quan pipeline (5 bước)](#m1)
2. [Bước nền: density-guided (xếp hạng ô)](#m2)
3. [RL Agent — STATE (trạng thái, 15 chiều, GT-free)](#m3)
4. [RL Agent — ACTION (hành động)](#m4)
5. [RL Agent — REWARD (phần thưởng)](#m5)
6. [Huấn luyện (training)](#m6)
7. [Chạy thật (inference)](#m7)
8. [Đảm bảo GT-free (chống "ăn gian")](#m8)
9. [Bảng siêu tham số](#m9)
10. [Bản đồ code](#m10)

---

<a id="m1"></a>
## 1. Tổng quan pipeline (5 bước)

```
Ảnh drone
   │
   ▼
[1] YOLO11s (đóng băng) soi CẢ ẢNH  ──►  detections gốc + "bản đồ nhiệt" mật độ
   │
   ▼
[2] Xếp hạng "ô nóng" (hotspot) theo mật độ proposal  ──►  danh sách ô ứng viên
   │
   ▼
[3] RL AGENT đi qua từng ô: quyết CẮT hay BỎ (dựa trên trạng thái GT-free)
   │        └─ mỗi lần CẮT: zoom ô đó → YOLO soi lại → quan sát "thu hoạch"
   ▼
[4] Gộp (merge + NMS) detection của cả-ảnh + các ô đã cắt
   │
   ▼
[5] Kết quả cuối: nhiều vật nhỏ hơn, ở ÍT ô hơn cắt-đều
```

**Điểm mấu chốt:** YOLO **không bị sửa**. "Bộ não" RL chỉ quyết định **cắt ô nào** — đó là toàn bộ phần "học".

[↑ đầu](#top)

---

<a id="m2"></a>
## 2. Bước nền: density-guided (xếp hạng ô)

Trước khi agent quyết định, ta cần một **danh sách ô ứng viên** đã xếp hạng. Cách làm (`compute_static_features`, `state_maps.py`):

1. Chia ảnh thành lưới (vd 16×16).
2. Với mỗi ô, đếm **mật độ proposal** của YOLO (số box YOLO nghi có vật, kể cả độ tin cậy thấp).
3. Ô mật độ cao = nghi nhiều vật nhỏ → xếp hạng trên. Mỗi ô → 1 ROI vuông (cạnh = 35% cạnh ngắn của ảnh), tâm tại ô.
4. Loại ô trùng nhau (dedup theo khoảng cách tâm).

→ Agent sẽ duyệt các ô này theo thứ tự mật độ giảm dần. **density-guided cũng là baseline** mạnh để so (cắt top-K cố định).

[↑ đầu](#top)

---

<a id="m3"></a>
## 3. RL Agent — STATE (trạng thái, **15 chiều**, GT-free)

Khi đứng trước ô thứ `i`, agent nhìn một vector **15 số** (`yield_state`, `yield_env.py:55`). Gồm 2 nhóm:

### 🟦 8 đặc trưng TĨNH của ô (từ YOLO nhìn cả ảnh — không cần nhãn):
| # | Đặc trưng | Ý nghĩa |
|---|---|---|
| 1 | `density` | Mật độ proposal trong ô (chuẩn hoá) |
| 2 | `objectness` | Điểm "có vật" của ô (từ feature map YOLO) |
| 3 | `cx/w` | Vị trí ngang của ô (0→1) |
| 4 | `cy/h` | Vị trí dọc của ô (0→1) |
| 5 | `dist_center` | Khoảng cách tới tâm ảnh (vật xa tâm thường nhỏ hơn) |
| 6 | `ncell/10` | Số proposal trong ô |
| 7 | `mean_conf` | Độ tin cậy **trung bình** của proposal trong ô |
| 8 | `max_conf` | Độ tin cậy **cao nhất** của proposal trong ô |

### 🟩 7 đặc trưng ĐỘNG (cập nhật khi agent đã cắt vài ô — đây là "trí nhớ"):
| # | Đặc trưng | Ý nghĩa |
|---|---|---|
| 9 | `n_cropped/k` | Đã cắt bao nhiêu ô (so với tối đa) |
| 10 | `mean_yield` | **Thu hoạch trung bình** của các ô đã cắt (số vật mới quan sát) |
| 11 | `nearest_yield` | Thu hoạch của ô đã cắt **gần ô hiện tại nhất** |
| 12 | `max_yield` | Thu hoạch cao nhất từng thấy |
| 13 | `i/k` | Đang ở ô thứ mấy trong danh sách |
| 14 | `remaining/k` | Còn bao nhiêu ô chưa duyệt |
| 15 | `skipped/k` | Đã bỏ qua bao nhiêu ô |

> 🔑 **Vì sao GT-free:** nhóm động (9-15) dùng **`raw_yield`** = số box-mới YOLO sinh ra khi cắt (đếm được lúc chạy thật, **không cần nhãn**). Nhãn (GT) **chỉ** xuất hiện trong reward lúc train (mục 5). → Trạng thái lúc train **giống hệt** lúc thi → không có lỗ hổng "học một đằng thi một nẻo". Có **test CI** chứng minh điều này (mục 8).

[↑ đầu](#top)

---

<a id="m4"></a>
## 4. RL Agent — ACTION (hành động)

Tại mỗi ô, agent chọn **1 trong 2** (`NUM_YIELD_ACTIONS = 2`):

| Hành động | Mã | Làm gì |
|---|---|---|
| **CROP** | 0 | Cắt ô này → zoom → YOLO soi lại → nhận vật mới |
| **SKIP** | 1 | Bỏ qua ô này, sang ô tiếp |

→ Agent duyệt hết danh sách ô (hoặc tới `k_max`). Số ô CẮT là **tự quyết theo từng ảnh** — đó chính là "phân bổ ngân sách thích nghi".

[↑ đầu](#top)

---

<a id="m5"></a>
## 5. RL Agent — REWARD (phần thưởng)

Công thức (`yield_env.py:151`), **chỉ dùng lúc train**:

```
Nếu CROP ô i:   reward = w_cov × real_yield[i]  −  crop_cost
Nếu SKIP:       reward = 0
```

- `real_yield[i]` = **số vật nhỏ THẬT (đúng nhãn) mới bắt được** nhờ cắt ô i (đây là chỗ duy nhất dùng GT, và chỉ lúc train).
- `w_cov` = thưởng cho mỗi vật bắt được. `crop_cost` = **phạt mỗi lần cắt** (để agent không cắt bừa).

> 💡 **Trực giác:** agent được thưởng khi cắt trúng ô nhiều vật, bị phạt cho mỗi nhát cắt → nó học **cắt ít mà trúng**. Đây là lý do nó đạt recall cao ở ít ô (xem REPORT §7).

[↑ đầu](#top)

---

<a id="m6"></a>
## 6. Huấn luyện (training)

**Thuật toán:** Deep Q-Network (DQN) — mạng MLP nhận state 15 chiều → ước lượng giá trị 2 hành động.

**Tăng tốc bằng precompute cache:** chạy YOLO trên mọi ô là rất chậm. Nên ta **tính trước** (`scripts/precompute_hotspot_yields.py`): với mỗi ô của mỗi ảnh train, lưu sẵn `raw_yield` (số box mới, GT-free) và `real_yield` (số vật thật, GT). → Lúc train agent **chỉ đọc cache**, không gọi YOLO → train cực nhanh (vài chục phút thay vì nhiều giờ).

**Vòng train (mỗi episode = 1 ảnh):**
1. Agent duyệt các ô, chọn CROP/SKIP bằng ε-greedy.
2. Nhận reward (từ `real_yield` trong cache).
3. Lưu (state, action, reward, next_state) vào replay buffer.
4. Cập nhật mạng Q (Double-DQN + n-step return).

**Sanity check kiểu Karpathy:** overfit 1 ảnh trước, theo dõi Q-value không phân kỳ, cố định seed.

[↑ đầu](#top)

---

<a id="m7"></a>
## 7. Chạy thật (inference)

Lúc chạy ảnh mới (`benchmark.py --yield-rl`), **không có cache, không có nhãn**:

1. YOLO soi cả ảnh → detections gốc + danh sách ô.
2. Agent duyệt từng ô, xem state (15 chiều **GT-free**) → quyết CROP/SKIP.
3. **Khi CROP:** zoom ô → gọi **YOLO thật** trên ô → đếm số box-mới → `set_observed_yield()` nạp con số này vào state (cập nhật nhóm động 9-12) cho ô tiếp theo.
4. Hết ô → gộp detections (cả-ảnh + các ô) bằng **class-aware NMS**.

→ Vì state lúc này **giống hệt** lúc train (đều dùng `raw_yield` quan sát được), agent hành xử nhất quán.

[↑ đầu](#top)

---

<a id="m8"></a>
## 8. Đảm bảo GT-free (chống "ăn gian" — điểm hội đồng chất vấn nặng nhất)

**Nguyên tắc:** trạng thái agent **tuyệt đối không chứa thông tin nhãn**. Nhãn chỉ vào **reward lúc train**.

- Nhóm động của state dùng `raw_yield` (đếm box, GT-free), **không** dùng `real_yield` (cần nhãn).
- Có **bài test tự động (CI):** dựng 2 môi trường — một có nhãn, một không — và kiểm tra **state ra giống hệt nhau từng bit** (`tests/test_yield_env_state_gtfree.py`). Nếu lỡ rò nhãn vào state → test fail ngay.

→ Đây là cách trả lời câu "làm sao biết agent không gian lận khi vào ảnh mới?".

[↑ đầu](#top)

---

<a id="m9"></a>
## 9. Bảng siêu tham số (từ `configs/`)

| Nhóm | Tham số | Giá trị |
|---|---|---|
| Detector | YOLO11s COCO, đóng băng · ảnh vào | 640 px |
| Detector | ngưỡng output · map lớp | 0.25 · 6 lớp COCO→VisDrone |
| Ô cắt | cạnh ROI · số ô tối đa `k_max` | 35% cạnh ngắn · 16 |
| Reward | `w_cov` (thưởng/vật) · `crop_cost` (phạt/ô) | (config) |
| DQN | thuật toán | Double-DQN + n-step(3) + (PER tùy chọn) |
| DQN | optimizer · γ · batch · hidden | AdamW · 0.95 · 128 · 512 |
| DQN | ε-decay · replay tối thiểu · reward clip | 15000 bước · 512 · 10 |
| Đo | tập · chỉ số | VAL 548 / TEST 1610 · mAP@0.5, recall, FP/ảnh, ô/ảnh |

> Mọi tham số nạp từ `configs/*.yaml` qua dataclass (`EnvConfig`, `StateConfig`, `TrainConfig`) — không hardcode.

[↑ đầu](#top)

---

<a id="m10"></a>
## 10. Bản đồ code (file → chức năng)

| File | Chức năng |
|---|---|
| `src/rl_sahi/rl/yield_env.py` | **Môi trường RL chính**: state 15-chiều, action CROP/SKIP, reward |
| `src/rl_sahi/rl/yield_trainer.py` | Vòng train DQN (đọc precompute cache) |
| `scripts/precompute_hotspot_yields.py` | Tính trước `raw_yield`/`real_yield` mỗi ô (tăng tốc train) |
| `src/rl_sahi/rl/state_maps.py` | Xếp hạng ô theo mật độ + đặc trưng tĩnh |
| `src/rl_sahi/eval/benchmark.py` | Đo mAP/recall/FP, chạy `--yield-rl` |
| `tests/test_yield_env_state_gtfree.py` | Test CI: chứng minh state GT-free |
| `configs/*.yaml` | Mọi siêu tham số |

---

> **Tóm 1 câu:** YOLO đóng băng soi cả ảnh → xếp hạng ô nóng theo mật độ → **DQN agent quyết CẮT/BỎ từng ô dựa trên trạng thái GT-free (15 chiều) + trí nhớ thu hoạch**, được thưởng theo vật-thật-bắt-được trừ chi-phí-cắt → học **cắt ít ô mà trúng nhiều vật**. Chi tiết số liệu: xem [REPORT.md](REPORT.md).
