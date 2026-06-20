# Train FULL trên Lightning.ai (T4 16GB) — Yield-Aware RL-SAHI

> Mục tiêu: train **yield-aware agent** (Cửa 2) trên **TOÀN BỘ** train split (6471 ảnh, vs 1200 local)
> để agent học sạch hơn: **chỉ ROI vùng vật-nhỏ YOLO bỏ lỡ, SKIP vùng đã detect**.
> Máy local (GTX 1650 4GB) chỉ đủ subset; T4 16GB build cache + train nhanh hơn nhiều.

---

## 0. Tổng quan pipeline (4 bước)
```
[Data + weights]  ->  [detect cache]  ->  [yield cache]  ->  [train yield-agent]  ->  [eval]
   upload/clone        detect.py         precompute_*.py     train.py --yield-aware   eval_yield_agent.py
```

## 1. Tạo Studio + clone code
Trên **lightning.ai** → New Studio (GPU = **T4**). Mở Terminal:
```bash
git clone https://github.com/<USER>/<REPO>.git
cd <REPO>
```

## 2. Cài môi trường
```bash
pip install numpy opencv-python PyYAML ultralytics
pip install torch --index-url https://download.pytorch.org/whl/cu121   # CUDA 12.1
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 3. Đưa DATA lên (chọn 1 trong 2)
Repo **KHÔNG** chứa ảnh/cache (quá nặng — xem `.gitignore`). Cần `data/raw/{images,labels}/{train,val,test}`.

- **Cách A — Upload data local (an toàn nhất, đúng format đã validate):**
  Nén local `data/raw` → kéo-thả vào Studio (hoặc `lightning upload`) → giải nén thành `data/raw/...`.
  `yolo11s.pt` đã có sẵn trong repo (detector đóng băng).

- **Cách B — Tải VisDrone-DET mới:** tải VisDrone2019-DET-{train,val,test-dev} từ trang chính thức,
  xếp thành `data/raw/images/{train,val,test}/*.jpg` + `data/raw/labels/{train,val,test}/*.txt`
  đúng format mà `detect.py`/`_read_gt` đọc (kiểm bằng 1 ảnh trước).

Kiểm: `ls data/raw/images/train | wc -l   # ky vong 6471`

## 4. Build cache (chạy 1 lần — đây là phần tốn GPU, T4 nhanh)
```bash
# 4a. Detection cache (YOLO full-image + features) — ca 3 split
for s in train val test; do PYTHONPATH=src python scripts/detect.py --split $s; done

# 4b. Yield cache (yield moi hotspot) — train FULL + val
PYTHONPATH=src python scripts/precompute_hotspot_yields.py --split train      # 6471 anh, ~30-50 phut/T4
PYTHONPATH=src python scripts/precompute_hotspot_yields.py --split val        # 548 anh
```
> Ước lượng: yield cache train ≈ 6471 × 16 crop ≈ 103k lần YOLO. T4 ~30-50 phút (chạy `tmux`/background).

## 5. Train yield-agent FULL
```bash
mkdir -p runs/dqn_yield_full
PYTHONPATH=src python scripts/train.py --yield-aware \
    --crop-cost 0.5 --episodes 25000 --trust-cache \
    --out-dir runs/dqn_yield_full
```
- `--crop-cost`: điều khiển "độ kén". **0.5** = crop mọi ô có vật mới, skip ô 0-vật. Muốn **ÍT ROI hơn** (đúng vision "vài ROI chính xác") → tăng **1.0 / 2.0** (kén hơn → ít ô, recall/ô cao hơn).
- Theo dõi: `runs/dqn_yield_full/train_log.csv` cột `val_recall` (% vật-thật bắt được) / `val_crops` (số ô).
- Sweep nhanh để chọn điểm Pareto: chạy lại với `--crop-cost 1.0` và `2.0`, `--out-dir` khác nhau.

## 6. Eval + xem kết quả
```bash
# So yield-agent vs density CUNG ngan sach o, tren full val:
PYTHONPATH=src python scripts/eval_yield_agent.py --checkpoint runs/dqn_yield_full/best.pt --split val

# Xuat 2 anh (YOLO goc | RL-SAHI) cho bao cao:
PYTHONPATH=src python scripts/visualize_compare.py 0000006_00611_d_0000002.jpg --split test
```

## 7. Kéo kết quả về
Tải về local: `runs/dqn_yield_full/best.pt` + `train_log.csv` + ảnh trong `runs/report/`.

---

## ⚠️ Kỳ vọng trung thực
Local (1200 ảnh) cho yield-agent **+0.35–1.0% recall** so density (marginal). Full 6471 ảnh + train kỹ
**có thể** sắc hơn (skip detected sạch hơn, đúng vision) **nhưng** trần cơ bản là *density-guided đã gần tối ưu*
(AUC tách clutter ~0.86). Đừng kỳ vọng nhảy vọt; mục tiêu thực tế = agent **đặt ROI sạch đúng vùng bỏ lỡ**
trên cả tập + bằng/nhỉnh density → đủ để trình "RL-SAHI có cơ sở, hoạt động đúng".

## Kiểm tra nhanh trước khi tin số
- `pytest tests/ -q` phải xanh (gồm `test_yield_env_state_gtfree` — khóa GT-free, no train↔infer gap).
- Cố định seed (đã set trong `configs/rl.yaml`).
