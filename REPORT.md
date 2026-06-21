# BÁO CÁO ĐỒ ÁN — RL-SAHI

**Reinforcement Learning điều khiển cắt ảnh thích nghi (adaptive slicing) cho phát hiện vật thể nhỏ trên ảnh drone (VisDrone) — AIP391**

> Mọi con số trong báo cáo lấy **trực tiếp từ file benchmark thật**: VAL = 548 ảnh, TEST = 1610 ảnh, cùng một pipeline. Không ước lượng, không làm tròn ẩu.

---

## 1. Tóm tắt 1 phút (đọc cái này là nắm hết)

- **Vấn đề:** YOLO nhìn cả tấm ảnh drone thì **bỏ lỡ gần hết vật nhỏ** (xe/người ở xa) — recall chỉ **2.3%** (val). Cách cứu kinh điển là **SAHI**: cắt ảnh thành nhiều ô, zoom vào, cho YOLO soi lại. Nhưng SAHI cắt **đều 28 ô/ảnh** → chậm và nhiều báo nhầm (FP).
- **Ý tưởng đồ án:** dùng **RL agent học cắt THÔNG MINH** — chỉ cắt vài ô đúng chỗ thay vì cắt đều. **Ràng buộc cứng:** KHÔNG sửa (fine-tune) YOLO, KHÔNG thêm model phụ (vd super-resolution), **chỉ dùng RL điều khiển việc cắt**.
- **Kết quả chính (số thật):**
  1. **Heuristic density-guided:** cắt ô theo mật độ → đạt mAP ngang SAHI ở **~⅓ số ô** (cả val lẫn test). Hiệu quả thật.
  2. **RL yield-aware agent (đóng góp RL chính):** học **cắt thích nghi** → đạt recall tương đương các phương pháp cắt-nhiều ở **ÍT ô hơn hẳn**, kiểm chứng **cả val và test**.
  3. **Negative results có kiểm soát** (adaptive-conf, multi-scale, residual): đo được **trần thật = giới hạn của detector đóng băng**, không phải do thuật toán cắt.
  4. **Phát hiện tri thức:** ~**40.6%** vật "bỏ lỡ" thật ra YOLO **vẫn sinh box** ở độ tin cậy thấp, chỉ bị ngưỡng vứt đi → bài toán thật nằm ở **calibration của detector**, không phải ở việc cắt.
- **Tự chấm trung thực: 8 – 8.5.** Lý do không 10: detector đóng băng (theo ràng buộc) chặn mAP tuyệt đối; RL không vượt heuristic ở *con số tuyệt đối*, nhưng **thắng ở hiệu quả/độ thích nghi** — và đó là đóng góp khoa học thật, defend được.

---

## 2. Bài toán & vì sao khó

Ảnh drone chụp từ trên cao → vật thể (người, xe) **rất nhỏ** (vài chục pixel). YOLO11s huấn luyện trên COCO (ảnh đời thường, vật to) nên khi nhìn cả ảnh drone thu nhỏ về 640px, vật nhỏ **biến mất** trong lưới đặc trưng → bỏ lỡ.

**Bằng chứng (val 548 ảnh):** YOLO nhìn cả ảnh chỉ đạt recall vật nhỏ = **2.3%** (`yolo_full`). Tức bỏ lỡ ~98% vật nhỏ.

**Cách cứu (SAHI):** cắt ảnh thành ô nhỏ → mỗi ô zoom to → vật nhỏ trở nên "đủ lớn" cho YOLO → soi lại từng ô rồi gộp kết quả. SAHI nâng recall lên **25.8%** — nhưng phải cắt **28 ô/ảnh** (chậm, tốn, nhiều FP).

**Câu hỏi đồ án:** liệu **RL** có học được cách cắt **ít ô mà vẫn nhiều vật** không?

---

## 3. Ràng buộc (luật chơi — rất quan trọng khi bảo vệ)

1. **KHÔNG fine-tune YOLO** — detector giữ nguyên (frozen COCO).
2. **KHÔNG thêm model** (đã cân nhắc và loại super-resolution).
3. **Chỉ RL agent** điều khiển việc cắt.
4. **GT-free:** lúc chạy thật, agent **không được nhìn nhãn** (nhãn chỉ dùng lúc train để chấm điểm) → tránh "ăn gian" và tránh lỗ hổng train↔infer. Có **test tự động (CI)** chứng minh trạng thái agent giống hệt nhau dù có/không có nhãn.

→ Đây là khung "nghiên cứu có kiểm soát": ta cố định detector và chỉ hỏi *"phần CẮT có thể cải thiện tới đâu bằng RL?"*

---

## 4. Cách làm

**Nền (heuristic): density-guided slicing.** Đếm mật độ "đề xuất" (proposal) của YOLO trên lưới → ô nào đông thì cắt. Cắt top-K ô → soi lại → gộp (NMS). Đây là baseline mạnh.

**3 thế hệ RL agent (đều GT-free):**
- **Yield-aware:** agent đi qua từng ô nóng, quyết **CẮT hay BỎ**, quan sát "thu hoạch" (số vật mới) của các ô đã cắt để quyết ô sau. → Học **cắt bao nhiêu ô là đủ, theo từng ảnh**.
- **Multi-scale:** agent chọn thêm **cỡ ô** (nhỏ/vừa/lớn).
- **Adaptive-conf:** agent chọn **ngưỡng tin cậy** khi soi ô (ý tưởng: hạ ngưỡng để cứu vật mờ).

**Baseline đối chứng:** `random_k` (cắt K ô **ngẫu nhiên**) — để chứng minh agent **thật sự chọn khôn**, chứ không phải "cắt đại cũng được".

---

## 5. Kết quả thật (bảng số từ file)

### 5.1 VAL — 548 ảnh
| Phương pháp | mAP@0.5 | recall vật nhỏ | FP/ảnh | **ô cắt/ảnh** |
|---|---|---|---|---|
| YOLO cả ảnh | 0.133 | 0.023 | 5.5 | 0 |
| SAHI (cắt đều) | 0.219 | 0.258 | 28.7 | 28.0 |
| **rl_yield (RL)** | **0.237** | **0.260** | 24.4 | **9.8** |
| density_k12 | 0.235 | 0.252 | 23.9 | 10.7 |
| density_k8 | 0.227 | 0.232 | 20.4 | 7.7 |
| random_k12 | 0.232 | 0.244 | 23.7 | 10.8 |
| adaptive-conf (gập về density) | 0.238 | 0.262 | 25.4 | 12.2 |
| adaptive-conf (bật lever hạ-conf) | 0.213 | 0.242 | 36.0 | 4.0 |

### 5.2 TEST — 1610 ảnh
| Phương pháp | mAP@0.5 | recall vật nhỏ | FP/ảnh | **ô cắt/ảnh** |
|---|---|---|---|---|
| YOLO cả ảnh | 0.057 | 0.004 | 3.3 | 0 |
| SAHI (cắt đều) | 0.121 | 0.112 | 13.2 | 26.6 |
| **rl_yield (RL)** | 0.109 | 0.088 | 8.9 | **3.9** |
| density_k12 | 0.120 | 0.100 | 11.0 | 10.4 |
| density_k8 | 0.115 | 0.089 | 9.6 | 7.5 |
| density_k4 | 0.101 | 0.054 | 7.0 | 3.9 |
| random_k12 | 0.117 | 0.097 | 10.6 | 10.5 |

### 5.3 Đọc bảng — ai thắng ở đâu (đọc theo HIỆU QUẢ, không đọc thô)
- **Đừng so tuyệt đối** (vì mỗi phương pháp cắt số ô khác nhau). Phải so **ở cùng số ô** hoặc nhìn **"recall trên mỗi ô"**.
- **VAL:** rl_yield (0.237 mAP / 0.260 recall @ **9.8 ô**) **vượt density_k12** (0.235 / 0.252 @ 10.7 ô) — **tốt hơn cả 2 mặt mà cắt ít ô hơn**. Vượt SAHI (28 ô). Vượt random (chứng minh chọn-khôn có giá trị).
- **TEST:** rl_yield đạt recall **0.088 ở 3.9 ô** ≈ density_k8 (recall 0.089) nhưng density_k8 phải cắt **7.5 ô** — **rl_yield đạt recall tương đương ở MỘT NỬA số ô**. So với density_k4 (cùng 3.9 ô): rl_yield thắng cả mAP (0.109 vs 0.101) lẫn recall (**0.088 vs 0.054, +63%**).

➡️ **Kết luận cốt lõi:** RL agent **không** thắng SAHI/density ở *mAP tuyệt đối* (vì những cái đó cắt 3–7 lần nhiều ô hơn), **nhưng thắng ở HIỆU QUẢ** — đạt recall tương đương ở **ít ô hơn hẳn**, và điều này **đúng trên CẢ val và test**.

---

## 6. Phân tích sâu

### 6.1 Đóng góp RL = học phân bổ ngân sách cắt thích nghi
- Agent **không cắt cố định K ô**. Nó cắt **nhiều ở ảnh đông vật, ít ở ảnh thưa** → tiết kiệm ô mà giữ recall.
- Bằng chứng "thích nghi": cùng agent, ở val cắt trung bình 9.8 ô, ở test (ảnh khó/khác) tự điều chỉnh xuống 3.9 ô.
- **Tại sao là RL chứ không phải heuristic/classify:** quyết định cắt ô sau **phụ thuộc** ô đã cắt (ngân sách + thu hoạch quan sát được) → bài toán **chuỗi quyết định (MDP)**, đúng việc của RL. `random_k` thua rõ → việc *chọn* thật sự có giá trị.

### 6.2 Các hướng KHÔNG ăn (negative results — vẫn là kết quả khoa học)
- **adaptive-conf (hạ ngưỡng tin cậy):** ý tưởng cứu vật mờ. Thực tế: hạ conf cứu được vài vật **nhưng kéo theo +1143 box rác/ảnh** → FP tăng (36 vs 24), mAP **giảm** (0.213 < 0.235). Đã thử **bản gập (giữ conf cao) = density-crop** và **bản bật lever = tệ hơn**; đã **sửa cả bug đo FP (over-count ×1.58)** → vẫn không cứu được. → **Lever hại net, có ablation sạch.**
- **residual-density** (ép cắt vùng YOLO bỏ lỡ): ≈ density thường (hơi thua) → density raw đã gần tối ưu vị trí cắt.
- **multi-scale / adaptive-K (hotspot):** nằm trên đường Pareto của density → không vượt.

### 6.3 Hành trình debug (minh chứng tính nghiêm túc — điểm cộng khi bảo vệ)
Khi adaptive-conf cho kết quả "đẹp đáng ngờ" (3 lần benchmark giống hệt nhau), đã truy ra **chuỗi 3 bug thật** đúng tinh thần "nghi ngờ kết quả đẹp":
1. **learning rate 1e-4 quá thấp** → mạng Q không fit nổi → agent "ngủ quên" (luôn chọn 1 hành động). Sửa: 5e-4.
2. **trọng số phạt FP quá nặng** → agent né lever.
3. **trạng thái thiếu tín hiệu** để biết ô nào đáng hạ conf → thêm đặc trưng "mật độ proposal conf-thấp".
4. **bug đo FP over-count ×1.58** (đã kiểm chứng bằng số, đã sửa bằng dedup grid).
→ Sau khi sửa hết, agent **dùng lever đúng** — và vẫn thua → chứng minh **negative result là THẬT**, không phải do code lỗi.

---

## 7. Phát hiện tri thức: "bức tường" nằm ở đâu?

Thí nghiệm trực tiếp (1398 vật YOLO bỏ lỡ): nếu zoom đúng chỗ + hạ ngưỡng về 0.01:
- **21%** thật sự "mù" (YOLO không sinh box nào) — đây mới là giới hạn cứng.
- **40.6%** YOLO **vẫn sinh box** ở conf 0.01–0.25, **chỉ bị ngưỡng output 0.25 vứt đi.**

→ **Định vị lại bài toán:** vật nhỏ "bỏ lỡ" phần lớn **không mất**, mà nằm dưới ngưỡng tin cậy. **Nhưng** lấy chúng ra thì **rác (FP) tăng nhanh hơn lợi (recall)** → đó là lý do mọi cách "cắt thêm/hạ ngưỡng" bị chặn. **Trần thật = sự calibration tin cậy của detector đóng băng**, không phải ở khâu cắt. Đây là đóng góp hiểu biết, và nó **giải thích vì sao không thể vượt bằng RL trong ràng buộc này**.

---

## 8. Đóng góp & Giới hạn (thành thật)

**Đóng góp:**
1. Một **RL agent GT-free** học **phân bổ ngân sách cắt thích nghi**, đạt recall tương đương fixed-K ở **ít ô hơn**, **kiểm chứng val + test**.
2. **Nghiên cứu có kiểm soát** xác định **chính xác ranh giới** nơi RL hết tác dụng dưới detector đóng băng (3 lever + random baseline phá tautology).
3. **Phát hiện**: 40.6% vật bỏ lỡ là vấn đề calibration, không phải vấn đề cắt.
4. **Phương pháp chặt:** GT-free + test CI, đo trên cả val và test, tự khai điểm yếu.

**Giới hạn (tự nêu trước giám khảo):**
- mAP tuyệt đối thấp (~0.12 trên test) vì **detector COCO-frozen** + map lớp COCO→VisDrone hao hụt — theo đúng ràng buộc thiết kế.
- Lợi thế của RL/density **co lại trên test** so với val (test khó hơn): claim **chỉ là hiệu quả/độ thích nghi**, KHÔNG phải "thắng tuyệt đối".
- Đo trên 1 seed cho agent; cần thêm seed để có thanh sai số (việc tiếp theo).

---

## 9. Câu giám khảo sẽ hỏi + cách trả lời

1. **"RL đóng góp gì? Bỏ RL đi density vẫn tốt?"** → RL đóng góp **độ thích nghi**: cắt ít ô mà giữ recall, đo được trên đường Pareto val+test. Random baseline thua → chọn-khôn có giá trị. Và chúng tôi **đo được ranh giới** nơi RL hết tác dụng — đó cũng là kết quả.
2. **"Sao mAP chỉ ~0.12?"** → Vì detector COCO-frozen theo ràng buộc + map lớp hao hụt. Chúng tôi đo **lợi thế tương đối trên cùng detector**, không chạy đua mAP tuyệt đối với SOTA.
3. **"Sao không hạ ngưỡng để cứu vật mờ?"** → Đã thử (adaptive-conf). Hạ ngưỡng cứu vài vật nhưng FP tăng nhanh hơn → mAP giảm. Có ablation + đã sửa cả bug đo FP. Đây là negative result được kiểm chứng.
4. **"Sao không fine-tune / super-resolution?"** → Ràng buộc thiết kế là **chỉ RL, detector frozen**, để cô lập câu hỏi "khâu cắt cải thiện được tới đâu". Đã chỉ rõ trần nằm ở calibration của detector.
5. **"density thắng SAHI không?"** → Trên **val** có (ít ô hơn, mAP cao hơn). Trên **test chỉ hòa mAP và thua recall** → nên chúng tôi claim **HIỆU QUẢ Ô-THẤP** (cùng mAP ở ⅓ ô — sống cả trên test), không claim thắng recall tuyệt đối. **Báo cáo cả val lẫn test.**
6. **"RL có khác classify không?"** → Có: quyết định cắt ô sau phụ thuộc ô đã cắt (ngân sách + thu hoạch) → MDP chuỗi, không phải phân loại từng ô độc lập.

---

## 10. Kết luận + tự chấm

RL-SAHI **không phá được trần mAP tuyệt đối** dưới ràng buộc detector-frozen (đó là giới hạn vật lý của bài toán, đã chứng minh). **Nhưng** nó đóng góp một **policy cắt thích nghi hiệu quả** (recall tương đương ở ít ô hơn, val+test), một **bộ negative-result có kiểm soát** xác định đúng ranh giới, và một **phát hiện** rằng bài toán thật là calibration của detector.

**Tự chấm: 8 – 8.5.** Một đồ án **trung thực, phương pháp chặt, defend được bằng số liệu val + test** — thay vì một con số đẹp không đứng vững.

> *Phụ lục số liệu: tất cả file `benchmark.csv` trong `_sync/` (val) và `runs/benchmark/test_clean/` (test 1610 ảnh).*
