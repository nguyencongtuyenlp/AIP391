# CLAUDE.md — Luật Đọc & Code cho RL-SAHI

> File này là **quy tắc làm việc** cho mọi agent (Claude) và thành viên nhóm trên đồ án RL-SAHI (AIP391).
> Triết lý code theo tinh thần **Andrej Karpathy**: đơn giản, đọc như văn xuôi, không trừu tượng thừa,
> baseline trước rồi mới tối ưu. Đọc hết file này một lần; tuân thủ mỗi lần code.

---

## 0. Bắt đầu mỗi session (BẮT BUỘC, đúng thứ tự)

```text
1. .agents/agent.md                      ← luật vận hành agent
2. .agents/context/long_term.md          ← đồ án là gì, "thành công" là gì, ràng buộc bất biến
3. .agents/context/mid_term.md           ← milestone + hypotheses + số liệu hiện tại
4. .agents/context/short_term.md         ← task ngay trước mắt
5. .agents/context/session.md            ← đọc entry gần nhất
6. .agents/context/lessons.md            ← quét bài học đã đúc kết (replay buffer); lỗi cũ KHÔNG lặp lại
7. CLAUDE.md (file này)                   ← quy tắc đọc & code
8. .agents/current_contruction.md        ← tham chiếu kiến trúc khi cần chi tiết
```

**Không đọc context → không được sửa code.** Nếu một yêu cầu mâu thuẫn với `long_term.md` / `current_contruction.md`:
dừng lại, nêu rõ mâu thuẫn, nêu tác động, đề xuất cách xử lý, chỉ làm tiếp khi người dùng xác nhận.

---

## 1. Triết lý code (tinh thần Karpathy)

> *"The best code is no code. The second best is code so simple you can hold it all in your head."*

1. **Đơn giản trước, thông minh sau.** Viết bản rõ ràng nhất chạy được. Chỉ tối ưu khi có số đo chứng minh cần.
2. **Đọc như văn xuôi.** Đặt tên biến/hàm để người đọc hiểu *ý định*, không cần comment giải thích.
3. **Không trừu tượng thừa.** Đừng tạo class/lớp abstraction "phòng khi cần". YAGNI. Một hàm làm một việc.
4. **Tường minh hơn khôn lỏi.** Tránh one-liner khó hiểu, magic, metaprogramming. Code để người sau sửa được.
5. **Ít phụ thuộc.** Mỗi dependency mới là một khoản nợ. Giữ đúng bộ trong `requirements.txt`.
6. **Đừng làm anh hùng.** Dùng default đã được kiểm chứng (AdamW, Huber loss, ε-greedy...) trước khi sáng chế.
7. **Hiểu shape của tensor.** Luôn biết rõ shape vào/ra. Lỗi RL/CV phần lớn là lỗi shape âm thầm.
8. **Xóa nhiều như viết.** Code chết, flag không dùng, nhánh thử nghiệm bỏ → xóa. Repo nhỏ là repo khỏe.

## 2. Quy ước code của dự án (theo đúng code hiện có — KHÔNG được lệch)

Quan sát từ `src/rl_sahi/`. Code mới phải khớp phong cách này:

- **Header:** mọi module bắt đầu bằng `from __future__ import annotations`.
- **Type hints ở mọi chữ ký hàm.** Dùng `np.ndarray`, `torch.Tensor`, `X | None`, `tuple[...]`. Không bỏ trống.
- **KHÔNG docstring / comment thừa.** Code dự án này cố tình sạch comment (xem `memory.md`). Chỉ thêm comment khi
  logic *thật sự* không thể tự giải thích, và khi đó comment giải thích **"tại sao"**, không phải "cái gì".
- **numpy vectorized, không loop từng phần tử** khi tránh được (xem `slice_env.py`: mask, broadcast, `np.where`).
- **Private helper bắt đầu bằng `_`** (`_apply_action`, `_reward`, `_roi_area_ratio`...). Public API tối thiểu.
- **Cấu hình ra `configs/*.yaml`, nạp qua dataclass** (`EnvConfig`, `StateConfig`). **Không hardcode hyperparameter.**
- **snake_case** cho hàm/biến, **PascalCase** cho class, **UPPER_CASE** cho hằng (`NUM_ACTIONS`).
- **Defensive numeric:** chia thì `max(x, 1e-6)`; làm sạch NaN/Inf bằng `np.nan_to_num`; `np.clip` khi cần chặn biên.
- **Đặt code đúng tầng module:**

  ```
  common/      → boxes, geometry, NMS, cache, config, device, actions  (dùng chung, không phụ thuộc RL)
  detection/   → wrapper YOLO, feature extraction
  hard_region/ → phân tích vùng khó từ GT
  inference/   → pipeline, crop, merge, rollout, visualize
  rl/          → agent, env, network, replay, trainer, state_*
  eval/        → benchmark (mAP50, small_recall, FP)
  ```
  Reward **chỉ** ở `rl/slice_env.py` + `rl/env_config.py`. Định nghĩa state **chỉ** ở `rl/state_*.py`.

## 3. Quy tắc thay đổi (mọi cải tiến)

- **Backward-compat bằng toggle flag.** Thêm tính năng → thêm cờ trong config (mặc định an toàn) để bật/tắt.
  Giữ đường cũ chạy được → vừa làm được ablation vừa có đường lui. (Xem `dueling`, `use_per`,
  `use_simplified_reward`, `use_soft_update`, `use_curriculum` trong `configs/rl.yaml`.)
- **Một thay đổi = một mục đích.** Đừng trộn refactor + đổi thuật toán + đổi hyperparameter trong một lần.
- **Đổi state layout → đồng bộ `state_*.py` và `network.py`** (qua `StateLayout`). Lệch nhau là vỡ ngầm.
- **Sửa reward → kiểm reward hacking** (agent đứng yên / STOP sớm / overflow ROI). Xem phân bố action sau khi sửa.
- **Đụng `configs/` → ghi rõ giá trị cũ → mới trong CHANGELOG.**
- **Trước khi sửa file/vùng → quét `lessons.md` theo tag khớp task** (tag ghi sẵn ở `short_term.md`): nêu rõ *đã quét tag nào* + bài học nào áp dụng (cấm để "không có" trống — phải nói quét tag gì). Bài học đổi hành vi phiên này → `+1 Hits` + ghi `Last-hit`. Vòng lặp đầy đủ: §8.

## 4. Kỷ luật RL (recipe của Karpathy — "A Recipe for Training Neural Nets")

> RL/DL **âm thầm thất bại**: không crash nhưng học sai. Trước khi tin một kết quả, chạy các sanity check sau.

- ✅ **Overfit một mẫu trước.** Cho agent học 1 ảnh duy nhất; nếu không "thắng" được nó thì logic/reward sai, không phải thiếu data.
- ✅ **Kiểm loss/Q lúc init** có hợp lý (không NaN, không nổ). Theo dõi Q-value trung bình không phân kỳ.
- ✅ **Cố định seed** (`seed: 42`), báo cáo số liệu kèm seed. Một lần chạy ≠ kết quả; cần ≥ 2 seed cho kết luận.
- ✅ **Log mọi thứ:** reward theo episode, eval mAP định kỳ, phân bố action, lý do STOP, số slice/ảnh. Không nhìn được thì không debug được.
- ✅ **Đổi một biến mỗi lần.** Ablation chỉ có giá trị khi giữ nguyên phần còn lại.
- ✅ **Nghi ngờ kết quả đẹp.** mAP tăng vọt thường là rò rỉ (train↔val), reward hack, hoặc so sánh lệch. Truy nguồn trước khi mừng.
- ⚠️ **Khoảng cách train↔infer:** train có HardRegionCache (GT), infer thì không. Luôn kiểm agent có generalize không — đây là điểm hội đồng sẽ chất vấn nặng nhất.

## 5. Test & "Definition of Done"

- **Test phần dễ sai:** reward (`tests/test_slice_env_reward.py`), NMS/merge, state layout, GPU/DirectML (`tests/test_gpu.py`).
- Một thay đổi coi là **xong** khi:
  - [ ] Chạy được; test liên quan pass.
  - [ ] Số liệu (nếu có) ghi kèm config + seed + đường dẫn `runs/...`.
  - [ ] `CHANGELOG.md` đã cập nhật.
  - [ ] Tầng context phù hợp đã cập nhật (`short_term` luôn; `mid_term`/`long_term` khi cần — xem `.agents/context/README.md`).

## 6. Khi cần đánh giá lớn / định hướng đồ án

Gọi **skill `research-council`** ("Hội đồng Giáo sư AI"): RL researcher, CV/detection expert, senior big-tech
engineer, paper scholar, thesis advisor cùng review đồ án, chấm điểm, chỉ lỗ hổng, định hướng giải pháp và
chuẩn bị bảo vệ. Dùng khi: trước mốc báo cáo, khi bí hướng đi, khi nghi phương pháp có lỗ hổng, hoặc khi cần
soi mình so với SOTA. (Skill sẽ tự đọc các file context ở trên.)

### 6.1 Tự động kích hoạt skill liên quan (RULE — user yêu cầu 2026-06-19)

Khi câu hỏi của user **liên quan** tới một skill → **CHỦ ĐỘNG kích hoạt ngay, KHÔNG đợi user gõ tên skill**:
- Hiểu / đánh giá / review đồ án · "method là gì / sai chỗ nào / yếu chỗ nào" · trước báo cáo/bảo vệ → **`research-council`** (chạy như workflow đa-persona nếu cần sâu).
- Cần SOTA / paper quốc tế / so với công bố mới → **`deep-research`** (hoặc WebSearch/WebFetch) để nối đất bằng dẫn chứng thật.
- Review/merge code · nghi bug/bảo mật → **`code-review`**.
- Thiết kế hệ thống / quyết định kiến trúc → **`system-design`** / **`architecture`**.
- Nhiều skill cùng liên quan → kích hoạt **tất cả** cái cần. Nếu skill không nằm trong danh sách gọi-được của phiên → thực thi **quy trình** của nó (đọc `SKILL.md` / chạy workflow tương ứng), đừng bỏ qua.

## 7. Phong cách trả lời (agent)

Ngắn gọn · chính xác · thiên hành động · nêu rõ giả định · không bịa yêu cầu/kiến trúc/lịch sử không có trong context.
Khi không chắc → hỏi, đừng đoán. Trả lời bằng **tiếng Việt** (thuật ngữ ML giữ tiếng Anh) trừ khi được yêu cầu khác.

---

## 8. Vòng lặp tự học "Đúc Kết" & Luật đã đúc kết

Cơ chế tự cải tiến kiểu **Reflexion** (verbal RL — *tài liệu, không chạy code*). Vòng 4 bước:
**CAPTURE → RETRIEVE+APPLY → PROMOTE → CONSOLIDATE.** Sơ đồ + RL-mapping ở `.agents/context/README.md §Vòng Lặp Đúc Kết`.

- **CAPTURE** (cuối phiên, *chỉ khi có trigger thật + bằng chứng*): lỗi/bẫy/insight sẽ tái dùng → thêm 1 dòng vào `lessons.md`. Trigger trùng bài cũ → `+1 Hits`, không thêm dòng.
- **RETRIEVE + APPLY**: đọc `lessons.md` ở §0 bước 6; trước khi sửa, lọc theo **tag khớp task** (§3) và ghi vết áp dụng vào dòng `🔁 Lessons` của `session.md` — *đọc/áp dụng để lại dấu vết → file không thể thành "file chết"*.
- **PROMOTE** (cổng có người duyệt): `Hits ≥ 3` qua **≥ 2 phiên**, bằng chứng đủ, **không** làm yếu denylist & không mâu thuẫn luật hiện có → agent **soạn** 1 dòng luật (vào §2 / §3 / §4 hoặc §8 dưới) + 1 mục `CHANGELOG` 'Decision', rồi **DỪNG**. Chỉ ghi khi người dùng viết token `✅ PROMOTE <ID> approved <user> <ngày>` trong `session.md`. Luật promoted bị ≥2 phiên phản chứng → xóa dòng, lesson `retired`, ghi `CHANGELOG` 'Decision (revert)'.
- **CONSOLIDATE** (gộp vào checklist cuối phiên, ~60s): gộp trùng, retire bài `Hits=0` quá ~5 phiên, ép cap 12.

**Guardrail (bất biến — đây là phần an toàn, không được nới):**
- Agent **WRITE tự do** vào `lessons.md` (low-trust) nhưng **chỉ ĐƯỢC ĐỀ XUẤT** sửa `CLAUDE.md`/`long_term.md` (high-trust). Van duy nhất giữa hai tầng là **token người duyệt** — không có đường tự động từ buffer lên hiến pháp.
- **Denylist không bao giờ bị làm yếu:** `long_term.md §3 (success) / §4 (architecture) / §5 (ràng buộc cứng) / §6 (non-goals)` + §5 DoD ở trên. Bài học **chỉ được THÊM** kỷ luật.
- **Provenance-or-nothing:** mỗi bài học phải cite artifact mở được (`session` / `runs/` / `CHANGELOG` / `file:line`). Không nguồn = không phải bài học (giết luật ảo tại gốc).
- **No self-serving:** cấm bài kiểu "bỏ test", "khỏi ghi CHANGELOG", nới lỏng so-sánh-công-bằng, giảm verification.
- Bài **provisional** = ε cao (tái kiểm trước khi tin); luật promoted ≈ deterministic. Lesson sống **DƯỚI** architecture contract — mâu thuẫn thì ràng buộc thắng (luật conflict-detection §0 / `agent.md`).

### Luật đã đúc kết (promoted lessons)

*(trống — bài học đủ `Hits` sẽ được người duyệt nâng lên đây, kèm back-link `← lessons.md L#`)*

---

> Tóm tắt một dòng: **Đọc context + quét lessons → code đơn giản & sạch như code hiện có → một thay đổi một mục đích có flag →
> sanity-check như Karpathy → ghi CHANGELOG + context + lesson.** Mục tiêu cuối luôn là: *đồ án bảo vệ được bằng số liệu.*
