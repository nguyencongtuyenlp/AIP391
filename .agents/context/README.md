# 🧠 Hệ Thống Context — RL-SAHI

> Mục tiêu: mọi agent (Claude) hoặc thành viên nhóm khi mở session đều load đúng ngữ cảnh,
> không phải đọc lại toàn bộ codebase, không "quên" quyết định cũ, không đi lạc khỏi mục tiêu đồ án.

## 4 tầng ngữ cảnh (đọc từ trên xuống)

| Tầng | File | Vòng đời | Trả lời câu hỏi |
|------|------|----------|-----------------|
| 🪨 **Dài hạn** | [`long_term.md`](long_term.md) | Tháng → cả đồ án. Ít khi đổi. | *Đồ án này là gì? Thành công nghĩa là gì? Ràng buộc bất biến?* |
| 🌊 **Trung hạn** | [`mid_term.md`](mid_term.md) | 1–2 tuần / milestone | *Giai đoạn này đang làm gì? Giả thuyết nào đang test? Số liệu hiện tại?* |
| ⚡ **Ngắn hạn** | [`short_term.md`](short_term.md) | Vài ngày / task hiện tại | *Ngay bây giờ đang làm gì? 3 bước kế tiếp? Đang kẹt ở đâu?* |
| 📒 **Session** | [`session.md`](session.md) | Append-only, mỗi phiên 1 entry | *Phiên trước làm gì, ra kết quả gì, để lại gì?* |
| 📓 **Lessons** | [`lessons.md`](lessons.md) | Append-only, bounded ~12 | *Bài học tái dùng nào áp dụng cho task này? Lỗi nào KHÔNG được lặp lại?* |

## Thứ tự đọc khi bắt đầu session

```text
1. .agents/agent.md                 ← luật vận hành agent (đã có sẵn)
2. .agents/context/long_term.md     ← north star, không được vi phạm
3. .agents/context/mid_term.md      ← milestone + hypotheses đang chạy
4. .agents/context/short_term.md    ← task ngay trước mắt
5. .agents/context/session.md       ← đọc entry gần nhất
6. .agents/context/lessons.md       ← quét bài học đã đúc kết (replay buffer); lỗi cũ KHÔNG lặp lại
7. CLAUDE.md (ở project root)        ← quy tắc đọc & code (style Karpathy)
8. .agents/current_contruction.md   ← kiến trúc kỹ thuật chi tiết (tham chiếu khi cần)
```

## Thứ tự cập nhật khi kết thúc một đơn vị công việc

```text
1. CHANGELOG.md        ← ghi MỌI thay đổi code/config (bắt buộc)
2. short_term.md       ← cập nhật task hiện tại + bước kế
3. mid_term.md         ← nếu số liệu / hypothesis / experiment queue thay đổi
4. session.md          ← thêm 1 entry tổng kết phiên (điền dòng 🔁 Lessons)
4.5 lessons.md         ← thêm/cập nhật bài học (chỉ khi có trigger thật + bằng chứng); groom ~60s
5. long_term.md        ← CHỈ khi đổi mục tiêu / ràng buộc / quyết định nền tảng
```

## 🔁 Vòng Lặp Đúc Kết (cơ chế tự học kiểu Reflexion)

> Cách agent **tự khá lên qua thời gian**: tài liệu/quy trình, KHÔNG chạy code. `lessons.md` là *replay buffer*
> kinh nghiệm; `CLAUDE.md` là *bộ luật* ổn định. Bài học lặp lại đủ nhiều → người duyệt nâng thành luật.

**4 bước:**
1. **CAPTURE** — cuối phiên, *chỉ khi có trigger thật + bằng chứng* (test fail→fix · số liệu trái giả thuyết · triệu chứng reward-hack/train↔infer · tường DirectML · `research-council` chỉ lỗ hổng · người dùng sửa điều tổng quát) → thêm 1 dòng vào `lessons.md`. Trigger trùng bài cũ → `+1 Hits`, không thêm dòng.
2. **RETRIEVE + APPLY** — đọc `lessons.md` ở bước 6 session-start; trước khi sửa file/vùng, lọc theo **tag khớp task** (tag ghi ở `short_term.md`), nêu *đã quét tag nào* + bài áp dụng. Áp dụng đổi hành vi → `+1 Hits`. *Đọc/áp dụng để lại dấu vết ở dòng `🔁 Lessons` của `session.md` → file không thể thành "file chết".*
3. **PROMOTE** — `Hits ≥ 3` qua ≥ 2 phiên, đủ bằng chứng, không làm yếu denylist → agent **soạn** luật + mục `CHANGELOG` 'Decision' rồi **DỪNG**; chỉ ghi khi người dùng dán token `✅ PROMOTE L# approved <user> <ngày>` (xem `CLAUDE.md §8`).
4. **CONSOLIDATE** — gộp vào checklist cuối phiên (~60s): gộp trùng, retire bài `Hits=0` quá ~5 phiên, ép cap 12.

**RL mapping — giàn giáo để bảo vệ trước hội đồng (*analogy, KHÔNG phải cơ chế chạy*):**

| RL (DQN của nhóm) | Tương ứng trong loop |
|-------------------|----------------------|
| ⚠️ **Function approximator / weights** | **KHÔNG có** — "agent" là một phiên LLM mới, không gradient/trọng số. Tổng quát hóa nhờ *luật bằng chữ người đọc được*, không nhờ weights. Đây là **verbal-RL** (Reflexion): đúng *vai trò*, không đúng số học. **Nêu thẳng điểm này trước khi giám khảo hỏi.** |
| Replay buffer | `lessons.md` (bounded; evict theo promote/staleness → *reservoir*, không phải FIFO/ring) |
| Reward / credit assignment | `Hits` (số lần bài học cứu mình) + trường Bằng chứng |
| Policy at inference | Áp dụng bài học khớp tag *trước khi* sửa code |
| Target-network update (rời rạc, có cổng) | Promotion → luật `CLAUDE.md` (chậm, có người duyệt — *không* phải Polyak liên tục) |
| ε-greedy | Bài provisional = ε cao (tái kiểm); luật promoted ≈ deterministic |
| Action masking / hard constraints | Denylist `long_term.md §3/§4/§5/§6` + DoD — không bao giờ bị làm yếu |

## Nguyên tắc vàng

- **Ghi ngay, ghi ngắn.** Context cũ mà sai còn tệ hơn không có context.
- **Số liệu phải có ngày + config + seed.** "mAP tăng" mà không có 3 thứ đó = vô nghĩa.
- **Một sự thật, một nơi.** Đừng copy số liệu vào 3 file; link tới nguồn.
- **Tầng cao ổn định, tầng thấp biến động.** Đừng nhét chi tiết task vào `long_term.md`.
