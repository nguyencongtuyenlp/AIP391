# 📒 Session Log — RL-SAHI

> Append-only. Entry mới ở TRÊN CÙNG. Mỗi phiên làm việc = 1 entry theo template.
> Đây là "nhật ký" ngắn để phiên sau hiểu phiên trước đã làm gì mà không cần đọc lại code.

---

## Template (copy khi mở phiên mới)

```markdown
## YYYY-MM-DD — <tiêu đề phiên>
- 🎯 Mục tiêu phiên:
- ✅ Đã làm:
- 📊 Kết quả / số liệu (kèm config + seed + đường dẫn runs/):
- 🧠 Học được / quyết định:  (lỗi/bẫy/insight tái dùng + có bằng chứng → thêm 1 dòng vào lessons.md as L#, kèm tag)
- 🔁 Lessons: đã quét tag <...> | áp dụng <ID hoặc KHÔNG CÓ>   ← BẮT BUỘC điền (dấu vết chống "file chết")
- ⏭️ Để lại cho phiên sau:  (nếu có lesson đạt Hits≥3 → dán sẵn: "✅ PROMOTE L# approved <user> <ngày>")
- 📝 Đã cập nhật: [ ] CHANGELOG [ ] short_term [ ] mid_term [ ] long_term [ ] lessons (🔁 đã điền + groom: gộp trùng/retire bài cũ)
```

---

## 2026-06-19 — Thiết lập hệ thống context & skill hội đồng
- 🎯 Mục tiêu phiên: Dựng bộ nhớ ngữ cảnh (ngắn/trung/dài/session), CLAUDE.md (style Karpathy),
  CHANGELOG, và skill `research-council` để hội đồng AI review đồ án.
- ✅ Đã làm:
  - Tạo `.agents/context/` 4 tầng: `long_term`, `mid_term`, `short_term`, `session` + `README`.
  - Tạo `CLAUDE.md` ở project root: quy tắc đọc context + quy tắc code style Karpathy + kỷ luật RL.
  - Tạo `CHANGELOG.md` (Keep a Changelog), seed lịch sử từ `memory.md` (session 2026-06-13).
  - Tạo skill `research-council` (Hội đồng Giáo sư AI: RL researcher, CV expert, senior big-tech eng,
    paper scholar, thesis advisor) để đánh giá & định hướng đồ án.
  - Thêm **Vòng Lặp Đúc Kết** (cơ chế tự học kiểu Reflexion): `lessons.md` (replay buffer, 5 seed thật) + móc vào
    `CLAUDE.md §0/§3/§8`, `session.md` (dòng 🔁 + checklist), `README.md`. Thiết kế qua workflow 4 chuyên gia + 3 phản biện.
- 📊 Kết quả / số liệu: chưa có (đây là phiên setup, không train).
- 🧠 Học được / quyết định:
  - Theo `memory.md`, phần RL coi như "đủ tính năng"; nút thắt thật của đồ án là **số liệu so sánh + ablation**, không phải thêm thuật toán.
  - Giữ nguyên `.agents/agent.md` và `current_contruction.md` cũ; hệ thống mới *bổ sung*, không thay thế.
  - Phản biện chốt: cơ chế tự học phải **cực nhẹ** (1 file + vài dòng), dấu vết nằm trong file không-thể-bỏ (`session.md`), nếu không sẽ thành "file chết".
- 🔁 Lessons: đã quét tag <—, phiên setup> | áp dụng KHÔNG CÓ (vừa tạo buffer; 5 seed sẵn cho task baseline kế)
- ⏭️ Để lại cho phiên sau: chạy **baseline 3 chiều** (full-image / fixed-grid SAHI / RL-SAHI) — xem `short_term.md`.
- 📝 Đã cập nhật: [x] CHANGELOG [x] short_term [x] mid_term [x] long_term [x] lessons
