# 🏆 Thể Lệ Cuộc Thi & Định Dạng Bài Nộp (Contest Rules & Submission)

Tài liệu tóm tắt quy định cuộc thi AI Challenge Ho Chi Minh City 2026 (AIC2026), cách tính điểm và định dạng file CSV nộp bài.

---

## 1. Các Tác Vụ Đánh Giá (Evaluation Tasks)

1. **KIS (Known-Item Search):**
   - Ban tổ chức cung cấp câu mô tả văn bản hoặc audio.
   - Hệ thống cần trả về đúng cặp `(video_id, frame_id)` của phân đoạn mục tiêu.
   - Thang điểm dựa trên thứ hạng (Rank 1 được điểm tối đa).

2. **VQA (Visual Question Answering):**
   - Trả về frame chứa câu trả lời kèm nội dung trả lời (text).

3. **TRAKE (Temporal Action & Key Event Retrieval):**
   - Trả về danh sách chuỗi keyframe `(kf_1, kf_2, ..., kf_N)` khớp với thứ tự các sự kiện trong câu truy vấn.

---

## 2. Định Dạng File CSV Nộp Bài Chuẩn

File CSV nộp bài phải tuân thủ nghiêm ngặt:
- Không có dòng tiêu đề (header).
- Không có khoảng trắng sau dấu phẩy.
- Định dạng: `query_id,video_id,frame_id` (hoặc chuỗi frame đối với TRAKE).

```csv
Q01,L25_V001,340
Q02,L26_V102,120,245,410
```

### Thao tác trực tiếp trên Web UI:
- Chọn keyframe và bấm phím tắt **`C`** để thêm vào giỏ bài nộp.
- Bấm nút **"Xuất CSV Nộp Bài"** trên giao diện để tự động tải về file CSV chuẩn.
