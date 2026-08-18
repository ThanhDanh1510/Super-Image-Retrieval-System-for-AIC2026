# 🧠 Prompt & Hướng Dẫn: Trích Xuất OCR & Mô Tả Ảnh Bằng Gemini Trong Antigravity

Tài liệu này cung cấp **Prompt chuẩn hóa 1-Click** để sử dụng trực tiếp với **Gemini trong Antigravity**. Antigravity sẽ tự động duyệt toàn bộ ảnh trong một folder video, đọc trực tiếp từng ảnh bằng công cụ thị giác, phân tích trích xuất dữ liệu chuẩn JSON rồi tự động dừng lại.

---

## 1. 📋 Cấu Trúc JSON Chuẩn Hóa (1 Trường `description` Duy Nhất)

Mỗi ảnh keyframe sau khi phân tích sẽ được ghi vào file JSON theo đúng cấu trúc 4 trường tinh gọn:

```json
{
  "18316": {
    "ocr_text": "Câu 4: Trong không gian Oxyz, cho điểm A(4; -2; 2) và mặt cầu (S): x^2 + y^2 + z^2 = 4... Chọn D.",
    "description": "Thầy giáo đang giảng bài toán hình học không gian Oxyz trên bảng đen. Bảng viết chi tiết các bước tính bán kính r của đường tròn ngoại tiếp tam giác BCD, bên phải có sơ đồ tọa độ và mặt cầu minh họa, không gian phòng học sáng rõ.",
    "objects": ["thầy giáo", "bảng đen", "phấn trắng", "sơ đồ hình học Oxyz", "mặt cầu"],
    "tags": ["hình học không gian", "mặt cầu", "bài giảng toán", "Oxyz", "math lecture"]
  }
}
```

---

## 2. 🎯 Prompt Copy-Paste Trực Tiếp Vào Antigravity

Khi muốn xử lý bất kỳ thư mục video nào, bạn chỉ cần copy toàn bộ đoạn prompt bên dưới, sửa lại đường dẫn thư mục ảnh ở dòng đầu tiên, rồi dán vào khung chat của Antigravity:

```markdown
Hãy thực hiện tác vụ trích xuất OCR và sinh mô tả hình ảnh cho toàn bộ các ảnh keyframe trong thư mục sau:
👉 Thư mục mục tiêu: `data/image/L25/L25/V017/`
👉 Thư mục lưu kết quả: `data/gemini_output/`

Quy trình thực hiện:
1. Sử dụng công cụ `list_dir` để liệt kê tất cả các file ảnh (.webp, .jpg, .png) trong thư mục mục tiêu.
2. Với từng ảnh, sử dụng công cụ `view_file` để xem và phân tích chi tiết nội dung thị giác:
   - `ocr_text`: Đọc CHÍNH XÁC 100% tất cả chữ viết, công thức toán học/hình học Oxyz, số hiệu, logo, biển báo, phụ đề trên ảnh. Giữ nguyên dấu tiếng Việt và ký hiệu toán học. Nếu không có chữ, trả về chuỗi rỗng `""`.
   - `description`: Viết 1 đoạn văn tiếng Việt hoàn chỉnh. Câu đầu tiên tóm tắt hành động/đối tượng trọng tâm; các câu tiếp theo mô tả chi tiết nhân vật, trang phục, hành động, đồ vật xung quanh, bối cảnh không gian, màu sắc và ánh sáng.
   - `objects`: Mảng danh sách các đồ vật, thực thể có mặt trong ảnh.
   - `tags`: Mảng các từ khóa tìm kiếm quan trọng bằng tiếng Việt và tiếng Anh.
3. Tổng hợp toàn bộ kết quả vào một đối tượng JSON với key là tên ảnh (frame_id, ví dụ "18316").
4. Sử dụng công cụ `write_to_file` để lưu kết quả vào file `data/gemini_output/{video_id}_descriptions.json` (ví dụ `data/gemini_output/V017_descriptions.json`).
5. Sau khi hoàn thành toàn bộ ảnh trong thư mục này, hãy báo cáo tổng số ảnh đã xử lý và DỪNG LẠI (không tự ý chạy sang thư mục video khác).
```

---

## 3. 💡 Quy Tắc Vàng Khi Phân Tích

1. **OCR Trung Thực 100%**:
   * Đọc chính xác từng ký tự, dấu ngoặc, tọa độ (ví dụ `A(4; -2; 2)`), công thức toán học.
   * Không tự ý thêm bớt từ ngữ không có trên ảnh.
2. **Description Có Cấu Trúc**:
   * *Câu 1 (Lead Sentence)*: Tóm tắt bức tranh toàn cảnh (phục vụ mô hình CLIP).
   * *Các câu tiếp theo*: Mô tả chi tiết các góc cạnh, màu sắc, vị trí vật thể (phục vụ mô hình Text Dense & Reranker).
3. **Tiết Kiệm Token & Tốc Độ Tối Đa**:
   * Việc dùng 1 trường `description` duy nhất giúp Antigravity đọc và xử lý nhanh gấp đôi, không bị lặp từ và dễ dàng nạp trực tiếp vào cơ sở dữ liệu vector.
