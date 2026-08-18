# 🧠 Prompt & Hướng Dẫn: Trích Xuất OCR & Mô Tả Ảnh Bằng Gemini Trong Antigravity

Tài liệu này cung cấp **Prompt chuẩn hóa 1-Click** để sử dụng trực tiếp với **Gemini trong Antigravity**. Antigravity sẽ tự động duyệt toàn bộ ảnh trong một folder video, đọc trực tiếp từng ảnh bằng công cụ thị giác, phân tích trích xuất dữ liệu chuẩn JSON rồi tự động dừng lại.

---

## 1. 📋 Cấu Trúc JSON Siêu Tinh Gọn (Chỉ 2 Trường Cốt Lõi)

Dữ liệu đầu ra được tối giản hóa 100% để khớp trực tiếp với cơ sở dữ liệu `unified` (Parquet/Qdrant):

```json
{
  "18316": {
    "ocr_text": "Câu 4: Trong không gian Oxyz, cho điểm A(4; -2; 2) và mặt cầu (S): x^2 + y^2 + z^2 = 4. Biết B, C, D là ba điểm phân biệt trên (S)... Chọn D.",
    "description": "Thầy giáo đang giảng bài toán hình học không gian Oxyz trên bảng đen. Bảng viết chi tiết các bước tính bán kính r của đường tròn ngoại tiếp tam giác BCD, bên góc phải có sơ đồ tọa độ và mặt cầu minh họa, không gian phòng học sáng rõ."
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
   - `ocr_text`: Đọc CHÍNH XÁC 100% tất cả chữ viết nội dung trong ảnh (bài giảng, bảng viết, công thức toán học/hình học Oxyz, biển hiệu cửa hàng, băng rôn, số xe, phụ đề nội dung, nhãn mác). Giữ nguyên dấu tiếng Việt và ký hiệu toán học.
     ⚠️ QUY TẮC LOẠI TRỪ QUAN TRỌNG: TUYỆT ĐỐI KHÔNG đọc logo/watermark của đài truyền hình, kênh tin tức hoặc báo chí ở các góc màn hình (ví dụ: "Thanh Niên", "Báo Thanh Niên", "VTV", "VTV1", "HTV", "Tuổi Trẻ", "VTC", "VOV", logo đồng hồ góc đài). Nếu không có chữ nội dung nào khác, trả về chuỗi rỗng `""`.
   - `description`: Viết 1 đoạn văn tiếng Việt hoàn chỉnh và chi tiết. Câu đầu tiên tóm tắt hành động/đối tượng trọng tâm của ảnh; các câu tiếp theo mô tả đầy đủ nhân vật, trang phục, hành động, các đồ vật xung quanh, bối cảnh không gian, màu sắc chủ đạo và ánh sáng.
3. Tổng hợp toàn bộ kết quả vào một đối tượng JSON với key là tên ảnh (frame_id, ví dụ "18316").
4. Sử dụng công cụ `write_to_file` để lưu kết quả vào file `data/gemini_output/{video_id}_descriptions.json` (ví dụ `data/gemini_output/V017_descriptions.json`).
5. Sau khi hoàn thành toàn bộ ảnh trong thư mục này, hãy báo cáo tổng số ảnh đã xử lý và DỪNG LẠI (không tự ý chạy sang thư mục video khác).
```

---

## 3. 💡 Quy Tắc Vàng Khi Phân Tích

1. **OCR Nội Dung & Loại Bỏ Logo Đài**:
   * ✅ **CẦN ĐỌC**: Bảng chữ bài giảng, tọa độ toán học `A(4; -2; 2)`, công thức, biển hiệu đường phố, phụ đề cuộc phỏng vấn, tên nhân vật được phỏng vấn.
   * ❌ **KHÔNG ĐỌC**: Logo nhà đài/kênh báo chí cố định ở các góc màn hình (ví dụ: chữ *"Thanh Niên"*, *"VTV1"*, *"HTV7"*, *"Tuổi Trẻ Online"*...). Việc loại trừ này giúp tránh nhầm lẫn khi người dùng tìm kiếm từ khóa "thanh niên" (người trẻ tuổi).
2. **Description Giàu Chi Tiết & Có Cấu Trúc**:
   * *Câu đầu tiên*: Tóm tắt bức tranh toàn cảnh (tối ưu cho mô hình Text-to-Image CLIP).
   * *Các câu tiếp theo*: Mô tả chi tiết các vật thể, màu sắc, vị trí tương đối (tối ưu cho mô hình Dense gte-Qwen2 và ViRanker).
3. **Tiết Kiệm Token & Tốc Độ Tối Đa**:
   * Định dạng 2 trường `ocr_text` và `description` giúp hệ thống nạp thẳng vào cơ sở dữ liệu vector mà không cần tiền xử lý phức tạp.
