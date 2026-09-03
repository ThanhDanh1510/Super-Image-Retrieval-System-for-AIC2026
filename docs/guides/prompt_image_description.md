# 🎯 Hướng Dẫn Prompt Sinh Image Description Cho VLM (AIC2026)

Tài liệu này cung cấp Prompt chuẩn hóa đã được kiểm chứng để Antigravity / Gemini / Qwen2.5-VL sinh mô tả keyframe video đạt độ chính xác cao nhất cho cuộc thi AIC2026.

👉 **File prompt gốc:** [`tools/vlm_captioning/prompt_image_description.md`](../../tools/vlm_captioning/prompt_image_description.md)

---

## 1. Tại Sao Mô Tả Tiếng Việt Lại Tối Ưu Hơn?

1. **Đề thi 100% tiếng Việt bản địa:** Đề thi của ban tổ chức AIC được đặt hoàn toàn bằng tiếng Việt với các thuật ngữ văn hóa, giáo dục, ẩm thực và đời sống Việt Nam (*bánh tét, nước cốt dừa, mộc nhĩ, áo bà ba, nón lá, bến phà, bùng binh, thầy Đỗ Đức Anh, bài giảng Ngữ văn...*).
2. **Loại bỏ sai lệch dịch thuật (Translation Artifacts):** Dịch qua tiếng Anh thường làm mất sắc thái ngữ nghĩa hoặc dịch sai từ khóa then chốt (ví dụ: *mộc nhĩ* bị dịch thành *wood ear / black fungus*, *hạt nêm* thành *seasoning powder / MSG*, *áo bà ba* thành *traditional shirt / pajamas*).
3. **Phát huy tối đa sức mạnh của ViRanker:** Model rerank `BAAI/bge-reranker-v2-m3` là mô hình Cross-Encoder đa ngữ cao cấp. Khi câu query tiếng Việt được so sánh trực tiếp với mô tả tiếng Việt giàu thực thể, điểm tương đồng sẽ vượt trội hơn hẳn so với việc so sánh qua ngôn ngữ trung gian.
4. **Kết hợp sức mạnh Song ngữ (Bilingual Tags):** Bằng cách bổ sung thêm 8-12 tags tiếng Anh ở cuối, nhánh Visual CLIP vẫn bắt trọn các visual concepts toàn cầu.

---

## 2. Câu Lệnh / Prompt Ngắn Để Gọi Gemini Flash Subagents Sinh Caption Song Song

Để Antigravity tự động kích hoạt các Subagent chạy bằng mô hình **Gemini Flash** đọc ảnh trực tiếp và sinh caption chuẩn xác (không cần nạp GPU local Qwen2.5-VL), bạn chỉ cần gửi câu lệnh ngắn sau:

```text
Hãy chia các subagent (sử dụng model 'flash') đọc trực tiếp danh sách ảnh keyframe trong thư mục [ĐƯỜNG DẪN ẢNH HOẶC BATCH], tuân thủ đúng prompt tại tools/vlm_captioning/prompt_image_description.md để sinh mô tả chuẩn 3 khối (MÔ TẢ TỔNG QUAN, CHI TIẾT THỰC THỂ & VỊ TRÍ, TAGS SONG NGỮ).
```
