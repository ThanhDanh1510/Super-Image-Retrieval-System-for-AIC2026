---
name: spiderman-ui
description: Skill thiết kế và xây dựng giao diện Spider-Man Theme High-Density Web UI cho hệ thống AI Video Keyframe Retrieval AIC2026.
---

# 🕷️ SPIDER-MAN THEME HIGH-DENSITY WEB UI SKILL

Skill này cung cấp quy chuẩn thiết kế giao diện Web UI mang phong cách Spider-Man Dark Theme với mật độ hiển thị ảnh cao (High-Density Grid) cho bài thi AI Challenge HCMC 2026.

## 1. Phong cách Thiết kế (Design Tokens & Color Palette)
- **Nền chính**: Midnight Black (`#080b11`) kết hợp hiệu ứng đường tơ nhện mờ (`rgba(230, 36, 41, 0.08)`).
- **Màu nhấn chính (Primary Crimson Red)**: `#e62429` (Spider Red Glow `#e6242966`).
- **Màu nhấn phụ (Electric Cyan Blue)**: `#00d2ff` (Spider Suit Glow).
- **Màu cảnh báo / Accent**: Neon Yellow (`#ffd700`).
- **Typography**: Font chữ Google Fonts `Outfit` cho headings, `Roboto Mono` cho mã frame/timestamp/OCR.

## 2. Quy chuẩn Mật độ Hiển thị Ảnh (High-Density Grid Layout)
- Mặc định sử dụng lưới 6 cột (`grid-cols-6` / `repeat(auto-fill, minmax(230px, 1fr))`) để hiển thị từ 24 - 36+ bức ảnh trên một màn hình mà không cần cuộn trang nhiều.
- Cho phép toggle linh hoạt giữa 6 cột (Dày - High Density) và 4 cột (Vừa - Medium Density).

## 3. Các Tính Năng Tối Ưu Trải Nghiệm (UX Features)
- **Tab Swapping**: Chuyển đổi linh hoạt giữa 3 nhiệm vụ KIS, VQA, và TRAKE.
- **Voice Search 🎙️**: Tự động nhận diện giọng nói tiếng Việt bằng Web Speech API.
- **Image Search 🖼️**: Kéo thả ảnh để tìm kiếm ảnh tương đồng (Image-to-Image).
- **Giỏ Nộp Bài (Submission Drawer)**: Giúp người dùng click chọn các bức ảnh ưng ý, tự nhập đáp án VQA và bấm 1-Click xuất file CSV nộp BTC.
- **Lightbox Preview**: Click phóng to ảnh sắc nét kèm đầy đủ thông tin OCR/ASR.
