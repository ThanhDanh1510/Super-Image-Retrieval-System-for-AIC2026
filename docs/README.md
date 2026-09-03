# 📚 Danh Mục Tài Liệu Kỹ Thuật (Documentation Index)

Chào mừng bạn đến với trung tâm tài liệu kỹ thuật của hệ thống **Super Image Retrieval System for AIC2026**.

---

## 📂 Mục Lục Tài Liệu

### 1. 🏗️ Kiến Trúc Hệ Thống (Architecture)
* [**Kiến Trúc Tổng Thể & Model Specs**](architecture/system_architecture.md): Sơ đồ luồng dữ liệu đa nhánh, Hybrid In-Memory Engine, Dual Rerankers và độ trễ truy vấn.
* [**Tài Liệu API Reference**](architecture/api_reference.md): Quy chuẩn endpoint, headers, schemas và mẫu JSON Request/Response cho KIS, VQA, TRAKE, Similar Image.

### 2. 🧠 Thuật Toán Đột Phá (Algorithms)
* [**Thuật Toán KIS (Known-Item Search)**](algorithms/kis_retrieval.md): Cơ chế In-Memory Matmul GPU, Exact-Match Tiered Scoring, Entity Coverage Scorer, và Temporal Diversity Filter.
* [**Thuật Toán Dòng Thời Gian TRAKE (DANTE DP)**](algorithms/trake_temporal.md): Quy hoạch động DANTE, Sub-event splitting, Targeted Dense Frame Sampling qua OpenCV.

### 3. 🛠️ Hướng Dẫn & Bộ Công Cụ (Guides)
* [**Quy Trình Sinh Vector & Dữ Liệu (ASR, OCR, CLIP, Descriptions)**](guides/vector_generation_and_pipeline.md): Hướng dẫn offline pipeline trích xuất văn bản, audio, VLM captioning và sinh vector đa nhánh.
* [**Bộ Công Cụ & Pipeline Tiện Ích**](guides/tools_and_pipelines.md): Hướng dẫn Web UI lọc ảnh trùng lặp (`tools/image_filter`), kiểm toán độ phủ video, và quản trị Zero-Downtime Alias Qdrant.
* [**Thể Lệ & Định Dạng Nộp Bài AIC2026**](guides/contest_rules.md): Quy chuẩn tác vụ, định dạng file CSV nộp bài chính thức.

### 4. 📑 Báo Cáo & Đề Xuất Kỹ Thuật (Proposals & Research)
* [**Thư Mục Nghiên Cứu & Đề Xuất**](proposals/): Lưu trữ các bản nháp nghiên cứu, phân tích tối ưu hóa DANTE DP và benchmark đánh giá chi tiết.
