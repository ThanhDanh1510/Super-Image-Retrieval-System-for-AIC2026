# 🛠️ AIC2026 Tools & Utilities

Thư mục chứa các công cụ hỗ trợ lọc dữ liệu, trích xuất đặc trưng, tiền xử lý và kiểm tra hệ thống AIC2026.

---

## 📁 Cấu trúc thư mục

```text
tools/
├── image_filter/                       # Web UI trực quan hóa & lọc ảnh rác (Port 8501)
│   ├── server.py                       # FastAPI server phục vụ tool lọc ảnh
│   ├── app.js                          # Client logic (Grid render, phím tắt, duplicate scan)
│   ├── index.html                      # Giao diện Spider-Man Cyberpunk
│   └── style.css                       # Stylesheet
├── data_prep/                          # Tiền xử lý dữ liệu video & keyframe
│   ├── fill_gap_keyframes.py           # Bù keyframe vào các khoảng trống timeline (>5s)
│   ├── calibrate_all_video_fps.py      # Hiệu chuẩn FPS thực tế cho video
│   ├── remap_l_json.py                 # Ánh xạ và chuẩn hóa file JSON metadata từ BTC
│   └── merge_additional_filters.py     # Hợp nhất danh sách lọc vào filtered_keyframes.json
├── ocr/                                # Pipeline nhận dạng chữ OCR trên GPU
│   ├── multiprocess_gpu_ocr.py         # Engine OCR đa tiến trình trên CUDA
│   ├── ocr_batch_dataset.py            # Chạy OCR theo từng batch dữ liệu
│   └── run_all_ocr_batches.py          # Runner tự động chạy toàn bộ L21 -> L30
├── embeddings/                         # Sinh vector biểu diễn ngữ nghĩa
│   └── build_description_embeddings.py # Sinh vector text GTE/Qwen2 từ scene_descriptions.parquet
├── download_models.py                  # Tải trước các mô hình AI từ HuggingFace về máy
├── report_all_batches_coverage.py      # Thống kê tỷ lệ bao phủ dữ liệu (Keyframe, OCR, ASR, VLM)
├── test_user_query.py                  # CLI kiểm tra nhanh câu truy vấn mẫu qua Search Pipeline
└── verify_qdrant_db.py                 # Kiểm tra số lượng point và trạng thái Qdrant DB
```

---

## 🚀 Hướng dẫn sử dụng từng công cụ

### 1. 🖼️ Web Tool Lọc Ảnh & Trùng Lặp (Image Filter Tool)
Khởi chạy giao diện Web UI tại cổng `8501` để xem, lọc ảnh đơn sắc/logo/rác và quét ảnh tương đồng:
```bash
python -m tools.image_filter.server
# Hoặc: make filter-ui
```
Truy cập trình duyệt: **http://localhost:8501**

---

### 2. 🎬 Bù Keyframe & Hiệu Chuẩn FPS (`data_prep`)
* **Dò và bù các khoảng trống timeline:**
  ```bash
  python -m tools.data_prep.fill_gap_keyframes --data-dir ./data --gap-threshold 5.0
  ```
* **Hiệu chuẩn FPS video gốc:**
  ```bash
  python -m tools.data_prep.calibrate_all_video_fps --data-dir ./data
  ```
* **Chuẩn hóa file JSON metadata từ BTC:**
  ```bash
  python -m tools.data_prep.remap_l_json --data-dir ./data
  ```

---

### 3. 📝 Chạy Batch OCR trên GPU (`ocr`)
* **Chạy OCR tự động cho toàn bộ các batch L21 -> L30:**
  ```bash
  python -m tools.ocr.run_all_ocr_batches
  ```
* **Chạy OCR cho 1 batch cụ thể (ví dụ L21):**
  ```bash
  python -m tools.ocr.ocr_batch_dataset --batch L21 --device cuda
  ```

---

### 4. 🧠 Sinh Vector từ Mô tả Cảnh (`embeddings`)
* **Trích xuất GTE/Qwen2 embeddings:**
  ```bash
  python -m tools.embeddings.build_description_embeddings --data-dir ./data
  ```

---

### 5. 🔍 Kiểm tra & Tiện ích Hệ thống
* **Thống kê độ phủ dữ liệu toàn diện:**
  ```bash
  python -m tools.report_all_batches_coverage
  ```
* **Kiểm tra trạng thái Qdrant Vector DB:**
  ```bash
  python -m tools.verify_qdrant_db
  ```
* **Test nhanh câu query qua terminal:**
  ```bash
  python -m tools.test_user_query --query "học sinh đứng chào cờ"
  ```
