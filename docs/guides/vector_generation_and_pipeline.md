# ⚡ Quy Trình Trích Xuất Dữ Liệu & Sinh Vector Embeddings (Data & Vector Pipeline)

Tài liệu hướng dẫn toàn diện quy trình chạy offline/local để trích xuất văn bản (OCR), âm thanh (ASR), sinh mô tả hình ảnh (VLM Captioning) và tạo ma trận vector đa nhánh (Multi-Branch Embeddings) phục vụ hệ thống.

---

## 📊 Tổng Quan Dữ Liệu & Các Nhánh Vector

Tất cả các file vector và metadata hợp nhất được lưu tại thư mục: `data/unified/`

| Loại Dữ Liệu | File Nguồn / Parquet | File Vector Output | Kích Thước | Model Sử Dụng |
|---|---|---|---|---|
| **Visual Image** | Keyframe images (`.webp`) | `image_vectors.npy` | `(88899, 1024)` | `openai/clip-vit-large-patch14` |
| **OCR Text** | `ocr_text.parquet` | `ocr_vectors.npy` | `(88899, 1536)` | `Alibaba-NLP/gte-Qwen2-1.5B-instruct` |
| **ASR Speech** | `asr_text.parquet` | `asr_vectors.npy` | `(88899, 1536)` | `Alibaba-NLP/gte-Qwen2-1.5B-instruct` |
| **Image Description** | `scene_descriptions.parquet` | `description_embeddings.npy` & `.pt` | `(88899, 1024)` | `BAAI/bge-large-en-v1.5` |
| **Scene Metadata** | `scene_metadata.parquet` | `scene_ids.txt` (88,899 dòng) | Metadata | Khớp 1-1 theo thứ tự hàng |

---

## 🚀 1. Hướng Dẫn Chạy Từng Nhánh Vector

### Nhánh 1: Sinh Mô Tả VLM & Tạo Vector Image Descriptions (`BGE-Large`)
Đây là nhánh trọng tâm giúp hệ thống hiểu sâu ngữ nghĩa chi tiết của từng scene video.

1. **Bước 1 (VLM Captioning):** Sinh mô tả hình ảnh từ model VLM (Gemini / Qwen2-VL):
   ```powershell
   # Sinh mô tả tự động cho các batch video:
   python -m tools.vlm_captioning.generate_descriptions --batch-size 16
   ```
2. **Bước 2 (Mã hóa Vector BGE):** Đọc `data/unified/scene_descriptions.parquet` và xuất ma trận `description_embeddings.npy` & `.pt`:
   ```powershell
   python -m tools.embeddings.build_description_embeddings
   ```
   *(Tự động chạy FP16 batch inference trên GPU CUDA, tốc độ ~800 scenes/giây).*

3. **Cập nhật riêng cho batch L25 (nếu có bổ sung):**
   ```powershell
   python tools/update_l25_descriptions.py
   ```

---

### Nhánh 2: Trích Xuất & Sinh Vector Text OCR (On-Screen Text)
Dành cho văn bản xuất hiện trong video (slide bài giảng, biển hiệu, tiêu đề):

1. **Trích xuất chữ từ ảnh (PaddleOCR GPU):**
   ```powershell
   python -m ingestion.skeletons.01b_text_extraction --mode ocr
   ```
2. **Sinh vector Dense Text (`gte-Qwen2-1.5B` - 1536d):**
   ```powershell
   python -m ingestion.skeletons.01c_embedding_generation --target ocr
   ```

---

### Nhánh 3: Trích Xuất & Sinh Vector Giọng Nói ASR (Speech-to-Text)
Dành cho lời giảng, thuyết minh và hội thoại trong video:

1. **Trích xuất âm thanh & chuyển giọng nói thành chữ (PhoWhisper-medium):**
   ```powershell
   python -m ingestion.skeletons.01b_text_extraction --mode asr
   ```
2. **Sinh vector Dense ASR (`gte-Qwen2-1.5B` - 1536d):**
   ```powershell
   python -m ingestion.skeletons.01c_embedding_generation --target asr
   ```

---

### Nhánh 4: Sinh Vector Hình Ảnh Visual CLIP (`CLIP ViT-L/14`)
Dành cho tìm kiếm visual concept và Image-to-Image search:

```powershell
python -m ingestion.skeletons.01c_embedding_generation --target image
```

---

## 🔍 2. Kiểm Tra & Đánh Giá Cấu Trúc `image_description`

Cấu trúc chuẩn của bảng mô tả hình ảnh trong hệ thống:
```text
data/unified/scene_descriptions.parquet
├── scene_id        : str        (Mã định danh duy nhất của scene, ví dụ: L21_V001_S006)
├── video_id        : str        (Mã video, ví dụ: L21_V001)
├── frame_id        : int64      (Frame index đại diện)
├── timestamp_mid   : float64    (Mốc thời gian giây ở giữa scene)
└── description     : str        (Mô tả chi tiết bằng tiếng Việt hoặc tiếng Anh)
```

### Tiêu Chuẩn Đạt Chuẩn 100%:
1. **Khớp thứ tự 1-1:** Thứ tự dòng của `scene_descriptions.parquet` phải khớp chính xác tuyệt đối 100% với `scene_ids.txt` và hàng thứ $i$ trong `description_embeddings.npy`.
2. **L2-Normalized:** Toàn bộ vector trong `description_embeddings.npy` đều được chuẩn hóa L2 ($\|v\|_2 = 1.0$) để phép tính cosine similarity tương đương với phép nhân vô hướng `torch.matmul`.
3. **Độ dài phong phú:** Độ dài mô tả trung bình từ 150 - 400 ký tự, mô tả đầy đủ chủ thể, hành động, bối cảnh không gian, màu sắc và đạo cụ.

---

## 🔄 3. Nạp Toàn Bộ Vector Vào Qdrant (Bulk Indexing)

Sau khi hoàn tất tạo các file vector trong `data/unified/`, tiến hành nạp vào Qdrant:

```powershell
# Tạo mới collection và index toàn bộ 88,899 keyframes:
python -m indexing.run_index --dataset-dir ./data/unified --collection aic2026_clean_v3 --create-collection
```
