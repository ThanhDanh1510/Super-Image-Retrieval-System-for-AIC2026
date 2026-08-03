# 🎯 Super Image Retrieval System for AIC2026

Hệ thống truy vấn video keyframe real-time phục vụ cuộc thi **AI Challenge HCMC 2026**.

Hỗ trợ 4 dạng task chính:
- **KIS** (Known-Item Search) — tìm keyframe theo mô tả text/voice
- **VQA** (Visual Question Answering) — tìm cảnh và trả lời câu hỏi
- **TRAKE** (Temporal Retrieval) — tìm chuỗi sự kiện theo thứ tự thời gian
- **Image Search** — tìm keyframe tương đồng với ảnh upload

---

## 🏗️ Kiến trúc tổng quan

```
ingestion (Kaggle GPU) → aggregation (Local) → indexing (Qdrant DB) → serving (FastAPI Backend)
```

| Tầng | Mô tả | Môi trường chạy |
|---|---|---|
| `ingestion/` | Trích xuất keyframes, OCR, ASR và sinh Vector Embeddings | Kaggle GPU (T4/P100) |
| `aggregation/` | Gộp output của tất cả member thành 1 unified dataset | Máy Local |
| `indexing/` | Nạp dữ liệu & 3 bộ Vector Embeddings vào Qdrant DB | Docker / Local Qdrant |
| `serving/` | FastAPI REST API real-time cho web client / UI thi đấu | Server local / Staging |

---

## 🚀 Hướng Dẫn Cài Đặt & Khởi Chạy Dự Án Siêu Tốc Bằng `uv` (Cho Windows & Apple macOS)

Dự án chuẩn hóa sử dụng **`uv`** (Trình quản lý gói Python bằng Rust siêu nhanh, gấp 10-100 lần pip thông thường) giúp cài đặt môi trường chỉ trong vài giây.

---

### 1. Clone Repository (Dùng chung cho Win & Mac)

```bash
git clone https://github.com/ThanhDanh1510/Super-Image-Retrieval-System-for-AIC2026.git
cd Super-Image-Retrieval-System-for-AIC2026
```

---

### 2. Cài Đặt `uv`

#### 🪟 Trên Windows:
```powershell
pip install uv
```

#### 🍎 Trên macOS (Apple Silicon M1/M2/M3/M4 & Intel Macs):
```bash
brew install uv
```

---

### 3. Khởi Tạo Môi Trường Ảo `.venv` & Cài Đặt PyTorch GPU Acceleration

#### 🪟 Dành cho Windows (Có Card GPU NVIDIA RTX / GTX):
```powershell
# 1. Tạo môi trường ảo .venv
uv venv .venv

# 2. Kích hoạt môi trường ảo (PowerShell)
.\.venv\Scripts\Activate.ps1

# 3. Cài đặt PyTorch CUDA GPU cho NVIDIA Card (RTX 5060 Ti, 40xx, 30xx...)
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 4. Cài toàn bộ dependencies còn lại & package dự án
uv pip install transformers accelerate sentencepiece qdrant-client fastapi "uvicorn[standard]" pydantic pydantic-settings python-multipart pyyaml python-dotenv pandas numpy pillow scipy deep-translator -e .
```

#### 🍎 Dành cho macOS / Apple Silicon (M1/M2/M3/M4):
```bash
# 1. Tạo môi trường ảo .venv
uv venv .venv

# 2. Kích hoạt môi trường ảo
source .venv/bin/activate

# 3. Cài đặt toàn bộ dependencies (Tự động tối ưu PyTorch cho Apple Metal MPS GPU)
uv pip install torch torchvision torchaudio transformers accelerate sentencepiece qdrant-client fastapi "uvicorn[standard]" pydantic pydantic-settings python-multipart pyyaml python-dotenv pandas numpy pillow scipy deep-translator -e .
```

> 💡 **Mẹo:** Mỗi lần mở terminal mới để làm việc với dự án, hãy kích hoạt `.venv`:
> - Windows: `.\.venv\Scripts\Activate.ps1`
> - macOS: `source .venv/bin/activate`

---

### 4. Khởi Động Qdrant Vector Database (Bằng Docker)

Yêu cầu máy tính đã bật **Docker Desktop**.

#### 🪟 Trên Windows (PowerShell / CMD):
```powershell
# 🔹 Nếu là LẦN ĐẦU TIÊN chạy:
docker run -d -p 6333:6333 -p 6334:6334 -v ${PWD}/data/qdrant_storage:/qdrant/storage --name qdrant_aic2026 qdrant/qdrant

# 🔹 Nếu container ĐÃ TỒN TẠI (những lần sau):
docker start qdrant_aic2026
```

#### 🍎 Trên macOS (zsh / bash):
```bash
# 🔹 Nếu là LẦN ĐẦU TIÊN chạy:
docker run -d -p 6333:6333 -p 6334:6334 -v $(pwd)/data/qdrant_storage:/qdrant/storage --name qdrant_aic2026 qdrant/qdrant

# 🔹 Nếu container ĐÃ TỒN TẠI (những lần sau):
docker start qdrant_aic2026
```

#### 🧪 Kiểm Tra Qdrant Server:
```bash
curl.exe http://localhost:6333/
```

> 💡 *Nếu chưa bật Docker Desktop, hệ thống indexing sẽ tự động fallback sang **Local Embedded Qdrant** tại `./data/qdrant_db/` mà vẫn chạy mượt 100%!*

---

### 📂 Cấu Trúc Dữ Liệu Yêu Cầu Trong Thư Mục `data/`

Để hệ thống hoạt động chính xác (tổng hợp dữ liệu thô, indexing, và serving API), bạn cần tổ chức cấu trúc thư mục `data/` ở thư mục gốc của dự án theo định dạng dưới đây:

```text
data/
├── image/                     # Chứa các folder ảnh keyframe thô phân theo Group
│   ├── L21/L21/V001/*.webp    # Cấu trúc thư mục ảnh: data/image/{Group}/{Group}/{Video}/*.webp
│   ├── L22/L22/V001/*.webp
│   └── ...
├── OCR/                       # Chứa các file JSON kết quả trích xuất chữ viết OCR
│   ├── L21/L21_V001.json      # File JSON OCR của từng video (L21_V001.json, L21_V002.json...)
│   ├── L22/L22_V001.json
│   └── ...
├── ASR/                       # Chứa file kết quả nhận dạng giọng nói ASR
│   └── asr.json               # File JSON chứa toàn bộ audio transcripts kèm start/end frame
├── unified/                   # [QUAN TRỌNG] Dataset hợp nhất và Vector Embeddings (cho Indexing)
│   ├── scene_metadata.parquet # Metadata toàn bộ keyframes (scene_id, video_id, timestamp_mid...)
│   ├── ocr_text.parquet       # Text OCR của từng scene (scene_id, video_id, ocr_text)
│   ├── asr_text.parquet       # Transcript ASR của từng scene (scene_id, video_id, asr_text)
│   ├── image_vectors.npy      # Dense vectors của keyframes (CLIP, kích thước: N x 1024)
│   ├── ocr_vectors.npy        # Dense vectors của OCR text (gte-Qwen2, kích thước: N x 1536)
│   ├── asr_vectors.npy        # Dense vectors của ASR transcript (gte-Qwen2, kích thước: N x 1536)
│   └── scene_ids.txt          # File text chứa danh sách scene_id tương ứng với index của các file .npy
└── unified_raw/               # Thư mục đầu ra được tạo tự động sau khi chạy aggregation
```

---

### 5. Nạp Dữ Liệu Unified Dataset Vào Qdrant DB (Indexing)

Tải file `unified_dataset.zip` hoặc tập dữ liệu unified hoàn chỉnh (gồm 3 file `.parquet`, 3 file `.npy` và file `scene_ids.txt`) và đặt vào thư mục `./data/unified/` theo đúng cấu trúc thư mục ở trên.

Chạy lệnh nạp toàn bộ 108,413 scenes vào Vector DB:

```bash
python -m indexing.run_index \
    --dataset-dir ./data/unified \
    --collection collection_v1 \
    --create-collection
```

Gắn alias `production` cho collection:
```bash
python -m indexing.collection_alias --action set --collection collection_v1
```

---

### 6. Khởi Chạy FastAPI Server (Serving API)

```bash
# Chạy API Server ở Dev mode (tự động reload khi sửa code)
python -m uvicorn serving.api.main:app --reload --host 0.0.0.0 --port 8000
```

* 🌐 **Swagger UI Interactive API Docs:** `http://localhost:8000/docs`
* 🩺 **Healthcheck API:** `http://localhost:8000/health`

---

## 📁 Cấu Trúc Thư Mục Dự Án

```
Super-Image-Retrieval-System-for-AIC2026/
├── .venv/                # Môi trường ảo Python được quản lý bởi uv (đã .gitignore)
├── configs/              # Cấu hình tập trung (model versions, shard, fusion)
├── ingestion/            # Pipeline trích xuất keyframes & sinh vector trên Kaggle
│   └── notebooks/        # Notebooks Kaggle (04_kaggle_generate_all_embeddings.ipynb)
├── aggregation/          # Gộp output từ các member thành unified raw
├── indexing/             # Quản lý schema, alias và index data vào Qdrant
├── serving/              # FastAPI API (controller → service → repo)
├── shared/               # Code dùng chung (schemas, model wrappers, text normalizer)
├── data/                 # Thư mục lưu dữ liệu local (unified, qdrant_storage, hf_cache,...)
├── scratch/              # Thư mục script chạy thử nghiệm/debug
├── AGENT_CONTEXT.md      # Quy chuẩn và ngữ cảnh dự án cho AI Assistant
├── pyproject.toml
└── README.md
```

---

## 🤖 AI Models Trong Baseline

| Model | Dùng cho | Dim | Gia tốc phần cứng |
|---|---|---|---|
| **BEiT-large + CLIP ViT-L/14** | Image embedding | 1024 | CUDA (Nvidia) / MPS (Apple Silicon) / CPU |
| **gte-Qwen2-1.5B-instruct** | Text embedding (OCR + ASR) | 1536 | CUDA (Nvidia) / MPS (Apple Silicon) / CPU |
| **PaddleOCR v6** | Nhận dạng chữ viết | — | CUDA / CPU |
| **PhoWhisper-medium** | Nhận dạng giọng nói tiếng Việt | — | CUDA / MPS / CPU |
| **TransNetV2** | Phân đoạn video shot boundary | — | CUDA / CPU |

---

