# Super Image Retrieval System for AIC2026

## 🎯 Tổng quan dự án (Project Overview)
- **Tên:** Super Image Retrieval System for AIC2026
- **Mục đích:** Truy xuất video keyframe thời gian thực cho AI Challenge HCMC 2026
- **4 tasks:** KIS (Known-Item Search), VQA (Visual Question Answering), TRAKE (Temporal Retrieval), Image Search
- **Kiến trúc:** ingestion (Kaggle GPU) → aggregation (Local) → indexing (Qdrant DB) → serving (FastAPI)
- **Package manager:** `uv` (Python virtual environment)
- **Python** >= 3.10

## 🚀 Hướng dẫn cài đặt nhanh (Quick Start)

### Yêu cầu hệ thống (Prerequisites)
- Python 3.10+
- NVIDIA GPU với CUDA 12.4+ (khuyến nghị: RTX 3060/4060/5060 Ti với 8GB+ VRAM)
- Docker Desktop (để chạy Qdrant vector database)
- Package manager `uv`

### Bước 1: Clone & Cài đặt thư viện
```powershell
git clone https://github.com/ThanhDanh1510/Super-Image-Retrieval-System-for-AIC2026.git
cd Super-Image-Retrieval-System-for-AIC2026

# Cài đặt uv
pip install uv

# Tạo môi trường ảo
uv venv .venv
.venv\Scripts\Activate.ps1

# Cài đặt PyTorch với CUDA
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Cài đặt toàn bộ dependencies
uv pip install -e ".[all]"
```

### Bước 2: Khởi động Qdrant (Docker)
```powershell
docker compose up -d
# Kiểm tra:
curl.exe http://localhost:6333/
```
Lưu ý: Dự án có sẵn `docker-compose.yml`. Nếu không có Docker, hệ thống sẽ tự động fallback sang Local Embedded Qdrant tại `./data/qdrant_db/`.

### Bước 3: Tải Dữ liệu
📥 Tải data tại: [Google Drive - TBD]()

Sau khi tải về, giải nén sao cho cấu trúc thư mục như sau:
```text
data/
├── image/           # Keyframes: data/image/{Group}/{Group}/{Video}/*.webp
├── unified/         # Unified dataset + embeddings
│   ├── scene_metadata.parquet
│   ├── ocr_text.parquet
│   ├── asr_text.parquet
│   ├── image_vectors.npy    # N x 1024 (CLIP)
│   ├── ocr_vectors.npy      # N x 1536 (gte-Qwen2)
│   ├── asr_vectors.npy      # N x 1536 (gte-Qwen2)
│   └── scene_ids.txt
├── OCR/             # JSON OCR per video
├── ASR/             # ASR transcripts
└── ocr_output/      # Newest OCR parquets per batch
```

### Bước 4: Cấu hình Môi trường
```powershell
copy .env.example .env
```
Bảng các biến môi trường quan trọng:

| Biến | Mặc định | Mô tả |
|---|---|---|
| AIC_QDRANT_URL | http://localhost:6333 | URL của server Qdrant |
| AIC_QDRANT_COLLECTION_ALIAS | production | Collection alias dùng cho query |
| AIC_DEVICE | cuda | cuda / cpu |
| AIC_PRELOAD_MODELS | false | true = khởi động trước tất cả model lúc startup |
| AIC_DATA_DIR | ./data | Thư mục chứa dữ liệu chung |
| AIC_KEYFRAME_DIR | ./data/image | Thư mục chứa video keyframes |

### Bước 5: Index Dữ liệu vào Qdrant
```powershell
python -m indexing.run_index --dataset-dir ./data/unified --collection aic2026_v1 --create-collection
```
Quá trình này mất khoảng ~7 phút. Nó sẽ tạo collection và thiết lập alias `production`.

### Bước 6: Tải AI Models (Pre-download)

Tất cả models được tải tự động từ HuggingFace Hub lần đầu chạy. Để tránh chờ lần đầu, nên tải trước:

```powershell
# Cách 1: Dùng script (khuyên dùng)
python tools/download_models.py

# Cách 2: Makefile
make download-models
```

**Danh sách models (~12GB):**

| Model | Mục đích | Device | RAM |
|---|---|---|---|
| **CLIP ViT-L/14** | Image embedding | GPU | ~1.7GB |
| **gte-Qwen2-1.5B** | Text embedding (OCR+ASR) | GPU | ~3GB |
| **bge-reranker-v2-m3** | Text rerank | GPU | ~2.3GB |
| **PhoWhisper-medium** | ASR tiếng Việt | CPU | ~1.5GB |
| **bge-large-en-v1.5** | Hybrid search | CPU | ~2GB |

> 💡 Models được cache tại `~/.cache/huggingface/` (hoặc `data/hf_cache/` nếu đặt `HF_HOME`).

### Bước 7: Khởi động Server
```powershell
python start_api.py
```
- **Web UI:** http://localhost:8000/ui/index.html
- **Swagger:** http://localhost:8000/docs
- **Health:** http://localhost:8000/health

## 🏗️ Kiến trúc Hệ thống (Architecture)
Pipeline sử dụng 5 loại vector chính để tìm kiếm:
- **image_dense** (1024, CLIP ViT-L/14)
- **ocr_dense** (1536, gte-Qwen2-1.5B)
- **asr_dense** (1536, gte-Qwen2-1.5B)
- **ocr_sparse** (BM42)
- **asr_sparse** (BM42)

Sử dụng Weighted RRF (k=60, prefetch=200), rerank với bge-reranker-v2-m3 (text, luôn bật, GPU) + jina-reranker-m0 (crossmodal, tắt mặc định).

## 📂 Cấu trúc Dự án (Project Structure)
```text
Super-Image-Retrieval-System-for-AIC2026/
├── configs/              # Centralized YAML configs (model_versions, fusion_config)
├── ingestion/            # Kaggle GPU pipeline (keyframe extraction, OCR, ASR, embeddings)
│   └── skeletons/        # Modular scripts: 01a keyframe, 01b text, 01c embeddings, 02 export
├── aggregation/          # Merge shard outputs into unified dataset
├── indexing/             # Qdrant schema + bulk indexing + alias management
├── serving/              # FastAPI REST API (strict layering: controller → service → repo)
│   └── api/
│       ├── controller/   # Route handlers
│       ├── service/      # Business logic + reranker
│       ├── repo/         # Qdrant client abstraction
│       ├── core/         # Config, ModelRegistry, middleware
│       └── schema/       # Pydantic request/response models
├── shared/               # Cross-module contracts: schemas, utils, model wrappers, OCR
├── eval/                 # Offline benchmark & metrics
├── tests/                # Pytest unit + integration tests
├── tools/                # Utility scripts (GPU OCR batch, VLM description, metadata fix)
├── web/                  # Spider-Man themed Web UI (index.html, app.js, style.css)
├── docs/                 # Reference documentation
├── docker-compose.yml    # Qdrant container setup
├── start_api.py          # Entry point: run server in-process
├── Makefile              # Common commands (make serve / test / lint)
├── pyproject.toml        # Dependencies (uv pip install -e ".[all]")
└── .env.example          # Environment variable template
```

## 🔌 API Endpoints

| Endpoint | Body | Mô tả |
|---|---|---|
| `POST /api/v1/kis` | `{query, input_type: "text"\|"voice", top_k (1-200), use_crossmodal_reranker?, user_choices?}` | Known-Item Search |
| `POST /api/v1/vqa` | `{query, question, input_type, top_k, use_crossmodal_reranker?, user_choices?}` | Visual QA — trả `results` + `answers` |
| `POST /api/v1/trake` | `{query, num_events (2-10), input_type, top_k}` | Temporal Retrieval — trả `list[list[CandidateResult]]` |
| `POST /api/v1/similar-image` | multipart `file` + query param `top_k` | Image Search |

Tất cả response trả về `CandidateResult`: `video_id`, `scene_id`, `frame_id`, `timestamp_start/mid/end`, `keyframe_path`, `ocr_text`, `asr_text`, `score`.

**Tham số `use_crossmodal_reranker`:** Mặc định `false`. Khi bật (`true`), Jina Reranker M0 chạy trên GPU để tăng độ chính xác (tốc độ chậm hơn ~8s). Text reranker (bge-reranker-v2-m3) **luôn chạy** vì rất nhẹ (~0.2s).

---

## 🧠 AI Models

Model **tự động tải về** từ HuggingFace Hub lần đầu được dùng (lazy-load).

| Model | Mục đích | HuggingFace | Thiết bị | VRAM/RAM |
|---|---|---|---|---|
| **CLIP ViT-L/14** | Image embedding (dim 1024) | `openai/clip-vit-large-patch14` | GPU | ~1.7GB |
| **gte-Qwen2-1.5B** | Text embedding OCR+ASR (dim 1536) | `Alibaba-NLP/gte-Qwen2-1.5B-instruct` | GPU | ~3GB |
| **bge-reranker-v2-m3** | Text rerank (luôn bật) | `BAAI/bge-reranker-v2-m3` | GPU | ~2.3GB |
| **jina-reranker-m0** | Crossmodal rerank (tắt mặc định) | `jinaai/jina-reranker-m0` | GPU | ~2.5GB |
| **PhoWhisper-medium** | ASR tiếng Việt | `vinai/PhoWhisper-medium` | CPU | ~1.5GB |
| **bge-large-en-v1.5** | Description hybrid search | `BAAI/bge-large-en-v1.5` | CPU | ~2GB |

Tất cả phiên bản model được khai báo tập trung tại `configs/model_versions.yaml` — **single source of truth** cho cả ingestion lẫn serving.

**Pre-download tất cả models (khuyên dùng — tổng ~15GB):**

Thay vì tự viết code, bạn có thể chạy 1 dòng lệnh duy nhất để script tự động parse đúng model versions từ `configs/model_versions.yaml` và tải về:

```powershell
# Cách 1: Sử dụng Makefile shortcut
make download-models

# Cách 2: Chạy trực tiếp script
python tools/download_models.py
```

---

## 🕸️ Web UI

Giao diện Web UI chủ đề **Spider-Man** mật độ cao tại `http://localhost:8000/ui/index.html`:

- 🎨 **Dark theme** với neon accents đỏ/xanh
- 📑 **Multi-task tabs**: KIS, VQA, TRAKE
- 🔍 **Tìm kiếm**: nhập text, ghi âm giọng nói, upload ảnh
- ⚙️ **Controls**: chọn nhánh (All/Embedding/OCR/ASR), Top-K, nút **🚀 Reranker** bật/tắt, thanh chỉnh mật độ card
- 🧺 **Giỏ nộp bài**: lưu keyframe, xuất CSV chính thức, hỗ trợ phím tắt (`C`)

---

## 📁 Indexing & Alias Management

### Nạp dữ liệu vào Qdrant

```powershell
# Tạo collection mới + nạp dữ liệu + gắn alias production
python -m indexing.run_index --dataset-dir ./data/unified --collection aic2026_v1 --create-collection

# Chỉ nạp thêm (collection đã tồn tại)
python -m indexing.run_index --dataset-dir ./data/unified --collection aic2026_v1

# Nạp xong nhưng KHÔNG đổi alias (để verify trước)
python -m indexing.run_index --dataset-dir ./data/unified --collection aic2026_v1 --skip-alias
```

### Quản lý Alias (Zero-downtime swap)

```powershell
# Xem collection nào đang ở production
python -m indexing.collection_alias --action get

# Chuyển production → collection mới
python -m indexing.collection_alias --action set --collection aic2026_v2

# Rollback về collection cũ
python -m indexing.collection_alias --action rollback --collection aic2026_v1
```

> 💡 **Tại sao dùng alias?** Khi update dữ liệu: index vào collection mới → verify → đổi alias. Nếu lỗi: rollback trong 1 lệnh, không cần re-index. Server không bao giờ downtime vì alias đổi atomic.

---

## ⚙️ Biến Môi Trường (`.env`)

Copy `.env.example` → `.env`. Các biến quan trọng:

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `AIC_QDRANT_URL` | `http://localhost:6333` | URL Qdrant server |
| `AIC_QDRANT_COLLECTION_ALIAS` | `production` | Alias collection dùng để query |
| `AIC_DEVICE` | `cuda` | `cuda` / `mps` / `cpu` |
| `AIC_PRELOAD_MODELS` | `false` | `true` = warmup toàn bộ model lúc startup |
| `AIC_MODEL_UNLOAD_TTL_S` | `0` | Unload model sau N giây idle (`0` = không unload) |
| `AIC_DATA_DIR` | `data/` | Thư mục chứa dữ liệu chung |
| `AIC_KEYFRAME_DIR` | `data/image` | Thư mục chứa ảnh keyframe |
| `AIC_CORS_ORIGINS` | `["*"]` | Allowed origins |

---

## 🧪 Test & Code Quality

```powershell
# Unit tests (không cần server):
pytest tests/ -v

# Hoặc dùng Makefile:
make test          # Chạy pytest
make lint          # Ruff linter
make lint-fix      # Ruff auto-fix
make typecheck     # Mypy type checker
```

> 💡 Gõ `make help` để xem toàn bộ lệnh có sẵn.

---

## 📊 Hiệu năng (Performance)

| Bước | Thời gian | Device |
|---|---|---|
| enhance_query (dịch EN/VI) | ~0.4s | API call |
| encode_text+image | ~0.8s | GPU |
| qdrant_search (RRF fusion) | ~0.3s | CPU |
| bge-reranker-v2-m3 (luôn bật) | ~0.2s | GPU |
| jina-reranker-m0 (tắt mặc định) | ~8s | GPU |
| **Tổng (không Jina)** | **~2s** | — |
| **Tổng (có Jina)** | **~10s** | — |

**GPU Memory:** ~7.9GB (không Jina) / ~10.4GB (có Jina)

---

## 🔧 Khắc phục sự cố (Troubleshooting)

### Docker chưa chạy / Qdrant không kết nối
```powershell
docker compose up -d
curl.exe http://localhost:6333/
```

### Collection not found
Cần chạy indexing trước (xem phần **Indexing & Alias Management**).

### GPU Out of Memory
Đã tối ưu — chỉ CLIP + Qwen2 + bge-reranker trên GPU (~7-9.5GB), còn lại chạy CPU.

### Search chậm (>10s)
Kiểm tra `use_crossmodal_reranker` có đang bật không. Tắt đi nến cần latency thấp:
```powershell
# Xem log profiling
Get-Content "server.log" | Select-String "PROFILE"
```

### Lỗi dtype mismatch (Float vs Half)
Đã sửa — models tự động ép sang float16 sau khi load.

---

## 🔄 Pipeline dữ liệu đầy đủ

1. **Ingestion (Kaggle GPU):** mỗi member chạy notebook trên shard → xuất keyframes, parquets, embeddings → upload lên Kaggle Dataset (`shard_assignment.yaml`).
2. **Aggregation (Local):** `pull_shards`, `verify_shards`, `merge_datasets`, `map_keyframes_to_unified` → `data/unified/`.
3. **Indexing (Qdrant):** `run_index` nạp vectors + metadata + BM42 sparse → gắn alias `production`.
4. **Serving (FastAPI):** query text/voice/ảnh → embeddings → Weighted RRF trên 5 branches → rerank → trả kết quả. TRAKE thêm bước temporal alignment (DANTE).

