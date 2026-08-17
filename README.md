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

#### Dành cho Windows:
```powershell
git clone https://github.com/ThanhDanh1510/Super-Image-Retrieval-System-for-AIC2026.git
cd Super-Image-Retrieval-System-for-AIC2026

# Cài đặt uv
pip install uv

# Tạo môi trường ảo & Kích hoạt
uv venv .venv
.venv\Scripts\Activate.ps1

# Cài đặt PyTorch hỗ trợ GPU CUDA (NVIDIA)
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Cài đặt toàn bộ dependencies
uv pip install -e ".[all]"
```

#### Dành cho macOS:
```bash
git clone https://github.com/ThanhDanh1510/Super-Image-Retrieval-System-for-AIC2026.git
cd Super-Image-Retrieval-System-for-AIC2026

# Cài đặt uv
pip3 install uv

# Tạo môi trường ảo & Kích hoạt
uv venv .venv
source .venv/bin/activate

# Cài đặt PyTorch mặc định (Tự động hỗ trợ Apple Silicon MPS hoặc Intel CPU)
uv pip install torch torchvision torchaudio

# Cài đặt toàn bộ dependencies
uv pip install -e ".[all]"
```
*Lưu ý cho macOS:* 
- Nếu bạn dùng máy Mac chip **Apple Silicon (M1/M2/M3)**, hãy sửa biến môi trường trong file `.env` thành `AIC_DEVICE=mps` để sử dụng GPU tích hợp (Metal Performance Shaders).
- Nếu dùng máy Mac chip **Intel**, hãy để `AIC_DEVICE=cpu`.


### Bước 2: Khởi động Qdrant (Docker)
```powershell
docker compose up -d
# Kiểm tra:
curl.exe http://localhost:6333/
```
Lưu ý: Dự án có sẵn `docker-compose.yml`. Nếu không có Docker, hệ thống sẽ tự động fallback sang Local Embedded Qdrant tại `./data/qdrant_db/`.

### Bước 3: Tải Dữ liệu
📥 Tải data tại: [Google Drive - TBD](https://drive.google.com/drive/folders/174LtGV8Zex9TSD0eTajtVaY59DSi23kF?usp=sharing)

Sau khi tải về, giải nén, ta sẽ có cấu trúc thư mục như sau:
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
Tùy chỉnh các đường dẫn lưu cache mô hình AI (tránh đầy ổ C) và thư mục dữ liệu trong file `.env`. Xem chi tiết tại mục [Biến Môi Trường (`.env`)](#️-biến-môi-trường-env).

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

**Danh sách models (~15GB):**

| Model | Mục đích | Device | RAM |
|---|---|---|---|
| **CLIP ViT-L/14** | Image embedding | GPU | ~1.7GB |
| **gte-Qwen2-1.5B** | Text embedding (OCR+ASR) | GPU | ~3GB |
| **bge-reranker-v2-m3** | Text rerank (luôn bật) | GPU | ~2.3GB |
| **jina-reranker-m0** | Crossmodal rerank (tắt mặc định) | GPU | ~2.5GB |
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

| Endpoint | Method | Body / Params | Mô tả |
|---|---|---|---|
| `/api/v1/kis` | POST | `{query, input_type: "text"\|"voice"\|"image", search_mode: "all", top_k (1-500), use_reranker?, user_choices?}` | Known-Item Search |
| `/api/v1/vqa` | POST | `{query, question, input_type, search_mode, top_k, use_reranker?, user_choices?}` | Visual QA — trả `results` + `answers` |
| `/api/v1/trake` | POST | `{query, num_events (2-10), input_type, search_mode, top_k}` | Temporal Retrieval — trả `list[list[CandidateResult]]` |
| `/api/v1/similar-image` | POST | multipart `file` + query param `top_k` (1-100), `use_reranker?` | Image Search |
| `/api/v1/search/hybrid` | GET | query params: `query_vi`, `query_en?`, `mode`, `top_k` | Hybrid Search (Weighted Convex + RRF) |
| `/api/v1/image/{video_id}/{frame_id}` | GET | path params: `video_id`, `frame_id` | Lấy file ảnh keyframe thực tế |
| `/api/v1/video/{video_id}` | GET | path param: `video_id` (hỗ trợ HTTP Range) | Lấy file video gốc |
| `/api/v1/video/{video_id}/fps` | GET | path param: `video_id` | Lấy FPS thực tế của video |

Tất cả response retrieval trả về danh sách `CandidateResult`: `video_id`, `scene_id`, `frame_id`, `timestamp_start`, `timestamp_mid`, `timestamp_end`, `keyframe_path`, `ocr_text`, `asr_text`, `description`, `score`.

**Tham số `use_reranker`:** Mặc định `false`.
- Khi `false`: Chỉ chạy text reranker **ViRanker (bge-reranker-v2-m3)** trên GPU (~0.2s, rất nhẹ).
- Khi `true`: Chạy song song cả **ViRanker** và **Jina Reranker M0** (crossmodal image-text) trên GPU để tối ưu độ chính xác.

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

## 🧹 Công cụ Lọc Keyframe & Tìm Ảnh Trùng Lặp (Image Filter Tool)

Hệ thống cung cấp Web UI độc lập để duyệt ảnh theo từng video, đánh dấu lọc các ảnh bị mờ/hỏng/thừa và quét tìm ảnh trùng lặp trên 108.000+ vector CLIP:

```powershell
# 1. Khởi chạy Web UI lọc ảnh (Port 8501)
python -m tools.image_filter.server
# Hoặc: make filter-tool
# Truy cập: http://localhost:8501
```

- 📂 **Duyệt theo Video**: Sidebar điều hướng Batch (`L21`–`L30`) ➔ Video (`V001`, `V002`,...).
- 🖱️ **Đánh dấu 1-Click**: Click vào card ảnh để bật/tắt đánh dấu lọc (Overlay ❌).
- 🔍 **Quét trùng lặp siêu tốc (<10ms)**: Bấm nút *🔍 Giống* để quét toàn bộ dataset, kéo thanh trượt ngưỡng tương đồng (ví dụ $\ge 90\%$) và chọn *Giữ lại ảnh gốc* hoặc *Xóa cả ảnh gốc*.
- 💾 **Auto-Save JSON**: Tự động lưu và đồng bộ vào file duy nhất `data/filtered_keyframes.json`.

```powershell
# 2. Áp dụng lọc (Xóa file ảnh trên ổ đĩa & Rebuild Unified Dataset)
python -m aggregation.apply_filter --delete-images
# Hoặc kiểm tra thử trước: make apply-filter-dry
```

---

## ⏱️ Công cụ Bù đắp Keyframe & Kiểm tra Độ phủ Timeline (Gap Filling & Coverage)

Trong quá trình thi AIC, việc video bị thiếu keyframe ở các phân đoạn dài hoặc sau khi lọc bỏ TVC/ảnh thừa có thể tạo ra các "khoảng trống" (timeline gaps $> 30\text{s}$). Bộ công cụ sau giúp tự động phát hiện, trích xuất bổ sung keyframe và chuẩn hóa $100\%$ độ phủ timeline cho toàn bộ dataset:

### 1. Báo cáo & Kiểm tra Độ phủ Timeline (Coverage Audit)
Quét toàn bộ các video từ batch `L21` đến `L30` (hoặc video mới thêm vào `data/video/`), phân loại trạng thái `PASS` (khoảng cách tối đa $\le 30\text{s}$), `WARNING` ($30\text{s} - 60\text{s}$), hoặc `FAILED` ($> 60\text{s}$):

```powershell
python tools/report_all_batches_coverage.py
```

### 2. Tự động Bù đắp Keyframe vào các khoảng Gap $> 30\text{s}$ (`tools/fill_gap_keyframes.py`)
Khi bạn thêm video mới vào `data/video/` hoặc sau khi lọc bớt ảnh rác/TVC, chạy script này để tự động bù kín toàn bộ timeline:

```powershell
python tools/fill_gap_keyframes.py
```

**Cơ chế hoạt động:**
- 🔍 **Tự động quét**: So sánh mốc thời gian giữa các keyframe hiện có trong `data/unified/scene_metadata.parquet` với tổng thời lượng thực tế của file video.
- ✂️ **Trích xuất ảnh WebP**: Bổ sung keyframe tại các mốc phân bố đều ($\le 20-30\text{s}$) cho các khoảng gap ở đầu, giữa hoặc cuối video, lưu trực tiếp vào thư mục `data/image/{Batch}/{Batch}/{Vxxx}/{frame_id}.webp`.
- ⚡ **Sinh Vector CLIP trên GPU**: Tự động đưa ảnh qua mô hình **CLIP ViT-L/14** để tạo visual embeddings 1024 chiều.
- 🔄 **Đồng bộ toàn bộ Unified Dataset**: Tự động cập nhật và ghi đè đồng bộ các file `scene_metadata.parquet`, `ocr_text.parquet`, `asr_text.parquet`, `scene_descriptions.parquet`, `image_vectors.npy`, `ocr_vectors.npy`, `asr_vectors.npy`, `description_embeddings.pt`, và `scene_ids.txt`.

### 3. Hiệu chuẩn FPS và Thời gian Video (`tools/calibrate_all_video_fps.py`)
Đảm bảo tất cả keyframe có mốc thời gian `timestamp_mid = frame_id / FPS_thực_tế` chính xác tuyệt đối (giải quyết triệt để vấn đề video 29.97/30 FPS bị tính nhầm theo 25 FPS):

```powershell
python tools/calibrate_all_video_fps.py
```

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

Copy file mẫu cấu hình:
```powershell
copy .env.example .env
```

### 1. Cấu hình Cache thư viện ML (Khuyến nghị cấu hình để tránh đầy ổ C)

| Biến | Mặc định trong `.env.example` | Ý nghĩa |
|---|---|---|
| `HF_HOME` | `./data/hf_cache` | Thư mục gốc lưu cache HuggingFace (models, tokenizers) |
| `HF_HUB_CACHE` | `./data/hf_cache/hub` | Thư mục lưu hub cache của HuggingFace |
| `TRANSFORMERS_CACHE` | `./data/hf_cache/hub` | Thư mục lưu cache mô hình Transformers |
| `PADDLE_HOME` | `./data/paddle_cache` | Thư mục lưu cache PaddlePaddle |
| `PADDLEOCR_BASE_DIR` | `./data/paddle_cache` | Thư mục lưu pretrained weights của PaddleOCR |
| `TORCH_HOME` | `./data/torch_cache` | Thư mục lưu cache weights của PyTorch |

> 💡 **Mẹo:** Nếu máy có ổ cứng phụ (ví dụ ổ `D:`), bạn có thể đổi giá trị các biến trên thành đường dẫn tuyệt đối (vd: `D:\huggingface_cache`) để giải phóng hoàn toàn dung lượng ổ `C:`.

### 2. Cấu hình Đường dẫn Dữ liệu (Tùy chọn)

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `AIC_DATA_DIR` | `./data` (thư mục `data/` tại root) | Thư mục gốc chứa toàn bộ dataset, unified parquet, vectors và keyframes |
| `AIC_KEYFRAME_DIR` | `./data/image` | Đường dẫn thư mục ảnh keyframe (tùy chọn ghi đè nếu tách riêng ổ đĩa) |

### 3. Cấu hình Nâng cao (Tùy chọn — Server có sẵn giá trị mặc định tối ưu)

Hệ thống cung cấp sẵn các giá trị mặc định chuẩn, không bắt buộc phải khai báo trong file `.env`:

| Biến | Mặc định trong code | Ý nghĩa |
|---|---|---|
| `AIC_DEVICE` | `cuda` | Thiết bị chạy model (`cuda`, `mps`, `cpu`) |
| `AIC_PRELOAD_MODELS` | `true` | Tự động warmup toàn bộ models lúc startup |
| `AIC_QDRANT_URL` | `http://localhost:6333` | URL Qdrant server |
| `AIC_QDRANT_COLLECTION_ALIAS` | `production` | Collection alias dùng cho truy vấn |
| `AIC_MODEL_UNLOAD_TTL_S` | `0` | Tự động unload model sau N giây không dùng (`0` = tắt) |
| `AIC_PORT` | `8000` | Cổng HTTP của FastAPI server |
| `AIC_HOST` | `0.0.0.0` | Địa chỉ bind của FastAPI server |
| `AIC_CORS_ORIGINS` | `["*"]` | Danh sách CORS allowed origins |

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
Kiểm tra `use_reranker` có đang bật không. Tắt đi nếu cần latency thấp:
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

