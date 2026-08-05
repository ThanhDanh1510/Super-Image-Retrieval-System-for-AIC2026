# 🎯 Super Image Retrieval System for AIC2026

Hệ thống truy vấn keyframe video real-time phục vụ cuộc thi **AI Challenge HCMC 2026**.

Hỗ trợ 4 dạng task chính:

| Task | Mô tả |
|---|---|
| **KIS** (Known-Item Search) | Tìm keyframe theo mô tả text hoặc giọng nói |
| **VQA** (Visual Question Answering) | Tìm cảnh liên quan rồi trả lời câu hỏi |
| **TRAKE** (Temporal Retrieval) | Tìm chuỗi sự kiện theo đúng thứ tự thời gian |
| **Image Search** | Tìm keyframe tương đồng với ảnh upload |

---

## 🏗️ Kiến trúc tổng quan

```
ingestion (Kaggle GPU) → aggregation (Local) → indexing (Qdrant DB) → serving (FastAPI)
```

| Tầng | Mô tả | Môi trường |
|---|---|---|
| `ingestion/` | Trích xuất keyframes, OCR (PaddleOCR v6), ASR (PhoWhisper), sinh vector embeddings | Kaggle GPU |
| `aggregation/` | Gộp output của tất cả member thành 1 unified dataset | Máy Local |
| `indexing/` | Định nghĩa schema, nạp unified dataset + embeddings vào Qdrant | Docker / Qdrant local |
| `serving/` | FastAPI REST API real-time với lazy model loading + RRF fusion + rerank | Server local / Staging |

Dữ liệu được biểu diễn bằng **5 named vectors** trong Qdrant (xem `shared/schemas/named_vector_spec.py`):

| Vector | Loại | Dim | Nguồn |
|---|---|---|---|
| `image_dense` | dense | 1024 | CLIP ViT-L/14 |
| `ocr_dense` | dense | 1536 | gte-Qwen2-1.5B (text OCR) |
| `asr_dense` | dense | 1536 | gte-Qwen2-1.5B (transcript ASR) |
| `ocr_sparse` | sparse | — | BM42 keyword (OCR) |
| `asr_sparse` | sparse | — | BM42 keyword (ASR) |

Kết quả retrieval tổng hợp bằng **Weighted RRF** (`k=60`, prefetch=200, trọng số đọc từ `configs/fusion_config.yaml`), sau đó rerank bằng `BAAI/bge-reranker-v2-m3` (text) và `jinaai/jina-reranker-m0` (crossmodal).

---

## 🚀 Cài Đặt & Khởi Chạy

Dự án dùng **`uv`** để quản lý 1 môi trường ảo `.venv` duy nhất cho cả serving + ingestion + dev.

### 1. Cài `uv`

- **Windows:** `pip install uv`
- **macOS:** `brew install uv`

### 2. Tạo môi trường & cài dependencies

```powershell
uv venv .venv

# Windows (kích hoạt)
.\.venv\Scripts\Activate.ps1
# macOS
# source .venv/bin/activate

# Windows có card NVIDIA (PyTorch CUDA) — xem thêm: https://pytorch.org/get-started/locally/
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Cài TOÀN BỘ dependencies (serving + ingestion + dev)
uv pip install -e ".[all]"
```

### 3. Khởi động Qdrant (Docker)

```powershell
# Lần đầu:
docker run -d -p 6333:6333 -p 6334:6334 -v ${PWD}/data/qdrant_storage:/qdrant/storage --name qdrant_aic2026 qdrant/qdrant
# Lần sau:
docker start qdrant_aic2026
# Kiểm tra:
curl.exe http://localhost:6333/
```

> 💡 Nếu không có Docker, indexing/serving tự động fallback sang **Local Embedded Qdrant** tại `./data/qdrant_db/`.

### 4. Cấu trúc dữ liệu `data/`

```text
data/
├── image/                     # Keyframes thô: data/image/{Group}/{Group}/{Video}/*.webp
├── OCR/                       # JSON OCR từng video: L21/L21_V001.json ...
├── ASR/                       # File JSON transcript: asr.json
├── unified/                   # [Quan trọng] Dataset hợp nhất + embeddings cho indexing
│   ├── scene_metadata.parquet # Metadata keyframes (scene_id, video_id, timestamps...)
│   ├── ocr_text.parquet       # Text OCR theo scene
│   ├── asr_text.parquet       # Transcript ASR theo scene
│   ├── image_vectors.npy      # N x 1024 (CLIP)
│   ├── ocr_vectors.npy        # N x 1536 (gte-Qwen2)
│   ├── asr_vectors.npy        # N x 1536 (gte-Qwen2)
│   ├── scene_ids.txt          # Danh sách scene_id ứng với index các file .npy
│   └── keyframes/             # Ảnh keyframe đã map vào unified (L01_V001-L01_V001_S000.webp)
└── unified_raw/               # Đầu ra của aggregation (gộp từ các shard)
```

### 5. Nạp dữ liệu vào Qdrant (Indexing)

```bash
python -m indexing.run_index \
    --dataset-dir ./data/unified \
    --collection collection_v1 \
    --create-collection
```

Gắn alias `production` (serving luôn truy vấn qua alias này để swap collection không downtime):

```bash
python -m indexing.collection_alias --action set --collection collection_v1
```

### 6. Chạy FastAPI server

```bash
# Cách 1 (khuyên dùng): in-process, tải model lazy (tiết kiệm RAM)
python start_api.py

# Cách 2: dev mode
python -m uvicorn serving.api.main:app --reload --host 0.0.0.0 --port 8000
```

- 🌐 **Swagger UI:** `http://localhost:8000/docs`
- 🩺 **Healthcheck:** `http://localhost:8000/health`

---

## 🔌 API Endpoints

| Endpoint | Body | Mô tả |
|---|---|---|
| `POST /api/v1/kis` | `{query, input_type: "text"\|"voice", top_k (1-200), user_choices?}` | Known-Item Search |
| `POST /api/v1/vqa` | `{query, question, input_type, top_k, user_choices?}` | Visual QA — trả `results` + `answers` |
| `POST /api/v1/trake` | `{query, num_events (2-10), input_type, top_k}` | Temporal Retrieval — trả `list[list[CandidateResult]]` |
| `POST /api/v1/similar-image` | multipart `file` + query param `top_k` | Image Search |

Tất cả response trả về `CandidateResult`: `video_id`, `scene_id`, `frame_id`, `timestamp_start/mid/end`, `keyframe_path`, `ocr_text`, `asr_text`, `score`.

---

## 🧠 AI Models

Model **tự động tải về** từ HuggingFace Hub lần đầu được dùng (lazy-load).

| Model | Dùng cho | HuggingFace / Nguồn |
|---|---|---|
| **TransNetV2** | Shot detection (tách scene) | `transnetv2-pytorch-weights.pth` |
| **PaddleOCR v6** | OCR tiếng Việt | `paddleocr` |
| **PhoWhisper-medium** | ASR tiếng Việt | `vinai/phowhisper-medium` |
| **CLIP ViT-L/14** | Image embedding (dim 1024) | `openai/clip-vit-large-patch14` |
| **gte-Qwen2-1.5B-instruct** | Text embedding OCR + ASR (dim 1536) | `Alibaba-NLP/gte-Qwen2-1.5B-instruct` |
| **bge-reranker-v2-m3** | Text rerank (KIS/VQA/TRAKE) | `BAAI/bge-reranker-v2-m3` |
| **jina-reranker-m0** | Crossmodal rerank (Image Search) | `jinaai/jina-reranker-m0` |

Tất cả phiên bản model được khai báo tập trung tại `configs/model_versions.yaml` — **single source of truth** cho cả ingestion lẫn serving. Ingestion và serving phải khớp version, nếu lệch thì vector không tương thích.

---

## 📁 Cấu trúc dự án

```
Super-Image-Retrieval-System-for-AIC2026/
├── configs/              # Cấu hình tập trung (model_versions, fusion_config, shard_assignment)
├── ingestion/            # Pipeline Kaggle GPU (skeletons + notebooks)
│   ├── skeletons/        # 01a keyframe, 01b text, 01c embeddings, 02 export shard
│   └── notebooks/        # Kaggle notebooks (gitignored)
├── aggregation/          # pull_shards, verify_shards, merge, publish_unified, map_keyframes...
├── indexing/             # index_data, schema, upsert, collection_alias
├── serving/              # FastAPI: controller → service → repo (+ temporal_aligner, reranker)
├── shared/               # Dùng chung: named_vector_spec, config_loader, model wrappers
├── eval/                 # Offline benchmark + metrics (r_score, final_score)
├── tests/                # Unit tests + integration tests (cần server :8000)
├── start_api.py          # Entry point run server in-process
├── Makefile              # Các lệnh thường dùng (make serve / lint / test / ...)
├── pyproject.toml        # Dependencies (uv/pip install -e ".[all]")
└── .env.example          # Template biến môi trường (prefix AIC_)
```

---

## ⚙️ Biến môi trường (`.env`)

Copy `.env.example` → `.env`. Một số biến quan trọng:

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `AIC_QDRANT_URL` | `http://localhost:6333` | URL Qdrant server |
| `AIC_QDRANT_COLLECTION_ALIAS` | `production` | Alias collection dùng để query |
| `AIC_DEVICE` | `cuda` | `cuda` / `mps` / `cpu` |
| `AIC_PRELOAD_MODELS` | `false` | `true` = warmup toàn bộ model lúc startup |
| `AIC_MODEL_UNLOAD_TTL_S` | `0` | Unload model sau N giây idle (`0` = không unload) |
| `AIC_CORS_ORIGINS` | `["*"]` | Allowed origins |
| `AIC_DATA_DIR` / `AIC_KEYFRAME_DIR` | `data/` | Mount dữ liệu nếu chạy server riêng |

---

## 🧪 Test & Code Quality

```bash
# Unit tests (không cần server):
pytest tests/test_retrieval_serving.py tests/test_query_parsing.py tests/test_rerank_service.py

# Integration tests (cần Qdrant + server đang chạy tại :8000):
pytest tests/test_search_services.py tests/test_edge_cases.py

# Lint & typecheck:
ruff check .          # hoặc: make lint
mypy shared/ serving/ indexing/ --ignore-missing-imports   # hoặc: make typecheck
```

> 💡 Toàn bộ lệnh hay dùng có sẵn trong `Makefile` (`make help` để xem danh sách).

---

## 🔄 Pipeline dữ liệu đầy đủ (tổng hợp)

1. **Ingestion (Kaggle GPU):** mỗi member chạy notebook trên shard của mình → xuất `keyframes/`, `scene_metadata.parquet`, `ocr_text.parquet`, `asr_text.parquet`, embeddings `.npy` → upload lên Kaggle Dataset (`shard_assignment.yaml`).
2. **Aggregation (Local):** `pull_shards --owner`, `verify_shards`, `merge_datasets`, `map_keyframes_to_unified` → `data/unified/`.
3. **Indexing (Qdrant):** `run_index` nạp toàn bộ + sparse (BM42) vectors → gắn alias `production`.
4. **Serving (FastAPI):** query text/voice/ảnh → embeddings → Weighted RRF trên 5 branches → rerank → trả kết quả. TRAKE thêm bước temporal alignment (DANTE) để đảm bảo thứ tự thời gian.