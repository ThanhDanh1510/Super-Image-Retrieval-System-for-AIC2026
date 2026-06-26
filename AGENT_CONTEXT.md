# AGENT_CONTEXT.md — Project Context Block cho AI Agent

> **Cách dùng:** Researcher copy toàn bộ nội dung file này → paste vào đầu prompt mỗi khi dùng AI Agent (Claude/Cursor) để implement code. Agent đọc được context toàn hệ thống và sinh code đúng constraint.

---

## 1. Tổng quan hệ thống

Hệ thống **Video Retrieval** phục vụ cuộc thi **AI Challenge HCMC 2026**, gồm 4 nhịp:

```
ingestion (Kaggle) → aggregation → indexing (Qdrant) → serving (FastAPI, lúc thi)
```

- **Ingestion**: Mỗi member chạy full pipeline trên shard video riêng (Kaggle Notebook).
- **Aggregation**: Gộp output tất cả member thành 1 dataset thống nhất.
- **Indexing**: Đẩy vectors vào Qdrant collection (5 named vectors).
- **Serving**: FastAPI API real-time lúc thi, trả kết quả cho 3 task KIS/VQA/TRAKE + Image Search.

---

## 2. Named Vectors (Qdrant)

Nguồn xác định duy nhất: `shared/schemas/named_vector_spec.py`

| Tên vector | Loại | Dimension | Model | Mô tả |
|---|---|---|---|---|
| `image_dense` | dense | 1024 | BEiT-large + CLIP ViT-L/14 | Embedding ảnh keyframe |
| `ocr_dense` | dense | 1536 | gte-Qwen2-1.5B | Embedding text OCR từ keyframe |
| `asr_dense` | dense | 1536 | gte-Qwen2-1.5B | Embedding transcript ASR |
| `ocr_sparse` | sparse | — | BM42 | Keyword matching cho OCR |
| `asr_sparse` | sparse | — | BM42 | Keyword matching cho ASR |

**KHÔNG hardcode dimension hay tên vector** — luôn import từ `shared/schemas/named_vector_spec.py`.

---

## 3. Model Versions (ghim cứng)

Nguồn xác định duy nhất: `configs/model_versions.yaml`

- **Text Embedding**: `Alibaba-NLP/gte-Qwen2-1.5B-instruct` — dim=1536 — dùng chung cho OCR lẫn ASR
- **Image Embedding**: `microsoft/BEiT-large-patch16-224-pt22k-ft22k` + `openai/clip-vit-large-patch14` — dim=1024
- **OCR**: PaddleOCR v6 (tiếng Việt)
- **ASR**: `vinai/PhoWhisper-medium` (tiếng Việt)
- **Shot Detection**: TransNetV2

**KHÔNG tự ý đổi model** — lệch version = vector không tương thích giữa ingestion và serving.

---

## 4. Kiến trúc Serving (controller → service → repo)

```
controller/            # Nhận HTTP request, validate, gọi service, trả response
  ├─ routes_kis.py     # POST /api/v1/kis, /vqa, /trake
  └─ routes_similar_image.py  # POST /api/v1/similar-image

service/retrieval/     # Logic nghiệp vụ — encode, route, orchestrate
  ├─ text_query.py     # Text → ocr_vector + asr_vector + image_vector + sparse → Qdrant
  ├─ image_query.py    # Image → image_vector → Qdrant
  ├─ voice_query.py    # Audio → transcript → delegate text_query
  └─ retrieval_pipeline.py  # Orchestrator: phân loại input type, gọi đúng module

service/reranker/      # Reranking sau retrieval
  ├─ text_branch/viranker.py
  ├─ crossmodal_branch/jina_reranker_m0.py
  └─ fusion_score.py

repo/                  # Giao tiếp với Qdrant
  └─ qdrant_repo.py    # search_by_text_vectors, search_by_image_vector, search_by_all_vectors
```

### Quy tắc bắt buộc

1. **Một chiều**: `controller → service → repo` — KHÔNG ngược lại
2. **KHÔNG khởi tạo model** trong function body hay module level — dùng dependency injection
3. **KHÔNG gọi Qdrant trực tiếp** từ service — chỉ qua `qdrant_repo`
4. **KHÔNG import từ layer cao hơn** (service không import controller, repo không import service)

---

## 5. Schema (Request/Response)

File: `serving/api/schema/retrieval.py`

- `CandidateResult` — kết quả nội bộ từ Qdrant (video_id, scene_id, frame_id, score, ...)
- `KISRequest/Response` — Known-Item Search
- `VQARequest/Response` — Visual Question Answering (thêm field `answers`)
- `TRAKERequest/Response` — Temporal Retrieval (list of list CandidateResult)
- `ImageSearchRequest/Response` — Image-to-Image Search

---

## 6. Query Routing (text query)

File cấu hình: `configs/fusion_config.yaml`

Khi người dùng gửi **text query**, hệ thống sinh **5 vectors** và search trên **5 nhánh**:

| Nhánh | Vector | Model encode |
|---|---|---|
| `image_dense` | CLIP text encoder → dim=1024 | CLIP ViT-L/14 (Text-to-Image) |
| `ocr_dense` | gte-Qwen2-1.5B → dim=1536 | Text embedding |
| `asr_dense` | gte-Qwen2-1.5B → dim=1536 | Text embedding |
| `ocr_sparse` | BM42 | Keyword matching |
| `asr_sparse` | BM42 | Keyword matching |

Kết quả từ 5 nhánh được **RRF fusion** (k=60) trước khi trả về.

---

## 7. Point ID Convention

```python
# KHÔNG random UUID — dùng deterministic hash
from shared.schemas.named_vector_spec import make_point_id
point_id = make_point_id(video_id="L01_V001", scene_id="L01_V001_S003")
# → UUID v5 deterministic
```

---

## 8. Config Loading

```python
from shared.utils.config_loader import load_model_versions, load_fusion_config
models = load_model_versions()     # dict từ configs/model_versions.yaml
fusion = load_fusion_config()      # dict từ configs/fusion_config.yaml
```

---

## 9. Payload Schema (Qdrant Point)

Mỗi point trong Qdrant có payload theo `KeyframePayload`:
- `video_id`, `scene_id`, `frame_id`
- `timestamp_start`, `timestamp_end`, `timestamp_mid`
- `keyframe_path` (relative: `{video_id}/{frame_idx:06d}.jpg`)
- `ocr_text`, `asr_text`
- `ingestion_version`, `member`

---

## 10. Constraints cứng khi viết code

```
✅ Luôn import tên vector từ shared/schemas/named_vector_spec.py
✅ Luôn đọc model version từ configs/model_versions.yaml
✅ Point ID = make_point_id(video_id, scene_id) — deterministic
✅ Skeleton functions: raise NotImplementedError
✅ Type hints cho tất cả function signatures
✅ L2-normalize tất cả dense vectors trước khi upsert/search

❌ KHÔNG hardcode dimension, tên vector, hay model ID
❌ KHÔNG import ngược chiều (repo → service, service → controller)
❌ KHÔNG khởi tạo model ở module level (OOM khi import)
❌ KHÔNG dùng random UUID cho Point ID
❌ KHÔNG tự ý thay đổi model version
```
