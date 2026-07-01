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

## 11. Bổ sung thông tin Tuần 2

### A. Reranker Constraints (Researcher B)
- Thư mục: `serving/api/service/reranker/skeletons/`
- Gồm 3 tasks chính:
  1. `text_branch/viranker.py` (ViRanker): Đo độ tương thích của văn bản.
  2. `crossmodal_branch/jina_reranker_m0.py` (Jina Reranker): Đối chiếu text query với ảnh keyframe thật.
  3. `fusion_score.py`: Hợp nhất điểm số 2 nhánh theo linear/RRF với `alpha` cấu hình trong `configs/fusion_config.yaml`.
- **Ràng buộc**:
  - **Lazy load**: Khởi tạo model khi được gọi lần đầu, cache lại model instance để tránh OOM khi import.
  - **Batch inference**: Thực hiện dự đoán song song cho toàn bộ candidates trong 1 forward pass. KHÔNG dùng loop từng candidate.

### B. Temporal Aligner cho TRAKE (Researcher B)
- File: `serving/api/service/temporal_aligner.py`
- **Quy tắc**:
  - Lọc ra chuỗi các keyframes đại diện thỏa mãn: Cùng `video_id`, thứ tự thời gian tăng dần (`timestamp_mid` tăng dần).
  - Ưu tiên video có số lượng events matched nhiều nhất, sau đó mới tối ưu hóa điểm số.

### C. Aggregation Flow (MLOps Lead)
- Thư mục: `aggregation/`
- Luồng chạy:
  `pull_shards` (tải data từ Kaggle) → `verify_shards` (xác thực tính nhất quán và trùng lặp) → `merge_datasets` (gộp metadata, vectors, copy keyframes) → `publish_unified` (publish unified dataset lên Kaggle).

### D. Evaluation (Researcher)
- Thư mục: `eval/skeletons/`
- Công thức `r_score`:
  - KIS: `1.0` nếu đúng video_id và timestamp trong khoảng ground truth, ngược lại `0.0`.
  - VQA: `1.0` nếu đúng video_id, timestamp và đúng answer, ngược lại `0.0`.
  - TRAKE: `0.0` nếu sai video_id, ngược lại `(1 / N) * count_matched_events`.
- `final_score` = trung bình cộng của `R@1, R@5, R@20, R@50, R@100`.

---

## 12. Constraints bổ sung (Tuần 2)

```
✅ Khởi tạo model lazy load (reranker) và cache instance.
✅ Batch inference cho reranker để tận dụng GPU.
✅ verify_shards bắt buộc kiểm tra trùng lặp và alignment chéo.
✅ eval/metrics/r_score.py bắt buộc tính đúng công thức quy định.

❌ KHÔNG loop từng candidate để chạy inference.
❌ KHÔNG merge shards nếu verify phát hiện lỗi trùng/sót video_id.
```

