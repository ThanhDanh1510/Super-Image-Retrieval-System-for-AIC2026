# AGENT_CONTEXT.md — SYSTEM PROMPT CONTEXT BLOCK FOR AI AGENTS

> **Usage for AI Agents**: Copy this entire context block into the system prompt or top of prompt when generating or refactoring code for AIC2026.

---

## 1. System Architecture & Flow

```
ingestion (Kaggle) ➔ aggregation ➔ indexing (Qdrant) ➔ serving (FastAPI)
```

### Layering Constraints (STRICT 1-WAY FLOW)
```
controller ➔ service ➔ repo ➔ qdrant_client
```
* **NO Reverse Imports**: Lower layers MUST NOT import upper layers (`repo` never imports `service`/`controller`).
* **NO Direct Instantiation**: DO NOT initialize models in function bodies or module level. Use dependency injection (`Depends(get_text_encoder)`).
* **NO Direct Qdrant Access**: Services MUST NOT call `QdrantClient` directly — route all DB access through `serving/api/repo/qdrant_repo.py`.

---

## 2. Named Vectors Spec (`shared/schemas/named_vector_spec.py`)

| Vector Name | Type | Dimension | Model Source | Purpose |
|---|---|---|---|---|
| `image_dense` | Dense | `1024` | BEiT-large + CLIP ViT-L/14 | Keyframe image embedding |
| `ocr_dense` | Dense | `1536` | gte-Qwen2-1.5B-instruct | Keyframe OCR text embedding |
| `asr_dense` | Dense | `1536` | gte-Qwen2-1.5B-instruct | Audio transcript ASR embedding |
| `ocr_sparse` | Sparse | Dynamic | BM42 | Keyword match for OCR |
| `asr_sparse` | Sparse | Dynamic | BM42 | Keyword match for ASR |

*Always import vector names & dims from `shared.schemas.named_vector_spec` — NEVER hardcode.*

---

## 3. Pinned Model Versions (`configs/model_versions.yaml`)

* **Text Embedding**: `Alibaba-NLP/gte-Qwen2-1.5B-instruct` (Dim=1536)
* **Image Embedding**: `microsoft/BEiT-large` + `openai/clip-vit-large-patch14` (Dim=1024)
* **OCR Engine**: PaddleOCR v6 (`lang='vi'`)
* **ASR Model**: `vinai/PhoWhisper-medium` (`lang='vi'`)
* **Shot Detection**: TransNetV2 (Fallback: Color Histogram Difference)

---

## 4. Deterministic Point ID & Payload Spec

```python
from shared.schemas.named_vector_spec import make_point_id
point_id = make_point_id(video_id="L01_V001", scene_id="L01_V001_S003") # UUID v5 deterministic
```
* **Payload Fields**: `video_id`, `scene_id`, `frame_id`, `timestamp_start`, `timestamp_end`, `timestamp_mid`, `keyframe_path`, `ocr_text`, `asr_text`, `ingestion_version`, `member`.

---

## 5. Query Routing & Reranking (`configs/fusion_config.yaml`)

* **Text Query 5-Branch Search**: `image_dense` (CLIP text encoder), `ocr_dense` (gte-Qwen2), `asr_dense` (gte-Qwen2), `ocr_sparse`, `asr_sparse` $\rightarrow$ **RRF Fusion** ($k=60$).
* **Reranker Constraints**:
  * **Lazy Load**: Initialize reranker models on first call and cache instance (NO module-level load to prevent OOM).
  * **Batch Inference**: Process ALL candidate results in a single forward pass batch. NO looping per candidate.
* **TRAKE Temporal Aligner** (`serving/api/service/temporal_aligner.py`):
  * Filter candidate sequences by same `video_id`, strictly ascending `timestamp_mid`. Prioritize videos matching highest event count.

---

## 6. Evaluation Rules (`eval/skeletons/metrics/`)

* **KIS `r_score`**: `1.0` if `(video_id == GT_v)` AND `(frame_id in [start, end])`, else `0.0`.
* **VQA `r_score`**: `1.0` if `(video_id == GT_v)` AND `(frame_id in [start, end])` AND `(answer == GT_a)`, else `0.0`.
* **TRAKE `r_score`**: `0.0` if wrong `video_id`; else `(1 / N) * sum(1[frame_j in [start_j, end_j]])`.
* **Final Score**: `mean(R@1, R@5, R@20, R@50, R@100)`.
