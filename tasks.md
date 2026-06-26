# AIC2026 — Task Breakdown & Team Workflow

## Bối cảnh team

| Vai trò | Thế mạnh | Công cụ |
|---|---|---|
| **MLOps Lead** (bạn) | Infrastructure, pipeline, code quality | Toàn bộ |
| **Researcher A** | Research, model selection, thực nghiệm | AI Agent (Claude/Cursor) |
| **Researcher B** | Research, model selection, thực nghiệm | AI Agent (Claude/Cursor) |

**Nguyên tắc quản lý:** AI Agent của researcher cần spec đủ chi tiết để sinh code đúng — không giao miệng, không giải thích sau. Nhiệm vụ của bạn là **viết skeleton + spec trước**, researcher dùng Agent để fill vào, bạn **review/merge**.

---

## Cách giao việc cho AI Agent

AI Agent hoạt động tốt nhất khi đọc code, không đọc prose. Quy trình chuẩn:

```
1. [Bạn] Viết skeleton file — function signature, docstring, type hint, raise NotImplementedError
2. [Researcher] Mở AGENT_CONTEXT.md → copy Project Context Block → paste vào đầu prompt Agent
3. [Agent] Đọc skeleton + context → fill implementation
4. [Researcher] Tạo PR theo branch naming convention
5. [Bạn] Review → merge
```

Skeleton là spec ở dạng máy đọc được — Agent bám vào đó tốt hơn bám vào mô tả bằng lời. File `AGENT_CONTEXT.md` ở root chứa Project Context Block dùng chung, không cần folder riêng cho Agent.

**Skeleton trông như thế nào:**
```python
# serving/service/retrieval/text_query.py  ← bạn tạo file này trước
from serving.api.schema import CandidateResult
from serving.repo import qdrant_repo
# KHÔNG import model trực tiếp — dùng dependency injection

async def query_by_text(
    query: str,
    encoder,  # injected từ dependencies.py
    top_k: int = 100,
) -> list[CandidateResult]:
    """
    Encode query text → dense vector → truy vấn Qdrant → trả về candidates.

    KHÔNG dịch thuật trong hàm này.
    KHÔNG gọi Qdrant trực tiếp — dùng qdrant_repo.search_by_text().
    KHÔNG khởi tạo model trong hàm này.
    """
    raise NotImplementedError


def split_trake_events(query: str) -> list[str]:
    """
    Tách multi-event TRAKE query thành list sự kiện riêng biệt.
    Ví dụ: "người đứng dậy, sau đó chạy ra cửa" → ["người đứng dậy", "chạy ra cửa"]
    """
    raise NotImplementedError
```

Agent đọc file này → hiểu ngay import path, interface, constraint — không cần bạn giải thích thêm.

---

## Phân chia trách nhiệm theo module

### 🔴 Bạn phụ trách (MLOps Lead) — KHÔNG delegate

Sai một chỗ → cả team bị block, hoặc cần hiểu hệ thống toàn cục mới làm đúng.

| Module | File/Component | Lý do giữ lại |
|---|---|---|
| `configs/` | `shard_assignment.yaml`, `model_versions.yaml`, `fusion_config.yaml` | Ghim version sai → toàn bộ vector không tương thích |
| `aggregation/` | Toàn bộ | Logic verify/merge cross-shard, dễ sinh bug lặng lẽ |
| `indexing/` | `collection_schema.py`, `collection_alias.py`, `run_index.py` | Schema Qdrant sai → phải re-index lại toàn bộ |
| `serving/core/` | `config.py`, `dependencies.py`, `lifespan.py` | Model singleton, startup/shutdown — sai là crash |
| `serving/repo/` | `qdrant_repo.py` | Query logic RRF/prefetch, ảnh hưởng trực tiếp recall |
| `infra/` | Docker, compose, CI/CD, monitoring | Deploy sai → mất điểm lúc thi |
| `shared/schemas/` | `named_vector_spec.py` | Contract giữa ingestion và serving — sai là không load được |
| `pyproject.toml`, `Makefile` | — | Dependency và entry point toàn hệ thống |
| `AGENT_CONTEXT.md` | — | Project Context Block cho AI Agent — bạn viết, không delegate |

---

### 🟡 Bạn viết skeleton + spec → Researcher dùng Agent implement

#### Module 1 — `ingestion/` (mỗi Researcher phụ trách shard của mình)

Mỗi người tự chạy toàn bộ pipeline trên Kaggle Notebook của mình. Skeleton nằm tại `ingestion/skeletons/`.

| Task | Skeleton file | Output cần đạt |
|---|---|---|
| **TASK-ING-01** | `01a_keyframe_extraction.ipynb` | Keyframe JPG + `scene_metadata.parquet` (scene_id, video_id, start_frame, end_frame, fps) |
| **TASK-ING-02** | `01b_text_extraction.ipynb` | `ocr_text.parquet` (scene_id, ocr_text), `asr_text.parquet` (scene_id, asr_text) |
| **TASK-ING-03** | `01c_embedding_generation.ipynb` | `image_vectors.npy`, `ocr_vectors.npy`, `asr_vectors.npy` — shape `(N, D)`, cùng thứ tự scene_id |
| **TASK-ING-04** | `02_export_shard_dataset.ipynb` | Kaggle Dataset `aic2026-shard-<member>-v<n>` publish public |
| **TASK-ING-05** | `kaggle_utils/checkpoint_manifest.py` | Class ghi/đọc `checkpoint.json` — track scene_id đã xong, resume khi session Kaggle bị ngắt |

**Constraints bắt buộc (đã nhúng trong skeleton, Agent không được override):**
```
- Model OCR: PaddleOCR v6 — version đọc từ configs/model_versions.yaml, không hardcode
- Model ASR: PhoWhisper — version đọc từ configs/model_versions.yaml, không hardcode
- Model text embedding: gte-Qwen2-1.5B — dùng chung cho cả OCR lẫn ASR text
- Model image embedding: BEiT3+CLIP — import từ shared/models/image_embedding_wrapper.py
- Point ID: sha256(video_id + "_" + scene_id)[:16] — KHÔNG random, KHÔNG uuid
- Output path: /kaggle/working/shard_output/ — KHÔNG hardcode path khác
- Checkpoint granularity: scene-level (không phải video-level)
```

---

#### Module 2 — `serving/service/retrieval/` (Researcher A)

Skeleton nằm tại `serving/service/retrieval/skeletons/`. Researcher A fill implementation, bạn review interface contract với repo.

| Task | Skeleton file | Interface |
|---|---|---|
| **TASK-SRV-01** | `text_query.py` | `async def query_by_text(query: str, encoder=Depends(get_text_encoder)) -> list[CandidateResult]` |
| **TASK-SRV-02** | `image_query.py` | `async def query_by_image(image_bytes: bytes, encoder=Depends(get_image_encoder)) -> list[CandidateResult]` |
| **TASK-SRV-03** | `voice_query.py` | `async def query_by_voice(audio_bytes: bytes) -> list[CandidateResult]` — transcribe rồi gọi lại `text_query`, không tạo luồng riêng |
| **TASK-SRV-04** | `retrieval_pipeline.py` | `async def retrieve(request: RetrievalRequest) -> list[CandidateResult]` — orchestrator phân loại input type |
| **TASK-SRV-05** | `text_query.py` (bổ sung) | `def split_trake_events(query: str) -> list[str]` — tách multi-event query thành N sự kiện |

**Constraints bắt buộc (đã nhúng trong skeleton):**
```python
# KHÔNG khởi tạo model trong function body hay module level
# PHẢI dùng dependency injection: Depends(get_text_encoder), Depends(get_image_encoder)
# KHÔNG gọi Qdrant trực tiếp — chỉ gọi qua qdrant_repo (import từ serving/repo/)
# KHÔNG import từ controller/ — luồng chỉ đi một chiều: controller → service → repo
```

---

#### Module 3 — `serving/service/reranker/` (Researcher B)

Skeleton nằm tại `serving/service/reranker/skeletons/`. Researcher B fill implementation, bạn review `fusion_score.py` và edge cases TRAKE.

| Task | Skeleton file | Interface |
|---|---|---|
| **TASK-RNK-01** | `text_branch/viranker.py` | `def score(query: str, candidates: list[CandidateResult]) -> list[float]` — ViRanker (BGE-M3 + BPT, MMARCO-VI) |
| **TASK-RNK-02** | `crossmodal_branch/jina_reranker_m0.py` | `def score(query: str, candidates: list[CandidateResult]) -> list[float]` — Jina Reranker M0, cross-modal |
| **TASK-RNK-03** | `fusion_score.py` | `def fuse(text_scores: list[float], crossmodal_scores: list[float], alpha: float) -> list[float]` — alpha đọc từ fusion_config.yaml |
| **TASK-RNK-04** | `../temporal_aligner.py` | `def align(event_candidates: list[list[CandidateResult]]) -> list[CandidateResult]` — TRAKE, 1 result per event, đảm bảo cùng video + thứ tự thời gian |

**Constraints bắt buộc (đã nhúng trong skeleton):**
```python
# Lazy load: khởi tạo model lần đầu gọi, cache lại — KHÔNG load ở module level (OOM khi import)
# Batch inference: xử lý toàn bộ candidates trong 1 forward pass — KHÔNG loop từng candidate
# TASK-RNK-04: ưu tiên video có nhiều events khớp nhất, sau đó mới tối ưu frame trong video đó
```

---

#### Module 4 — `eval/` (Researcher A hoặc B)

Skeleton nằm tại `eval/skeletons/`. Ai rảnh trước làm — không cần đợi nhau.

| Task | Skeleton file | Interface |
|---|---|---|
| **TASK-EVL-01** | `metrics/r_score.py` | `def r_score(prediction: Prediction, ground_truth: GroundTruth, task_type: Literal["KIS","VQA","TRAKE"]) -> float` |
| **TASK-EVL-02** | `metrics/final_score.py` | `def final_score(ranked_list: list[Prediction], ground_truth: GroundTruth) -> float` — trung bình R@{1,5,20,50,100} |
| **TASK-EVL-03** | `run_benchmark.py` | Script so sánh cấu hình: đổi fusion weight, bật/tắt reranker — in bảng kết quả |

**Constraints (đã nhúng trong skeleton):**
```
Công thức r_score — KHÔNG được suy diễn, implement đúng theo đây:
  KIS:   1.0 nếu (video_id == GT_v) AND (frame_id in [s, e]) else 0.0
  VQA:   1.0 nếu cả 3 đúng: video_id, frame_id in [s,e], answer == GT_a; else 0.0
  TRAKE: 0.0 nếu sai video; nếu đúng video → (1/N) * Σ 1[frame_j in [s_j, e_j]]

final_score = mean([R@1, R@5, R@20, R@50, R@100])
R@k = max(r_score(pred) for pred in ranked_list[:k])
```

---

#### Module 5 — `tests/` (ai implement feature thì viết test cho feature đó)

| Task | File | Người viết | Gắn với task |
|---|---|---|---|
| **TASK-TST-01** | `test_fusion_score.py` | Researcher B | Cùng lúc với TASK-RNK-03 |
| **TASK-TST-02** | `test_query_parsing.py` | Researcher A | Cùng lúc với TASK-SRV-05 |
| **TASK-TST-03** | `test_api_schema.py` | Bạn | Khi viết `serving/api/schema/` |
| **TASK-TST-04** | `test_merge_shards.py` | Bạn | Cùng lúc với `aggregation/verify_shards.py` |
| **TASK-TST-05** | `test_r_score.py` | Researcher (ai làm EVL) | Cùng lúc với TASK-EVL-01 — test edge cases mỗi task type |

---

## Workflow Git

### Branch naming
```
feature/<TASK-ID>-<slug>    # ví dụ: feature/TASK-SRV-01-text-query
fix/<TASK-ID>-<slug>        # ví dụ: fix/TASK-RNK-02-batch-inference
```

### PR Checklist (AI Agent tự điền trước khi tạo PR)
```markdown
## PR Checklist
- [ ] Code chạy được, không chỉ "trông có vẻ đúng" (paste output hoặc log vào PR description)
- [ ] Không import từ layer cao hơn (controller không import repo, service không import controller)
- [ ] Model không khởi tạo ở module level
- [ ] Point ID dùng hash deterministic — không random, không uuid
- [ ] Không hardcode path hay version model — đọc từ config
- [ ] Không tạo file ngoài file được yêu cầu trong skeleton
- [ ] Test đi kèm (nếu task yêu cầu)
```

### Review ownership
| PR thuộc module | Reviewer | Focus điểm |
|---|---|---|
| `ingestion/`, `aggregation/`, `indexing/`, `infra/`, `shared/` | **Bạn** | Toàn bộ |
| `serving/service/retrieval/` | **Bạn** | Interface contract với `qdrant_repo` |
| `serving/service/reranker/` | **Bạn** | `fusion_score.py` và edge cases TRAKE |
| `eval/metrics/` | **Bạn** | Verify công thức theo `contest_info.md` |
| `ingestion/notebooks/` (nội bộ shard) | Researcher tự review | Bạn chỉ review TASK-ING-04 (publish step) |

---

## Critical Path

```
Tuần 1 — Foundation (bạn unblock mọi người)
├── [Bạn]          configs/ — shard_assignment + model_versions + fusion_config
├── [Bạn]          shared/schemas/named_vector_spec.py
├── [Bạn]          serving/api/schema/ — Pydantic models (CandidateResult, RetrievalRequest...)
├── [Bạn]          AGENT_CONTEXT.md + skeleton files cho ING-01..05
├── [Researcher]   TASK-ING-01, 02, 03 (song song, mỗi người shard riêng)
└── [Bạn]          serving/repo/qdrant_repo.py

Tuần 2 — Core serving
├── [Bạn]          Skeleton files cho SRV-01..05, RNK-01..04
├── [Bạn]          aggregation/ (sau khi có ≥1 shard xong)
├── [Bạn]          indexing/ (sau aggregation)
├── [Researcher A] TASK-SRV-01, 02, 03, 04
└── [Researcher B] TASK-RNK-01, 02, 03

Tuần 3 — TRAKE + eval
├── [Researcher A] TASK-SRV-05 (event_splitter)
├── [Researcher B] TASK-RNK-04 (temporal_aligner)
├── [Researcher]   TASK-ING-04 (publish Kaggle Dataset)
├── [Bạn]          serving/core/ — model singleton, lifespan
└── [Researcher]   TASK-EVL-01, 02, 03

Tuần 4 — Hardening
├── [Bạn]          infra/ — Docker, CI/CD, smoke test
├── [Team]         Integration test end-to-end
└── [Bạn]          Collection alias + rollback plan trước ngày thi
```

---

## Quy ước đặt tên

| Thứ | Quy ước |
|---|---|
| Kaggle Dataset | `aic2026-shard-<tên_member>-v<n>` và `aic2026-unified-v<n>` |
| Qdrant collection | `collection_v<n>` — alias `production` trỏ vào version mới nhất |
| Point ID | `sha256(video_id + "_" + scene_id)[:16]` — không random |
| Branch | `feature/<TASK-ID>-<slug>` |
| Parquet schema | Luôn có cột `scene_id` (str) và `video_id` (str) làm key join |
| Skeleton file | Đặt trong `<module>/skeletons/` — bạn viết, không commit implementation vào đây |

---

*File này là source of truth cho phân công. Khi có task mới: bạn viết skeleton trước → thêm task vào đây → giao cho Researcher. Không giao miệng.*
