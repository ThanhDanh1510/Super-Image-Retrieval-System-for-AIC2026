# AIC2026 — Project Task List & Member Assignments

> **Source of Truth**: Danh sách toàn bộ nhiệm vụ của hệ thống. 
> Các task đã hoàn thành `[x]` sẽ **không xuất hiện** trong bảng phân công ở cuối file để các thành viên tập trung làm các công việc còn lại.

---

## 📑 I. Danh Sách Nhiệm Vụ Toàn Hệ Thống

### 1. Module Ingestion (Xử lý Dữ liệu trên Kaggle)
- `[x]` **TASK-ING-01**: Trích xuất Keyframe từ video (`ingestion/skeletons/01a_keyframe_extraction.py`) — *Đã xong*
- `[x]` **TASK-ING-02a**: Trích xuất Text - OCR từ keyframe bằng PaddleOCR v6 (`ingestion/skeletons/01b_text_extraction.py`) — *Đã xong*
- `[ ]` **TASK-ING-02b**: Trích xuất Text - ASR giọng nói bằng PhoWhisper (`ingestion/skeletons/01b_text_extraction.py`)
- `[ ]` **TASK-ING-03a**: Sinh Vector Embedding Ảnh bằng BEiT3 + CLIP ViT-L/14 (`ingestion/skeletons/01c_embedding_generation.py`)
- `[ ]` **TASK-ING-03b**: Sinh Vector Embedding Text bằng gte-Qwen2-1.5B (`ingestion/skeletons/01c_embedding_generation.py`)
- `[x]` **TASK-ING-04**: Validate & Export Shard Dataset lên Kaggle (`ingestion/skeletons/02_export_shard_dataset.py`) — *Đã xong*
- `[x]` **TASK-ING-05**: Checkpoint Manifest quản lý trạng thái (`ingestion/skeletons/checkpoint_manifest.py`) — *Đã xong*

### 2. Module Retrieval Serving (Truy vấn Tìm kiếm Real-time)
- `[x]` **TASK-SRV-01**: Text Query Retrieval 5 nhánh vectors (`serving/service/retrieval/skeletons/text_query.py`) — *Đã xong*
- `[x]` **TASK-SRV-02**: Image-to-Image Query Retrieval (`serving/service/retrieval/skeletons/image_query.py`) — *Đã xong*
- `[x]` **TASK-SRV-03**: Voice Query Retrieval (`serving/service/retrieval/skeletons/voice_query.py`) — *Đã xong*
- `[x]` **TASK-SRV-04**: Retrieval Pipeline Orchestrator (`serving/service/retrieval/skeletons/retrieval_pipeline.py`) — *Đã xong*
- `[x]` **TASK-SRV-05**: Tách chuỗi sự kiện TRAKE multi-event (`serving/service/retrieval/skeletons/text_query.py`) — *Đã xong*

### 3. Module Reranker & Temporal Alignment (Xếp hạng & Căn chỉnh)
- `[ ]` **TASK-RNK-01**: ViRanker Text Reranker (`serving/service/reranker/skeletons/text_branch/viranker.py`)
- `[ ]` **TASK-RNK-02**: Jina Reranker M0 Cross-modal (`serving/service/reranker/skeletons/crossmodal_branch/jina_reranker_m0.py`)
- `[ ]` **TASK-RNK-03**: Fusion Score Integration (`serving/service/reranker/skeletons/fusion_score.py`)
- `[ ]` **TASK-RNK-04**: Temporal Aligner cho sự kiện TRAKE (`serving/service/temporal_aligner.py`)

### 4. Module Evaluation & Benchmark (Đo đạc Đánh giá)
- `[x]` **TASK-EVL-01**: Tính chỉ số r_score cho KIS/VQA/TRAKE (`eval/skeletons/metrics/r_score.py`) — *Đã xong*
- `[x]` **TASK-EVL-02**: Tính điểm tổng hợp Final Score R@{1,5,20,50,100} (`eval/skeletons/metrics/final_score.py`) — *Đã xong*
- `[x]` **TASK-EVL-03**: Script chạy Benchmark thử nghiệm (`eval/skeletons/run_benchmark.py`) — *Đã xong*

### 5. Module Unit Testing (Kiểm thử Tự động)
- `[x]` **TASK-TST-03**: Test API Request/Response Schemas (`tests/test_api_schema.py`) — *Đã xong*
- `[x]` **TASK-TST-04**: Test Verification & Merge Shards (`tests/test_merge_shards.py`) — *Đã xong*
- `[x]` **TASK-TST-06**: Test Keyframe Extraction & Checkpoint (`tests/test_keyframe_extraction.py`) — *Đã xong*
- `[x]` **TASK-TST-07**: Test Export Shard Dataset & Validation (`tests/test_export_shard_dataset.py`) — *Đã xong*
- `[ ]` **TASK-TST-01**: Test Fusion Score Reranking (`tests/test_fusion_score.py`)
- `[ ]` **TASK-TST-02**: Test Query Parsing & TRAKE Alignment (`tests/test_query_parsing.py`)
- `[x]` **TASK-TST-05**: Test Evaluation R-Score Metrics (`tests/test_r_score.py`) — *Đã xong*

---

## 🎯 II. Bảng Phân Công Nhiệm Vụ (Chỉ chứa các Task CẦN LÀM)

> Các công việc dưới đây được phân chia cho từng thành viên. Mỗi thành viên mở file skeleton tương ứng, copy `AGENT_CONTEXT.md` và nhờ AI Agent (Claude/Cursor) điền mã nguồn hoàn chỉnh.

| Thành viên | Mã Task | File Skeleton cần làm | Chi tiết công việc |
|---|---|---|---|
| **Member_1** | `TASK-ING-02a` | `ingestion/skeletons/01b_text_extraction.py` | Implement hàm `run_ocr()` dùng PaddleOCR v6 |
| | `TASK-SRV-01` | `serving/service/retrieval/skeletons/text_query.py` | Implement `query_by_text()` truy vấn 5 vectors trên Qdrant |
| | `TASK-RNK-01` | `serving/service/reranker/skeletons/text_branch/viranker.py` | Implement ViRanker text reranking |
| | `TASK-EVL-01` | `eval/skeletons/metrics/r_score.py` | Implement công thức tính `r_score` cho KIS/VQA/TRAKE |
| | `TASK-TST-05` | `tests/test_r_score.py` | Viết unit tests kiểm thử r_score |
|---|---|---|---|
| **Member_2** | `TASK-ING-02b` | `ingestion/skeletons/01b_text_extraction.py` | Implement hàm `run_asr()` dùng PhoWhisper |
| | `TASK-SRV-02` | `serving/service/retrieval/skeletons/image_query.py` | Implement `query_by_image()` tìm kiếm theo ảnh |
| | `TASK-RNK-02` | `serving/service/reranker/skeletons/crossmodal_branch/jina_reranker_m0.py` | Implement Jina Reranker M0 cross-modal |
| | `TASK-EVL-02` | `eval/skeletons/metrics/final_score.py` | Implement công thức `final_score` mean R@k |
|---|---|---|---|
| **Member_3** | `TASK-ING-03a` | `ingestion/skeletons/01c_embedding_generation.py` | Implement `generate_image_embeddings()` dùng BEiT3 + CLIP |
| | `TASK-SRV-03` | `serving/service/retrieval/skeletons/voice_query.py` | Implement `query_by_voice()` xử lý audio query |
| | `TASK-SRV-05` | `serving/service/retrieval/skeletons/text_query.py` | Implement `split_trake_events()` tách sự kiện TRAKE |
| | `TASK-RNK-03` | `serving/service/reranker/skeletons/fusion_score.py` | Implement `fuse()` kết hợp điểm từ 2 nhánh rerank |
| | `TASK-EVL-03` | `eval/skeletons/run_benchmark.py` | Implement script chạy benchmark thử nghiệm |
| | `TASK-TST-01` | `tests/test_fusion_score.py` | Viết unit tests cho Fusion Score |
|---|---|---|---|
| **Member_4** | `TASK-ING-03b` | `ingestion/skeletons/01c_embedding_generation.py` | Implement `generate_text_embeddings()` dùng gte-Qwen2-1.5B |
| | `TASK-SRV-04` | `serving/service/retrieval/skeletons/retrieval_pipeline.py` | Implement `retrieve()` orchestrator |
| | `TASK-RNK-04` | `serving/service/temporal_aligner.py` | Implement `align()` căn chỉnh thứ tự thời gian cho TRAKE |
| | `TASK-TST-02` | `tests/test_query_parsing.py` | Viết unit tests cho query parsing & TRAKE alignment |
