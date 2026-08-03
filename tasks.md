# AIC2026 — Project Task List & Member Assignments

> **Source of Truth**: Danh sách toàn bộ nhiệm vụ của hệ thống. 
> Các task đã hoàn thành `[x]` sẽ **không xuất hiện** trong bảng phân công ở cuối file để các thành viên tập trung làm các công việc còn lại.

---

## 📑 I. Danh Sách Nhiệm Vụ Toàn Hệ Thống

### 1. Module Ingestion (Xử lý Dữ liệu trên Kaggle)
- `[x]` **TASK-ING-01**: Trích xuất Keyframe từ video (`ingestion/skeletons/01a_keyframe_extraction.py`) — *Đã xong*
- `[x]` **TASK-ING-02a**: Trích xuất Text - OCR từ keyframe bằng PaddleOCR v6 (`ingestion/skeletons/01b_text_extraction.py`) — *Đã xong*
- `[x]` **TASK-ING-02b**: Trích xuất Text - ASR giọng nói bằng PhoWhisper (`ingestion/skeletons/01b_text_extraction.py`) — *Đã xong*
- `[x]` **TASK-ING-03a**: Sinh Vector Embedding Ảnh bằng BEiT3 + CLIP ViT-L/14 (`ingestion/skeletons/01c_embedding_generation.py`) — *Đã xong*
- `[x]` **TASK-ING-03b**: Sinh Vector Embedding Text bằng gte-Qwen2-1.5B (`ingestion/skeletons/01c_embedding_generation.py`) — *Đã xong*
- `[x]` **TASK-ING-04**: Validate & Export Shard Dataset lên Kaggle (`ingestion/skeletons/02_export_shard_dataset.py`) — *Đã xong*
- `[x]` **TASK-ING-05**: Checkpoint Manifest quản lý trạng thái (`ingestion/skeletons/checkpoint_manifest.py`) — *Đã xong*

### 2. Module Retrieval Serving (Truy vấn Tìm kiếm Real-time)
- `[x]` **TASK-SRV-01**: Text Query Retrieval 5 nhánh vectors (`serving/service/retrieval/skeletons/text_query.py`) — *Đã xong*
- `[x]` **TASK-SRV-02**: Image-to-Image Query Retrieval (`serving/service/retrieval/skeletons/image_query.py`) — *Đã xong*
- `[x]` **TASK-SRV-03**: Voice Query Retrieval (`serving/service/retrieval/skeletons/voice_query.py`) — *Đã xong*
- `[x]` **TASK-SRV-04**: Retrieval Pipeline Orchestrator (`serving/service/retrieval/skeletons/retrieval_pipeline.py`) — *Đã xong*
- `[x]` **TASK-SRV-05**: Tách chuỗi sự kiện TRAKE multi-event (`serving/service/retrieval/skeletons/text_query.py`) — *Đã xong*

### 3. Module Reranker & Temporal Alignment (Xếp hạng & Căn chỉnh)
- `[x]` **TASK-RNK-01**: ViRanker Text Reranker (`serving/service/reranker/skeletons/text_branch/viranker.py`) — *Đã xong*
- `[x]` **TASK-RNK-02**: Jina Reranker M0 Cross-modal (`serving/service/reranker/skeletons/crossmodal_branch/jina_reranker_m0.py`) — *Đã xong*
- `[x]` **TASK-RNK-03**: Fusion Score Integration (`serving/service/reranker/skeletons/fusion_score.py`) — *Đã xong*
- `[x]` **TASK-RNK-04**: Temporal Aligner cho sự kiện TRAKE (`serving/service/temporal_aligner.py`) — *Đã xong*

### 4. Module Evaluation & Benchmark (Đo đạc Đánh giá)
- `[x]` **TASK-EVL-01**: Tính chỉ số r_score cho KIS/VQA/TRAKE (`eval/skeletons/metrics/r_score.py`) — *Đã xong*
- `[x]` **TASK-EVL-02**: Tính điểm tổng hợp Final Score R@{1,5,20,50,100} (`eval/skeletons/metrics/final_score.py`) — *Đã xong*
- `[x]` **TASK-EVL-03**: Script chạy Benchmark thử nghiệm (`eval/skeletons/run_benchmark.py`) — *Đã xong*

### 5. Module Unit Testing (Kiểm thử Tự động)
- `[x]` **TASK-TST-01**: Test Fusion Score Reranking (`tests/test_fusion_score.py`) — *Đã xong*
- `[x]` **TASK-TST-02**: Test Query Parsing & TRAKE Alignment (`tests/test_query_parsing.py`) — *Đã xong*
- `[x]` **TASK-TST-03**: Test API Request/Response Schemas (`tests/test_api_schema.py`) — *Đã xong*
- `[x]` **TASK-TST-04**: Test Verification & Merge Shards (`tests/test_merge_shards.py`) — *Đã xong*
- `[x]` **TASK-TST-05**: Test Evaluation R-Score Metrics (`tests/test_r_score.py`) — *Đã xong*
- `[x]` **TASK-TST-06**: Test Keyframe Extraction & Checkpoint (`tests/test_keyframe_extraction.py`) — *Đã xong*
- `[x]` **TASK-TST-07**: Test Export Shard Dataset & Validation (`tests/test_export_shard_dataset.py`) — *Đã xong*

---

## 🎯 II. Bảng Phân Công Nhiệm Vụ

🎉 **TẤT CẢ CAC NHIỆM VỤ TRONG DỰ ÁN ĐÃ ĐƯỢC HOÀN THÀNH 100%!**

