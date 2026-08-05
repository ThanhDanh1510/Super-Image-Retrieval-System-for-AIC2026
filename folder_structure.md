# Cấu trúc thư mục dự án — Video Retrieval Pipeline (AI Challenge HCMC)

Dự án vận hành theo **4 nhịp**: **ingestion theo shard** (mỗi member chạy full pipeline trên 50–100 video của mình, trên Kaggle Notebook) → **aggregation** (gộp output của cả team thành 1 dataset thống nhất) → **indexing** (đẩy vào Qdrant) → **serving** (real-time lúc thi). Nguyên tắc xuyên suốt: không ai cần tải/chia sẻ video gốc (300GB+) sau khi đã xử lý — chỉ trao đổi keyframe đã crop, metadata, và vector, vì đó mới là dữ liệu phục vụ retrieval.

```
project-root/
├── AGENT_CONTEXT.md  # Project Context Block cho AI Agent — Leader viết và giữ, không delegate
├── configs/          # Cấu hình tập trung (model versions, shard assignments, fusion weights)
├── ingestion/        # Mỗi member chạy full pipeline trên shard riêng (Kaggle Notebook)
├── aggregation/      # Gộp output của tất cả member thành 1 dataset thống nhất
├── indexing/         # Pull dataset thống nhất, đẩy named vectors vào Qdrant
├── serving/          # Backend API real-time lúc thi (controller-service-repo)
├── shared/           # Code dùng chung (models, schemas, specs) giữa ingestion và serving
├── eval/             # Benchmark offline theo metric chấm điểm cuộc thi
├── infra/            # Docker, CI/CD, monitoring
├── data/             # Local, không commit git
├── docs/
├── tests/
├── pyproject.toml
└── Makefile
```

---

## `ingestion/` — Mỗi member chạy full pipeline trên shard riêng (Kaggle Notebook)

Mỗi thành viên nhận 1 shard video (theo `video_id`), tự chạy **toàn bộ** pipeline (shot detection → keyframe → OCR → ASR → embedding) trên Kaggle Notebook của mình, rồi publish kết quả thành 1 Kaggle Dataset riêng. Không ai cần đụng tới video của người khác.

```
ingestion/
├── skeletons/
│   ├── 01a_keyframe_extraction.py     # [MLOps viết] Skeleton script — function signatures, output schema, constraints nhúng sẵn
│   ├── 01b_text_extraction.py         # [MLOps viết] Skeleton script
│   ├── 01c_embedding_generation.py    # [MLOps viết] Skeleton script
│   ├── 02_export_shard_dataset.py     # [MLOps viết] Skeleton script
│   └── checkpoint_manifest.py         # [MLOps viết] Skeleton class — Member dùng Agent fill implementation
│                                      # (Chạy offline bằng CLI / notebook nháp trên Kaggle, không commit file .ipynb)
│
├── kaggle_utils/
│   ├── shard_loader.py        # Đọc shard_assignment.yaml, biết shard này gồm video nào
│   ├── checkpoint_manifest.py # Ghi scene_id đã xử lý — resume khi Kaggle session bị ngắt (granularity: scene, không phải video)
│   └── dataset_io.py          # Helper đọc/ghi đúng path Kaggle (/kaggle/working)
│
└── configs/
    └── ingest_runtime.yaml    # Cấu hình runtime cục bộ (batch size, device...) cho từng máy chạy shard
```

**Quy ước bắt buộc giữa các shard:**
- Tên Kaggle Dataset output theo convention `aic2026-shard-<member>-v<n>` (ví dụ `aic2026-shard-an-v1`) — để `aggregation/` biết pull đủ và đúng version của ai.
- Mọi shard phải tuân thủ đúng `configs/model_versions.yaml` ở root (cùng version model OCR/ASR/embedding) — nếu 1 member tự đổi model khác, vector của shard đó sẽ không tương thích với phần còn lại khi gộp.
- Point ID cho mỗi keyframe phải là hash deterministic từ `video_id + scene_id` (không random) — đảm bảo gộp nhiều shard không trùng ID và idempotent khi chạy lại.

---

## `aggregation/` — Gộp output của cả team thành 1 dataset thống nhất

Bước "bàn giao" duy nhất giữa các member: mount toàn bộ Kaggle Dataset đã publish, kiểm tra không thiếu/trùng video, gộp lại, publish thành 1 dataset duy nhất để `indexing/` dùng.

```
aggregation/
├── pull_shards.py          # Dùng Kaggle API pull toàn bộ dataset theo danh sách trong shard_assignment.yaml
├── verify_shards.py        # Check đủ video_id theo plan, phát hiện shard thiếu hoặc bị trùng video_id giữa 2 member
├── merge_datasets.py       # Script tổng hợp: gộp parquet (metadata+text) và .npy/.npz (vector) từ mọi shard
└── publish_unified.py      # Publish dataset đã gộp: `aic2026-unified-v<n>`
```

**Lưu ý:** `verify_shards.py` chạy trước khi merge — nếu phát hiện 2 member vô tình xử lý trùng `video_id`, hoặc thiếu video so với `shard_assignment.yaml`, dừng lại và báo trước khi gộp, tránh phải merge lại từ đầu.

---

## `indexing/` — Pull dataset thống nhất, đẩy named vectors vào Qdrant

```
indexing/
├── pull_unified/
│   └── download_dataset.py     # Pull đúng version `aic2026-unified-v<n>` từ aggregation/ về máy chạy Qdrant
│
├── schema/
│   └── collection_schema.py    # Named vectors: image_dense (BEiT3+CLIP), ocr_dense (gte-Qwen2-1.5B),
│                                # asr_dense (gte-Qwen2-1.5B), ocr_sparse, asr_sparse — định nghĩa 1 lần, dùng cho mọi lần index.
│                                # (Tham số RRF đọc từ configs/fusion_config.yaml ở root)
│
├── upsert/
│   ├── upsert_vectors.py       # Batch upsert (100–500 point/lần), point ID = hash(video_id+scene_id)
│   └── snapshot.py             # Trigger Qdrant snapshot ngay sau mỗi lần upsert thành công — để rollback khi cần
│
├── collection_alias.py         # Alias "production" → collection_vN, đổi tức thì khi rollback
└── run_index.py                  # Entry point: `python run_index.py --dataset aic2026-unified-v2 --collection collection_v2`
```

**Thêm batch sau (video mới phát sinh, không re-run pipeline cũ):** member phụ trách video mới chạy lại `ingestion/` chỉ cho phần mới → publish shard dataset mới → `aggregation/` merge thêm (không đụng dataset cũ) → `indexing/` upsert thêm vào collection hiện có → snapshot lại.

---

## `serving/` — Backend API chạy lúc thi (real-time), layer hoá controller-service-repo

```
serving/
└── api/
    ├── core/                    # config.py, dependencies.py (DB client, model singleton), lifespan.py (load model lúc startup)
    ├── schema/                  # Pydantic request/response cho KIS/VQA/TRAKE & Similar Image Search
    ├── controller/              # routes_kis.py, routes_similar_image.py... — chỉ gọi service/; nếu input_type=voice, gọi voice_to_text trước
    │                            # khi truyền tiếp vào retrieval_service, không tạo route riêng cho voice
    ├── service/
    │   ├── retrieval/
    │   │   ├── text_query.py           # Dịch thuật, tách sự kiện (event splitter cho TRAKE), tạo vector và truy vấn text
    │   │   ├── image_query.py          # Trích xuất vector ảnh và truy vấn ảnh tương đồng (Image-to-Image)
    │   │   ├── voice_query.py          # Chuyển giọng nói → text (voice-to-text) và tái sử dụng luồng của text_query
    │   │   └── retrieval_pipeline.py   # Orchestrator: nhận query đầu vào, điều hướng, trả về danh sách candidates thô
    │   ├── temporal_aligner.py         # Căn chỉnh thời gian (TRAKE) sau khi đã retrieve candidates
    │   └── reranker/
    │       ├── skeletons/              # Skeleton files cho Member_3
    │       │   ├── text_branch/viranker.py
    │       │   ├── crossmodal_branch/jina_reranker_m0.py
    │       │   └── fusion_score.py
    │       ├── rerank_service.py       # Seam duy nhất controller gọi cho rerank (text + crossmodal + fusion)
    ├── repo/                    # qdrant_repo.py (prefetch + FusionQuery RRF, hỗ trợ cả truy vấn text & image), gg_image_search_repo.py (fallback)
    └── main.py                  # FastAPI app, mount router
```

**Quy ước một chiều:** `controller → service → repo`, không ngược lại — giúp test từng layer độc lập (ví dụ test `fusion_score.py` không cần FastAPI app hay Qdrant thật).

---

## `shared/` — Dùng chung giữa ingestion (Kaggle) và serving

```
shared/
├── models/
│   ├── text_embedding_wrapper.py   # Load gte-Qwen2 — dùng y nguyên trong notebook Kaggle và serving
│   └── image_embedding_wrapper.py  # Load BEiT3/CLIP
└── schemas/
    └── named_vector_spec.py        # Spec named vector dùng chung giữa indexing/ và serving/repo/
```

> Vì `ingestion/` chạy trên Kaggle (môi trường tách biệt), `shared/` nên đóng gói thành 1 package nhỏ (hoặc cài đặt dạng editable `pip install -e shared/` ở local) để mọi member và serving import đúng cùng phiên bản, tránh copy-paste lệch nhau giữa các shard.

---

## `configs/` — Quản lý cấu hình tập trung cho toàn bộ hệ thống (Root)

```
configs/
├── shard_assignment.yaml   # Map member → danh sách video_id phụ trách (tránh trùng/sót)
├── model_versions.yaml     # Ghim phiên bản các model AI (OCR, ASR, Embeddings) dùng chung
└── fusion_config.yaml      # Cấu hình tham số Fusion (RRF) phục vụ indexing và retrieval
```

> Mọi module (`ingestion`, `indexing`, `serving`) đều đọc cấu hình chung từ thư mục `configs/` ở root này để đảm bảo tính đồng nhất tham số và phiên bản model.

---

## `eval/` — Benchmark offline theo đúng metric chấm điểm cuộc thi

```
eval/
├── skeletons/
│   ├── metrics/
│   │   ├── r_score.py      # [MLOps viết] Skeleton — công thức nhúng trong docstring, Agent không được suy diễn
│   │   └── final_score.py  # [MLOps viết] Skeleton
│   └── run_benchmark.py    # [MLOps viết] Skeleton
├── queries/           # sample_queries_kis.json, sample_queries_vqa.json, sample_queries_trake.json (kèm ground truth)
├── metrics/           # r_score.py, final_score.py (Final Score = trung bình R@{1,5,20,50,100})
└── run_benchmark.py   # So sánh phương án: đổi model, đổi trọng số fusion, có/không reranker
```

---

## `infra/` — Vận hành, CI/CD, rollback

```
infra/
├── docker/Dockerfile.serving       # Image nhẹ cho serving/ (text encoder + reranker) — KHÔNG có Dockerfile.ingestion (chạy Kaggle)
├── compose/                        # docker-compose.dev.yml (Qdrant + serving local), docker-compose.contest.yml (server thi)
├── monitoring/                     # prometheus.yml, grafana_dashboard.json — theo dõi latency lúc thi
├── ci/                             # lint_test.yml, build_push.yml, smoke_test.py (gọi thử query mẫu trước deploy)
└── kaggle/kaggle.json.example      # Mẫu credential Kaggle API cho aggregation/pull_shards.py
```

---

## `data/` — Local, không commit git, không có video chia sẻ giữa member

```
data/
├── raw_shard/          # Video gốc CHỈ thuộc shard của mình — KHÔNG upload/chia sẻ nguyên video
├── shard_output_v3/    # Output (keyframe, metadata, vector) trước khi publish Kaggle Dataset riêng
├── unified_v3/         # Bản đã gộp từ aggregation/ (gồm metadata parquet và vector npy/npz)
├── unified/            # Toàn bộ ảnh keyframe của tất cả các shard phục vụ hiển thị UI — tên chuẩn
│                       # {video_id}-{scene_id}-{frame_id}.webp (flat, không thư mục con)
└── unified_keyframes/  # (CŨ, đã thay bằng unified/) — xóa sau khi mapping
```

---

## `docs/`

```
docs/
├── contest_info.md             # Timeline, thể lệ, giải thưởng
├── task_scoring_context.md     # Mô tả 3 task (KIS/VQA/TRAKE) + công thức chấm điểm
└── architecture_decisions/     # ADR — lý do chọn Qdrant/RRF, lý do chia shard theo video_id...
```

---

## `tests/`

```
tests/
├── test_fusion_score.py     # Logic fusion 2 nhánh reranker
├── test_query_parsing.py    # event_splitter, translate_enhance
├── test_api_schema.py       # Endpoint trả đúng schema response
├── test_merge_shards.py     # verify_shards.py không bỏ sót/trùng video_id khi gộp dataset
└── test_r_score.py          # Edge cases r_score theo từng task type (KIS/VQA/TRAKE)
```

---

## File gốc ở root

```
AGENT_CONTEXT.md   # Project Context Block cho AI Agent — Leader viết và giữ
                   # Member copy block này vào đầu prompt mỗi khi dùng Agent
                   # Nội dung: pipeline overview, named vector spec, import rules, constraint cứng
pyproject.toml     # Dependency, version Python, entry points
Makefile           # make ingest-shard / make aggregate / make index / make serve / make benchmark
```

---

### Nguyên tắc thiết kế

`ingestion/` tách theo **shard**, không theo **giai đoạn xử lý** — mỗi member tự chạy hết toàn bộ chain (shot detection → OCR/ASR → embedding) trên video của mình, vì cách này không đòi hỏi đồng bộ giữa các member trong lúc chạy (chỉ cần đồng bộ ở `configs/shard_assignment.yaml` và `configs/model_versions.yaml` ở root trước khi bắt đầu), và mỗi người có thể chạy độc lập, không chờ nhau.

`aggregation/` là lớp mới, đứng giữa `ingestion/` và `indexing/`, vì giờ có N dataset (N member) cần gộp thành 1 trước khi index — tách riêng để `verify_shards.py` bắt lỗi thiếu/trùng video_id sớm, trước khi tốn công upsert vào Qdrant.

`indexing/` chỉ làm việc với 1 dataset thống nhất duy nhất (`aic2026-unified-v<n>`), không cần biết gì về việc dataset đó được gộp từ bao nhiêu shard — giữ logic upsert đơn giản, và việc thêm batch video mới sau này (dù từ 1 hay nhiều member) đều đi qua đúng 1 luồng `aggregation → indexing`.

`skeletons/` trong mỗi module là ranh giới rõ ràng giữa **Leader** (viết skeleton) và **Member + AI Agent** (fill implementation). Skeleton chứa function signature, type hint, docstring với constraint cứng, và `raise NotImplementedError` — Agent đọc code tốt hơn đọc prose, nên skeleton chính là spec. Không commit implementation vào `skeletons/` — đó là nơi Leader đặt contract, không phải nơi chứa code chạy thật.

`AGENT_CONTEXT.md` ở root thay thế hoàn toàn cho ý tưởng tạo folder `agents/` riêng — Agent không phải thành phần của hệ thống, chỉ là công cụ viết code. File này chứa Project Context Block mà Member paste vào đầu mỗi prompt, giúp Agent hiểu constraint toàn hệ thống mà không cần giải thích miệng.

Nguyên tắc dữ liệu xuyên suốt: **video gốc không bao giờ rời khỏi máy/Kaggle account của member phụ trách**; thứ duy nhất di chuyển giữa người với người, và giữa các tầng trong hệ thống, là keyframe đã crop, metadata, và vector — vì đó là tất cả những gì retrieval cần.
