# 🎯 Super Image Retrieval System for AIC2026

Hệ thống truy vấn video keyframe real-time phục vụ cuộc thi **AI Challenge HCMC 2026**.

Hỗ trợ 3 dạng task chính:
- **KIS** (Known-Item Search) — tìm keyframe theo mô tả text/voice
- **VQA** (Visual Question Answering) — tìm cảnh và trả lời câu hỏi
- **TRAKE** (Temporal Retrieval) — tìm chuỗi sự kiện theo thứ tự thời gian
- **Image Search** — tìm keyframe tương đồng với ảnh upload

---

## 🏗️ Kiến trúc tổng quan

```
ingestion (Kaggle) → aggregation → indexing (Qdrant) → serving (FastAPI)
```

| Tầng | Mô tả | Chạy ở đâu |
|---|---|---|
| `ingestion/` | Mỗi member chạy full pipeline trên shard video riêng | Kaggle Notebook |
| `aggregation/` | Gộp output tất cả member thành 1 dataset thống nhất | Local |
| `indexing/` | Đẩy vectors vào Qdrant (5 named vectors) | Local/Server |
| `serving/` | FastAPI API real-time (controller → service → repo) | Server thi |

---

## ⚡ Quick Start

### 1. Clone và cài đặt

```bash
git clone https://github.com/ThanhDanh1510/Super-Image-Retrieval-System-for-AIC2026.git
cd Super-Image-Retrieval-System-for-AIC2026

# Cài đặt dependencies
pip install -e ".[dev]"

# Copy file cấu hình môi trường
cp .env.example .env
# → Sửa .env theo máy của bạn (QDRANT_URL, DEVICE, ...)
```

### 2. Chạy Qdrant (Docker)

```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant:latest
```

### 3. Tạo Collection

```bash
# Xem trước schema (không tạo)
make create-collection-dry

# Tạo collection
make create-collection
```

### 4. Chạy API Server

```bash
# Dev mode (auto-reload)
make serve-reload

# Production mode
make serve
```

API docs: http://localhost:8000/docs

---

## 📁 Cấu trúc thư mục

```
project-root/
├── configs/              # Cấu hình tập trung (model versions, shard, fusion)
├── ingestion/            # Pipeline xử lý video trên Kaggle
├── aggregation/          # Gộp output từ các member
├── indexing/             # Tạo collection và upsert vectors vào Qdrant
├── serving/              # FastAPI API (controller → service → repo)
├── shared/               # Code dùng chung (schemas, model wrappers, utils)
├── eval/                 # Benchmark offline
├── infra/                # Docker, CI/CD, monitoring
├── data/                 # Local, KHÔNG commit git
├── tests/
├── AGENT_CONTEXT.md      # Context cho AI Agent (Researcher paste vào prompt)
├── pyproject.toml
├── Makefile
└── .env.example
```

Chi tiết: xem [`folder_structure.md`](folder_structure.md).

---

## 🔧 Các lệnh thường dùng

| Lệnh | Mô tả |
|---|---|
| `make install` | Cài đặt dependencies |
| `make serve-reload` | Chạy API server (dev mode) |
| `make create-collection` | Tạo Qdrant collection |
| `make lint` | Chạy ruff linter |
| `make test` | Chạy pytest |
| `make clean` | Xóa cache |

---

## 🤖 AI Models

| Model | Dùng cho | Dim |
|---|---|---|
| BEiT-large + CLIP ViT-L/14 | Image embedding | 1024 |
| gte-Qwen2-1.5B | Text embedding (OCR + ASR) | 1536 |
| PaddleOCR v6 | Nhận dạng chữ trong ảnh | — |
| PhoWhisper-medium | Nhận dạng giọng nói tiếng Việt | — |
| TransNetV2 | Shot detection | — |

Phiên bản chi tiết: xem [`configs/model_versions.yaml`](configs/model_versions.yaml).

---

## 📖 Tài liệu

- [`folder_structure.md`](folder_structure.md) — Cấu trúc thư mục chi tiết
- [`contest_info.md`](contest_info.md) — Thể lệ và timeline cuộc thi
- [`tasks.md`](tasks.md) — Phân công task và workflow
- [`AGENT_CONTEXT.md`](AGENT_CONTEXT.md) — Context cho AI Agent

---

## 👥 Team

| Vai trò | Trách nhiệm |
|---|---|
| **Leader** | Infrastructure, skeleton, review, deploy |
| **Member_1** | Retrieval (Text/Image/Voice Query) implementation |
| **Member_2** | Retrieval Pipeline & TRAKE (Event Splitter, Temporal Aligner) |
| **Member_3** | Reranker (ViRanker, Jina Reranker, Fusion) |
| **Member_4** | Evaluation (Metric calculation, Benchmark script) |
