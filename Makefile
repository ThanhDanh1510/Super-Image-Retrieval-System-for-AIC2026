# =============================================================================
# Makefile — AIC2026 Super Image Retrieval System
# =============================================================================
# Entry point nhanh cho các thao tác thường xuyên.
# Chạy: make <target>
# =============================================================================

.PHONY: help install install-dev install-ingestion serve serve-reload \
        lint typecheck test test-cov \
        create-collection verify-collection \
        clean

# --- Biến mặc định ---
PYTHON ?= python
UVICORN ?= uvicorn
HOST ?= 0.0.0.0
PORT ?= 8000
QDRANT_URL ?= http://localhost:6333

# =============================================================================
# Help
# =============================================================================

help: ## Hiển thị danh sách commands
	@echo "=== AIC2026 Makefile ==="
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

# =============================================================================
# Install
# =============================================================================

install: ## Cài đặt dependencies chính (serving + shared)
	pip install -e ".[dev]"

install-dev: ## Cài đặt dependencies + dev tools (pytest, ruff, mypy)
	pip install -e ".[dev]"

install-ingestion: ## Cài đặt thêm dependencies cho ingestion (Kaggle)
	pip install -e ".[ingestion]"

# =============================================================================
# Serving
# =============================================================================

serve: ## Chạy API server (production mode)
	$(UVICORN) serving.api.main:app --host $(HOST) --port $(PORT)

serve-reload: ## Chạy API server (dev mode, auto-reload)
	$(UVICORN) serving.api.main:app --host $(HOST) --port $(PORT) --reload

# =============================================================================
# Indexing & Aggregation
# =============================================================================

create-collection: ## Tạo Qdrant collection với schema đầy đủ
	$(PYTHON) -m indexing.schema.collection_schema --url $(QDRANT_URL)

create-collection-dry: ## In ra schema collection mà không tạo
	$(PYTHON) -m indexing.schema.collection_schema --url $(QDRANT_URL) --dry-run

verify-collection: ## Kiểm tra collection đã tạo đúng schema chưa
	$(PYTHON) -c "from indexing.schema.collection_schema import *; \
		client = QdrantClient(url='$(QDRANT_URL)'); \
		print('OK' if verify_collection(client) else 'FAILED')"

pull-shards: ## Tải toàn bộ shard datasets từ Kaggle (yêu cầu --owner)
	$(PYTHON) -m aggregation.pull_shards --owner $(OWNER)

verify-shards: ## Xác thực các shard datasets
	$(PYTHON) -m aggregation.verify_shards

merge-datasets: ## Gộp các shard datasets thành bộ thống nhất
	$(PYTHON) -m aggregation.merge_datasets

publish-unified: ## Đóng gói và publish Unified Dataset lên Kaggle (yêu cầu --owner)
	$(PYTHON) -m aggregation.publish_unified --owner $(OWNER)

# =============================================================================
# Evaluation
# =============================================================================

benchmark: ## Chạy offline benchmark đánh giá độ chính xác và latency
	$(PYTHON) -m eval.skeletons.run_benchmark


# =============================================================================
# Code Quality
# =============================================================================

lint: ## Chạy ruff linter
	ruff check .

lint-fix: ## Chạy ruff linter và tự động fix
	ruff check . --fix

typecheck: ## Chạy mypy type checker
	mypy shared/ serving/ indexing/ --ignore-missing-imports

# =============================================================================
# Testing
# =============================================================================

test: ## Chạy pytest
	pytest tests/ -v

test-cov: ## Chạy pytest với coverage report
	pytest tests/ -v --cov=serving --cov=shared --cov-report=html

# =============================================================================
# Cleanup
# =============================================================================

clean: ## Xóa cache và build artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf dist/ build/ htmlcov/ .coverage
