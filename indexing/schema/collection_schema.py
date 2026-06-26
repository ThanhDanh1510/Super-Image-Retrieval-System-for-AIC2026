"""
indexing/schema/collection_schema.py
=====================================
Script tạo và cấu hình Qdrant collection cho dự án AIC2026.

Chạy script này MỘT LẦN trước khi bắt đầu upsert dữ liệu.
Script sẽ tạo collection với tất cả named vectors (dense + sparse)
theo đúng spec từ shared/schemas/named_vector_spec.py.

CÁCH CHẠY:
    python -m indexing.schema.collection_schema

    # Hoặc chỉ in ra config mà không tạo:
    python -m indexing.schema.collection_schema --dry-run

IMPORT RULES:
    - Import spec từ: shared/schemas/named_vector_spec.py
    - Import config từ: shared/utils/config_loader.py
    - KHÔNG import từ serving/ (indexing và serving là independent).
"""

from __future__ import annotations

import argparse
import logging
import sys

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    SparseVectorParams,
    Distance,
    CollectionStatus,
    OptimizersConfigDiff,
    HnswConfigDiff,
)

from shared.schemas.named_vector_spec import (
    ALL_VECTOR_SPECS,
    VectorSpec,
    IMAGE_DENSE,
    OCR_DENSE,
    ASR_DENSE,
    OCR_SPARSE,
    ASR_SPARSE,
)

logger = logging.getLogger(__name__)

# Tên collection chính — tạo alias "production" trỏ về collection này
COLLECTION_NAME = "aic2026_v1"
PRODUCTION_ALIAS = "production"


# =============================================================================
# Schema Builder
# =============================================================================

def build_vectors_config() -> dict[str, VectorParams]:
    """
    Tạo cấu hình dense vectors từ ALL_VECTOR_SPECS.

    Chỉ lấy các VectorSpec có vector_type == 'dense'.
    Map distance string ('Cosine', 'Dot', 'Euclid') sang qdrant Distance enum.

    Returns:
        Dict mapping tên vector → VectorParams.
        Ví dụ:
            {
                'image_dense': VectorParams(size=1024, distance=Distance.COSINE),
                'ocr_dense': VectorParams(size=1536, distance=Distance.COSINE),
                'asr_dense': VectorParams(size=1536, distance=Distance.COSINE),
            }
    """
    raise NotImplementedError


def build_sparse_vectors_config() -> dict[str, SparseVectorParams]:
    """
    Tạo cấu hình sparse vectors từ ALL_VECTOR_SPECS.

    Chỉ lấy các VectorSpec có vector_type == 'sparse'.

    Returns:
        Dict mapping tên vector → SparseVectorParams.
        Ví dụ:
            {
                'ocr_sparse': SparseVectorParams(),
                'asr_sparse': SparseVectorParams(),
            }
    """
    raise NotImplementedError


# =============================================================================
# Collection Management
# =============================================================================

def create_collection(
    client: QdrantClient,
    collection_name: str = COLLECTION_NAME,
    recreate: bool = False,
) -> None:
    """
    Tạo Qdrant collection với schema đầy đủ (dense + sparse vectors).

    Args:
        client: QdrantClient đã kết nối.
        collection_name: Tên collection. Mặc định 'aic2026_v1'.
        recreate: Nếu True, xóa collection cũ và tạo lại.
                  Nếu False và collection đã tồn tại, skip.

    Raises:
        RuntimeError: Nếu collection đã tồn tại và recreate=False.
    """
    raise NotImplementedError


def create_production_alias(
    client: QdrantClient,
    collection_name: str = COLLECTION_NAME,
    alias: str = PRODUCTION_ALIAS,
) -> None:
    """
    Tạo alias 'production' trỏ về collection_name.

    Serving layer luôn truy vấn qua alias 'production'.
    Khi cần swap collection mới (ví dụ: re-index), chỉ cần đổi alias.

    Args:
        client: QdrantClient đã kết nối.
        collection_name: Tên collection thực. Mặc định 'aic2026_v1'.
        alias: Tên alias. Mặc định 'production'.
    """
    raise NotImplementedError


def verify_collection(
    client: QdrantClient,
    collection_name: str = COLLECTION_NAME,
) -> bool:
    """
    Kiểm tra collection đã tồn tại và có schema đúng không.

    Validate:
        - Collection tồn tại và status == GREEN
        - Đủ 3 dense vectors (image_dense, ocr_dense, asr_dense) với đúng dim
        - Đủ 2 sparse vectors (ocr_sparse, asr_sparse)

    Args:
        client: QdrantClient đã kết nối.
        collection_name: Tên collection cần verify.

    Returns:
        True nếu collection hợp lệ, False nếu không.
    """
    raise NotImplementedError


def print_collection_info(
    client: QdrantClient,
    collection_name: str = COLLECTION_NAME,
) -> None:
    """
    In ra thông tin chi tiết của collection (debug/logging).

    Bao gồm: tên, số points, danh sách vectors, status, aliases.

    Args:
        client: QdrantClient đã kết nối.
        collection_name: Tên collection.
    """
    raise NotImplementedError


# =============================================================================
# CLI Entry Point
# =============================================================================

def main() -> None:
    """
    CLI entry point — chạy khi gọi `python -m indexing.schema.collection_schema`.

    Options:
        --url: Qdrant server URL (mặc định: http://localhost:6333)
        --api-key: Qdrant API key (nếu có)
        --recreate: Xóa collection cũ và tạo lại
        --dry-run: Chỉ in ra config mà không tạo collection
    """
    parser = argparse.ArgumentParser(
        description="Tạo Qdrant collection cho AIC2026"
    )
    parser.add_argument(
        "--url", default="http://localhost:6333",
        help="Qdrant server URL",
    )
    parser.add_argument(
        "--api-key", default=None,
        help="Qdrant API key (nếu có)",
    )
    parser.add_argument(
        "--recreate", action="store_true",
        help="Xóa collection cũ và tạo lại từ đầu",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Chỉ in ra config mà không tạo collection",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # In cấu hình
    vectors_config = build_vectors_config()
    sparse_config = build_sparse_vectors_config()

    logger.info("=== Dense Vectors ===")
    for name, params in vectors_config.items():
        logger.info(f"  {name}: dim={params.size}, distance={params.distance}")

    logger.info("=== Sparse Vectors ===")
    for name in sparse_config:
        logger.info(f"  {name}: sparse")

    if args.dry_run:
        logger.info("Dry-run mode — không tạo collection.")
        return

    # Tạo collection
    client = QdrantClient(url=args.url, api_key=args.api_key)
    create_collection(client, recreate=args.recreate)
    create_production_alias(client)

    # Verify
    if verify_collection(client):
        logger.info("✅ Collection tạo thành công và schema hợp lệ.")
        print_collection_info(client)
    else:
        logger.error("❌ Collection tạo thất bại hoặc schema không khớp.")
        sys.exit(1)


if __name__ == "__main__":
    main()
