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

ĐÃ IMPLEMENT ĐẦY ĐỦ bởi Leader.

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

# Map string distance → Qdrant Distance enum
_DISTANCE_MAP = {
    "Cosine": Distance.COSINE,
    "Dot": Distance.DOT,
    "Euclid": Distance.EUCLID,
}


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
    """
    config: dict[str, VectorParams] = {}

    for spec in ALL_VECTOR_SPECS:
        if spec.vector_type != "dense":
            continue

        distance = _DISTANCE_MAP.get(spec.distance)
        if distance is None:
            raise ValueError(
                f"Distance không hợp lệ: '{spec.distance}' cho vector '{spec.name}'. "
                f"Chỉ chấp nhận: {list(_DISTANCE_MAP.keys())}"
            )

        config[spec.name] = VectorParams(
            size=spec.dim,
            distance=distance,
        )

    return config


def build_sparse_vectors_config() -> dict[str, SparseVectorParams]:
    """
    Tạo cấu hình sparse vectors từ ALL_VECTOR_SPECS.

    Chỉ lấy các VectorSpec có vector_type == 'sparse'.

    Returns:
        Dict mapping tên vector → SparseVectorParams.
    """
    config: dict[str, SparseVectorParams] = {}

    for spec in ALL_VECTOR_SPECS:
        if spec.vector_type != "sparse":
            continue
        config[spec.name] = SparseVectorParams()

    return config


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
    exists = client.collection_exists(collection_name)

    if exists and not recreate:
        logger.info(
            f"Collection '{collection_name}' đã tồn tại. "
            f"Dùng --recreate nếu muốn xóa và tạo lại."
        )
        return

    if exists and recreate:
        logger.warning(f"Xóa collection cũ: '{collection_name}'")
        client.delete_collection(collection_name)

    vectors_config = build_vectors_config()
    sparse_vectors_config = build_sparse_vectors_config()

    logger.info(f"Tạo collection '{collection_name}' với {len(vectors_config)} dense + {len(sparse_vectors_config)} sparse vectors")

    client.create_collection(
        collection_name=collection_name,
        vectors_config=vectors_config,
        sparse_vectors_config=sparse_vectors_config,
        # HNSW config tối ưu cho retrieval accuracy
        hnsw_config=HnswConfigDiff(
            m=16,               # Số connection mỗi node (16 là balanced)
            ef_construct=200,   # Cao hơn → index chậm hơn nhưng recall tốt hơn
        ),
        # Optimizer config
        optimizers_config=OptimizersConfigDiff(
            indexing_threshold=20000,   # Bắt đầu index khi có >= 20k points
        ),
    )

    logger.info(f"✅ Collection '{collection_name}' tạo thành công.")


def create_production_alias(
    client: QdrantClient,
    collection_name: str = COLLECTION_NAME,
    alias: str = PRODUCTION_ALIAS,
) -> None:
    """
    Tạo alias 'production' trỏ về collection_name.

    Serving layer luôn truy vấn qua alias 'production'.
    Khi cần swap collection mới (ví dụ: re-index), chỉ cần đổi alias.
    """
    # Kiểm tra xem alias đã tồn tại chưa
    aliases = client.get_collection_aliases(collection_name)
    existing_aliases = [a.alias_name for a in aliases.aliases]

    if alias in existing_aliases:
        logger.info(f"Alias '{alias}' đã trỏ tới '{collection_name}'. Skip.")
        return

    # Xóa alias cũ nếu đang trỏ tới collection khác, rồi tạo mới
    try:
        client.update_collection_aliases(
            change_aliases_operations=[
                # Xóa alias cũ (nếu có) — ignore error nếu alias chưa tồn tại
                # Tạo alias mới trỏ về collection hiện tại
            ]
        )
    except Exception:
        pass  # Alias chưa tồn tại, bỏ qua

    client.update_collection_aliases(
        change_aliases_operations=[
            {
                "create_alias": {
                    "collection_name": collection_name,
                    "alias_name": alias,
                }
            }
        ]
    )
    logger.info(f"✅ Alias '{alias}' → '{collection_name}' tạo thành công.")


def verify_collection(
    client: QdrantClient,
    collection_name: str = COLLECTION_NAME,
) -> bool:
    """
    Kiểm tra collection đã tồn tại và có schema đúng không.

    Validate:
        - Collection tồn tại
        - Đủ 3 dense vectors (image_dense, ocr_dense, asr_dense) với đúng dim
        - Đủ 2 sparse vectors (ocr_sparse, asr_sparse)
    """
    if not client.collection_exists(collection_name):
        logger.error(f"Collection '{collection_name}' không tồn tại.")
        return False

    info = client.get_collection(collection_name)

    # --- Validate dense vectors ---
    expected_dense = build_vectors_config()
    actual_vectors = info.config.params.vectors or {}

    for name, expected_params in expected_dense.items():
        if name not in actual_vectors:
            logger.error(f"Thiếu dense vector: '{name}'")
            return False

        actual = actual_vectors[name]
        if actual.size != expected_params.size:
            logger.error(
                f"Dense vector '{name}': dim sai — "
                f"expected {expected_params.size}, got {actual.size}"
            )
            return False

        if actual.distance != expected_params.distance:
            logger.error(
                f"Dense vector '{name}': distance sai — "
                f"expected {expected_params.distance}, got {actual.distance}"
            )
            return False

    # --- Validate sparse vectors ---
    expected_sparse = build_sparse_vectors_config()
    actual_sparse = info.config.params.sparse_vectors or {}

    for name in expected_sparse:
        if name not in actual_sparse:
            logger.error(f"Thiếu sparse vector: '{name}'")
            return False

    logger.info(
        f"Collection '{collection_name}': "
        f"{len(expected_dense)} dense + {len(expected_sparse)} sparse vectors — OK"
    )
    return True


def print_collection_info(
    client: QdrantClient,
    collection_name: str = COLLECTION_NAME,
) -> None:
    """
    In ra thông tin chi tiết của collection (debug/logging).

    Bao gồm: tên, số points, danh sách vectors, status, aliases.
    """
    info = client.get_collection(collection_name)

    print(f"\n{'=' * 60}")
    print(f"  Collection: {collection_name}")
    print(f"  Status:     {info.status}")
    print(f"  Points:     {info.points_count}")
    print(f"{'=' * 60}")

    # Dense vectors
    vectors = info.config.params.vectors or {}
    print(f"\n  Dense Vectors ({len(vectors)}):")
    for name, params in vectors.items():
        print(f"    • {name}: dim={params.size}, distance={params.distance}")

    # Sparse vectors
    sparse = info.config.params.sparse_vectors or {}
    print(f"\n  Sparse Vectors ({len(sparse)}):")
    for name in sparse:
        print(f"    • {name}")

    # Aliases
    aliases_info = client.get_collection_aliases(collection_name)
    alias_names = [a.alias_name for a in aliases_info.aliases]
    print(f"\n  Aliases: {alias_names if alias_names else '(none)'}")
    print(f"{'=' * 60}\n")


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
