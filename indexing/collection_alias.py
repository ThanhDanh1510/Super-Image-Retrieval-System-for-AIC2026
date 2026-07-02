"""
indexing/collection_alias.py
=============================
Quản lý Qdrant collection aliases — zero-downtime collection swap và rollback.

Alias "production" luôn trỏ tới collection mới nhất đã được verify.
Khi cần rollback, chỉ cần đổi alias sang collection cũ — không cần re-index.

CÁC TÌNH HUỐNG SỬ DỤNG:
    # 1. Index xong collection_v2, chuyển production sang đó:
    python -m indexing.collection_alias --action set --collection collection_v2

    # 2. Kiểm tra collection nào đang ở production:
    python -m indexing.collection_alias --action get

    # 3. Rollback về collection cũ khi phát hiện lỗi:
    python -m indexing.collection_alias --action rollback --collection collection_v1

    # 4. Xem tất cả aliases:
    python -m indexing.collection_alias --action list

ĐÃ IMPLEMENT ĐẦY ĐỦ bởi Leader.
"""

from __future__ import annotations

import argparse
import logging
import sys

from qdrant_client import QdrantClient
from qdrant_client.models import (
    AliasOperations,
    CreateAliasOperation,
    CreateAlias,
    DeleteAliasOperation,
    DeleteAlias,
)

logger = logging.getLogger(__name__)

# Alias cố định — KHÔNG thay đổi tên này
PRODUCTION_ALIAS = "production"


# =============================================================================
# Core Alias Management Functions
# =============================================================================

def set_production_alias(client: QdrantClient, collection_name: str) -> None:
    """
    Chuyển alias "production" sang trỏ vào collection_name.

    Thao tác này là atomic trên Qdrant — không có downtime.
    Nếu alias "production" chưa tồn tại, sẽ được tạo mới.
    Nếu đã tồn tại, sẽ được cập nhật (xóa cũ + tạo mới trong 1 operation).

    Args:
        client: QdrantClient đã kết nối.
        collection_name: Tên collection sẽ được gắn alias "production".
            Ví dụ: 'collection_v2'

    Raises:
        ValueError: Nếu collection_name không tồn tại trong Qdrant.
        RuntimeError: Nếu thao tác alias thất bại.
    """
    # Kiểm tra collection tồn tại trước khi đổi alias
    existing_collections = [c.name for c in client.get_collections().collections]
    if collection_name not in existing_collections:
        raise ValueError(
            f"Collection '{collection_name}' không tồn tại trong Qdrant. "
            f"Collections hiện có: {existing_collections}"
        )

    # Lấy collection đang gắn alias hiện tại (nếu có) để log
    current = get_current_production(client)
    if current:
        logger.info(f"Đang chuyển alias 'production': {current} → {collection_name}")
    else:
        logger.info(f"Tạo alias 'production' → {collection_name}")

    # Xóa alias cũ (nếu có) và tạo alias mới trong 1 atomic operation
    operations: list[AliasOperations] = []
    if current:
        operations.append(
            AliasOperations(
                delete_alias=DeleteAliasOperation(
                    delete_alias=DeleteAlias(alias_name=PRODUCTION_ALIAS)
                )
            )
        )
    operations.append(
        AliasOperations(
            create_alias=CreateAliasOperation(
                create_alias=CreateAlias(
                    collection_name=collection_name,
                    alias_name=PRODUCTION_ALIAS,
                )
            )
        )
    )

    client.update_collection_aliases(change_aliases_operations=operations)
    logger.info(f"✅ Alias 'production' đã được set → '{collection_name}'")


def get_current_production(client: QdrantClient) -> str | None:
    """
    Trả về tên collection hiện đang gắn alias "production".

    Args:
        client: QdrantClient đã kết nối.

    Returns:
        Tên collection (ví dụ: 'collection_v2') hoặc None nếu alias chưa tồn tại.
    """
    try:
        aliases = client.get_collection_aliases(collection_name=None)
        for alias_info in aliases.aliases:
            if alias_info.alias_name == PRODUCTION_ALIAS:
                return alias_info.collection_name
        return None
    except Exception:
        # Thử cách khác: get alias qua collections
        for collection in client.get_collections().collections:
            try:
                col_aliases = client.get_collection_aliases(
                    collection_name=collection.name
                )
                for alias_info in col_aliases.aliases:
                    if alias_info.alias_name == PRODUCTION_ALIAS:
                        return collection.name
            except Exception:
                continue
        return None


def rollback_alias(client: QdrantClient, previous_collection: str) -> None:
    """
    Rollback alias "production" về collection cũ hơn.

    Gọi hàm này khi phát hiện lỗi ở collection mới vừa được set.
    Thao tác atomic — không có downtime.

    Args:
        client: QdrantClient đã kết nối.
        previous_collection: Tên collection cũ cần rollback về.
            Ví dụ: 'collection_v1'

    Raises:
        ValueError: Nếu previous_collection không tồn tại.
    """
    current = get_current_production(client)
    logger.warning(
        f"⚠️ ROLLBACK: Đang chuyển alias 'production' từ '{current}' → '{previous_collection}'"
    )
    set_production_alias(client, previous_collection)
    logger.warning(f"✅ ROLLBACK hoàn tất. 'production' → '{previous_collection}'")


def list_all_aliases(client: QdrantClient) -> list[dict[str, str]]:
    """
    Liệt kê tất cả aliases hiện có và collection chúng trỏ tới.

    Args:
        client: QdrantClient đã kết nối.

    Returns:
        List of dicts: [{'alias_name': str, 'collection_name': str}, ...]
    """
    result = []
    for collection in client.get_collections().collections:
        try:
            col_aliases = client.get_collection_aliases(collection_name=collection.name)
            for alias_info in col_aliases.aliases:
                result.append({
                    "alias_name": alias_info.alias_name,
                    "collection_name": collection.name,
                })
        except Exception:
            continue
    return result


# =============================================================================
# CLI Entry Point
# =============================================================================

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Quản lý Qdrant collection aliases cho AIC2026.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  # Chuyển production → collection_v2
  python -m indexing.collection_alias --action set --collection collection_v2

  # Xem collection đang ở production
  python -m indexing.collection_alias --action get

  # Rollback về collection_v1
  python -m indexing.collection_alias --action rollback --collection collection_v1

  # Xem tất cả aliases
  python -m indexing.collection_alias --action list
        """,
    )
    parser.add_argument(
        "--action",
        required=True,
        choices=["set", "get", "rollback", "list"],
        help="Hành động cần thực hiện.",
    )
    parser.add_argument(
        "--collection",
        default=None,
        help="Tên collection (dùng với --action set hoặc rollback).",
    )
    parser.add_argument(
        "--url",
        default="http://localhost:6333",
        help="URL của Qdrant server. Mặc định: http://localhost:6333",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key của Qdrant (nếu có authentication).",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = _build_parser()
    args = parser.parse_args(argv)

    client = QdrantClient(url=args.url, api_key=args.api_key, timeout=30)

    if args.action == "set":
        if not args.collection:
            parser.error("--collection bắt buộc khi --action set")
        set_production_alias(client, args.collection)

    elif args.action == "get":
        current = get_current_production(client)
        if current:
            print(f"production → {current}")
        else:
            print("Alias 'production' chưa được set.")

    elif args.action == "rollback":
        if not args.collection:
            parser.error("--collection bắt buộc khi --action rollback")
        rollback_alias(client, args.collection)

    elif args.action == "list":
        aliases = list_all_aliases(client)
        if not aliases:
            print("Không có alias nào.")
        else:
            print(f"{'Alias':<30} {'Collection':<30}")
            print("-" * 60)
            for a in aliases:
                marker = " ← production" if a["alias_name"] == PRODUCTION_ALIAS else ""
                print(f"{a['alias_name']:<30} {a['collection_name']:<30}{marker}")


if __name__ == "__main__":
    main()
