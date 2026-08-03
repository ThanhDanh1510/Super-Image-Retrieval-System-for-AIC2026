"""
indexing/run_index.py
======================
Entry point chính để nạp dữ liệu (vectors + metadata) vào Qdrant collection.

CÁCH CHẠY:
    # Full pipeline: tạo collection (nếu cần) + upsert + set production alias
    python -m indexing.run_index \\
        --dataset-dir ./data/unified \\
        --collection collection_v2 \\
        --create-collection

    # Chỉ upsert thêm (collection đã tồn tại):
    python -m indexing.run_index \\
        --dataset-dir ./data/unified_patch \\
        --collection collection_v2

    # Upsert xong nhưng KHÔNG đổi alias (để verify trước):
    python -m indexing.run_index \\
        --dataset-dir ./data/unified \\
        --collection collection_v2 \\
        --skip-alias

FOLDER STRUCTURE của --dataset-dir:
    dataset-dir/
    ├── scene_metadata.parquet   (scene_id, video_id, frame_id, timestamps, keyframe_path, member, ingestion_version)
    ├── ocr_text.parquet         (scene_id, video_id, ocr_text)
    ├── asr_text.parquet         (scene_id, video_id, asr_text)
    ├── image_vectors.npy        (N, 1024) float32 L2-normalized
    ├── ocr_vectors.npy          (N, 1536) float32 L2-normalized
    ├── asr_vectors.npy          (N, 1536) float32 L2-normalized
    └── scene_ids.txt            (N lines — thứ tự scene_id tương ứng với rows trong .npy)

ĐÃ IMPLEMENT ĐẦY ĐỦ bởi Leader.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from shared.schemas.named_vector_spec import (
    IMAGE_DENSE,
    OCR_DENSE,
    ASR_DENSE,
    make_point_id,
    KeyframePayload,
)
from shared.utils.config_loader import load_model_versions
from indexing.schema.collection_schema import create_collection, COLLECTION_NAME
from indexing.collection_alias import set_production_alias, get_current_production

logger = logging.getLogger(__name__)

# Số lần thử lại khi upsert thất bại
_MAX_RETRIES = 3
# Thời gian chờ ban đầu giữa các lần retry (giây) — exponential backoff
_RETRY_BACKOFF_BASE = 2.0


# =============================================================================
# Data Loading
# =============================================================================

def load_unified_dataset(dataset_dir: str) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """
    Load toàn bộ parquet metadata và các file .npy vectors từ dataset_dir.

    Tự động merge scene_metadata + ocr_text + asr_text theo scene_id.
    Đảm bảo thứ tự rows trong metadata khớp với thứ tự trong scene_ids.txt và .npy files.

    Args:
        dataset_dir: Thư mục chứa unified dataset (xem FOLDER STRUCTURE ở trên).

    Returns:
        Tuple (metadata_df, vectors_dict):
            - metadata_df: pd.DataFrame đã merge đủ thông tin, thứ tự theo scene_ids.txt
            - vectors_dict: {'image_dense': ndarray (N,1024), 'ocr_dense': (N,1536), 'asr_dense': (N,1536)}

    Raises:
        FileNotFoundError: Nếu bất kỳ file bắt buộc nào không tồn tại.
        ValueError: Nếu số rows không khớp giữa metadata và .npy files.
    """
    base = Path(dataset_dir)

    # --- Kiểm tra sự tồn tại của tất cả files bắt buộc ---
    required_files = [
        "scene_metadata.parquet",
        "ocr_text.parquet",
        "asr_text.parquet",
        "image_vectors.npy",
        "ocr_vectors.npy",
        "asr_vectors.npy",
        "scene_ids.txt",
    ]
    missing = [f for f in required_files if not (base / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Thiếu các file bắt buộc trong {dataset_dir}: {missing}"
        )

    # --- Load scene_ids.txt (thứ tự chuẩn) ---
    with open(base / "scene_ids.txt", "r", encoding="utf-8") as f:
        scene_ids = [line.strip() for line in f if line.strip()]
    logger.info(f"Loaded {len(scene_ids)} scene_ids từ scene_ids.txt")

    # --- Load vectors ---
    image_vecs = np.load(base / "image_vectors.npy").astype(np.float32)
    ocr_vecs = np.load(base / "ocr_vectors.npy").astype(np.float32)
    asr_vecs = np.load(base / "asr_vectors.npy").astype(np.float32)

    # --- Validate shape ---
    n = len(scene_ids)
    if image_vecs.shape[0] != n:
        raise ValueError(f"image_vectors.npy có {image_vecs.shape[0]} rows, nhưng scene_ids.txt có {n} entries")
    if ocr_vecs.shape[0] != n:
        raise ValueError(f"ocr_vectors.npy có {ocr_vecs.shape[0]} rows, nhưng scene_ids.txt có {n} entries")
    if asr_vecs.shape[0] != n:
        raise ValueError(f"asr_vectors.npy có {asr_vecs.shape[0]} rows, nhưng scene_ids.txt có {n} entries")

    logger.info(
        f"Vectors loaded — image: {image_vecs.shape}, "
        f"ocr: {ocr_vecs.shape}, asr: {asr_vecs.shape}"
    )

    # --- Load và merge metadata ---
    meta_df = pd.read_parquet(base / "scene_metadata.parquet")
    ocr_df = pd.read_parquet(base / "ocr_text.parquet")[["scene_id", "ocr_text"]]
    asr_df = pd.read_parquet(base / "asr_text.parquet")[["scene_id", "asr_text"]]

    # Merge text vào metadata
    meta_df = meta_df.merge(ocr_df, on="scene_id", how="left")
    meta_df = meta_df.merge(asr_df, on="scene_id", how="left")

    # Điền chuỗi rỗng cho text bị thiếu (không dùng NaN)
    meta_df["ocr_text"] = meta_df["ocr_text"].fillna("").astype(str)
    meta_df["asr_text"] = meta_df["asr_text"].fillna("").astype(str)

    # Sắp xếp metadata theo đúng thứ tự scene_ids.txt
    scene_order = {sid: i for i, sid in enumerate(scene_ids)}
    if not set(scene_ids).issubset(set(meta_df["scene_id"])):
        missing_in_meta = set(scene_ids) - set(meta_df["scene_id"])
        raise ValueError(
            f"{len(missing_in_meta)} scene_ids trong scene_ids.txt không có trong scene_metadata.parquet. "
            f"Ví dụ: {list(missing_in_meta)[:5]}"
        )
    meta_df = meta_df.set_index("scene_id").loc[scene_ids].reset_index()

    vectors_dict = {
        IMAGE_DENSE.name: image_vecs,
        OCR_DENSE.name: ocr_vecs,
        ASR_DENSE.name: asr_vecs,
    }

    return meta_df, vectors_dict


# =============================================================================
# Streamed Point Building & Batch Upsert
# =============================================================================

def build_and_upsert_batches(
    client: QdrantClient,
    collection_name: str,
    metadata: pd.DataFrame,
    vectors: dict[str, np.ndarray],
    ingestion_version: str = "1.0.0",
    batch_size: int = 200,
) -> int:
    """
    Build PointStruct và upsert theo từng batch nhỏ để tiết kiệm RAM tối đa (Memory Efficient).

    Args:
        client: QdrantClient đã kết nối.
        collection_name: Tên collection Qdrant.
        metadata: pd.DataFrame từ load_unified_dataset().
        vectors: dict từ load_unified_dataset().
        ingestion_version: Version string ghi vào payload.
        batch_size: Số points mỗi lần upsert.

    Returns:
        Tổng số points đã upsert thành công.
    """
    total = len(metadata)
    upserted = 0
    start_time = time.time()
    total_batches = (total + batch_size - 1) // batch_size

    image_vecs = vectors[IMAGE_DENSE.name]
    ocr_vecs = vectors[OCR_DENSE.name]
    asr_vecs = vectors[ASR_DENSE.name]

    logger.info(f"Bắt đầu upsert {total} points vào '{collection_name}' theo từng batch {batch_size}...")

    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch_meta = metadata.iloc[batch_start:batch_end]
        batch_num = batch_start // batch_size + 1

        batch_points = []
        for i, row in enumerate(batch_meta.itertuples(index=False)):
            idx = batch_start + i
            point_id = make_point_id(row.video_id, row.scene_id)
            payload = {
                "video_id": str(row.video_id),
                "scene_id": str(row.scene_id),
                "frame_id": int(row.frame_id),
                "timestamp_start": float(row.timestamp_start),
                "timestamp_end": float(row.timestamp_end),
                "timestamp_mid": float(row.timestamp_mid),
                "keyframe_path": str(row.keyframe_path),
                "ocr_text": str(row.ocr_text),
                "asr_text": str(row.asr_text),
                "ingestion_version": ingestion_version,
                "member": str(getattr(row, "member", "unknown")),
            }
            batch_points.append(
                PointStruct(
                    id=point_id,
                    vector={
                        IMAGE_DENSE.name: image_vecs[idx].tolist(),
                        OCR_DENSE.name: ocr_vecs[idx].tolist(),
                        ASR_DENSE.name: asr_vecs[idx].tolist(),
                    },
                    payload=payload,
                )
            )

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                client.upsert(collection_name=collection_name, points=batch_points)
                upserted += len(batch_points)
                elapsed = time.time() - start_time
                speed = upserted / elapsed if elapsed > 0 else 0
                if batch_num % 10 == 0 or batch_num == total_batches or batch_num <= 5:
                    logger.info(
                        f"Batch {batch_num}/{total_batches} — "
                        f"{upserted}/{total} points ({speed:.1f} pts/s)"
                    )
                break
            except Exception as e:
                if attempt == _MAX_RETRIES:
                    raise RuntimeError(
                        f"Upsert batch {batch_num} thất bại sau {_MAX_RETRIES} lần thử: {e}"
                    ) from e
                wait = _RETRY_BACKOFF_BASE ** attempt
                logger.warning(
                    f"Batch {batch_num} attempt {attempt} thất bại: {e}. "
                    f"Thử lại sau {wait:.1f}s..."
                )
                time.sleep(wait)

    elapsed_total = time.time() - start_time
    logger.info(
        f"✅ Upsert hoàn tất: {upserted}/{total} points trong {elapsed_total:.1f}s "
        f"({upserted/elapsed_total:.1f} pts/s)"
    )
    return upserted


# =============================================================================
# Main Orchestrator
# =============================================================================

def run(
    dataset_dir: str,
    collection_name: str,
    qdrant_url: str = "http://localhost:6333",
    qdrant_api_key: str | None = None,
    qdrant_path: str | None = None,
    batch_size: int = 200,
    create_if_missing: bool = False,
    skip_alias: bool = False,
) -> None:
    """
    Orchestrator chính: kết nối Qdrant → load data → build points → upsert → set alias.

    Args:
        dataset_dir: Thư mục chứa unified dataset.
        collection_name: Tên collection Qdrant đích (ví dụ: 'collection_v2').
        qdrant_url: URL của Qdrant server.
        qdrant_api_key: API key nếu Qdrant có authentication.
        batch_size: Số points mỗi lần upsert.
        create_if_missing: True → tạo collection nếu chưa tồn tại.
        skip_alias: True → không đổi alias sau khi upsert (để verify trước).

    Raises:
        FileNotFoundError: Nếu dataset_dir không tồn tại hoặc thiếu file.
        ValueError: Nếu data không hợp lệ.
        RuntimeError: Nếu upsert thất bại hoàn toàn.
    """
    run_start = time.time()
    logger.info(f"=== AIC2026 Indexing Run ===")
    logger.info(f"Dataset dir : {dataset_dir}")
    logger.info(f"Collection  : {collection_name}")
    logger.info(f"Qdrant URL  : {qdrant_url}")
    logger.info(f"Batch size  : {batch_size}")

    # 1. Kết nối Qdrant (Thử URL, nếu lỗi thì fallback sang Local Embedded Path)
    if qdrant_path:
        logger.info(f"Kết nối Embedded Qdrant tại path: {qdrant_path}")
        client = QdrantClient(path=qdrant_path)
    else:
        try:
            client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, timeout=10)
            client.get_collections()
            logger.info(f"Qdrant client connected to server {qdrant_url}.")
        except Exception as e:
            fallback_path = "./data/qdrant_db"
            logger.warning(
                f"Không thể kết nối Qdrant server tại {qdrant_url} ({e}). "
                f"Tự động chuyển sang Local Embedded Qdrant tại '{fallback_path}'..."
            )
            client = QdrantClient(path=fallback_path)

    # 2. Tạo collection nếu cần
    existing = [c.name for c in client.get_collections().collections]
    if collection_name not in existing:
        if create_if_missing:
            logger.info(f"Collection '{collection_name}' chưa tồn tại. Đang tạo...")
            create_collection(client, collection_name)
        else:
            raise ValueError(
                f"Collection '{collection_name}' không tồn tại. "
                f"Chạy lại với --create-collection để tạo tự động."
            )
    else:
        logger.info(f"Collection '{collection_name}' đã tồn tại. Upsert thêm dữ liệu.")

    # 3. Load dataset
    logger.info("Loading dataset...")
    metadata, vectors = load_unified_dataset(dataset_dir)
    logger.info(f"Loaded {len(metadata)} scenes.")

    # Lấy ingestion_version từ config
    try:
        model_cfg = load_model_versions()
        ingestion_version = model_cfg.get("schema_version", "1.0.0")
    except Exception:
        ingestion_version = "1.0.0"

    # 4. Stream Build & Upsert Points
    total_upserted = build_and_upsert_batches(
        client=client,
        collection_name=collection_name,
        metadata=metadata,
        vectors=vectors,
        ingestion_version=ingestion_version,
        batch_size=batch_size,
    )

    # 6. Set production alias
    if not skip_alias:
        current = get_current_production(client)
        logger.info(
            f"Đang chuyển alias 'production': "
            f"'{current or 'N/A'}' → '{collection_name}'"
        )
        set_production_alias(client, collection_name)
    else:
        logger.info("--skip-alias được set. Alias 'production' KHÔNG được cập nhật.")
        logger.info(
            f"Khi muốn set alias, chạy: "
            f"python -m indexing.collection_alias --action set --collection {collection_name}"
        )

    elapsed = time.time() - run_start
    logger.info(f"=== Indexing hoàn tất: {total_upserted} points trong {elapsed:.1f}s ===")


# =============================================================================
# CLI Entry Point
# =============================================================================

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Nạp unified dataset vào Qdrant collection.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  # Tạo collection mới và index toàn bộ dataset:
  python -m indexing.run_index \\
      --dataset-dir ./data/unified \\
      --collection collection_v2 \\
      --create-collection

  # Index thêm batch mới vào collection đã tồn tại:
  python -m indexing.run_index \\
      --dataset-dir ./data/unified_patch \\
      --collection collection_v2

  # Index xong nhưng chưa đổi alias (để verify trước):
  python -m indexing.run_index \\
      --dataset-dir ./data/unified \\
      --collection collection_v2 \\
      --create-collection --skip-alias
        """,
    )
    parser.add_argument(
        "--dataset-dir",
        required=True,
        help="Thư mục chứa unified dataset (parquet + npy + scene_ids.txt).",
    )
    parser.add_argument(
        "--collection",
        required=True,
        help="Tên collection Qdrant đích (ví dụ: collection_v2).",
    )
    parser.add_argument(
        "--qdrant-url",
        default="http://localhost:6333",
        help="URL của Qdrant server. Mặc định: http://localhost:6333",
    )
    parser.add_argument(
        "--qdrant-api-key",
        default=None,
        help="API key của Qdrant (nếu có authentication).",
    )
    parser.add_argument(
        "--qdrant-path",
        default=None,
        help="Đường dẫn lưu Local Embedded Qdrant Database (không cần Docker).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=200,
        help="Số points upsert mỗi lần. Mặc định: 200.",
    )
    parser.add_argument(
        "--create-collection",
        action="store_true",
        help="Tự động tạo collection nếu chưa tồn tại.",
    )
    parser.add_argument(
        "--skip-alias",
        action="store_true",
        help="Không đổi alias 'production' sau khi upsert xong.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        run(
            dataset_dir=args.dataset_dir,
            collection_name=args.collection,
            qdrant_url=args.qdrant_url,
            qdrant_api_key=args.qdrant_api_key,
            qdrant_path=args.qdrant_path,
            batch_size=args.batch_size,
            create_if_missing=args.create_collection,
            skip_alias=args.skip_alias,
        )
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"❌ Lỗi dữ liệu: {e}")
        sys.exit(1)
    except RuntimeError as e:
        logger.error(f"❌ Lỗi runtime: {e}")
        sys.exit(2)


if __name__ == "__main__":
    main()
