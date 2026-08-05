"""
aggregation/map_keyframes_to_unified.py
========================================
Ánh xạ (mapping) toàn bộ ảnh keyframe sang định dạng tên mới
[video_id]-[scene_id]-[frame_id] và lưu vào thư mục unified/.

TÊN CŨ (trên đĩa):
    data/unified_keyframes/{video_id}/{video_id}_S{idx:03d}.webp
    ví dụ: data/unified_keyframes/L21_V001/L21_V001_S000.webp

TÊN MỚI (trong data/unified/):
    {video_id}-{scene_id}-{frame_id}.webp
    ví dụ: data/unified/L21_V001-L21_V001_S000-0.webp

Nguồn ánh xạ: scene_metadata.parquet (video_id, scene_id, frame_id, keyframe_path)
— file metadata là source of truth, đảm bảo tên mới khớp chính xác với payload
Qdrant (video_id/scene_id/frame_id) và scene_metadata.

CHẾ ĐỘ CHẠY:
    # Xem trước những gì sẽ được ánh xạ (không ghi gì)
    python aggregation/map_keyframes_to_unified.py --dry-run

    # Ánh xạ thật: copy + verify (kích thước file) + xóa ảnh gốc
    python aggregation/map_keyframes_to_unified.py --delete-source

CÁC CỜ:
    --dry-run        Chỉ liệt kê, không copy/xóa.
    --delete-source  Xóa ảnh gốc sau khi đã copy+verify thành công.
                     (Mặc định: KHÔNG xóa — chỉ copy để an toàn.)
    --limit N        Chỉ xử lý N dòng đầu (test nhanh).
    --parquet PATH   Thay path parquet nguồn (mặc định data/data/unified/scene_metadata.parquet).
    --src-dir PATH   Thư mục ảnh cũ (mặc định data/unified_keyframes).
    --dst-dir PATH   Thư mục đích unified (mặc định data/unified).

TÍNH IDEMPOTENT:
    - Nếu file đích đã tồn tại VÀ kích thước khớp file nguồn → bỏ qua (coi như đã map).
    - Nếu file đích tồn tại nhưng kích thước KHÁC → copy lại (overwrite).
    - Chạy lại nhiều lần an toàn, không tạo trùng.
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def build_mapping(meta_df: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo bảng mapping (src_path -> dst_path) từ scene_metadata.

    Quy ước tên mới: {video_id}-{scene_id}-{frame_id}{ext}
    Với ext lấy từ keyframe_path gốc (thường .webp).

    Args:
        meta_df: DataFrame chứa ít nhất cột video_id, scene_id, frame_id, keyframe_path.

    Returns:
        DataFrame với 2 cột: src_path (Path), dst_path (Path).
    """
    required = {"video_id", "scene_id", "frame_id", "keyframe_path"}
    missing = required - set(meta_df.columns)
    if missing:
        raise ValueError(f"Thiếu cột trong scene_metadata: {sorted(missing)}")

    ext = meta_df["keyframe_path"].str.rsplit(".", n=1).str[-1]
    new_name = (
        meta_df["scene_id"]
        + "-"
        + meta_df["frame_id"].astype(str)
        + "."
        + ext
    )
    mapping = pd.DataFrame(
        {
            "src_path": [Path(p) for p in meta_df["keyframe_path"]],
            "dst_path": [Path(p) for p in new_name],
        }
    )
    return mapping


def main():
    parser = argparse.ArgumentParser(description="Map keyframes sang unified/ với tên [video_id]-[scene_id]-[frame_id]")
    parser.add_argument("--parquet", default="data/data/unified/scene_metadata.parquet")
    parser.add_argument("--src-dir", default="data/unified_keyframes")
    parser.add_argument("--dst-dir", default="data/unified")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--delete-source", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    src_root = Path(args.src_dir)
    dst_root = Path(args.dst_dir)

    meta_df = pd.read_parquet(args.parquet)
    logger.info(f"scene_metadata: {len(meta_df)} rows từ {args.parquet}")

    mapping = build_mapping(meta_df)
    if args.limit > 0:
        mapping = mapping.head(args.limit).copy()

    ok = skipped = missing_src = mismatched = 0
    total_bytes = 0

    if not args.dry_run:
        dst_root.mkdir(parents=True, exist_ok=True)

    for idx, row in mapping.iterrows():
        src = src_root / row["src_path"]
        dst = dst_root / row["dst_path"]

        if not src.exists():
            logger.warning(f"MISSING source: {src}")
            missing_src += 1
            continue

        src_size = src.stat().st_size
        if dst.exists() and dst.stat().st_size == src_size:
            skipped += 1
            continue
        if dst.exists() and dst.stat().st_size != src_size:
            mismatched += 1

        if args.dry_run:
            logger.info(f"[dry-run] {src} -> {dst}")
            ok += 1
            total_bytes += src_size
            continue

        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        if dst.exists() and dst.stat().st_size == src_size:
            ok += 1
            total_bytes += src_size
            if args.delete_source:
                src.unlink()
        else:
            logger.error(f"VERIFY FAILED: {dst} (size mismatch sau copy)")
            mismatched += 1

        if (idx + 1) % 20000 == 0:
            logger.info(f"Progress: {idx + 1}/{len(mapping)} processed...")

    logger.info("=" * 60)
    logger.info(f"RESULT (limit={args.limit or 'all'}):")
    logger.info(f"  copied   : {ok}")
    logger.info(f"  skipped  : {skipped} (đã tồn tại, kích thước khớp)")
    logger.info(f"  missing  : {missing_src} (không thấy file nguồn)")
    logger.info(f"  mismatch : {mismatched} (đích sai kích thước — cần kiểm tra)")
    logger.info(f"  delete   : {'YES' if args.delete_source else 'NO (giữ ảnh gốc)'}")
    if ok:
        logger.info(f"  bytes    : {total_bytes / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
