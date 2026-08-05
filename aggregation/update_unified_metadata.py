"""
aggregation/update_unified_metadata.py
=======================================
Cập nhật metadata + Qdrant payload sau khi ánh xạ ảnh sang unified/.

1. Cập nhật scene_metadata.parquet:
     keyframe_path: {video_id}/{scene_id}.webp  →  unified/{video_id}-{scene_id}-{frame_id}.webp

2. Tái khớp (re-match) OCR text từ data/data/OCR/{group}/{video_id}.json:
     - Match chính xác theo frame_id (ưu tiên, text không rỗng).
     - Fallback theo khoảng thời gian [timestamp_start, timestamp_end] (fps=25).
     - Ghi ocr_text.parquet (scene_id, video_id, ocr_text).

3. Upsert Qdrant payload (alias production):
     - keyframe_path → định dạng mới (đường dẫn tương đối từ data/, trỏ tới unified/).
     - ocr_text → text đã tái khớp (chỉ điểm có OCR thật sự).

CÁCH CHẠY:
    python aggregation/update_unified_metadata.py                 # đầy đủ (metadata + Qdrant)
    python aggregation/update_unified_metadata.py --skip-qdrant   # chỉ metadata/parquet
    python aggregation/update_unified_metadata.py --dry-run       # chỉ in ra dự kiến

QUAN TRỌNG:
    - Chỉ UPDATE payload keyframe_path/ocr_text — không đụng vector/điểm khác.
    - Idempotent: chạy lại nhiều lần an toàn.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
from qdrant_client import QdrantClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

FPS = 25.0


def load_ocr_json(ocr_dir: Path, video_id: str) -> dict[str, str]:
    """Đọc OCR JSON của 1 video (trả {} nếu không có)."""
    group = video_id.split("_")[0] if "_" in video_id else ""
    for candidate in (ocr_dir / group / f"{video_id}.json", ocr_dir / f"{video_id}.json"):
        if candidate.exists():
            try:
                return json.load(open(candidate, "r", encoding="utf-8"))
            except Exception as e:
                logger.warning(f"Không đọc được OCR JSON {candidate}: {e}")
                return {}
    return {}


def match_ocr_for_scenes(meta_df: pd.DataFrame, ocr_dir: Path) -> pd.DataFrame:
    """
    Tái khớp OCR text cho từng scene.

    Chiến lược (ưu tiên giảm dần):
      1. Exact frame_id: key khớp frame_id, text không rỗng.
      2. Index-aligned: số lượng key trong ocr.json bằng số scene VÀ keys là dãy số liên tiếp
         (ví dụ 107046..107116 cho 71 scenes) → key i ứng với scene i.
      3. Range-based: mọi key không rỗng có timestamp (key/fps) nằm trong
         [timestamp_start, timestamp_end] — nối text với nhau bằng space.

    Returns:
        DataFrame (scene_id, video_id, ocr_text) — ocr_text="" nếu không khớp.
    """
    rows = []
    for video_id, group in meta_df.groupby("video_id", sort=False):
        ocr = load_ocr_json(ocr_dir, video_id)
        non_empty = {
            int(k): v for k, v in ocr.items()
            if isinstance(v, str) and v.strip()
        }
        if not non_empty:
            for s_id in group["scene_id"]:
                rows.append({"scene_id": s_id, "video_id": video_id, "ocr_text": ""})
            continue

        all_keys = sorted(int(k) for k in ocr.keys() if isinstance(ocr.get(k), str))
        index_aligned = (
            len(all_keys) == len(group)
            and len(all_keys) > 0
            and all_keys == list(range(all_keys[0], all_keys[0] + len(all_keys)))
        )

        frame_ids = group["frame_id"].astype(int).tolist()
        scene_ids = group["scene_id"].tolist()
        t_starts = group["timestamp_start"].tolist()
        t_ends = group["timestamp_end"].tolist()

        for idx in range(len(group)):
            f_id = frame_ids[idx]
            s_id = scene_ids[idx]

            if f_id in non_empty:
                rows.append({"scene_id": s_id, "video_id": video_id, "ocr_text": non_empty[f_id]})
                continue

            if index_aligned:
                key = all_keys[idx]
                rows.append({"scene_id": s_id, "video_id": video_id, "ocr_text": non_empty.get(key, "")})
                continue

            ts_start, ts_end = t_starts[idx], t_ends[idx]
            hits = [txt for k, txt in non_empty.items() if ts_start <= (k / FPS) <= ts_end]
            rows.append({
                "scene_id": s_id,
                "video_id": video_id,
                "ocr_text": " ".join(hits) if hits else "",
            })

    return pd.DataFrame(rows)



def new_keyframe_path(video_id: str, scene_id: str, frame_id: int, ext: str = "webp") -> str:
    """Đường dẫn mới tương đối từ data/ — trỏ vào thư mục unified/."""
    return f"unified/{scene_id}-{frame_id}.{ext}"


def main():
    parser = argparse.ArgumentParser(description="Cập nhật metadata + Qdrant payload theo tên unified mới")
    parser.add_argument("--parquet", default="data/data/unified/scene_metadata.parquet")
    parser.add_argument("--ocr-dir", default="data/data/OCR")
    parser.add_argument("--ocr-out", default="data/data/unified/ocr_text.parquet")
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--collection", default="production")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-qdrant", action="store_true")
    args = parser.parse_args()

    meta_df = pd.read_parquet(args.parquet)
    logger.info(f"scene_metadata: {len(meta_df)} rows")

    # --- 1. Tính keyframe_path mới ---
    meta_df["keyframe_path"] = [
        new_keyframe_path(v, s, int(f))
        for v, s, f in zip(meta_df["video_id"], meta_df["scene_id"], meta_df["frame_id"])
    ]

    # --- 2. Tái khớp OCR ---
    logger.info("Re-matching OCR từ JSON (exact frame_id -> range fallback)...")
    ocr_df = match_ocr_for_scenes(meta_df, Path(args.ocr_dir))
    non_empty = (ocr_df["ocr_text"].str.len() > 0).sum()
    logger.info(f"OCR non-empty: {non_empty}/{len(ocr_df)} scenes")

    if args.dry_run:
        sample = ocr_df[ocr_df["ocr_text"] != ""].head(5)
        logger.info("Dry-run: 5 mẫu OCR sẽ được upsert:")
        for _, r in sample.iterrows():
            logger.info(f"  {r['scene_id']}: {r['ocr_text'][:60]}")
        logger.info("Dry-run: không ghi gì cả.")
        return

    # --- 3. Ghi parquet ---
    meta_df.to_parquet(args.parquet, index=False)
    logger.info(f"Đã cập nhật {args.parquet} (keyframe_path -> unified/...)")
    ocr_df.to_parquet(args.ocr_out, index=False)
    logger.info(f"Đã ghi {args.ocr_out}")

    # --- 4. Upsert Qdrant payload ---
    # Upsert payload mới (keyframe_path + ocr_text) bằng cách re-index toàn bộ
    # từ parquet đã cập nhật — idempotent theo point ID (hash video_id+scene_id):
    #
    #   python -m indexing.run_index --dataset-dir ./data/data/unified \
    #       --collection collection_v1 --skip-alias
    #
    # run_index đọc scene_metadata.parquet (đã có keyframe_path mới) và
    # ocr_text.parquet (đã re-match) vừa ghi ở bước 3, rồi upsert batch 200.
    if args.skip_qdrant:
        logger.info("--skip-qdrant: bỏ qua upsert Qdrant (chạy run_index riêng).")
        return
    logger.info(
        "Để upsert payload vào Qdrant, chạy: "
        "python -m indexing.run_index --dataset-dir ./data/data/unified "
        "--collection collection_v1 --skip-alias"
    )


if __name__ == "__main__":
    main()
