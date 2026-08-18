"""
aggregation/apply_filter.py
===========================
Áp dụng danh sách lọc từ filtered_keyframes.json:
1. Xóa vĩnh viễn các file ảnh keyframe đã lọc trên đĩa cứng (khi có --delete-images).
2. Lọc bỏ các dòng tương ứng trong metadata parquet (scene_metadata, ocr_text, asr_text, scene_descriptions).
3. Cắt và lưu lại các ma trận vector (image_vectors.npy, ocr_vectors.npy, asr_vectors.npy, description_embeddings.pt).
4. Cập nhật lại scene_ids.txt.

CÁCH CHẠY:
    # Kiểm tra thử (Dry-run, không xóa file, không ghi đè dữ liệu):
    python -m aggregation.apply_filter --dry-run

    # Áp dụng thực tế (Xóa ảnh thật và ghi đè unified dataset):
    python -m aggregation.apply_filter --delete-images
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from aggregation.common import slice_and_save_vector_file

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ApplyFilter")


def parse_args():
    parser = argparse.ArgumentParser(description="Áp dụng filter JSON để xóa ảnh & rebuild unified dataset.")
    parser.add_argument(
        "--filter-json",
        default="./data/filtered_keyframes.json",
        help="Đường dẫn file filtered_keyframes.json. Mặc định: ./data/filtered_keyframes.json",
    )
    parser.add_argument(
        "--image-dir",
        default="./data/image",
        help="Thư mục chứa ảnh keyframe. Mặc định: ./data/image",
    )
    parser.add_argument(
        "--unified-dir",
        default="./data/unified",
        help="Thư mục chứa unified dataset. Mặc định: ./data/unified",
    )
    parser.add_argument(
        "--delete-images",
        action="store_true",
        help="Thực sự xóa các file ảnh .webp trên đĩa cứng.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Chỉ in báo cáo thống kê, KHÔNG xóa file hay ghi đè dữ liệu.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    filter_json_path = Path(args.filter_json).resolve()
    image_dir = Path(args.image_dir).resolve()
    unified_dir = Path(args.unified_dir).resolve()

    if not filter_json_path.exists():
        old_path = filter_json_path.parent / "filtered_images_366.json"
        if old_path.exists():
            logger.info(f"{filter_json_path.name} chưa tồn tại. Tự động đọc từ {old_path.name}...")
            filter_json_path = old_path
        else:
            logger.error(f"File filter JSON không tồn tại: {filter_json_path}")
            sys.exit(1)

    # 1. Đọc filter JSON
    with open(filter_json_path, "r", encoding="utf-8") as f:
        filter_data = json.load(f)

    filtered_images = filter_data.get("filtered_images", [])
    total_to_filter = len(filtered_images)
    logger.info(f"Loaded {total_to_filter} entries to filter from {filter_json_path.name}")

    if total_to_filter == 0:
        logger.info("Danh sách lọc rỗng. Không có gì cần xử lý.")
        return

    # Build set các frame bị lọc theo (batch, video_id, frame_id) và keyframe_path
    excluded_keys: set[str] = set()
    excluded_tuples: set[tuple[str, str, int]] = set()

    for item in filtered_images:
        b = str(item.get("batch", ""))
        vid = str(item.get("video_id", ""))
        fid = int(item.get("frame_id", 0))
        fname = str(item.get("filename", f"{fid}.webp"))
        kpath = item.get("keyframe_path") or f"{b}/{vid}/{fname}"

        excluded_keys.add(kpath)
        excluded_tuples.add((b, vid, fid))
        # Support full_video_id format L21_V001
        if "_" in vid:
            parts = vid.split("_")
            excluded_tuples.add((parts[0], parts[1], fid))

    # 2. Xóa các file ảnh vật lý nếu được yêu cầu
    deleted_files_count = 0
    missing_files_count = 0

    if args.delete_images:
        logger.info(f"--- BẮT ĐẦU XÓA {total_to_filter} FILE ẢNH KEYFRAME TRÊN ĐĨA CỨNG ---")
        for item in filtered_images:
            b = str(item.get("batch", ""))
            vid = str(item.get("video_id", ""))
            fid = int(item.get("frame_id", 0))
            fname = str(item.get("filename", f"{fid}.webp"))

            # Kiểm tra các dạng đường dẫn có thể có
            candidates = [
                image_dir / b / b / vid / fname,
                image_dir / b / vid / fname,
                image_dir / b / b / f"{b}_{vid}" / fname,
            ]
            found = False
            for p in candidates:
                if p.exists() and p.is_file():
                    found = True
                    if not args.dry_run:
                        try:
                            p.unlink()
                            deleted_files_count += 1
                        except Exception as e:
                            logger.error(f"Không xóa được file {p}: {e}")
                    else:
                        deleted_files_count += 1
                    break

            if not found:
                missing_files_count += 1

        action_word = "Sẽ xóa" if args.dry_run else "Đã xóa"
        logger.info(f"{action_word} thành công {deleted_files_count} file ảnh (Không tìm thấy {missing_files_count} file).")
    else:
        logger.info("Bỏ qua bước xóa file ảnh (chạy với --delete-images để xóa thật).")

    # 3. Xử lý Unified Dataset
    if not unified_dir.exists():
        logger.warning(f"Thư mục unified không tồn tại: {unified_dir}")
        return

    meta_file = unified_dir / "scene_metadata.parquet"
    scene_txt_file = unified_dir / "scene_ids.txt"

    if not meta_file.exists() or not scene_txt_file.exists():
        logger.error(f"Thiếu file scene_metadata.parquet hoặc scene_ids.txt trong {unified_dir}")
        return

    # Đọc scene_ids.txt
    with open(scene_txt_file, "r", encoding="utf-8") as f:
        original_scene_ids = [line.strip() for line in f if line.strip()]
    total_scenes_original = len(original_scene_ids)
    logger.info(f"Tổng số scenes ban đầu: {total_scenes_original}")

    # Đọc scene_metadata.parquet
    meta_df = pd.read_parquet(meta_file)
    logger.info(f"scene_metadata.parquet ban đầu: {len(meta_df)} rows")

    # Xác định các scene_id bị loại trừ
    scenes_to_exclude: set[str] = set()
    for _, row in meta_df.iterrows():
        sid = str(row["scene_id"])
        vid = str(row["video_id"])
        fid = int(row["frame_id"])
        parts = vid.split("_")
        b = parts[0] if len(parts) == 2 else vid
        sub_v = parts[1] if len(parts) == 2 else vid

        if (b, sub_v, fid) in excluded_tuples or (b, vid, fid) in excluded_tuples:
            scenes_to_exclude.add(sid)

    logger.info(f"Số lượng scene_id khớp để loại bỏ khỏi Unified: {len(scenes_to_exclude)}")

    # Tạo mask giữ lại
    keep_mask_txt = np.array([sid not in scenes_to_exclude for sid in original_scene_ids], dtype=bool)
    new_scene_ids = [sid for sid in original_scene_ids if sid not in scenes_to_exclude]
    logger.info(f"Số lượng scenes sau khi lọc: {len(new_scene_ids)} (giảm {total_scenes_original - len(new_scene_ids)})")

    if args.dry_run:
        logger.info("=== DRY-RUN HOÀN TẤT (Không có file nào bị thay đổi) ===")
        return

    # Ghi lại scene_ids.txt mới
    logger.info("Đang cập nhật scene_ids.txt...")
    with open(scene_txt_file, "w", encoding="utf-8") as f:
        for sid in new_scene_ids:
            f.write(f"{sid}\n")

    # Cập nhật scene_metadata.parquet
    logger.info("Đang cập nhật scene_metadata.parquet...")
    filtered_meta_df = meta_df[~meta_df["scene_id"].isin(scenes_to_exclude)].copy()
    filtered_meta_df.to_parquet(meta_file, index=False)
    logger.info(f"Đã lưu scene_metadata.parquet mới: {len(filtered_meta_df)} rows")

    # Cập nhật ocr_text.parquet nếu có
    ocr_meta = unified_dir / "ocr_text.parquet"
    if ocr_meta.exists():
        logger.info("Đang cập nhật ocr_text.parquet...")
        ocr_df = pd.read_parquet(ocr_meta)
        ocr_df = ocr_df[~ocr_df["scene_id"].isin(scenes_to_exclude)]
        ocr_df.to_parquet(ocr_meta, index=False)
        logger.info(f"Đã lưu ocr_text.parquet mới: {len(ocr_df)} rows")

    # Cập nhật asr_text.parquet nếu có
    asr_meta = unified_dir / "asr_text.parquet"
    if asr_meta.exists():
        logger.info("Đang cập nhật asr_text.parquet...")
        asr_df = pd.read_parquet(asr_meta)
        asr_df = asr_df[~asr_df["scene_id"].isin(scenes_to_exclude)]
        asr_df.to_parquet(asr_meta, index=False)
        logger.info(f"Đã lưu asr_text.parquet mới: {len(asr_df)} rows")

    # Cập nhật scene_descriptions.parquet nếu có
    desc_meta = unified_dir / "scene_descriptions.parquet"
    if desc_meta.exists():
        logger.info("Đang cập nhật scene_descriptions.parquet...")
        desc_df = pd.read_parquet(desc_meta)
        desc_df = desc_df[~desc_df["scene_id"].isin(scenes_to_exclude)]
        desc_df.to_parquet(desc_meta, index=False)
        logger.info(f"Đã lưu scene_descriptions.parquet mới: {len(desc_df)} rows")

    # Cắt và lưu lại các file vector .npy & .pt
    keep_indices = np.where(keep_mask_txt)[0]
    for vec_name in ["image_vectors.npy", "ocr_vectors.npy", "asr_vectors.npy", "description_vectors.npy", "description_embeddings.pt"]:
        vec_path = unified_dir / vec_name
        if vec_path.exists():
            logger.info(f"Đang cắt và cập nhật {vec_name}...")
            ok = slice_and_save_vector_file(vec_path, keep_indices)
            if ok:
                logger.info(f"Đã cập nhật {vec_name} thành công.")
            else:
                logger.warning(f"Không thể cắt hoặc cập nhật {vec_name}.")

    # Đồng bộ xóa trong thư mục data/ocr_output/ (L21_ocr.parquet, L22_ocr.parquet, ...)
    ocr_output_dir = unified_dir.parent / "ocr_output"
    if ocr_output_dir.exists() and ocr_output_dir.is_dir():
        logger.info("Đang đồng bộ và loại bỏ ảnh đã lọc trong data/ocr_output/...")
        for ocr_p in ocr_output_dir.glob("*_ocr.parquet"):
            try:
                batch_ocr_df = pd.read_parquet(ocr_p)
                if "scene_id" in batch_ocr_df.columns:
                    init_len = len(batch_ocr_df)
                    batch_ocr_df = batch_ocr_df[~batch_ocr_df["scene_id"].isin(scenes_to_exclude)]
                    if len(batch_ocr_df) < init_len:
                        batch_ocr_df.to_parquet(ocr_p, index=False)
                        logger.info(f"Đã cập nhật {ocr_p.name}: {init_len} -> {len(batch_ocr_df)} rows")
            except Exception as e:
                logger.warning(f"Không xử lý được file {ocr_p.name}: {e}")

    logger.info("=" * 60)
    logger.info("🎉 ÁP DỤNG FILTER & REBUILD UNIFIED DATASET THÀNH CÔNG!")
    logger.info(f"- Đã xử lý xóa {deleted_files_count} file ảnh")
    logger.info(f"- Unified dataset còn lại {len(new_scene_ids)} keyframes")
    logger.info("Sẵn sàng re-index vào Qdrant với lệnh:")
    logger.info("    python -m indexing.run_index --dataset-dir ./data/unified --collection aic2026_v2 --create-collection")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
