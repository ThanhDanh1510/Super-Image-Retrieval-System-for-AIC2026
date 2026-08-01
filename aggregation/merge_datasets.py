"""
aggregation/merge_datasets.py
==============================
Gộp tất cả shards đã được xác thực thành một Unified Dataset duy nhất.

Các bước thực hiện:
  1. Đọc và gộp các file metadata Parquet (scene_metadata, ocr_text, asr_text).
  2. Gộp danh sách scene_ids.txt.
  3. Gộp các file vectors (.npy) tương ứng cho image, ocr, và asr.
  4. Sao chép toàn bộ ảnh keyframe từ các thư mục shard về thư mục chung.

Chạy:
    python -m aggregation.merge_datasets \\
        --shards-dir ./data/raw_shards \\
        --output-dir ./data/unified \\
        --keyframes-dir ./data/unified_keyframes
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from shared.utils.config_loader import load_shard_assignment

logger = logging.getLogger(__name__)

def copy_keyframes(shard_keyframes_dir: Path, target_keyframes_dir: Path) -> int:
    """Sao chép toàn bộ ảnh keyframe của shard sang thư mục chung, giữ nguyên cấu trúc thư mục con."""
    count = 0
    if not shard_keyframes_dir.exists():
        return 0
        
    for jpg_path in shard_keyframes_dir.rglob("*.jpg"):
        # Lấy path tương đối so với shard keyframes dir (ví dụ: L01_V001/L01_V001_S001.jpg)
        rel_path = jpg_path.relative_to(shard_keyframes_dir)
        dest_path = target_keyframes_dir / rel_path
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        shutil.copy2(jpg_path, dest_path)
        count += 1
        
    return count

def merge_all(shards_dir: str, output_dir: str, keyframes_dir: str, partial: bool = False) -> bool:
    """Gộp toàn bộ shards."""
    shards_path = Path(shards_dir)
    out_path = Path(output_dir)
    keys_path = Path(keyframes_dir)
    
    out_path.mkdir(parents=True, exist_ok=True)
    keys_path.mkdir(parents=True, exist_ok=True)
    
    metadata_dfs = []
    ocr_dfs = []
    asr_dfs = []
    
    image_vectors_list = []
    ocr_vectors_list = []
    asr_vectors_list = []
    
    total_scene_ids = []
    total_keyframes_copied = 0
    
    # Lấy danh sách shard directories
    if partial:
        shard_paths = [d for d in shards_path.iterdir() if d.is_dir()]
    else:
        try:
            shard_cfg = load_shard_assignment()
            members = shard_cfg.get("members", {})
            shard_paths = []
            for member_name, info in members.items():
                dataset_name = info.get("dataset_name")
                if dataset_name and (shards_path / dataset_name).exists():
                    shard_paths.append(shards_path / dataset_name)
        except Exception as e:
            logger.error(f"Không thể load shard_assignment.yaml: {e}")
            shard_paths = [d for d in shards_path.iterdir() if d.is_dir()]

    if not shard_paths:
        logger.error(f"Không tìm thấy shard nào để gộp trong: {shards_dir}")
        return False

    for shard_path in shard_paths:
        logger.info(f"Đang xử lý shard: {shard_path.name}")
        
        # 1. Đọc scene_metadata.parquet
        meta_file = shard_path / "scene_metadata.parquet"
        if not meta_file.exists():
            # Dự phòng nếu file nằm ở subfolder
            possible = list(shard_path.rglob("scene_metadata.parquet"))
            if possible:
                meta_file = possible[0]
            else:
                logger.warning(f"Bỏ qua {shard_path.name}: Không tìm thấy scene_metadata.parquet")
                continue

        meta_df = pd.read_parquet(meta_file)
        metadata_dfs.append(meta_df)
        
        if "scene_id" in meta_df.columns:
            total_scene_ids.extend(meta_df["scene_id"].tolist())

        # 2. Đọc OCR / ASR parquets nếu không phải partial mode
        if not partial:
            ocr_file = shard_path / "ocr_text.parquet"
            if ocr_file.exists():
                ocr_dfs.append(pd.read_parquet(ocr_file))
            asr_file = shard_path / "asr_text.parquet"
            if asr_file.exists():
                asr_dfs.append(pd.read_parquet(asr_file))

            # 3. Đọc vectors nếu có
            if (shard_path / "image_vectors.npy").exists():
                image_vectors_list.append(np.load(shard_path / "image_vectors.npy"))
            if (shard_path / "ocr_vectors.npy").exists():
                ocr_vectors_list.append(np.load(shard_path / "ocr_vectors.npy"))
            if (shard_path / "asr_vectors.npy").exists():
                asr_vectors_list.append(np.load(shard_path / "asr_vectors.npy"))
                
    # Gộp metadata parquet
    logger.info("Đang hợp nhất Parquet DataFrames...")
    if metadata_dfs:
        final_meta = pd.concat(metadata_dfs, ignore_index=True)
        # Bỏ duplicate scene_id nếu có
        if "scene_id" in final_meta.columns:
            final_meta = final_meta.drop_duplicates(subset=["scene_id"], keep="first")
        final_meta.to_parquet(out_path / "scene_metadata.parquet", index=False)
        logger.info(f"✅ Đã lưu {len(final_meta)} rows chuẩn vào scene_metadata.parquet")

        # Copy CHÍNH XÁC các ảnh keyframe có tên trong scene_metadata.parquet (loại bỏ ảnh rác)
        logger.info("Đang sao chép đúng các ảnh keyframes có trong metadata...")
        for row in final_meta.itertuples():
            rel_path = getattr(row, "keyframe_path", None)
            if not rel_path:
                continue
                
            # Tìm file ảnh tương ứng trong các raw_shards
            found = False
            for shard_path in shard_paths:
                possible_src = shard_path / "keyframes" / rel_path
                if not possible_src.exists():
                    possible_src = shard_path / rel_path
                    
                if possible_src.exists():
                    dest_path = keys_path / rel_path
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(possible_src, dest_path)
                    total_keyframes_copied += 1
                    found = True
                    break
                    
            if not found:
                logger.warning(f"Không tìm thấy file ảnh nguồn cho: {rel_path}")

    if not partial:
        if ocr_dfs:
            pd.concat(ocr_dfs, ignore_index=True).drop_duplicates(subset=["scene_id"], keep="first").to_parquet(out_path / "ocr_text.parquet", index=False)
        if asr_dfs:
            pd.concat(asr_dfs, ignore_index=True).drop_duplicates(subset=["scene_id"], keep="first").to_parquet(out_path / "asr_text.parquet", index=False)
            
        if image_vectors_list:
            final_image_vecs = np.concatenate(image_vectors_list, axis=0)
            np.save(out_path / "image_vectors.npy", final_image_vecs)
        if ocr_vectors_list:
            final_ocr_vecs = np.concatenate(ocr_vectors_list, axis=0)
            np.save(out_path / "ocr_vectors.npy", final_ocr_vecs)
        if asr_vectors_list:
            final_asr_vecs = np.concatenate(asr_vectors_list, axis=0)
            np.save(out_path / "asr_vectors.npy", final_asr_vecs)

    # Ghi scene_ids.txt
    unique_scene_ids = list(dict.fromkeys(total_scene_ids))
    with open(out_path / "scene_ids.txt", "w", encoding="utf-8") as f:
        for sid in unique_scene_ids:
            f.write(f"{sid}\n")

    logger.info("=== Hợp nhất hoàn tất ===")
    logger.info(f"  - Tổng số unique scenes: {len(unique_scene_ids)}")
    logger.info(f"  - Tổng số keyframes đã copy: {total_keyframes_copied}")
    logger.info(f"  - Thư mục output: {out_path.resolve()}")
    logger.info(f"  - Thư mục keyframes: {keys_path.resolve()}")
    
    return True

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    
    parser = argparse.ArgumentParser(description="Gộp các shard datasets thành một bộ dữ liệu thống nhất.")
    parser.add_argument(
        "--shards-dir",
        default="./data/raw_shards",
        help="Thư mục chứa các shards. Mặc định: ./data/raw_shards"
    )
    parser.add_argument(
        "--output-dir",
        default="./data/unified",
        help="Thư mục ghi kết quả gộp. Mặc định: ./data/unified"
    )
    parser.add_argument(
        "--keyframes-dir",
        default="./data/unified_keyframes",
        help="Thư mục chung lưu trữ keyframes phục vụ UI. Mặc định: ./data/unified_keyframes"
    )
    parser.add_argument(
        "--partial",
        action="store_true",
        help="Gộp chế độ Partial Keyframe Shards (chỉ gộp scene_metadata và keyframes)."
    )
    
    args = parser.parse_args()
    
    merge_all(args.shards_dir, args.output_dir, args.keyframes_dir, partial=args.partial)

if __name__ == "__main__":
    main()
