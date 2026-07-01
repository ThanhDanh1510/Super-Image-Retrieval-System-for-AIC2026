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

def merge_all(shards_dir: str, output_dir: str, keyframes_dir: str) -> bool:
    """Gộp toàn bộ shards."""
    shards_path = Path(shards_dir)
    out_path = Path(output_dir)
    keys_path = Path(keyframes_dir)
    
    out_path.mkdir(parents=True, exist_ok=True)
    keys_path.mkdir(parents=True, exist_ok=True)
    
    try:
        shard_cfg = load_shard_assignment()
    except Exception as e:
        logger.error(f"Không thể load shard_assignment.yaml: {e}")
        return False
        
    members = shard_cfg.get("members", {})
    
    # Danh sách để chứa các dataframe
    metadata_dfs = []
    ocr_dfs = []
    asr_dfs = []
    
    # Danh sách để chứa các vectors
    image_vectors_list = []
    ocr_vectors_list = []
    asr_vectors_list = []
    
    # Danh sách scene_ids tổng hợp
    total_scene_ids = []
    
    total_keyframes_copied = 0
    
    for member_name, info in members.items():
        dataset_name = info.get("dataset_name")
        if not dataset_name:
            continue
            
        shard_path = shards_path / dataset_name
        if not shard_path.exists():
            logger.error(f"Thư mục shard của {member_name} không tồn tại: {dataset_name}. Vui lòng chạy pull_shards trước.")
            return False
            
        logger.info(f"Đang xử lý shard: {dataset_name} ({member_name})")
        
        # 1. Đọc parquets
        meta_df = pd.read_parquet(shard_path / "scene_metadata.parquet")
        ocr_df = pd.read_parquet(shard_path / "ocr_text.parquet")
        asr_df = pd.read_parquet(shard_path / "asr_text.parquet")
        
        # Thêm thông tin member và version vào metadata phục vụ truy vết
        meta_df["member"] = member_name
        meta_df["ingestion_version"] = info.get("dataset_name", "v1").split("-")[-1]
        
        metadata_dfs.append(meta_df)
        ocr_dfs.append(ocr_df)
        asr_dfs.append(asr_df)
        
        # 2. Đọc scene_ids
        with open(shard_path / "scene_ids.txt", "r", encoding="utf-8") as f:
            sids = [line.strip() for line in f if line.strip()]
        total_scene_ids.extend(sids)
        
        # 3. Đọc vectors
        image_vecs = np.load(shard_path / "image_vectors.npy")
        ocr_vecs = np.load(shard_path / "ocr_vectors.npy")
        asr_vecs = np.load(shard_path / "asr_vectors.npy")
        
        image_vectors_list.append(image_vecs)
        ocr_vectors_list.append(ocr_vecs)
        asr_vectors_list.append(asr_vecs)
        
        # 4. Copy keyframes
        # Cấu trúc của shard: dataset_name/keyframes/
        # Một số member có thể lưu dataset_name/keyframes/ hoặc dataset_name/keyframes/{video_id}/...
        # Chúng ta tìm thư mục keyframes
        shard_keys_dir = shard_path / "keyframes"
        if not shard_keys_dir.exists():
            # Dự phòng nếu không có folder keyframes trực tiếp mà keyframes nằm ở root của shard
            # (Thường keyframes_path trong metadata trỏ tương đối)
            shard_keys_dir = shard_path
            
        copied = copy_keyframes(shard_keys_dir, keys_path)
        total_keyframes_copied += copied
        logger.info(f"  - Đã sao chép {copied} ảnh keyframes từ {dataset_name}")

    # Gộp parquets
    logger.info("Đang hợp nhất Parquet DataFrames...")
    final_meta = pd.concat(metadata_dfs, ignore_index=True)
    final_ocr = pd.concat(ocr_dfs, ignore_index=True)
    final_asr = pd.concat(asr_dfs, ignore_index=True)
    
    # Gộp vectors
    logger.info("Đang hợp nhất numpy vectors...")
    final_image_vecs = np.concatenate(image_vectors_list, axis=0)
    final_ocr_vecs = np.concatenate(ocr_vectors_list, axis=0)
    final_asr_vecs = np.concatenate(asr_vectors_list, axis=0)
    
    # 5. Ghi dữ liệu gộp ra output
    logger.info(f"Đang ghi kết quả gộp ra: {out_path}")
    final_meta.to_parquet(out_path / "scene_metadata.parquet")
    final_ocr.to_parquet(out_path / "ocr_text.parquet")
    final_asr.to_parquet(out_path / "asr_text.parquet")
    
    np.save(out_path / "image_vectors.npy", final_image_vecs)
    np.save(out_path / "ocr_vectors.npy", final_ocr_vecs)
    np.save(out_path / "asr_vectors.npy", final_asr_vecs)
    
    with open(out_path / "scene_ids.txt", "w", encoding="utf-8") as f:
        for sid in total_scene_ids:
            f.write(f"{sid}\n")
            
    logger.info("=== Hợp nhất hoàn tất ===")
    logger.info(f"  - Tổng số scenes: {len(total_scene_ids)}")
    logger.info(f"  - Shape image_vectors: {final_image_vecs.shape}")
    logger.info(f"  - Shape ocr_vectors: {final_ocr_vecs.shape}")
    logger.info(f"  - Shape asr_vectors: {final_asr_vecs.shape}")
    logger.info(f"  - Tổng số keyframes đã copy: {total_keyframes_copied}")
    
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
    
    args = parser.parse_args()
    
    merge_all(args.shards_dir, args.output_dir, args.keyframes_dir)

if __name__ == "__main__":
    main()
