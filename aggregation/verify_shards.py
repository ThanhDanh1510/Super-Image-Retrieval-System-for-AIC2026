"""
aggregation/verify_shards.py
=============================
Xác thực tính toàn vẹn và nhất quán của các shard datasets trước khi gộp.

Các kiểm tra:
  1. Kiểm tra trùng lặp video_id giữa các thành viên (video_id chỉ được gán cho 1 member).
  2. Kiểm tra xem các video_id được gán trong shard_assignment.yaml có bị thiếu không.
  3. Kiểm tra định dạng schema của các file parquet (scene_metadata, ocr_text, asr_text).
  4. Kiểm tra sự khớp nhau giữa metadata (parquet), list scene_ids (txt), và vectors (.npy).

Chạy:
    python -m aggregation.verify_shards --shards-dir ./data/raw_shards
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from shared.utils.config_loader import load_shard_assignment

logger = logging.getLogger(__name__)

# Schema các cột bắt buộc
REQUIRED_COLUMNS = {
    "scene_metadata.parquet": [
        "scene_id", "video_id", "frame_id",
        "timestamp_start", "timestamp_end", "timestamp_mid",
        "fps", "keyframe_path"
    ],
    "ocr_text.parquet": ["scene_id", "video_id", "ocr_text"],
    "asr_text.parquet": ["scene_id", "video_id", "asr_text"]
}

@dataclass
class VerifyReport:
    is_valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    shards_checked: list[str] = field(default_factory=list)

def verify_shard_integrity(shard_dir: Path, expected_video_ids: set[str]) -> list[str]:
    """Kiểm tra tính toàn vẹn của một shard đơn lẻ."""
    errors = []
    
    # 1. Kiểm tra sự tồn tại của các file bắt buộc
    required_files = [
        "scene_metadata.parquet",
        "ocr_text.parquet",
        "asr_text.parquet",
        "image_vectors.npy",
        "ocr_vectors.npy",
        "asr_vectors.npy",
        "scene_ids.txt"
    ]
    
    for f in required_files:
        if not (shard_dir / f).exists():
            errors.append(f"Thiếu file bắt buộc: {f}")
            
    if errors:
        return errors  # Nếu thiếu file, dừng kiểm tra sâu hơn cho shard này
        
    # 2. Kiểm tra schema của Parquets
    try:
        meta_df = pd.read_parquet(shard_dir / "scene_metadata.parquet")
        for col in REQUIRED_COLUMNS["scene_metadata.parquet"]:
            if col not in meta_df.columns:
                errors.append(f"scene_metadata.parquet thiếu cột: {col}")
                
        ocr_df = pd.read_parquet(shard_dir / "ocr_text.parquet")
        for col in REQUIRED_COLUMNS["ocr_text.parquet"]:
            if col not in ocr_df.columns:
                errors.append(f"ocr_text.parquet thiếu cột: {col}")
                
        asr_df = pd.read_parquet(shard_dir / "asr_text.parquet")
        for col in REQUIRED_COLUMNS["asr_text.parquet"]:
            if col not in asr_df.columns:
                errors.append(f"asr_text.parquet thiếu cột: {col}")
    except Exception as e:
        errors.append(f"Lỗi khi đọc file parquet: {e}")
        return errors

    # 3. Kiểm tra tính khớp dữ liệu (Alignment)
    try:
        with open(shard_dir / "scene_ids.txt", "r", encoding="utf-8") as f:
            scene_ids = [line.strip() for line in f if line.strip()]
            
        n_scenes = len(scene_ids)
        
        # Đọc vectors shape
        image_vecs = np.load(shard_dir / "image_vectors.npy", mmap_mode="r")
        ocr_vecs = np.load(shard_dir / "ocr_vectors.npy", mmap_mode="r")
        asr_vecs = np.load(shard_dir / "asr_vectors.npy", mmap_mode="r")
        
        if image_vecs.shape[0] != n_scenes:
            errors.append(f"image_vectors.npy có {image_vecs.shape[0]} samples, khác với {n_scenes} scene_ids")
        if ocr_vecs.shape[0] != n_scenes:
            errors.append(f"ocr_vectors.npy có {ocr_vecs.shape[0]} samples, khác với {n_scenes} scene_ids")
        if asr_vecs.shape[0] != n_scenes:
            errors.append(f"asr_vectors.npy có {asr_vecs.shape[0]} samples, khác với {n_scenes} scene_ids")
            
        if len(meta_df) != n_scenes:
            errors.append(f"scene_metadata.parquet có {len(meta_df)} rows, khác với {n_scenes} scene_ids")
            
        # Kiểm tra video_id thực tế so với expected
        actual_video_ids = set(meta_df["video_id"].unique())
        extra_videos = actual_video_ids - expected_video_ids
        missing_videos = expected_video_ids - actual_video_ids
        
        if extra_videos:
            errors.append(f"Chứa các video_id không được phân công: {list(extra_videos)}")
        if missing_videos:
            errors.append(f"Thiếu các video_id được phân công: {list(missing_videos)}")
            
    except Exception as e:
        errors.append(f"Lỗi trong quá trình kiểm tra alignment: {e}")
        
    return errors

def verify_all_shards(shards_dir: str) -> VerifyReport:
    """Xác thực toàn bộ các shards."""
    report = VerifyReport()
    shards_path = Path(shards_dir)
    
    if not shards_path.exists():
        report.is_valid = False
        report.errors.append(f"Thư mục shards không tồn tại: {shards_dir}")
        return report
        
    try:
        shard_cfg = load_shard_assignment()
    except Exception as e:
        report.is_valid = False
        report.errors.append(f"Không thể load shard_assignment.yaml: {e}")
        return report

    members = shard_cfg.get("members", {})
    
    # 1. Kiểm tra chéo trùng lặp phân công trong config trước
    video_assignment: dict[str, str] = {} # video_id -> member_name
    for member_name, info in members.items():
        vids = info.get("video_ids", [])
        for vid in vids:
            if vid in video_assignment:
                report.is_valid = False
                report.errors.append(
                    f"Config lỗi: video_id '{vid}' bị trùng phân công giữa "
                    f"'{video_assignment[vid]}' và '{member_name}'"
                )
            video_assignment[vid] = member_name

    # 2. Kiểm tra từng shard thực tế
    actual_shard_videos: dict[str, list[str]] = {} # video_id -> list of shard_names
    
    for member_name, info in members.items():
        dataset_name = info.get("dataset_name")
        if not dataset_name:
            continue
            
        shard_dir = shards_path / dataset_name
        report.shards_checked.append(dataset_name)
        
        if not shard_dir.exists():
            report.is_valid = False
            report.errors.append(f"Thư mục shard của '{member_name}' không tồn tại: {dataset_name}")
            continue
            
        # expected video_ids cho member này
        expected_vids = set(info.get("video_ids", []))
        
        shard_errors = verify_shard_integrity(shard_dir, expected_vids)
        if shard_errors:
            report.is_valid = False
            for err in shard_errors:
                report.errors.append(f"[{dataset_name}] {err}")
                
        # Lưu lại video_id thực tế có trong shard để check trùng chéo thực tế
        try:
            if (shard_dir / "scene_metadata.parquet").exists():
                meta_df = pd.read_parquet(shard_dir / "scene_metadata.parquet")
                for vid in meta_df["video_id"].unique():
                    actual_shard_videos.setdefault(vid, []).append(dataset_name)
        except Exception:
            pass

    # 3. Kiểm tra chéo trùng lặp video_id thực tế giữa các file tải về
    for vid, shard_names in actual_shard_videos.items():
        if len(shard_names) > 1:
            report.is_valid = False
            report.errors.append(
                f"Trùng lặp dữ liệu: video_id '{vid}' xuất hiện ở nhiều shards: {shard_names}"
            )
            
    return report

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    
    parser = argparse.ArgumentParser(description="Xác thực các shard datasets trước khi merge.")
    parser.add_argument(
        "--shards-dir",
        default="./data/raw_shards",
        help="Thư mục chứa các shards. Mặc định: ./data/raw_shards"
    )
    
    args = parser.parse_args()
    
    logger.info("Bắt đầu xác thực các shards...")
    report = verify_all_shards(args.shards_dir)
    
    logger.info(f"Đã kiểm tra {len(report.shards_checked)} shards: {report.shards_checked}")
    
    if report.warnings:
        logger.warning(f"=== Có {len(report.warnings)} Cảnh báo ===")
        for w in report.warnings:
            logger.warning(f"  [WARN] {w}")
            
    if not report.is_valid:
        logger.error(f"=== Xác thực THẤT BẠI (Có {len(report.errors)} lỗi nghiêm trọng) ===")
        for e in report.errors:
            logger.error(f"  [ERROR] {e}")
        import sys
        sys.exit(1)
    else:
        logger.info("✅ Xác thực THÀNH CÔNG! Sẵn sàng gộp dữ liệu.")

if __name__ == "__main__":
    main()
