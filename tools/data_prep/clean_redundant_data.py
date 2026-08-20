"""
tools/data_prep/clean_redundant_data.py
======================================
Script dọn dẹp an toàn các file và thư mục dư thừa trong data/:
1. Xóa data/qdrant_db/ (3.85 GB)
2. Xóa data/backup_L25_raw/ (568.41 MB)
3. Xóa các thư mục phẳng data/image/L25/V001..V088 (568 MB), bảo toàn data/image/L25/L25/
4. Xóa data/L.json (46.35 MB)
5. Dọn dẹp các report trung gian cũ trong data/reports/
"""

import os
import shutil
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("CleanData")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

def remove_dir(path: Path):
    if path.exists() and path.is_dir():
        logger.info(f"🗑️ Đang xóa thư mục: {path.relative_to(PROJECT_ROOT)}...")
        try:
            shutil.rmtree(path)
            logger.info(f"✅ Đã xóa: {path.name}")
        except Exception as e:
            logger.error(f"❌ Lỗi xóa {path}: {e}")

def remove_file(path: Path):
    if path.exists() and path.is_file():
        logger.info(f"🗑️ Đang xóa file: {path.relative_to(PROJECT_ROOT)}...")
        try:
            path.unlink()
            logger.info(f"✅ Đã xóa file: {path.name}")
        except Exception as e:
            logger.error(f"❌ Lỗi xóa {path}: {e}")

def main():
    logger.info("=" * 60)
    logger.info("🧹 BẮT ĐẦU DỌN DẸP THƯ MỤC DATA/")
    logger.info("=" * 60)

    # 1. Xóa data/qdrant_db/
    remove_dir(DATA_DIR / "qdrant_db")

    # 2. Xóa data/backup_L25_raw/
    remove_dir(DATA_DIR / "backup_L25_raw")

    # 3. Xóa các folder phẳng Vxxx trong data/image/L25 (giữ data/image/L25/L25)
    l25_dir = DATA_DIR / "image" / "L25"
    if l25_dir.exists():
        nested_l25 = l25_dir / "L25"
        if nested_l25.exists():
            deleted_v_count = 0
            for item in list(l25_dir.iterdir()):
                if item.is_dir() and item.name != "L25":
                    try:
                        shutil.rmtree(item)
                        deleted_v_count += 1
                    except Exception as e:
                        logger.error(f"Lỗi xóa {item}: {e}")
            logger.info(f"✅ Đã xóa {deleted_v_count} thư mục phẳng Vxxx dư thừa trong data/image/L25/ (Đã bảo toàn data/image/L25/L25/)")

    # 4. Xóa data/L.json
    remove_file(DATA_DIR / "L.json")

    # 5. Dọn dẹp reports trung gian cũ trong data/reports/
    reports_dir = DATA_DIR / "reports"
    old_reports = [
        "l25_duplicate_report.json",
        "L25_all_distinct_ocr.json",
        "filtered_images_366.json",
        "filtered_noise_frames.json",
        "l25_pixel_duplicate_report.json",
        "l25_video_file_duplicate_report.json",
        "duplicate_and_discarded_videos.json",
    ]
    for r in old_reports:
        remove_file(reports_dir / r)

    logger.info("=" * 60)
    logger.info("🎉 DỌN DẸP DỮ LIỆU DƯ THỪA HOÀN TẤT!")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
