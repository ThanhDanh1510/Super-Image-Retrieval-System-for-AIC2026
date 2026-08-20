"""
tools/data_prep/export_clean_data_packages.py
=============================================
Nén các thư mục dữ liệu sạch cần thiết cho đồng đội / teammate (trừ video):
1. Package 1: AIC2026_data_clean_unified_and_qdrant.zip
   - data/unified/ (Metadata, Parquet, 4 bộ Vectors .npy & .pt)
   - data/qdrant_storage/ (Qdrant DB aic2026_clean_v3 sẵn sàng chạy)
   - data/filtered_keyframes.json & data/reports/
   - data/gemini_output/, data/ocr_output/, data/ASR/
2. Package 2: AIC2026_data_clean_images.zip
   - data/image/ (90,551 ảnh keyframe sạch L21 - L30)
"""

import os
import sys
import time
import zipfile
import subprocess
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ExportData")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
EXPORT_DIR = PROJECT_ROOT / "data_export"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)


def fmt_size(size_bytes):
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f'{size_bytes:.2f} {unit}'
        size_bytes /= 1024.0
    return f'{size_bytes:.2f} PB'


def main():
    logger.info("=" * 70)
    logger.info("📦 BẮT ĐẦU NÉN DỮ LIỆU SẠCH CHO ĐỒNG ĐỘI (TRỪ VIDEO)")
    logger.info("=" * 70)

    # -------------------------------------------------------------------------
    # PACKAGE 1: UNIFIED VECTORS + QDRANT STORAGE + METADATA
    # -------------------------------------------------------------------------
    pkg1_path = EXPORT_DIR / "AIC2026_data_clean_unified_and_qdrant.zip"
    logger.info(f"\n[1/2] Đang nén Package 1: {pkg1_path.name}...")
    t0 = time.time()

    # List of files and folders to include in Package 1
    pkg1_targets = [
        "unified",
        "qdrant_storage",
        "gemini_output",
        "ocr_output",
        "ASR",
        "reports",
        "filtered_keyframes.json",
    ]

    with zipfile.ZipFile(pkg1_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for item_name in pkg1_targets:
            item_path = DATA_DIR / item_name
            if not item_path.exists():
                continue
            if item_path.is_file():
                arcname = f"data/{item_name}"
                zf.write(item_path, arcname)
                logger.info(f"  + Added file: {arcname}")
            elif item_path.is_dir():
                logger.info(f"  + Đang nén thư mục: data/{item_name}...")
                for p in item_path.rglob("*"):
                    if p.is_file():
                        arcname = f"data/{p.relative_to(DATA_DIR).as_posix()}"
                        zf.write(p, arcname)

    pkg1_size = pkg1_path.stat().st_size
    logger.info(f"✅ Hoàn thành Package 1 ({fmt_size(pkg1_size)}) trong {time.time()-t0:.1f}s")

    # -------------------------------------------------------------------------
    # PACKAGE 2: KEYFRAME IMAGES (data/image/)
    # -------------------------------------------------------------------------
    pkg2_path = EXPORT_DIR / "AIC2026_data_clean_images.zip"
    logger.info(f"\n[2/2] Đang nén Package 2: {pkg2_path.name} (Keyframe Images)...")
    t1 = time.time()

    image_dir = DATA_DIR / "image"
    img_files = list(image_dir.rglob("*.webp"))
    total_imgs = len(img_files)
    logger.info(f"  Tìm thấy {total_imgs:,d} clean keyframes trong data/image/")

    with zipfile.ZipFile(pkg2_path, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        for idx, p in enumerate(img_files):
            arcname = f"data/{p.relative_to(DATA_DIR).as_posix()}"
            zf.write(p, arcname)
            if (idx + 1) % 15000 == 0 or (idx + 1) == total_imgs:
                logger.info(f"  + Đã nén {idx+1:,d}/{total_imgs:,d} images ({((idx+1)/total_imgs)*100:.1f}%) in {time.time()-t1:.1f}s")

    pkg2_size = pkg2_path.stat().st_size
    logger.info(f"✅ Hoàn thành Package 2 ({fmt_size(pkg2_size)}) trong {time.time()-t1:.1f}s")

    logger.info("\n" + "=" * 70)
    logger.info("🎉 TỔNG KẾT CÁC GÓI DỮ LIỆU ĐÃ NÉN TẠI data_export/:")
    logger.info("=" * 70)
    logger.info(f"1. {pkg1_path.name:<45} | Size: {fmt_size(pkg1_size)}")
    logger.info(f"2. {pkg2_path.name:<45} | Size: {fmt_size(pkg2_size)}")
    logger.info(f"📁 Thư mục lưu: {EXPORT_DIR}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
