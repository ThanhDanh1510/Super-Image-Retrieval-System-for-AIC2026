"""
aggregation/pipeline.py
========================
Coordinator điều phối toàn bộ chu trình tổng hợp dữ liệu (Aggregation Pipeline).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from aggregation.build_unified_from_raw import run_build_unified
from aggregation.merge_datasets import merge_all
from aggregation.verify_shards import verify_all_shards

logger = logging.getLogger(__name__)


def build_from_raw_pipeline(
    image_dir: str | Path = "./data/image",
    ocr_dir: str | Path = "./data/OCR",
    asr_json: str | Path = "./data/ASR/asr.json",
    output_dir: str | Path = "./data/unified",
    fps: float = 25.0,
    copy_images: bool = False,
) -> bool:
    """
    Xây dựng Unified Dataset trực tiếp từ dữ liệu thô (Images, OCR JSONs, ASR JSON).
    """
    logger.info("Bắt đầu build Unified Dataset từ raw data...")
    try:
        run_build_unified(
            image_dir=image_dir,
            ocr_dir=ocr_dir,
            asr_json=asr_json,
            output_dir=output_dir,
            fps=fps,
            copy_images=copy_images,
        )
        logger.info("Hoàn tất build Unified Dataset!")
        return True
    except Exception as e:
        logger.error(f"Lỗi khi build Unified Dataset: {e}")
        return False


def merge_shards_pipeline(
    shards_dir: str | Path = "./data/raw_shards",
    output_dir: str | Path = "./data/unified",
    keyframes_dir: str | Path = "./data/unified_keyframes",
    partial: bool = False,
) -> bool:
    """
    Xác thực toàn bộ shards và gộp thành Unified Dataset duy nhất.
    """
    logger.info("Bước 1: Xác thực tính toàn vẹn của các shards...")
    report = verify_all_shards(shards_dir=Path(shards_dir), partial=partial)
    if not report.is_valid:
        logger.error(f"Xác thực shards thất bại với {len(report.errors)} lỗi:")
        for err in report.errors:
            logger.error(f"  - {err}")
        return False

    logger.info("Bước 2: Tiến hành gộp các shards...")
    success = merge_all(
        shards_dir=str(shards_dir),
        output_dir=str(output_dir),
        keyframes_dir=str(keyframes_dir),
        partial=partial,
    )
    return success
