"""
aggregation/filter_monochrome.py
=================================
Lọc bỏ ảnh đơn sắc / ít thông tin và đồng bộ OCR JSON.

Quy trình:
  1. Quét toàn bộ ảnh .webp trong data/image/{Lxx}/{Lxx}/{Vxxx}/
  2. Phân tích từng ảnh: std deviation RGB + mean saturation HSV
  3. Ảnh bị loại -> di chuyển sang data/image_filtered/ (backup)
  4. OCR JSON -> chuyển key từ index thứ tự sang frame_id, loại bỏ entry ảnh đã lọc

Chạy:
    python -m aggregation.filter_monochrome \\
        --image-dir ./data/image \\
        --ocr-dir ./data/OCR \\
        --backup-dir ./data/image_filtered \\
        --std-threshold 15 \\
        --sat-threshold 20 \\
        --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def is_low_info(
    image_path: Path,
    std_threshold: float = 15.0,
    sat_threshold: float = 20.0,
) -> bool:
    """Kiểm tra ảnh có phải đơn sắc / ít thông tin hay không.

    Tiêu chí:
      - Std deviation trung bình trên 3 kênh RGB < std_threshold
      - Mean saturation (kênh S trong HSV) < sat_threshold

    Ảnh bị đánh dấu khi CÙNG LÚC thỏa mãn CẢ HAI điều kiện.
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        return False  # Không đọc được -> giữ lại, không xóa

    # Tính std deviation trung bình trên 3 kênh BGR
    mean_std = np.mean([np.std(img[:, :, c]) for c in range(3)])

    # Tính mean saturation (kênh S trong HSV)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mean_sat = np.mean(hsv[:, :, 1])

    return mean_std < std_threshold and mean_sat < sat_threshold


def get_sorted_images(video_dir: Path) -> list[Path]:
    """Lấy danh sách ảnh .webp trong folder video, sắp xếp theo frame_id tăng dần."""
    images = list(video_dir.glob("*.webp"))
    images.sort(key=lambda f: int(f.stem))
    return images


def process_video(
    video_dir: Path,
    ocr_file: Path | None,
    backup_dir: Path | None,
    std_threshold: float,
    sat_threshold: float,
    dry_run: bool,
) -> dict:
    """Xử lý lọc ảnh đơn sắc và đồng bộ OCR cho một video.

    Returns:
        dict với các thống kê: total, filtered, kept
    """
    sorted_images = get_sorted_images(video_dir)
    total = len(sorted_images)

    if total == 0:
        return {"total": 0, "filtered": 0, "kept": 0}

    # Đọc OCR JSON gốc (nếu có)
    original_ocr: dict[str, str] = {}
    if ocr_file and ocr_file.exists():
        try:
            with open(ocr_file, "r", encoding="utf-8") as f:
                original_ocr = json.load(f)
        except Exception as e:
            logger.warning(f"Không đọc được OCR {ocr_file.name}: {e}")

    # Quét từng ảnh và quyết định giữ/loại
    filtered_indices: list[int] = []
    new_ocr: dict[str, str] = {}

    for idx, img_path in enumerate(sorted_images):
        frame_id = img_path.stem  # Ví dụ: "17", "54", "150"

        if is_low_info(img_path, std_threshold, sat_threshold):
            filtered_indices.append(idx)

            if not dry_run and backup_dir:
                # Di chuyển ảnh bị loại sang backup
                rel_path = img_path.relative_to(img_path.parents[3])  # Relative từ data/image/
                dest = backup_dir / rel_path
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(img_path), str(dest))
        else:
            # Giữ lại -> thêm vào OCR mới với key = frame_id
            ocr_text = original_ocr.get(str(idx), "")
            new_ocr[frame_id] = ocr_text

    # Ghi đè OCR JSON mới (key = frame_id)
    if ocr_file and not dry_run:
        with open(ocr_file, "w", encoding="utf-8") as f:
            json.dump(new_ocr, f, ensure_ascii=False, indent=2)

    filtered_count = len(filtered_indices)
    return {
        "total": total,
        "filtered": filtered_count,
        "kept": total - filtered_count,
    }


def filter_and_sync(
    image_dir: str,
    ocr_dir: str,
    backup_dir: str,
    std_threshold: float = 15.0,
    sat_threshold: float = 20.0,
    dry_run: bool = False,
) -> dict:
    """Main entry point: lọc ảnh đơn sắc và đồng bộ OCR cho toàn bộ L21-L30.

    Returns:
        dict tổng kết toàn cục
    """
    image_base = Path(image_dir)
    ocr_base = Path(ocr_dir)
    backup_base = Path(backup_dir) if backup_dir else None

    if backup_base and not dry_run:
        backup_base.mkdir(parents=True, exist_ok=True)

    grand_total = 0
    grand_filtered = 0
    grand_kept = 0
    video_reports: list[dict] = []

    # Duyệt qua từng thư mục Lxx
    for lxx_dir in sorted(image_base.iterdir()):
        if not lxx_dir.is_dir():
            continue
        lxx_name = lxx_dir.name  # "L21"

        # Cấu trúc: data/image/L21/L21/V001/
        inner_dir = lxx_dir / lxx_name
        if not inner_dir.exists():
            continue

        for video_dir in sorted(inner_dir.iterdir()):
            if not video_dir.is_dir():
                continue
            vxxx_name = video_dir.name  # "V001"
            video_label = f"{lxx_name}_{vxxx_name}"

            # Tìm OCR file tương ứng: data/OCR/L21/L21_V001.json
            ocr_file = ocr_base / lxx_name / f"{video_label}.json"

            result = process_video(
                video_dir=video_dir,
                ocr_file=ocr_file,
                backup_dir=backup_base,
                std_threshold=std_threshold,
                sat_threshold=sat_threshold,
                dry_run=dry_run,
            )

            grand_total += result["total"]
            grand_filtered += result["filtered"]
            grand_kept += result["kept"]

            if result["filtered"] > 0:
                video_reports.append({
                    "video": video_label,
                    **result,
                })
                logger.info(
                    f"  {'[DRY-RUN] ' if dry_run else ''}"
                    f"{video_label}: {result['filtered']}/{result['total']} ảnh bị loại"
                )

    # Tổng kết
    mode_label = "[DRY-RUN] " if dry_run else ""
    logger.info(f"\n{'=' * 60}")
    logger.info(f"{mode_label}=== TỔNG KẾT LỌC ẢNH ĐƠN SẮC ===")
    logger.info(f"  Tổng số ảnh quét: {grand_total:,}")
    logger.info(f"  Ảnh bị loại:      {grand_filtered:,} ({grand_filtered/max(grand_total,1)*100:.1f}%)")
    logger.info(f"  Ảnh giữ lại:      {grand_kept:,} ({grand_kept/max(grand_total,1)*100:.1f}%)")
    logger.info(f"  Số video bị ảnh hưởng: {len(video_reports)}")
    if not dry_run:
        logger.info(f"  OCR JSON đã chuyển sang format key=frame_id")
        logger.info(f"  Ảnh bị loại đã chuyển sang: {backup_base}")
    logger.info(f"{'=' * 60}")

    return {
        "total": grand_total,
        "filtered": grand_filtered,
        "kept": grand_kept,
        "videos_affected": len(video_reports),
        "details": video_reports,
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Lọc ảnh đơn sắc / ít thông tin và đồng bộ OCR JSON."
    )
    parser.add_argument(
        "--image-dir",
        default="./data/image",
        help="Thư mục gốc chứa ảnh keyframe. Mặc định: ./data/image",
    )
    parser.add_argument(
        "--ocr-dir",
        default="./data/OCR",
        help="Thư mục gốc chứa OCR JSON. Mặc định: ./data/OCR",
    )
    parser.add_argument(
        "--backup-dir",
        default="./data/image_filtered",
        help="Thư mục backup ảnh bị loại. Mặc định: ./data/image_filtered",
    )
    parser.add_argument(
        "--std-threshold",
        type=float,
        default=15.0,
        help="Ngưỡng std deviation RGB. Mặc định: 15.0",
    )
    parser.add_argument(
        "--sat-threshold",
        type=float,
        default=20.0,
        help="Ngưỡng mean saturation HSV. Mặc định: 20.0",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Chỉ in kết quả phân tích, không xóa/di chuyển file thật.",
    )

    args = parser.parse_args()

    filter_and_sync(
        image_dir=args.image_dir,
        ocr_dir=args.ocr_dir,
        backup_dir=args.backup_dir,
        std_threshold=args.std_threshold,
        sat_threshold=args.sat_threshold,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
