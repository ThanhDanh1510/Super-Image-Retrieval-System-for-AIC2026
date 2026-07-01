"""
aggregation/publish_unified.py
==============================
Đóng gói và publish Unified Dataset lên Kaggle.

Dữ liệu sẽ được đẩy lên Kaggle dưới tên: aic2026-unified-v{version}

Chạy:
    python -m aggregation.publish_unified --unified-dir ./data/unified --version 1 --owner thanh-danh
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

def package_metadata(unified_dir: Path, dataset_name: str, owner: str) -> None:
    """Tạo file dataset-metadata.json phục vụ Kaggle CLI."""
    meta = {
        "title": dataset_name,
        "id": f"{owner}/{dataset_name}",
        "licenses": [{"name": "CC0-1.0"}]
    }
    
    meta_file = unified_dir / "dataset-metadata.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
        
    logger.info(f"Đã tạo file metadata tại: {meta_file}")

def publish_dataset(unified_dir: Path, dataset_name: str, is_new: bool = False) -> bool:
    """Gọi Kaggle API để đẩy dataset lên."""
    if is_new:
        cmd = [
            "kaggle", "datasets", "create",
            "-p", str(unified_dir),
            "--dir-mode", "zip"
        ]
    else:
        cmd = [
            "kaggle", "datasets", "version",
            "-p", str(unified_dir),
            "-m", f"Cập nhật unified dataset {dataset_name}",
            "--dir-mode", "zip"
        ]
        
    logger.info(f"Đang thực thi lệnh publish lên Kaggle (is_new={is_new})...")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info(f"Publish thành công: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Lỗi khi publish lên Kaggle: {e.stderr.strip()}")
        # Nếu cố gắng cập nhật mà dataset chưa tồn tại, thử tạo mới
        if not is_new and "Could not find dataset" in e.stderr:
            logger.info("Dataset chưa tồn tại trên Kaggle. Thử tạo mới...")
            return publish_dataset(unified_dir, dataset_name, is_new=True)
        return False

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    
    parser = argparse.ArgumentParser(description="Publish Unified Dataset lên Kaggle.")
    parser.add_argument(
        "--unified-dir",
        default="./data/unified",
        help="Thư mục chứa Unified Dataset. Mặc định: ./data/unified"
    )
    parser.add_argument(
        "--version",
        type=int,
        default=1,
        help="Version của unified dataset. Mặc định: 1"
    )
    parser.add_argument(
        "--owner",
        required=True,
        help="Kaggle username của owner."
    )
    parser.add_argument(
        "--new",
        action="store_true",
        help="Tạo mới hoàn toàn dataset thay vì đẩy phiên bản mới."
    )
    
    args = parser.parse_args()
    
    unified_path = Path(args.unified_dir)
    if not unified_path.exists():
        logger.error(f"Thư mục Unified Dataset không tồn tại: {args.unified_dir}")
        return
        
    dataset_name = f"aic2026-unified-v{args.version}"
    
    package_metadata(unified_path, dataset_name, args.owner)
    success = publish_dataset(unified_path, dataset_name, is_new=args.new)
    
    if success:
        logger.info(f"✅ Đã publish thành công bộ dữ liệu thống nhất {dataset_name} lên Kaggle.")
    else:
        logger.error("❌ Publish thất bại.")

if __name__ == "__main__":
    main()
