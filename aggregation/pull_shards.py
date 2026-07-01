"""
aggregation/pull_shards.py
===========================
Tải toàn bộ shard datasets của các thành viên từ Kaggle về local.

Sử dụng Kaggle CLI:
    kaggle datasets download -d <owner>/<dataset_name> -p <shards_dir> --unzip

Chạy:
    python -m aggregation.pull_shards --shards-dir ./data/raw_shards --owner thanh-danh
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
from pathlib import Path

from shared.utils.config_loader import load_shard_assignment

logger = logging.getLogger(__name__)

def pull_one_shard(owner: str, dataset_name: str, output_dir: Path) -> bool:
    """Tải một shard dataset cụ thể từ Kaggle và giải nén."""
    dataset_slug = f"{owner}/{dataset_name}"
    dest_path = output_dir / dataset_name
    dest_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Đang tải {dataset_slug} về {dest_path}...")
    
    cmd = [
        "kaggle", "datasets", "download",
        "-d", dataset_slug,
        "-p", str(dest_path),
        "--unzip"
    ]
    
    try:
        # Chạy lệnh download
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info(f"Tải thành công {dataset_slug}: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Lỗi khi tải {dataset_slug}: {e.stderr.strip()}")
        return False
    except FileNotFoundError:
        logger.error("Không tìm thấy lệnh 'kaggle'. Vui lòng cài đặt kaggle CLI (pip install kaggle) và cấu hình kaggle.json.")
        return False

def pull_all_shards(shards_dir: str, owner: str) -> dict[str, bool]:
    """Đọc shard_assignment.yaml và tải toàn bộ shard."""
    try:
        shard_cfg = load_shard_assignment()
    except Exception as e:
        logger.error(f"Không thể load shard_assignment.yaml: {e}")
        return {}
        
    members = shard_cfg.get("members", {})
    results = {}
    output_path = Path(shards_dir)
    
    for member_name, info in members.items():
        dataset_name = info.get("dataset_name")
        if not dataset_name:
            logger.warning(f"Thành viên {member_name} không có dataset_name cấu hình.")
            continue
            
        success = pull_one_shard(owner, dataset_name, output_path)
        results[member_name] = success
        
    return results

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    
    parser = argparse.ArgumentParser(description="Tải toàn bộ shard datasets từ Kaggle.")
    parser.add_argument(
        "--shards-dir",
        default="./data/raw_shards",
        help="Thư mục chứa các shards tải về. Mặc định: ./data/raw_shards"
    )
    parser.add_argument(
        "--owner",
        required=True,
        help="Kaggle username của owner sở hữu các datasets."
    )
    
    args = parser.parse_args()
    
    logger.info(f"Bắt đầu tải các shards về thư mục: {args.shards_dir}")
    results = pull_all_shards(args.shards_dir, args.owner)
    
    success_count = sum(1 for s in results.values() if s)
    total_count = len(results)
    
    logger.info(f"=== Hoàn thành: Tải thành công {success_count}/{total_count} shards ===")
    for m, success in results.items():
        status = "Thành công" if success else "Thất bại"
        logger.info(f"  - {m}: {status}")

if __name__ == "__main__":
    main()
