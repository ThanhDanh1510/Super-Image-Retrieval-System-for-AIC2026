"""
ingestion/skeletons/02_export_shard_dataset.py
================================================
TASK-ING-04: Đóng gói và publish shard dataset lên Kaggle.

SKELETON: Implement các hàm có `raise NotImplementedError`.

INPUT:
    - Toàn bộ output của TASK-ING-01, 02, 03 tại output_dir/:
        scene_metadata.parquet
        ocr_text.parquet
        asr_text.parquet
        image_vectors.npy
        ocr_vectors.npy
        asr_vectors.npy
        scene_ids.txt
        keyframes/{video_id}/{scene_id}.jpg

OUTPUT:
    - Kaggle Dataset `aic2026-shard-{member}-v{n}` được publish public.
    - Tất cả files trên được đóng gói vào dataset này.

CONSTRAINTS (KHÔNG ĐƯỢC THAY ĐỔI):
    - Dataset name format: aic2026-shard-{member}-v{n}  (ví dụ: aic2026-shard-an-v1)
    - Dataset visibility: public (để aggregation/ có thể pull về)
    - Phải chạy validate_shard_output() TRƯỚC khi publish — nếu thiếu file → báo lỗi, không publish

CÁCH CHẠY (Kaggle Notebook cell):
    !python -m ingestion.skeletons.02_export_shard_dataset \
        --output-dir /kaggle/working/shard_output \
        --member an \
        --version 1
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Danh sách files bắt buộc phải có trước khi publish
_REQUIRED_FILES = [
    "scene_metadata.parquet",
    "ocr_text.parquet",
    "asr_text.parquet",
    "image_vectors.npy",
    "ocr_vectors.npy",
    "asr_vectors.npy",
    "scene_ids.txt",
]


def validate_shard_output(output_dir: str) -> tuple[bool, list[str]]:
    """
    Xác thực output_dir có đầy đủ file và đúng định dạng trước khi publish.

    Kiểm tra:
        1. Tất cả _REQUIRED_FILES tồn tại.
        2. scene_metadata.parquet có đủ 8 cột bắt buộc.
        3. Số rows trong scene_metadata.parquet = số dòng trong scene_ids.txt
           = axis-0 size của tất cả .npy files.
        4. Thư mục keyframes/ tồn tại và không rỗng.

    Args:
        output_dir: Thư mục output của shard.

    Returns:
        Tuple (is_valid, error_messages):
            - is_valid: True nếu tất cả kiểm tra đều qua.
            - error_messages: List các lỗi tìm thấy (rỗng nếu is_valid=True).
    """
    raise NotImplementedError


def package_dataset(output_dir: str, dataset_name: str) -> str:
    """
    Tạo Kaggle dataset-metadata.json và chuẩn bị thư mục để push lên Kaggle.

    Kaggle API yêu cầu một file dataset-metadata.json ở thư mục gốc của dataset
    trước khi có thể publish bằng `kaggle datasets create/version`.

    Gợi ý implementation:
        metadata = {
            'title': dataset_name,
            'id': f'<your_kaggle_username>/{dataset_name}',
            'licenses': [{'name': 'CC0-1.0'}],
        }
        metadata_path = Path(output_dir) / 'dataset-metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        return output_dir

    Args:
        output_dir: Thư mục chứa output của shard.
        dataset_name: Tên dataset (ví dụ: 'aic2026-shard-an-v1').

    Returns:
        Đường dẫn tuyệt đối tới thư mục đã chuẩn bị (= output_dir).

    Raises:
        FileNotFoundError: Nếu output_dir không tồn tại.
    """
    raise NotImplementedError


def publish_to_kaggle(
    dataset_path: str,
    dataset_slug: str,
    is_new: bool = False,
) -> None:
    """
    Publish hoặc update dataset lên Kaggle bằng Kaggle API CLI.

    Gợi ý implementation:
        import subprocess
        if is_new:
            cmd = ['kaggle', 'datasets', 'create', '-p', dataset_path, '--dir-mode', 'zip']
        else:
            cmd = ['kaggle', 'datasets', 'version', '-p', dataset_path, '-m', 'Updated shard']
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info(f'Kaggle publish output: {result.stdout}')

    ĐIỀU KIỆN TIÊN QUYẾT:
        - File ~/.kaggle/kaggle.json phải tồn tại với API key hợp lệ.
        - Dataset phải đã được tạo trên Kaggle UI lần đầu (nếu is_new=False).

    Args:
        dataset_path: Thư mục dataset (chứa dataset-metadata.json).
        dataset_slug: Slug của dataset (ví dụ: 'aic2026-shard-an-v1').
        is_new: True nếu tạo dataset mới, False nếu tạo version mới.

    Raises:
        RuntimeError: Nếu Kaggle CLI không khả dụng hoặc publish thất bại.
    """
    raise NotImplementedError


def run_export(
    output_dir: str,
    member: str,
    version: int = 1,
) -> None:
    """
    Entry point chính: Validate, package và publish shard dataset.

    Luồng xử lý:
        1. validate_shard_output() → nếu không hợp lệ → raise ValueError với danh sách lỗi
        2. Xây dựng dataset_name = f'aic2026-shard-{member}-v{version}'
        3. package_dataset() → tạo dataset-metadata.json
        4. publish_to_kaggle()
        5. Log URL dataset đã publish

    Args:
        output_dir: Thư mục output của shard (/kaggle/working/shard_output/).
        member: Tên member (ví dụ: 'an', 'binh').
        version: Số version của dataset (bắt đầu từ 1).

    Raises:
        ValueError: Nếu validate_shard_output() trả về lỗi.
        RuntimeError: Nếu publish thất bại.
    """
    raise NotImplementedError


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="TASK-ING-04: Đóng gói và publish shard dataset lên Kaggle.")
    parser.add_argument("--output-dir", default="/kaggle/working/shard_output", help="Thư mục output của shard.")
    parser.add_argument("--member", required=True, help="Tên thành viên chạy pipeline.")
    parser.add_argument("--version", type=int, default=1, help="Số version của dataset.")
    
    args = parser.parse_args()
    
    run_export(
        output_dir=args.output_dir,
        member=args.member,
        version=args.version,
    )
