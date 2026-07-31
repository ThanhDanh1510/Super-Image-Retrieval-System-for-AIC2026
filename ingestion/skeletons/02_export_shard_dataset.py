"""
ingestion/skeletons/02_export_shard_dataset.py
================================================
TASK-ING-04: Đóng gói và publish shard dataset lên Kaggle.

INPUT:
    - Toàn bộ output của TASK-ING-01, 02, 03 tại output_dir/:
        scene_metadata.parquet
        ocr_text.parquet
        asr_text.parquet
        image_vectors.npy
        ocr_vectors.npy
        asr_vectors.npy
        scene_ids.txt
        keyframes/{video_id}/{scene_id}_{frame_id:06d}.jpg

OUTPUT:
    - Kaggle Dataset `aic2026-shard-{member}-v{n}` được publish public.

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
import os
import shutil
import subprocess

from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
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

# Các cột bắt buộc trong scene_metadata.parquet
_REQUIRED_METADATA_COLS = [
    "scene_id",
    "video_id",
    "frame_id",
    "timestamp_start",
    "timestamp_end",
    "timestamp_mid",
    "fps",
    "keyframe_path",
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
    errors: list[str] = []
    base_path = Path(output_dir)

    if not base_path.exists():
        return False, [f"Thư mục output_dir không tồn tại: {output_dir}"]

    # 1. Kiểm tra sự tồn tại của các file bắt buộc
    for filename in _REQUIRED_FILES:
        fpath = base_path / filename
        if not fpath.exists():
            errors.append(f"Thiếu file bắt buộc: {filename}")

    # Nếu thiếu file cơ bản thì không thể kiểm tra sâu hơn
    if errors:
        return False, errors

    # 2. Kiểm tra schema của scene_metadata.parquet
    meta_path = base_path / "scene_metadata.parquet"
    try:
        df_meta = pd.read_parquet(meta_path)
        missing_cols = [col for col in _REQUIRED_METADATA_COLS if col not in df_meta.columns]
        if missing_cols:
            errors.append(f"scene_metadata.parquet thiếu các cột: {missing_cols}")
        num_scenes_meta = len(df_meta)
    except Exception as e:
        errors.append(f"Không thể đọc file scene_metadata.parquet: {e}")
        num_scenes_meta = -1

    # 3. Kiểm tra scene_ids.txt
    scene_ids_path = base_path / "scene_ids.txt"
    try:
        with open(scene_ids_path, "r", encoding="utf-8") as f:
            scene_ids = [line.strip() for line in f if line.strip()]
        num_scenes_txt = len(scene_ids)
    except Exception as e:
        errors.append(f"Không thể đọc file scene_ids.txt: {e}")
        num_scenes_txt = -2

    # 4. Kiểm tra shape của các file numpy vector (.npy)
    vector_files = ["image_vectors.npy", "ocr_vectors.npy", "asr_vectors.npy"]
    for vec_file in vector_files:
        vec_path = base_path / vec_file
        try:
            arr = np.load(vec_path)
            if num_scenes_meta >= 0 and arr.shape[0] != num_scenes_meta:
                errors.append(
                    f"Kích thước vector trong {vec_file} ({arr.shape[0]}) không khớp với số scenes trong metadata ({num_scenes_meta})"
                )
        except Exception as e:
            errors.append(f"Không thể load file vector {vec_file}: {e}")

    # Kiểm tra sự nhất quán giữa metadata và scene_ids.txt
    if num_scenes_meta >= 0 and num_scenes_txt >= 0 and num_scenes_meta != num_scenes_txt:
        errors.append(
            f"Số dòng trong scene_metadata.parquet ({num_scenes_meta}) không khớp với scene_ids.txt ({num_scenes_txt})"
        )

    # 5. Kiểm tra thư mục keyframes/
    keyframes_dir = base_path / "keyframes"
    if not keyframes_dir.exists():
        errors.append("Thư mục keyframes/ không tồn tại.")
    else:
        # Kiểm tra xem có file jpg nào không
        jpg_files = list(keyframes_dir.rglob("*.jpg"))
        if len(jpg_files) == 0:
            errors.append("Thư mục keyframes/ rỗng (không chứa file .jpg nào).")

    is_valid = len(errors) == 0
    return is_valid, errors


def package_dataset(output_dir: str, dataset_name: str) -> str:
    """
    Tạo Kaggle dataset-metadata.json và chuẩn bị thư mục để push lên Kaggle.

    Args:
        output_dir: Thư mục chứa output của shard.
        dataset_name: Tên dataset (ví dụ: 'aic2026-shard-an-v1').

    Returns:
        Đường dẫn tuyệt đối tới thư mục đã chuẩn bị (= output_dir).
    """
    base_path = Path(output_dir).resolve()
    if not base_path.exists():
        raise FileNotFoundError(f"Output directory not found: {output_dir}")

    # Lấy kaggle username từ ~/.kaggle/kaggle.json nếu có
    username = "aic2026team"
    kaggle_json_path = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json_path.exists():
        try:
            with open(kaggle_json_path, "r", encoding="utf-8") as f:
                kdata = json.load(f)
                username = kdata.get("username", username)
        except Exception:
            pass
    elif "KAGGLE_USERNAME" in os.environ:
        username = os.environ["KAGGLE_USERNAME"]

    metadata = {
        "title": dataset_name,
        "id": f"{username}/{dataset_name}",
        "licenses": [{"name": "CC0-1.0"}],
    }

    metadata_path = base_path / "dataset-metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"✅ Đã khởi tạo dataset-metadata.json tại: {metadata_path}")
    return str(base_path)


def publish_to_kaggle(
    dataset_path: str,
    dataset_slug: str,
    is_new: bool = False,
) -> None:
    """
    Publish hoặc update dataset lên Kaggle bằng Kaggle API CLI.

    Args:
        dataset_path: Thư mục dataset (chứa dataset-metadata.json).
        dataset_slug: Slug của dataset (ví dụ: 'aic2026-shard-an-v1').
        is_new: True nếu tạo dataset mới, False nếu tạo version mới.
    """
    path = Path(dataset_path).resolve()
    metadata_file = path / "dataset-metadata.json"
    if not metadata_file.exists():
        raise FileNotFoundError(f"Không tìm thấy dataset-metadata.json tại: {path}")

    # Kiểm tra xem CLI `kaggle` có sẵn không
    kaggle_cli = shutil.which("kaggle")
    if not kaggle_cli:
        logger.warning(
            "⚠️ Không tìm thấy CLI `kaggle` trong môi trường hệ thống.\n"
            "Hãy cài đặt bằng: pip install kaggle và đặt API Key vào ~/.kaggle/kaggle.json"
        )
        logger.info(f"Dataset đã được chuẩn bị đầy đủ tại: {path}. Bạn có thể upload thủ công lên Kaggle UI.")
        return

    # Thử tạo dataset mới trước, nếu báo lỗi đã tồn tại thì thử tạo version mới
    try:
        logger.info(f"Đang tiến hành push dataset lên Kaggle API: {dataset_slug}...")
        if is_new:
            cmd = ["kaggle", "datasets", "create", "-p", str(path), "--dir-mode", "zip", "-u"]
        else:
            cmd = ["kaggle", "datasets", "version", "-p", str(path), "-m", "Auto-updated shard data", "--dir-mode", "zip"]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            # Nếu create thất bại do đã tồn tại -> thử version
            if "already exists" in result.stderr or "already exists" in result.stdout or is_new:
                logger.info("Dataset đã tồn tại trên Kaggle. Đang tạo version mới...")
                cmd_v = ["kaggle", "datasets", "version", "-p", str(path), "-m", "Updated shard data", "--dir-mode", "zip"]
                res_v = subprocess.run(cmd_v, capture_output=True, text=True)
                if res_v.returncode == 0:
                    logger.info(f"✅ Tạo version mới thành công!\n{res_v.stdout}")
                    return
                else:
                    logger.error(f"Lỗi khi update version trên Kaggle: {res_v.stderr}")
                    raise RuntimeError(f"Kaggle publish version failed: {res_v.stderr}")
            else:
                logger.error(f"Lỗi khi publish lên Kaggle: {result.stderr}")
                raise RuntimeError(f"Kaggle publish failed: {result.stderr}")

        logger.info(f"✅ Publish Kaggle Dataset thành công!\n{result.stdout}")

    except Exception as e:
        logger.error(f"Lỗi khi gọi Kaggle API CLI: {e}")
        raise


def run_export(
    output_dir: str,
    member: str,
    version: int = 1,
) -> None:
    """
    Entry point chính: Validate, package và publish shard dataset.

    Args:
        output_dir: Thư mục output của shard (/kaggle/working/shard_output/).
        member: Tên member (ví dụ: 'an', 'binh').
        version: Số version của dataset (bắt đầu từ 1).
    """
    logger.info(f"=== BẮT ĐẦU EXPORT SHARD DATASET DÀNH CHO MEMBER [{member}] ===")

    # 1. Validate output
    is_valid, errors = validate_shard_output(output_dir)
    if not is_valid:
        error_msg = "\n".join([f"  - {e}" for e in errors])
        logger.error(f"❌ Kiểm tra Shard Output thất bại. Danh sách lỗi:\n{error_msg}")
        raise ValueError(f"Shard output không hợp lệ. Vui lòng sửa các lỗi sau trước khi export:\n{error_msg}")

    logger.info("✅ Xác thực Shard Output thành công! Tất cả các file và schema đều hợp lệ.")

    # 2. Xây dựng tên dataset
    dataset_name = f"aic2026-shard-{member}-v{version}"

    # 3. Đóng gói metadata.json
    dataset_dir = package_dataset(output_dir, dataset_name)

    # 4. Publish lên Kaggle
    publish_to_kaggle(dataset_dir, dataset_name, is_new=True)
    logger.info(f"🎉 Hoàn tất Export Shard Dataset: {dataset_name}")


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
