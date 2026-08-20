"""
aggregation/common.py
=====================
Các hằng số, schema chuẩn và tiện ích dùng chung cho tầng aggregation.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Hằng số chuẩn
DEFAULT_FPS: float = 25.0
IMAGE_EXTENSIONS: set[str] = {".webp", ".jpg", ".jpeg", ".png"}

# Danh sách các file Parquet và Vector trong Unified Dataset
METADATA_FILES: set[str] = {
    "scene_metadata.parquet",
    "ocr_text.parquet",
    "asr_text.parquet",
    "scene_descriptions.parquet",
}

VECTOR_FILES: set[str] = {
    "image_vectors.npy",
    "ocr_vectors.npy",
    "asr_vectors.npy",
    "description_vectors.npy",
    "description_embeddings.pt",
}

# Schema các cột bắt buộc cho từng file Parquet
REQUIRED_COLUMNS: dict[str, list[str]] = {
    "scene_metadata.parquet": [
        "scene_id",
        "video_id",
        "frame_id",
        "timestamp_start",
        "timestamp_end",
        "timestamp_mid",
        "fps",
        "keyframe_path",
    ],
    "ocr_text.parquet": ["scene_id", "video_id", "ocr_text"],
    "asr_text.parquet": ["scene_id", "video_id", "asr_text"],
    "scene_descriptions.parquet": ["scene_id", "video_id", "description"],
}


def parse_video_and_frame_id(file_path: Path | str) -> tuple[str | None, int | None]:
    """
    Trích xuất (video_id, frame_id) từ đường dẫn file ảnh.

    Hỗ trợ các định dạng cấu trúc thư mục:
      - .../L21/L21/V001/10141.webp -> ('L21_V001', 10141)
      - .../L21_V001/10141.jpg       -> ('L21_V001', 10141)
      - .../L21_V001-L21_V001_S000-10141.webp -> ('L21_V001', 10141)
    """
    p = Path(file_path)
    stem = p.stem

    # 1. Trích xuất frame_id
    frame_id: int | None = None
    try:
        frame_id = int(stem)
    except ValueError:
        numbers = re.findall(r"\d+", stem)
        if numbers:
            frame_id = int(numbers[-1])

    # 2. Trích xuất video_id
    video_id: str | None = None
    parts = p.parts

    for i in range(len(parts) - 1, 0, -1):
        part = parts[i]
        if re.match(r"^L\d+_V\d+$", part, re.IGNORECASE):
            video_id = part.upper()
            break
        elif re.match(r"^V\d+$", part, re.IGNORECASE) and i >= 1:
            parent_part = parts[i - 1]
            if re.match(r"^L\d+$", parent_part, re.IGNORECASE):
                video_id = f"{parent_part.upper()}_{part.upper()}"
                break

    if not video_id:
        v_match = re.search(r"(L\d+_V\d+|V\d+)", p.as_posix(), re.IGNORECASE)
        if v_match:
            matched = v_match.group(1).upper()
            if not matched.startswith("L") and len(parts) >= 2:
                for parent_dir in parts:
                    if re.match(r"^L\d+$", parent_dir, re.IGNORECASE):
                        matched = f"{parent_dir.upper()}_{matched}"
                        break
            video_id = matched

    return video_id, frame_id


def load_ocr_json_dict(ocr_dir: Path | str, video_id: str) -> dict[str, str]:
    """
    Đọc dữ liệu OCR JSON cho một video_id (tìm trong group folder hoặc root ocr_dir).
    Trả về dict rỗng nếu không tìm thấy file.
    """
    ocr_path = Path(ocr_dir)
    if not ocr_path.exists():
        return {}

    group = video_id.split("_")[0] if "_" in video_id else ""
    candidates = [
        ocr_path / group / f"{video_id}.json",
        ocr_path / f"{video_id}.json",
        ocr_path / f"{video_id.lower()}.json",
    ]

    for candidate in candidates:
        if candidate.exists():
            try:
                with open(candidate, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Lỗi khi đọc file OCR JSON {candidate}: {e}")
                return {}
    return {}


def slice_and_save_vector_file(
    matrix_path: Path,
    keep_indices: list[int] | np.ndarray,
    output_path: Path | None = None,
) -> bool:
    """
    Cắt lát (slice) file vector (.npy hoặc .pt) theo danh sách index cần giữ và lưu lại.
    Nếu output_path là None, ghi đè an toàn lên matrix_path bằng file tạm (tương thích Windows).
    """
    import gc
    import shutil

    if not matrix_path.exists():
        return False

    out_file = output_path or matrix_path
    indices = np.asarray(keep_indices, dtype=np.int64)

    try:
        if matrix_path.suffix == ".npy":
            mat = np.load(matrix_path)
            sliced = np.take(mat, indices, axis=0)
            del mat
            gc.collect()

            tmp_out = out_file.with_suffix(".tmp.npy")
            np.save(tmp_out, sliced)
            del sliced
            gc.collect()

            shutil.move(str(tmp_out), str(out_file))
            return True
        elif matrix_path.suffix == ".pt":
            mat = torch.load(matrix_path, map_location="cpu")
            if isinstance(mat, torch.Tensor):
                idx_tensor = torch.from_numpy(indices)
                sliced = mat[idx_tensor]
                del mat
                gc.collect()

                tmp_out = out_file.with_suffix(".tmp.pt")
                torch.save(sliced, tmp_out)
                del sliced
                gc.collect()

                shutil.move(str(tmp_out), str(out_file))
                return True
    except Exception as e:
        logger.error(f"Lỗi khi cắt vector matrix {matrix_path}: {e}")
        return False

    return False
