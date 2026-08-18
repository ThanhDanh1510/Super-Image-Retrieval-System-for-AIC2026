"""
AIC2026 Aggregation Package.
============================
Module tổng hợp, chuẩn hóa và đóng gói dữ liệu Unified Dataset.
"""

from aggregation.common import (
    DEFAULT_FPS,
    IMAGE_EXTENSIONS,
    METADATA_FILES,
    REQUIRED_COLUMNS,
    VECTOR_FILES,
    load_ocr_json_dict,
    parse_video_and_frame_id,
    slice_and_save_vector_file,
)
from aggregation.pipeline import (
    build_from_raw_pipeline,
    merge_shards_pipeline,
)

__all__ = [
    "DEFAULT_FPS",
    "IMAGE_EXTENSIONS",
    "METADATA_FILES",
    "REQUIRED_COLUMNS",
    "VECTOR_FILES",
    "parse_video_and_frame_id",
    "load_ocr_json_dict",
    "slice_and_save_vector_file",
    "build_from_raw_pipeline",
    "merge_shards_pipeline",
]
