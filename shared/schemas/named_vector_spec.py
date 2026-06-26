"""
shared/schemas/named_vector_spec.py
====================================
CONTRACT TRUNG TÂM — Spec cho tất cả named vectors trong Qdrant collection.

File này là nguồn xác định duy nhất cho:
  - Tên của từng named vector (dùng trong indexing/ và serving/repo/)
  - Dimension của từng vector (phải khớp với model trong configs/model_versions.yaml)
  - Loại vector (dense hay sparse)
  - Metadata fields lưu trong payload của mỗi Qdrant point

IMPORT RULES:
  - indexing/schema/collection_schema.py  → import từ đây
  - serving/repo/qdrant_repo.py           → import từ đây
  - ingestion/ (notebooks)                → import từ đây (qua package shared)

KHÔNG ĐƯỢC:
  - Hardcode tên vector hay dimension ở bất kỳ nơi nào khác.
  - Thay đổi file này mà không thông báo team (breaking change cho cả indexing và serving).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


# =============================================================================
# Named Vector Definitions
# =============================================================================

@dataclass(frozen=True)
class VectorSpec:
    """Mô tả một named vector trong Qdrant collection."""
    name: str
    dim: int
    distance: Literal["Cosine", "Dot", "Euclid"] = "Cosine"
    vector_type: Literal["dense", "sparse"] = "dense"
    description: str = ""


# --- Danh sách tất cả named vectors ------------------------------------------
# Phải khớp với configs/model_versions.yaml :: named_vectors

IMAGE_DENSE = VectorSpec(
    name="image_dense",
    dim=1024,
    distance="Cosine",
    vector_type="dense",
    description="BEiT-large + CLIP ViT-L/14 image embedding, L2-normalized",
)

OCR_DENSE = VectorSpec(
    name="ocr_dense",
    dim=1536,
    distance="Cosine",
    vector_type="dense",
    description="gte-Qwen2-1.5B embedding của text OCR từ keyframe, L2-normalized",
)

ASR_DENSE = VectorSpec(
    name="asr_dense",
    dim=3584,
    distance="Cosine",
    vector_type="dense",
    description="gte-Qwen2-7B embedding của transcript ASR từ audio, L2-normalized",
)

OCR_SPARSE = VectorSpec(
    name="ocr_sparse",
    dim=0,          # sparse: dim không cố định, để 0 làm marker
    vector_type="sparse",
    description="BM42 sparse vector của text OCR, dùng cho keyword matching",
)

ASR_SPARSE = VectorSpec(
    name="asr_sparse",
    dim=0,
    vector_type="sparse",
    description="BM42 sparse vector của transcript ASR, dùng cho keyword matching",
)

# Tập hợp tất cả specs — dùng để validate và tạo collection schema
ALL_VECTOR_SPECS: list[VectorSpec] = [
    IMAGE_DENSE,
    OCR_DENSE,
    ASR_DENSE,
    OCR_SPARSE,
    ASR_SPARSE,
]

# Map từ tên vector → spec (tiện tra cứu nhanh)
VECTOR_SPEC_MAP: dict[str, VectorSpec] = {
    spec.name: spec for spec in ALL_VECTOR_SPECS
}


# =============================================================================
# Payload Schema — Metadata lưu kèm mỗi Qdrant point
# =============================================================================

@dataclass
class KeyframePayload:
    """
    Schema của payload cho mỗi Qdrant point (1 point = 1 keyframe).

    Fields này được lưu dạng JSON trong Qdrant và có thể filter/facet.
    Tất cả fields đều REQUIRED — ingestion phải điền đủ trước khi upsert.
    """

    # Định danh
    video_id: str
    """ID video gốc, ví dụ: 'L01_V001'. Dùng để group results theo video."""

    scene_id: str
    """ID scene trong video, ví dụ: 'L01_V001_S003'. Format: {video_id}_S{scene_idx:03d}."""

    frame_id: int
    """Frame number của keyframe này trong video gốc (0-indexed)."""

    # Thời gian
    timestamp_start: float
    """Timestamp bắt đầu của scene (giây)."""

    timestamp_end: float
    """Timestamp kết thúc của scene (giây)."""

    timestamp_mid: float
    """Timestamp của keyframe (giữa scene). Dùng cho TRAKE temporal alignment."""

    # Đường dẫn ảnh
    keyframe_path: str
    """
    Đường dẫn tương đối của file ảnh keyframe từ thư mục data/unified_keyframes/.
    Format: '{video_id}/{scene_id}.jpg'
    Ví dụ: 'L01_V001/L01_V001_S003.jpg'
    """

    # Text content
    ocr_text: str
    """Text nhận dạng được từ OCR. Rỗng string '' nếu không có text trong frame."""

    asr_text: str
    """Transcript ASR của đoạn audio tương ứng với scene. Rỗng string '' nếu không có audio."""

    # Metadata ingestion (để debug và reproductibility)
    ingestion_version: str
    """Version của configs/model_versions.yaml dùng khi tạo ra point này."""

    member: str
    """Tên member đã chạy ingestion cho video này. Dùng để trace lỗi."""


# =============================================================================
# Point ID Convention
# =============================================================================

def make_point_id(video_id: str, scene_id: str) -> str:
    """
    Tạo deterministic Point ID cho một keyframe.

    RÀNG BUỘC CỨNG:
      - Phải deterministic: cùng input → cùng output, mọi lúc, mọi nơi.
      - Không dùng random UUID — sẽ gây trùng lặp khi upsert lại.
      - Format phải là string UUID v5 (namespace-based) để Qdrant chấp nhận.

    Args:
        video_id: ID video, ví dụ 'L01_V001'
        scene_id: ID scene, ví dụ 'L01_V001_S003'

    Returns:
        UUID v5 string duy nhất cho cặp (video_id, scene_id).
    """
    import uuid
    # Namespace cố định cho toàn project — KHÔNG THAY ĐỔI
    _NAMESPACE = uuid.UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890")
    key = f"{video_id}::{scene_id}"
    return str(uuid.uuid5(_NAMESPACE, key))
