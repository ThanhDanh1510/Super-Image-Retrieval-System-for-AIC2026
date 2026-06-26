"""
serving/api/schema/retrieval.py
================================
Pydantic models cho tất cả request/response của Retrieval API.

IMPORT RULES:
  - Controller (routes_*.py) dùng để validate request và serialize response.
  - Service (retrieval/*.py) dùng CandidateResult làm kiểu trả về nội bộ.
  - Không import từ repo/ hay model/ trong file này.

KHÔNG ĐƯỢC:
  - Thay đổi field name của CandidateResult mà không cập nhật serving/repo/qdrant_repo.py.
  - Bỏ field score — controller dùng score để sort trước khi trả về.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated

from pydantic import BaseModel, Field


# =============================================================================
# Enums
# =============================================================================

class InputType(str, Enum):
    """Loại đầu vào của query."""
    TEXT = "text"
    VOICE = "voice"
    IMAGE = "image"


class TaskType(str, Enum):
    """Loại task theo luật thi của AIC."""
    KIS = "kis"       # Known-Item Search
    VQA = "vqa"       # Visual Question Answering
    TRAKE = "trake"   # Temporal Retrieval / multi-event


# =============================================================================
# Shared Internal Model
# =============================================================================

class CandidateResult(BaseModel):
    """
    Kết quả trả về từ Qdrant cho 1 keyframe ứng viên.

    Đây là kiểu nội bộ dùng trong tầng service và repo.
    Controller sẽ map sang KISResponse / VQAResponse / TRAKEResponse tương ứng.
    """

    video_id: str = Field(..., description="ID video gốc, ví dụ: 'L01_V001'")
    scene_id: str = Field(..., description="ID scene, ví dụ: 'L01_V001_S003'")
    frame_id: int = Field(..., description="Frame number của keyframe trong video gốc")

    timestamp_start: float = Field(..., description="Timestamp bắt đầu scene (giây)")
    timestamp_end: float = Field(..., description="Timestamp kết thúc scene (giây)")
    timestamp_mid: float = Field(..., description="Timestamp của keyframe (giây)")

    keyframe_path: str = Field(
        ...,
        description="Đường dẫn tương đối tới ảnh keyframe từ data/unified_keyframes/",
    )

    ocr_text: str = Field(default="", description="Text OCR nhận dạng được từ keyframe")
    asr_text: str = Field(default="", description="Transcript ASR của scene")

    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Điểm relevance tổng hợp sau RRF (hoặc sau reranker). Cao hơn = tốt hơn.",
    )


# =============================================================================
# Request Models
# =============================================================================

class KISRequest(BaseModel):
    """Request cho Known-Item Search (text hoặc voice query)."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Câu mô tả tự nhiên hoặc transcript giọng nói (sau khi voice_query xử lý)",
    )
    input_type: InputType = Field(
        default=InputType.TEXT,
        description="Loại đầu vào: 'text' (gõ tay) hoặc 'voice' (ghi âm, controller xử lý trước)",
    )
    top_k: Annotated[int, Field(ge=1, le=100)] = Field(
        default=100,
        description="Số kết quả trả về. Tối đa 100 theo luật thi.",
    )


class VQARequest(BaseModel):
    """Request cho Visual Question Answering."""

    query: str = Field(..., min_length=1, max_length=2000, description="Câu mô tả cảnh")
    question: str = Field(..., min_length=1, max_length=500, description="Câu hỏi về cảnh đó")
    input_type: InputType = Field(default=InputType.TEXT)
    top_k: Annotated[int, Field(ge=1, le=100)] = Field(default=100)


class TRAKERequest(BaseModel):
    """Request cho Temporal Retrieval / multi-event (TRAKE)."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Mô tả chuỗi sự kiện theo thứ tự thời gian",
    )
    num_events: Annotated[int, Field(ge=2, le=10)] = Field(
        ...,
        description="Số lượng khoảnh khắc/sự kiện trong query (N). Controller tự động tách từ query.",
    )
    input_type: InputType = Field(default=InputType.TEXT)
    top_k: Annotated[int, Field(ge=1, le=100)] = Field(default=100)


class ImageSearchRequest(BaseModel):
    """
    Request cho Image-to-Image Search (Similar Image Search).

    Lưu ý: Controller nhận file upload (multipart/form-data), extract base64,
    rồi tạo ImageSearchRequest này để truyền vào service.
    """

    image_base64: str = Field(..., description="Ảnh query dạng base64-encoded string")
    top_k: Annotated[int, Field(ge=1, le=100)] = Field(default=100)


# =============================================================================
# Response Models
# =============================================================================

class KISResponse(BaseModel):
    """Response cho KIS — danh sách (video_id, frame_id) xếp hạng."""

    results: list[CandidateResult] = Field(
        ...,
        description="Danh sách kết quả xếp hạng giảm dần theo score. Tối đa top_k items.",
    )
    total: int = Field(..., description="Tổng số kết quả trả về (≤ top_k)")
    query_text: str = Field(..., description="Query text sau khi đã enhance/dịch")


class VQAResponse(BaseModel):
    """Response cho VQA — thêm field answer so với KIS."""

    results: list[CandidateResult]
    answers: list[str] = Field(
        ...,
        description="Câu trả lời tương ứng với mỗi CandidateResult. Cùng index với results.",
    )
    total: int
    query_text: str


class TRAKEResponse(BaseModel):
    """Response cho TRAKE — mỗi result là 1 tuple (video_id, frame_id_1, ..., frame_id_N)."""

    results: list[list[CandidateResult]] = Field(
        ...,
        description=(
            "Danh sách các chuỗi sự kiện. Mỗi phần tử là list N CandidateResult "
            "tương ứng với N khoảnh khắc, theo đúng thứ tự thời gian."
        ),
    )
    total: int
    query_text: str
    num_events: int


class ImageSearchResponse(BaseModel):
    """Response cho Image-to-Image Search."""

    results: list[CandidateResult]
    total: int
