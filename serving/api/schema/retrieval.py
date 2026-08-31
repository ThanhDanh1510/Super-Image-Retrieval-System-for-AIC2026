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
from typing import Annotated, Any

from pydantic import BaseModel, Field, field_validator


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
        description="Đường dẫn tương đối tới ảnh keyframe từ data/ (tên mới unified/{video_id}-{scene_id}-{frame_id}.webp)",
    )

    ocr_text: str = Field(default="", description="Text OCR nhận dạng được từ keyframe")
    asr_text: str = Field(default="", description="Transcript ASR của scene")
    description: str = Field(default="", description="Mô tả cảnh AI VLM từ scene_descriptions.parquet")
    answer: str = Field(default="", description="Câu trả lời VQA của user choice nếu có")

    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Điểm relevance tổng hợp sau RRF (hoặc sau reranker). Cao hơn = tốt hơn.",
    )

    @field_validator("score", mode="before")
    @classmethod
    def clamp_score(cls, v: Any) -> float:
        """Tự động kẹp giá trị điểm số về khoảng an toàn [0.0, 1.0] tránh lỗi float precision."""
        if v is None:
            return 0.0
        try:
            val = float(v)
            return max(0.0, min(1.0, val))
        except (ValueError, TypeError):
            return 0.0


# =============================================================================
# Request Models
# =============================================================================

class UserChoice(BaseModel):
    """
    Lựa chọn frame thủ công của người dùng.
    
    Chứa thông tin frame người dùng tự chọn để ưu tiên xếp lên đầu.
    """
    video_id: str = Field(..., description="ID video gốc, ví dụ: 'L01_V001'")
    frame_id: int = Field(..., description="Frame number của keyframe trong video gốc")
    scene_id: str = Field(default="", description="ID scene, ví dụ: 'L01_V001_S003'")
    timestamp_start: float = Field(default=0.0)
    timestamp_end: float = Field(default=0.0)
    timestamp_mid: float = Field(default=0.0)
    keyframe_path: str = Field(default="", description="Đường dẫn tương đối tới ảnh keyframe")
    ocr_text: str = Field(default="")
    asr_text: str = Field(default="")
    answer: str = Field(default="", description="Câu trả lời do người dùng gõ (chỉ dùng cho VQA)")


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
    search_mode: str = Field(
        default="all",
        description="Chế độ tìm kiếm: 'all' (Hybrid RRF), 'embedding' (Visual Clip), 'ocr' (Văn bản trên ảnh), 'asr' (Lời thoại)",
    )
    top_k: Annotated[int, Field(ge=1, le=500)] = Field(
        default=100,
        description="Số kết quả trả về. Tối đa 500.",
    )
    user_choices: list[UserChoice] = Field(
        default_factory=list,
        description="Danh sách các frame do người dùng tự chọn bằng tay để xếp lên đầu."
    )
    use_viranker: bool = Field(
        default=True,
        description="Bật/Tắt ViRanker (bge-reranker-v2-m3) chấm điểm ngữ nghĩa văn bản (OCR/ASR/Caption)."
    )
    use_jina_m0: bool = Field(
        default=False,
        description="Bật/Tắt Jina Reranker M0 chấm điểm trực quan trên pixel ảnh keyframe."
    )
    use_reranker: bool | None = Field(
        default=None,
        description="Tương thích ngược: Nếu truyền True -> bật Jina M0; nếu False -> giữ theo use_viranker."
    )


class VQARequest(BaseModel):
    """Request cho Visual Question Answering."""

    query: str = Field(..., min_length=1, max_length=2000, description="Câu mô tả cảnh")
    question: str = Field(..., min_length=1, max_length=500, description="Câu hỏi về cảnh đó")
    input_type: InputType = Field(default=InputType.TEXT)
    search_mode: str = Field(default="all")
    top_k: Annotated[int, Field(ge=1, le=500)] = Field(default=100)
    user_choices: list[UserChoice] = Field(
        default_factory=list,
        description="Danh sách các frame và câu trả lời do người dùng tự chọn bằng tay."
    )
    use_viranker: bool = Field(
        default=True,
        description="Bật/Tắt ViRanker (bge-reranker-v2-m3) chấm điểm ngữ nghĩa văn bản (OCR/ASR/Caption)."
    )
    use_jina_m0: bool = Field(
        default=False,
        description="Bật/Tắt Jina Reranker M0 chấm điểm trực quan trên pixel ảnh keyframe."
    )
    use_reranker: bool | None = Field(
        default=None,
        description="Tương thích ngược"
    )


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
    search_mode: str = Field(default="all")
    top_k: Annotated[int, Field(ge=1, le=500)] = Field(default=100)
    use_viranker: bool = Field(default=True)
    use_jina_m0: bool = Field(default=False)
    use_reranker: bool | None = Field(default=None)


class ImageSearchRequest(BaseModel):
    """
    Request cho Image-to-Image Search (Similar Image Search).
    """

    image_base64: str = Field(..., description="Ảnh query dạng base64-encoded string")
    top_k: Annotated[int, Field(ge=1, le=200)] = Field(default=100)
    use_viranker: bool = Field(default=False)
    use_jina_m0: bool = Field(default=False)
    use_reranker: bool | None = Field(default=None)


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
    metrics: dict[str, Any] | None = Field(default=None, description="Chi tiết thời gian thực thi (ms) của pipeline")


class VQAResponse(BaseModel):
    """Response cho VQA — thêm field answer so với KIS."""

    results: list[CandidateResult]
    answers: list[str] = Field(
        ...,
        description="Câu trả lời tương ứng với mỗi CandidateResult. Cùng index với results.",
    )
    total: int
    query_text: str
    metrics: dict[str, Any] | None = Field(default=None, description="Chi tiết thời gian thực thi (ms) của pipeline")


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
    metrics: dict[str, Any] | None = Field(default=None, description="Chi tiết thời gian thực thi (ms) của pipeline")


class ImageSearchResponse(BaseModel):
    """Response cho Image-to-Image Search."""

    results: list[CandidateResult]
    total: int
    metrics: dict[str, Any] | None = Field(default=None, description="Chi tiết thời gian thực thi (ms) của pipeline")
