"""
serving/api/service/retrieval/retrieval_pipeline.py
====================================================
Orchestrator trung tâm — nhận query từ controller, điều hướng sang đúng module,
và trả về danh sách candidates thô (chưa qua reranker hay temporal_aligner).

TRÁCH NHIỆM CỦA FILE NÀY:
  - Nhận InputType (text/voice/image) và TaskType (KIS/VQA/TRAKE).
  - Gọi đúng module: text_query / voice_query / image_query.
  - Trả về candidates thô cho controller.
  - KHÔNG chạy reranker hay temporal_aligner ở đây.

TRÁCH NHIỆM KHÔNG THUỘC FILE NÀY:
  - Reranking (ViRanker, Jina) — controller gọi reranker/ riêng.
  - Temporal alignment (TRAKE) — controller gọi temporal_aligner.py riêng.
  - Encode model logic — thuộc text_query/image_query/voice_query.

IMPORT RULES:
  - Được gọi bởi: serving/api/controller/routes_*.py
  - Gọi: text_query, image_query, voice_query (trong cùng package retrieval/)
  - Không gọi qdrant_repo trực tiếp — đó là việc của các sub-modules.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from serving.api.schema.retrieval import (
    CandidateResult,
    InputType,
    TaskType,
    KISRequest,
    VQARequest,
    TRAKERequest,
    ImageSearchRequest,
)
from serving.api.service.retrieval import text_query, image_query, voice_query
from serving.api.service.retrieval.text_query import TextQueryInput, TextQueryOutput
from serving.api.service.retrieval.image_query import ImageQueryInput, ImageQueryOutput
from serving.api.service.retrieval.voice_query import VoiceQueryInput, VoiceQueryOutput

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class PipelineResult:
    """
    Kết quả trả về từ pipeline orchestrator.

    Controller nhận PipelineResult, sau đó quyết định có chạy reranker hay không
    trước khi build response cuối.
    """
    candidates: list[CandidateResult]
    """Danh sách candidates thô từ Qdrant, chưa rerank."""

    task_type: TaskType
    """TaskType gốc — controller dùng để chọn response schema."""

    enhanced_query: str | None = None
    """Query sau khi enhance (text/voice). None với image query."""

    transcript: str | None = None
    """Transcript ASR (chỉ với voice query). None với text/image."""

    sub_events: list[str] | None = None
    """Sub-events đã tách (chỉ với TRAKE). None với KIS/VQA/image."""


# =============================================================================
# Core Orchestrator Function
# =============================================================================

async def run(
    input_type: InputType,
    task_type: TaskType,
    request,           # KISRequest | VQARequest | TRAKERequest | ImageSearchRequest
    qdrant_client,
    text_embedder,
    image_embedder,
    asr_model,
) -> PipelineResult:
    """
    Entry point duy nhất cho toàn bộ retrieval pipeline.

    Controller gọi hàm này và nhận về PipelineResult. Controller sau đó
    quyết định có chạy reranker (và temporal_aligner với TRAKE) hay không.

    Routing logic:
      - InputType.TEXT  → text_query.process()
      - InputType.VOICE → voice_query.process()   (sau đó delegate sang text_query)
      - InputType.IMAGE → image_query.process()

    Args:
        input_type: Loại đầu vào của query.
        task_type: KIS / VQA / TRAKE.
        request: Pydantic request model từ controller.
        qdrant_client: AsyncQdrantClient singleton.
        text_embedder: TextEmbeddingWrapper singleton.
        image_embedder: ImageEmbeddingWrapper singleton.
        asr_model: ASR model singleton (PhoWhisper). None nếu không load.

    Returns:
        PipelineResult với candidates thô và metadata.

    Raises:
        ValueError: Nếu input_type không hỗ trợ.
        NotImplementedError: Nếu combination (input_type, task_type) chưa implement.
    """
    logger.info(f"Pipeline run: input_type={input_type}, task_type={task_type}")

    if input_type == InputType.TEXT:
        return await _run_text(request, task_type, qdrant_client, text_embedder)

    elif input_type == InputType.VOICE:
        return await _run_voice(request, task_type, qdrant_client, text_embedder, asr_model)

    elif input_type == InputType.IMAGE:
        return await _run_image(request, qdrant_client, image_embedder)

    else:
        raise ValueError(f"Unsupported input_type: {input_type}")


# =============================================================================
# Private Routing Functions (implement these)
# =============================================================================

async def _run_text(
    request,
    task_type: TaskType,
    qdrant_client,
    text_embedder,
) -> PipelineResult:
    """
    Route text query sang text_query.process() và wrap kết quả thành PipelineResult.

    Args:
        request: KISRequest, VQARequest, hoặc TRAKERequest.
        task_type: TaskType để text_query biết có cần tách sub-events không.
        qdrant_client: AsyncQdrantClient singleton.
        text_embedder: TextEmbeddingWrapper singleton.

    Returns:
        PipelineResult với candidates, enhanced_query, và sub_events (nếu TRAKE).
    """
    raise NotImplementedError


async def _run_voice(
    request,
    task_type: TaskType,
    qdrant_client,
    text_embedder,
    asr_model,
) -> PipelineResult:
    """
    Route voice query sang voice_query.process() và wrap kết quả.

    Lưu ý: request.query đối với voice là audio base64 hoặc controller
    đã decode thành bytes và truyền riêng — cần thống nhất với controller.

    Args:
        request: Request với audio data (đã decode thành bytes bởi controller).
        task_type: KIS / VQA / TRAKE.
        qdrant_client: AsyncQdrantClient singleton.
        text_embedder: TextEmbeddingWrapper singleton.
        asr_model: PhoWhisper model singleton.

    Returns:
        PipelineResult với candidates, transcript, enhanced_query, sub_events.
    """
    raise NotImplementedError


async def _run_image(
    request: ImageSearchRequest,
    qdrant_client,
    image_embedder,
) -> PipelineResult:
    """
    Route image query sang image_query.process() và wrap kết quả.

    Lưu ý: ImageSearch chỉ hỗ trợ task_type=KIS (không có VQA/TRAKE với image input).

    Args:
        request: ImageSearchRequest với image_base64.
        qdrant_client: AsyncQdrantClient singleton.
        image_embedder: ImageEmbeddingWrapper singleton.

    Returns:
        PipelineResult với candidates. enhanced_query và sub_events là None.
    """
    raise NotImplementedError
