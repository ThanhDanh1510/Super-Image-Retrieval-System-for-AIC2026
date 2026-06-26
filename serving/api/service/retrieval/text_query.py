"""
serving/api/service/retrieval/text_query.py
============================================
Service xử lý query dạng text cho KIS, VQA, và TRAKE.

LUỒNG XỬ LÝ:
  1. Dịch query sang tiếng Anh (nếu cần) + enhance
  2. Tách thành N sub-events (chỉ cho TRAKE)
  3. Encode text → ocr_vector + asr_vector + sparse vectors
  4. Gọi qdrant_repo.search_by_text_vectors() → danh sách CandidateResult thô

IMPORT RULES:
  - Được gọi bởi: retrieval_pipeline.py
  - Gọi: serving/api/repo/qdrant_repo.py
  - Không gọi reranker hay temporal_aligner — đó là việc của pipeline orchestrator.

SKELETON: Implement các hàm có `raise NotImplementedError`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from serving.api.schema.retrieval import CandidateResult, TaskType
from serving.api.repo import qdrant_repo
from shared.models.text_embedding_wrapper import TextEmbeddingWrapper

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class TextQueryInput:
    """Input chuẩn hóa cho text_query.process()."""
    raw_query: str
    """Query gốc từ người dùng (tiếng Việt hoặc tiếng Anh)."""

    task_type: TaskType
    """KIS / VQA / TRAKE — ảnh hưởng đến cách xử lý (TRAKE cần tách sub-events)."""

    top_k: int = 100
    """Số candidates trả về. Tối đa 100."""


@dataclass
class TextQueryOutput:
    """Output trả về từ text_query.process()."""
    candidates: list[CandidateResult]
    """Danh sách candidates thô từ Qdrant, chưa qua reranker."""

    enhanced_query: str
    """Query sau khi đã dịch và enhance (dùng để log và hiển thị trên UI)."""

    sub_events: list[str] | None = None
    """
    Danh sách sub-events đã tách (chỉ có giá trị nếu task_type == TRAKE).
    None với KIS và VQA.
    """


# =============================================================================
# Core Functions
# =============================================================================

async def process(
    query_input: TextQueryInput,
    qdrant_client,
    text_embedder: TextEmbeddingWrapper,
) -> TextQueryOutput:
    """
    Xử lý toàn bộ luồng text query: enhance → encode → retrieve.

    Đây là hàm duy nhất mà retrieval_pipeline.py gọi vào module này.

    Args:
        query_input: TextQueryInput chứa raw query và metadata.
        qdrant_client: AsyncQdrantClient đã inject từ dependencies.py.
        text_embedder: TextEmbeddingWrapper đã load model (singleton từ dependencies.py).

    Returns:
        TextQueryOutput với danh sách candidates thô và enhanced query.
    """
    enhanced = await _enhance_query(query_input.raw_query)

    sub_events: list[str] | None = None
    if query_input.task_type == TaskType.TRAKE:
        sub_events = _split_events(enhanced)
        logger.info(f"TRAKE: tách được {len(sub_events)} sub-events từ query.")

    ocr_vector, asr_vector = await _encode_text(enhanced, text_embedder)
    ocr_sparse, asr_sparse = _encode_sparse(enhanced)

    candidates = await qdrant_repo.search_by_text_vectors(
        client=qdrant_client,
        ocr_vector=ocr_vector,
        asr_vector=asr_vector,
        ocr_sparse_vector=ocr_sparse,
        asr_sparse_vector=asr_sparse,
        top_k=query_input.top_k,
    )

    return TextQueryOutput(
        candidates=candidates,
        enhanced_query=enhanced,
        sub_events=sub_events,
    )


# =============================================================================
# Internal Helper Functions (implement these)
# =============================================================================

async def _enhance_query(raw_query: str) -> str:
    """
    Dịch query sang tiếng Anh (nếu là tiếng Việt) và enhance chất lượng.

    Gợi ý implementation:
      - Dùng model dịch nhẹ (MarianMT vi→en) hoặc API (DeepL/Google Translate).
      - Sau khi dịch, có thể thêm bước paraphrase/expand để tăng recall.
      - Nếu query đã là tiếng Anh, bỏ qua bước dịch.

    Args:
        raw_query: Query gốc từ người dùng.

    Returns:
        Query đã được dịch và enhanced (luôn là tiếng Anh).
    """
    raise NotImplementedError


def _split_events(query: str) -> list[str]:
    """
    Tách TRAKE query thành N sub-event descriptions.

    Ví dụ input:
        "A person jumps over a fence, then lands on the ground, then runs away"
    Ví dụ output:
        ["A person jumps over a fence",
         "lands on the ground",
         "runs away"]

    Gợi ý implementation:
      - Rule-based: tách theo connector words ("then", "after that", "followed by", ...)
      - Hoặc dùng LLM để tách có context-aware hơn.

    Args:
        query: Query tiếng Anh đã enhanced.

    Returns:
        List các sub-event strings. Mỗi string mô tả 1 khoảnh khắc.

    Raises:
        ValueError: Nếu không thể tách được ít nhất 2 sub-events từ query.
    """
    raise NotImplementedError


async def _encode_text(
    text: str,
    embedder: TextEmbeddingWrapper,
) -> tuple[list[float], list[float]]:
    """
    Encode text thành 2 dense vectors: ocr_vector và asr_vector.

    Lý do dùng 2 vector khác dim:
      - ocr_vector (dim=1536, gte-Qwen2-1.5B): match với OCR text ngắn trong keyframe.
      - asr_vector (dim=3584, gte-Qwen2-7B): match với ASR transcript dài hơn.

    Args:
        text: Text đã enhanced (tiếng Anh).
        embedder: TextEmbeddingWrapper singleton (đã load cả 2 model).

    Returns:
        Tuple (ocr_vector, asr_vector).
        - ocr_vector: list[float], len=1536, L2-normalized.
        - asr_vector: list[float], len=3584, L2-normalized.
    """
    raise NotImplementedError


def _encode_sparse(text: str) -> tuple[object | None, object | None]:
    """
    Encode text thành sparse vectors (BM42) cho keyword matching.

    Args:
        text: Text đã enhanced (tiếng Anh).

    Returns:
        Tuple (ocr_sparse, asr_sparse).
        Mỗi phần tử là qdrant_client.models.SparseVector hoặc None nếu không dùng sparse.
    """
    raise NotImplementedError
