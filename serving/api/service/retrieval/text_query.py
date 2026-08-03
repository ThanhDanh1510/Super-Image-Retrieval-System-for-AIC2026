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
from shared.models.image_embedding_wrapper import ImageEmbeddingWrapper

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
    image_embedder: ImageEmbeddingWrapper,
) -> TextQueryOutput:
    """
    Xử lý toàn bộ luồng text query: enhance → encode (text + image vectors) → retrieve.

    Đây là hàm duy nhất mà retrieval_pipeline.py gọi vào module này.

    Args:
        query_input: TextQueryInput chứa raw query và metadata.
        qdrant_client: AsyncQdrantClient đã inject từ dependencies.py.
        text_embedder: TextEmbeddingWrapper đã load model (singleton từ dependencies.py).
        image_embedder: ImageEmbeddingWrapper đã load model (singleton từ dependencies.py).

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
    image_vector = await _encode_text_to_image_vector(enhanced, image_embedder)

    candidates = await qdrant_repo.search_by_text_vectors(
        client=qdrant_client,
        ocr_vector=ocr_vector,
        asr_vector=asr_vector,
        ocr_sparse_vector=ocr_sparse,
        asr_sparse_vector=asr_sparse,
        image_vector=image_vector,
        top_k=query_input.top_k,
    )


    return TextQueryOutput(
        candidates=candidates,
        enhanced_query=enhanced,
        sub_events=sub_events,
    )


# =============================================================================
# Internal Helper Functions (implementations)
# =============================================================================

async def _enhance_query(raw_query: str) -> str:
    """
    Dịch query sang tiếng Anh (nếu là tiếng Việt) và enhance chất lượng.

    Args:
        raw_query: Query gốc từ người dùng.

    Returns:
        Query đã được dịch và enhanced (luôn là tiếng Anh).
    """
    from shared.utils.text_normalizer import sanitize_search_query
    cleaned_query = sanitize_search_query(raw_query)
    if not cleaned_query:
        return ""

    # Kiểm tra xem có chứa tiếng Việt (ký tự có dấu) hay không
    import re
    has_vietnamese_accent = bool(re.search(r'[àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]', cleaned_query, re.IGNORECASE))
    
    if has_vietnamese_accent:
        try:
            from deep_translator import GoogleTranslator
            translated = GoogleTranslator(source="auto", target="en").translate(cleaned_query)
            if translated and translated.strip():
                return sanitize_search_query(translated)
        except Exception as e:
            logger.warning(f"Không thể dịch query '{cleaned_query}' qua deep_translator ({e}), dùng query gốc.")

    return cleaned_query


def _split_events(query: str) -> list[str]:
    """
    Tách TRAKE query thành N sub-event descriptions.

    Priority 1: Nếu query chứa xuống dòng '\\n', tách theo từng dòng.
    Priority 2: Rule-based connectors ('then', 'after that', 'sau đó', ...).

    Args:
        query: Query tiếng Anh đã enhanced (hoặc tiếng Việt).

    Returns:
        List các sub-event strings. Mỗi string mô tả 1 khoảnh khắc.

    Raises:
        ValueError: Nếu không thể tách được ít nhất 2 sub-events từ query.
    """
    import re

    # 1. Ưu tiên 1 — Newline split
    if "\n" in query:
        lines = [line.strip() for line in query.split("\n") if line.strip()]
        if len(lines) >= 2:
            return lines

    # 2. Fallback — Rule-based connectors
    connectors_pattern = (
        r"\b(?:then|after that|followed by|next|afterwards|subsequently|later|finally|sau đó|tiếp theo|rồi)\b"
    )
    segments = re.split(connectors_pattern, query, flags=re.IGNORECASE)
    sub_events = [s.strip(" .,;-") for s in segments if s and s.strip(" .,;-")]

    # 3. Nếu connectors không tách được, thử tách theo dấu chấm hoặc chấm phẩy
    if len(sub_events) < 2:
        punct_segments = re.split(r"[.;]", query)
        sub_events = [s.strip(" .,;-") for s in punct_segments if s and s.strip(" .,;-")]

    if len(sub_events) < 2:
        raise ValueError(f"Không thể tách query thành ít nhất 2 sub-events: '{query}'")

    return sub_events


async def _encode_text(
    text: str,
    embedder: TextEmbeddingWrapper,
) -> tuple[list[float], list[float]]:
    """
    Encode text thành 2 dense vectors: ocr_vector và asr_vector (dim=1536).

    Args:
        text: Text đã enhanced.
        embedder: TextEmbeddingWrapper singleton (gte-Qwen2-1.5B).

    Returns:
        Tuple (ocr_vector, asr_vector) list[float], len=1536.
    """
    vec = embedder.encode(text)
    if hasattr(vec, "tolist"):
        vec_list = vec.tolist()
    else:
        vec_list = list(vec)

    return vec_list, vec_list


def _encode_sparse(text: str) -> tuple[object | None, object | None]:
    """
    Encode text thành sparse vectors (BM42) cho keyword matching.
    Phase 1: Trả về (None, None).
    """
    return None, None


async def _encode_text_to_image_vector(
    text: str,
    embedder: ImageEmbeddingWrapper,
) -> list[float]:
    """
    Encode text query thành image dense vector (Text-to-Image) dùng CLIP text encoder (dim=1024).

    Args:
        text: Text đã enhanced.
        embedder: ImageEmbeddingWrapper singleton.

    Returns:
        list[float], len=1024, L2-normalized.
    """
    vec = embedder.encode_text(text)
    if hasattr(vec, "tolist"):
        return vec.tolist()
    return list(vec)

