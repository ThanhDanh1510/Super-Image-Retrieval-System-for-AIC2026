"""
serving/api/service/retrieval/skeletons/text_query.py
======================================================
Service xử lý query dạng text cho KIS, VQA, và TRAKE (Skeleton).

LUỒNG XỬ LÝ:
  1. Dịch query sang tiếng Anh (nếu cần) + enhance
  2. Tách thành N sub-events (chỉ cho TRAKE)
  3. Encode text → ocr_vector + asr_vector + sparse vectors
  4. Gọi qdrant_repo.search_by_text_vectors() → danh sách CandidateResult thô

CONSTRAINTS:
  - KHÔNG khởi tạo model trong function body hay ở module level
  - PHẢI dùng dependency injection: Depends(get_text_encoder), Depends(get_image_encoder)
  - KHÔNG gọi Qdrant trực tiếp — chỉ gọi qua qdrant_repo
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from serving.api.schema.retrieval import CandidateResult, TaskType
from shared.models.text_embedding_wrapper import TextEmbeddingWrapper
from shared.models.image_embedding_wrapper import ImageEmbeddingWrapper

logger = logging.getLogger(__name__)

@dataclass
class TextQueryInput:
    raw_query: str
    task_type: TaskType
    top_k: int = 100

@dataclass
class TextQueryOutput:
    candidates: list[CandidateResult]
    enhanced_query: str
    sub_events: list[str] | None = None

async def process(
    query_input: TextQueryInput,
    qdrant_client,
    text_embedder: TextEmbeddingWrapper,
    image_embedder: ImageEmbeddingWrapper,
) -> TextQueryOutput:
    """
    Xử lý toàn bộ luồng text query: enhance → encode (text + image vectors) → retrieve.
    """
    raise NotImplementedError

async def _enhance_query(raw_query: str) -> str:
    """
    Dịch query sang tiếng Anh (nếu là tiếng Việt) và enhance chất lượng.
    """
    raise NotImplementedError

def _split_events(query: str) -> list[str]:
    """
    Tách TRAKE query thành N sub-event descriptions.
    """
    raise NotImplementedError

async def _encode_text(
    text: str,
    embedder: TextEmbeddingWrapper,
) -> tuple[list[float], list[float]]:
    """
    Encode text thành 2 dense vectors: ocr_vector và asr_vector (dim=1536).
    """
    raise NotImplementedError

def _encode_sparse(text: str) -> tuple[Any, Any]:
    """
    Encode text thành sparse vectors (BM42) cho keyword matching.
    """
    raise NotImplementedError

async def _encode_text_to_image_vector(
    text: str,
    embedder: ImageEmbeddingWrapper,
) -> list[float]:
    """
    Encode text query thành image dense vector (Text-to-Image) dùng CLIP text encoder (dim=1024).
    """
    raise NotImplementedError
