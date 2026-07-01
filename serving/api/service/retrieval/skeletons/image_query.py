"""
serving/api/service/retrieval/skeletons/image_query.py
=======================================================
Service xử lý query dạng ảnh (Image-to-Image / Instance Search) (Skeleton).

LUỒNG XỬ LÝ:
  1. Decode ảnh từ base64 hoặc bytes
  2. Preprocess ảnh (resize, normalize)
  3. Encode ảnh → image_vector (BEiT-large + CLIP)
  4. Gọi qdrant_repo.search_by_image_vector() → danh sách CandidateResult thô
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from serving.api.schema.retrieval import CandidateResult
from shared.models.image_embedding_wrapper import ImageEmbeddingWrapper

logger = logging.getLogger(__name__)

@dataclass
class ImageQueryInput:
    image_bytes: bytes
    top_k: int = 100

@dataclass
class ImageQueryOutput:
    candidates: list[CandidateResult]

async def process(
    query_input: ImageQueryInput,
    qdrant_client,
    image_embedder: ImageEmbeddingWrapper,
) -> ImageQueryOutput:
    """
    Xử lý toàn bộ luồng image query: decode → preprocess → encode → retrieve.
    """
    raise NotImplementedError

async def _encode_image(
    image_bytes: bytes,
    embedder: ImageEmbeddingWrapper,
) -> list[float]:
    """
    Decode và encode ảnh thành image_vector (dim=1024, L2-normalized).
    """
    raise NotImplementedError
