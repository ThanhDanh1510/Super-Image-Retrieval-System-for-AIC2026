"""
serving/api/service/retrieval/image_query.py
=============================================
Service xử lý query dạng ảnh (Image-to-Image / Instance Search).

LUỒNG XỬ LÝ:
  1. Decode ảnh từ base64 hoặc bytes
  2. Preprocess ảnh (resize, normalize)
  3. Encode ảnh → image_vector (CLIP ViT-L/14)
  4. Gọi qdrant_repo.search_by_image_vector() → danh sách CandidateResult thô

IMPORT RULES:
  - Được gọi bởi: retrieval_pipeline.py
  - Gọi: serving/api/repo/qdrant_repo.py
  - Không gọi reranker — đó là việc của pipeline orchestrator.

SKELETON: Implement các hàm có `raise NotImplementedError`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from serving.api.schema.retrieval import CandidateResult
from serving.api.repo import qdrant_repo
from shared.models.image_embedding_wrapper import ImageEmbeddingWrapper

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ImageQueryInput:
    """Input chuẩn hóa cho image_query.process()."""
    image_bytes: bytes
    """Raw bytes của ảnh query (JPEG/PNG). Controller đã decode từ base64."""

    top_k: int = 100
    """Số candidates trả về. Tối đa 100."""


@dataclass
class ImageQueryOutput:
    """Output trả về từ image_query.process()."""
    candidates: list[CandidateResult]
    """Danh sách candidates thô từ Qdrant, chưa qua reranker."""


# =============================================================================
# Core Functions
# =============================================================================

async def process(
    query_input: ImageQueryInput,
    qdrant_client,
    image_embedder: ImageEmbeddingWrapper,
) -> ImageQueryOutput:
    """
    Xử lý toàn bộ luồng image query: decode → preprocess → encode → retrieve.

    Đây là hàm duy nhất mà retrieval_pipeline.py gọi vào module này.

    Args:
        query_input: ImageQueryInput chứa raw image bytes.
        qdrant_client: AsyncQdrantClient đã inject từ dependencies.py.
        image_embedder: ImageEmbeddingWrapper đã load model (singleton).

    Returns:
        ImageQueryOutput với danh sách candidates thô.
    """
    image_vector = await _encode_image(query_input.image_bytes, image_embedder)

    candidates = await qdrant_repo.search_by_image_vector(
        client=qdrant_client,
        image_vector=image_vector,
        top_k=query_input.top_k,
    )

    return ImageQueryOutput(candidates=candidates)


# =============================================================================
# Internal Helper Functions (implement these)
# =============================================================================

async def _encode_image(
    image_bytes: bytes,
    embedder: ImageEmbeddingWrapper,
) -> list[float]:
    """
    Decode và encode ảnh thành image_vector (dim=1024, L2-normalized).

    Args:
        image_bytes: Raw bytes của ảnh (JPEG/PNG/WebP).
        embedder: ImageEmbeddingWrapper đã load model.

    Returns:
        list[float], len=1024, L2-normalized.

    Raises:
        ValueError: Nếu image_bytes không phải ảnh hợp lệ.
    """
    if not image_bytes:
        raise ValueError("image_bytes không được rỗng")

    vec = embedder.encode_from_bytes(image_bytes)
    if hasattr(vec, "tolist"):
        return vec.tolist()
    return list(vec)
