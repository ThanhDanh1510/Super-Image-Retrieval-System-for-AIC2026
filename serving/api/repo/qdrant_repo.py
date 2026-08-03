"""
serving/api/repo/qdrant_repo.py
=================================
Repository layer — giao tiếp duy nhất với Qdrant Vector Database.

INTERFACE CONTRACT:
  - Service layer gọi các function trong module này.
  - Không import gì từ serving/api/service/ hay serving/api/controller/.
  - Tất cả function nhận vector (list[float]) và trả về list[CandidateResult].

ĐÃ IMPLEMENT ĐẦY ĐỦ bởi Leader.
Members KHÔNG cần sửa file này trừ khi có yêu cầu từ Leader.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client.models import (
    NamedVector,
    NamedSparseVector,
    SparseVector,
    Prefetch,
    FusionQuery,
    Fusion,
    Filter,
    QueryResponse,
)

from serving.api.schema.retrieval import CandidateResult
from shared.schemas.named_vector_spec import (
    IMAGE_DENSE,
    OCR_DENSE,
    ASR_DENSE,
    OCR_SPARSE,
    ASR_SPARSE,
    VECTOR_SPEC_MAP,
    KeyframePayload,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Tên Qdrant collection alias "production" → collection_vN (xem indexing/collection_alias.py)
PRODUCTION_ALIAS = "production"

# Prefetch limit mặc định — lấy đủ candidates cho RRF fusion
_DEFAULT_PREFETCH_LIMIT = 200


# =============================================================================
# Client Initialization
# =============================================================================

def get_qdrant_client(url: str, api_key: str | None = None) -> QdrantClient:
    """
    Tạo synchronous Qdrant client.

    Args:
        url: URL của Qdrant server, ví dụ 'http://localhost:6333'
        api_key: API key nếu Qdrant được bảo vệ bằng authentication

    Returns:
        QdrantClient instance đã kết nối.
    """
    logger.info(f"Connecting to Qdrant (sync): {url}")
    return QdrantClient(url=url, api_key=api_key, timeout=30)


def get_async_qdrant_client(url: str, api_key: str | None = None) -> AsyncQdrantClient:
    """
    Tạo async Qdrant client cho FastAPI async endpoints.

    Args:
        url: URL của Qdrant server
        api_key: API key nếu cần

    Returns:
        AsyncQdrantClient instance.
    """
    logger.info(f"Connecting to Qdrant (async): {url}")
    return AsyncQdrantClient(url=url, api_key=api_key, timeout=30)


# =============================================================================
# Core Retrieval Functions
# =============================================================================

async def search_by_text_vectors(
    client: AsyncQdrantClient,
    ocr_vector: list[float],
    asr_vector: list[float],
    ocr_sparse_vector: SparseVector | None,
    asr_sparse_vector: SparseVector | None,
    image_vector: list[float] | None = None,
    top_k: int = 100,
    collection_name: str = PRODUCTION_ALIAS,
    score_threshold: float | None = None,
) -> list[CandidateResult]:
    """
    Tìm kiếm keyframe bằng text vectors (dense + sparse) và image vector qua Weighted RRF fusion.

    Thực hiện multi-vector query từ các nhánh:
      - image_dense (CLIP text vector mô phỏng ảnh, dim=1024)
      - ocr_dense  (gte-Qwen2-1.5B embedding của text OCR, dim=1536)
      - asr_dense  (gte-Qwen2-1.5B embedding của transcript ASR, dim=1536)
      - ocr_sparse (BM42 keyword matching cho OCR)
      - asr_sparse (BM42 keyword matching cho ASR)

    Áp dụng trọng số nhánh (Weighted RRF) đọc từ configs/fusion_config.yaml:
      RRF_score(d) = Σ w_branch * (1 / (k + rank_branch(d)))
    """
    import asyncio
    from shared.utils.config_loader import load_fusion_config

    try:
        cfg = load_fusion_config()
        weights = cfg.get("weights", {})
        rrf_k = float(cfg.get("rrf", {}).get("k", 60))
    except Exception:
        weights = {"image_dense": 2.5, "ocr_dense": 0.5, "asr_dense": 0.3, "ocr_sparse": 0.2, "asr_sparse": 0.2}
        rrf_k = 60.0

    w_image = float(weights.get("image_dense", 2.5))
    w_ocr_dense = float(weights.get("ocr_dense", 0.5))
    w_asr_dense = float(weights.get("asr_dense", 0.3))
    w_ocr_sparse = float(weights.get("ocr_sparse", 0.2))
    w_asr_sparse = float(weights.get("asr_sparse", 0.2))

    tasks = []
    task_weights = []

    # Nhánh image_dense (Text-to-Image via CLIP text encoder)
    if image_vector is not None:
        tasks.append(
            client.query_points(
                collection_name=collection_name,
                query=image_vector,
                using=IMAGE_DENSE.name,
                limit=_DEFAULT_PREFETCH_LIMIT,
                score_threshold=score_threshold,
                with_payload=True,
            )
        )
        task_weights.append(w_image)

    # Nhánh ocr_dense
    tasks.append(
        client.query_points(
            collection_name=collection_name,
            query=ocr_vector,
            using=OCR_DENSE.name,
            limit=_DEFAULT_PREFETCH_LIMIT,
            score_threshold=score_threshold,
            with_payload=True,
        )
    )
    task_weights.append(w_ocr_dense)

    # Nhánh asr_dense
    tasks.append(
        client.query_points(
            collection_name=collection_name,
            query=asr_vector,
            using=ASR_DENSE.name,
            limit=_DEFAULT_PREFETCH_LIMIT,
            score_threshold=score_threshold,
            with_payload=True,
        )
    )
    task_weights.append(w_asr_dense)

    # Nhánh ocr_sparse
    if ocr_sparse_vector is not None:
        tasks.append(
            client.query_points(
                collection_name=collection_name,
                query=ocr_sparse_vector,
                using=OCR_SPARSE.name,
                limit=_DEFAULT_PREFETCH_LIMIT,
                with_payload=True,
            )
        )
        task_weights.append(w_ocr_sparse)

    # Nhánh asr_sparse
    if asr_sparse_vector is not None:
        tasks.append(
            client.query_points(
                collection_name=collection_name,
                query=asr_sparse_vector,
                using=ASR_SPARSE.name,
                limit=_DEFAULT_PREFETCH_LIMIT,
                with_payload=True,
            )
        )
        task_weights.append(w_asr_sparse)

    # Thực thi tất cả nhánh bất đồng bộ
    branch_results = await asyncio.gather(*tasks)

    # Tích lũy điểm Weighted RRF
    rrf_scores: dict[str, float] = {}
    point_map: dict[str, any] = {}

    for branch_res, weight in zip(branch_results, task_weights):
        points = getattr(branch_res, "points", branch_res)
        for rank, point in enumerate(points, start=1):
            pid = str(point.id)
            score_contrib = weight * (1.0 / (rrf_k + rank))
            rrf_scores[pid] = rrf_scores.get(pid, 0.0) + score_contrib
            if pid not in point_map:
                point_map[pid] = point

    # Sắp xếp theo Weighted RRF score giảm dần
    sorted_ids = sorted(rrf_scores.keys(), key=lambda pid: rrf_scores[pid], reverse=True)[:top_k]

    final_candidates = []
    for pid in sorted_ids:
        sp = point_map[pid]
        cand = _scored_point_to_candidate(sp)
        cand.score = round(float(rrf_scores[pid]), 6)
        final_candidates.append(cand)

    logger.debug(f"search_by_text_vectors: {len(final_candidates)} candidates via Weighted RRF from {len(tasks)} branches")
    return final_candidates


async def search_by_image_vector(
    client: AsyncQdrantClient,
    image_vector: list[float],
    top_k: int = 100,
    collection_name: str = PRODUCTION_ALIAS,
    score_threshold: float | None = None,
) -> list[CandidateResult]:
    """
    Tìm kiếm keyframe tương đồng bằng image vector (Image-to-Image Search).

    Chỉ dùng nhánh image_dense (BEiT-large + CLIP), không RRF vì chỉ có 1 nhánh.
    """
    results = await client.query_points(
        collection_name=collection_name,
        query=image_vector,
        using=IMAGE_DENSE.name,
        limit=top_k,
        score_threshold=score_threshold,
        with_payload=True,
    )

    logger.debug(f"search_by_image_vector: {len(results.points)} results")
    return [_scored_point_to_candidate(p) for p in results.points]


async def search_by_all_vectors(
    client: AsyncQdrantClient,
    image_vector: list[float],
    ocr_vector: list[float],
    asr_vector: list[float],
    ocr_sparse_vector: SparseVector | None,
    asr_sparse_vector: SparseVector | None,
    top_k: int = 100,
    collection_name: str = PRODUCTION_ALIAS,
) -> list[CandidateResult]:
    """
    Multimodal search: kết hợp cả image và text vectors với RRF.

    Dùng khi query bao gồm cả ảnh và text mô tả.
    """
    # Tận dụng search_by_text_vectors — đã hỗ trợ image_vector parameter
    return await search_by_text_vectors(
        client=client,
        ocr_vector=ocr_vector,
        asr_vector=asr_vector,
        ocr_sparse_vector=ocr_sparse_vector,
        asr_sparse_vector=asr_sparse_vector,
        image_vector=image_vector,
        top_k=top_k,
        collection_name=collection_name,
    )


# =============================================================================
# Helper Functions
# =============================================================================

# Các field bắt buộc trong payload — khớp với KeyframePayload
_REQUIRED_PAYLOAD_FIELDS = [
    "video_id", "scene_id", "frame_id",
    "timestamp_start", "timestamp_end", "timestamp_mid",
    "keyframe_path",
]


def _scored_point_to_candidate(scored_point) -> CandidateResult:
    """
    Convert Qdrant ScoredPoint → CandidateResult.

    RÀNG BUỘC: Payload phải có đủ các field của KeyframePayload.
    Nếu thiếu field, raise ValueError với tên field bị thiếu.
    """
    payload = scored_point.payload or {}

    # Validate required fields
    missing = [f for f in _REQUIRED_PAYLOAD_FIELDS if f not in payload]
    if missing:
        raise ValueError(
            f"Payload thiếu field bắt buộc: {missing}. "
            f"Point ID: {scored_point.id}. "
            f"Kiểm tra lại quá trình ingestion/upsert."
        )

    return CandidateResult(
        video_id=payload["video_id"],
        scene_id=payload["scene_id"],
        frame_id=int(payload["frame_id"]),
        timestamp_start=float(payload["timestamp_start"]),
        timestamp_end=float(payload["timestamp_end"]),
        timestamp_mid=float(payload["timestamp_mid"]),
        keyframe_path=payload["keyframe_path"],
        ocr_text=payload.get("ocr_text", ""),
        asr_text=payload.get("asr_text", ""),
        score=float(scored_point.score) if scored_point.score is not None else 0.0,
    )


async def get_point_by_id(
    client: AsyncQdrantClient,
    point_id: str,
    collection_name: str = PRODUCTION_ALIAS,
) -> CandidateResult | None:
    """
    Lấy thông tin 1 keyframe cụ thể theo Point ID.

    Dùng cho trường hợp cần fetch metadata của 1 keyframe cụ thể
    (ví dụ: hiển thị chi tiết khi user click vào 1 kết quả).
    """
    points = await client.retrieve(
        collection_name=collection_name,
        ids=[point_id],
        with_payload=True,
        with_vectors=False,
    )

    if not points:
        logger.warning(f"Point not found: {point_id} in {collection_name}")
        return None

    point = points[0]

    # retrieve() trả về Record (không có score), tạo fake scored_point-like object
    class _FakeScored:
        def __init__(self, p):
            self.id = p.id
            self.payload = p.payload
            self.score = 1.0  # không có score thật khi retrieve by ID

    return _scored_point_to_candidate(_FakeScored(point))
