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
import socket
from urllib.parse import urlparse

from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client.models import (
    SparseVector,
    Filter,
    FieldCondition,
    MatchText,
    MatchValue,
)

from serving.api.schema.retrieval import CandidateResult
from shared.schemas.named_vector_spec import (
    IMAGE_DENSE,
    OCR_DENSE,
    ASR_DENSE,
    OCR_SPARSE,
    ASR_SPARSE,
)

logger = logging.getLogger(__name__)

# Tên Qdrant collection alias "production" → collection_vN (xem indexing/collection_alias.py)
PRODUCTION_ALIAS = "production"

# Prefetch limit mặc định — lấy đủ candidates cho RRF fusion
_DEFAULT_PREFETCH_LIMIT = 200


# =============================================================================
# Client Initialization
# =============================================================================

def _is_server_reachable(url: str) -> bool:
    try:
        parsed = urlparse(url)
        host = parsed.hostname or "127.0.0.1"
        if host == "localhost":
            host = "127.0.0.1"
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        with socket.create_connection((host, port), timeout=1.5):
            return True
    except Exception:
        return False


def get_qdrant_client(url: str, api_key: str | None = None) -> QdrantClient:
    """
    Tạo synchronous Qdrant client. Tự động fallback sang Local Embedded Qdrant nếu remote server không kết nối được.
    """
    if (url.startswith("http://") or url.startswith("https://")) and _is_server_reachable(url):
        logger.info(f"Connected to Qdrant server (sync): {url}")
        return QdrantClient(url=url, api_key=api_key, timeout=30)
    
    if url.startswith("http://") or url.startswith("https://"):
        logger.warning(
            f"Không thể kết nối Qdrant server tại {url}. "
            f"Tự động chuyển sang Local Embedded Qdrant tại './data/qdrant_db'..."
        )
        return QdrantClient(path="./data/qdrant_db")
    return QdrantClient(path=url)


def get_async_qdrant_client(url: str, api_key: str | None = None) -> AsyncQdrantClient:
    """
    Tạo async Qdrant client cho FastAPI async endpoints. Tự động fallback sang Local Embedded Qdrant nếu remote server không kết nối được.
    """
    if (url.startswith("http://") or url.startswith("https://")) and _is_server_reachable(url):
        logger.info(f"Connected to Qdrant server (async): {url}")
        return AsyncQdrantClient(url=url, api_key=api_key, timeout=30)

    if url.startswith("http://") or url.startswith("https://"):
        logger.warning(
            f"Không thể kết nối Qdrant server tại {url}. "
            f"Tự động chuyển sang Local Embedded Qdrant tại './data/qdrant_db'..."
        )
        return AsyncQdrantClient(path="./data/qdrant_db")
async def _resolve_collection(client: AsyncQdrantClient, collection_name: str) -> str:
    """Tự động chuyển 'production' sang 'collection_v1' nếu alias chưa khởi tạo."""
    try:
        if not await client.collection_exists(collection_name):
            if await client.collection_exists("collection_v1"):
                return "collection_v1"
    except Exception:
        pass
    return collection_name


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
    search_mode: str = "all",
    raw_query_vi: str = "",
    raw_query_en: str = "",
) -> list[CandidateResult]:
    """
    Tìm kiếm keyframe bằng text vectors (dense + sparse) và image vector qua Weighted RRF fusion.
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

    collection_name = await _resolve_collection(client, collection_name)

    w_image = float(weights.get("image_dense", 2.5))
    w_ocr_dense = float(weights.get("ocr_dense", 0.5))
    w_asr_dense = float(weights.get("asr_dense", 0.3))
    w_ocr_sparse = float(weights.get("ocr_sparse", 0.2))
    w_asr_sparse = float(weights.get("asr_sparse", 0.2))

    tasks = []
    task_weights = []

    # Nhánh image_dense (Visual CLIP)
    if (search_mode in ["all", "embedding"]) and (image_vector is not None):
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
    if search_mode in ["all", "ocr"]:
        tasks.append(
            client.query_points(
                collection_name=collection_name,
                query=ocr_vector,
                using=OCR_DENSE.name,
                limit=_DEFAULT_PREFETCH_LIMIT,
                score_threshold=score_threshold,
                query_filter=Filter(
                    must_not=[
                        FieldCondition(key="ocr_text", match=MatchValue(value=""))
                    ]
                ),
                with_payload=True,
            )
        )
        task_weights.append(w_ocr_dense)

    # Nhánh asr_dense
    if search_mode in ["all", "asr"]:
        tasks.append(
            client.query_points(
                collection_name=collection_name,
                query=asr_vector,
                using=ASR_DENSE.name,
                limit=_DEFAULT_PREFETCH_LIMIT,
                score_threshold=score_threshold,
                query_filter=Filter(
                    must_not=[
                        FieldCondition(key="asr_text", match=MatchValue(value=""))
                    ]
                ),
                with_payload=True,
            )
        )
        task_weights.append(w_asr_dense)

    # Nhánh ocr_sparse
    if (search_mode in ["all", "ocr"]) and (ocr_sparse_vector is not None):
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
    if (search_mode in ["all", "asr"]) and (asr_sparse_vector is not None):
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
    try:
        branch_results = await asyncio.gather(*tasks)
    except Exception as err:
        logger.warning(f"Qdrant search failed ({err}). Falling back to HybridSearchEngine (Parquet vector index)...")
        try:
            from serving.api.service.retrieval.hybrid_search import get_hybrid_search_engine
            engine = get_hybrid_search_engine()
            q_vi = raw_query_vi or "query"
            q_en = raw_query_en or q_vi
            raw_results = engine.search(query_vi=q_vi, query_en=q_en, mode=search_mode, top_k=top_k)
            cands = []
            for item in raw_results:
                cands.append(CandidateResult(
                    video_id=item["video_id"],
                    scene_id=item["scene_id"],
                    frame_id=int(item["frame_id"]),
                    timestamp_start=float(item.get("timestamp_start", 0.0)),
                    timestamp_end=float(item.get("timestamp_end", 0.0)),
                    timestamp_mid=float(item.get("timestamp_mid", 0.0)),
                    keyframe_path=item["keyframe_path"],
                    ocr_text=item.get("ocr_text", ""),
                    asr_text=item.get("asr_text", ""),
                    score=float(item["score"]),
                ))
            return cands
        except Exception as fallback_err:
            logger.error(f"Fallback search_by_text_vectors failed: {fallback_err}")
            return []

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

    Chỉ dùng nhánh image_dense (CLIP ViT-L/14), không RRF vì chỉ có 1 nhánh.
    """
    collection_name = await _resolve_collection(client, collection_name)
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

    scene_id = payload["scene_id"]
    frame_id = int(payload["frame_id"])
    keyframe_path = payload.get("keyframe_path", "")
    
    # Chuẩn hóa keyframe_path về dạng unified/{scene_id}-{frame_id}.webp (không bị lặp video_id 2 lần)
    ext = keyframe_path.rsplit(".", 1)[-1] if "." in keyframe_path else "webp"
    clean_keyframe_path = f"unified/{scene_id}-{frame_id}.{ext}"

    return CandidateResult(
        video_id=payload["video_id"],
        scene_id=scene_id,
        frame_id=frame_id,
        timestamp_start=float(payload["timestamp_start"]),
        timestamp_end=float(payload["timestamp_end"]),
        timestamp_mid=float(payload["timestamp_mid"]),
        keyframe_path=clean_keyframe_path,
        ocr_text=payload.get("ocr_text", ""),
        asr_text=payload.get("asr_text", ""),
        score=float(getattr(scored_point, "score", 0.0) or 0.0),
    )



async def get_keyframes_by_video_id(
    client: AsyncQdrantClient,
    video_id: str,
    limit: int = 100,
    collection_name: str = PRODUCTION_ALIAS,
) -> list[CandidateResult]:
    """
    Lấy tất cả keyframe thuộc 1 video cụ thể theo video_id.
    """
    try:
        collection_name = await _resolve_collection(client, collection_name)
        res, _ = await client.scroll(
            collection_name=collection_name,
            scroll_filter=Filter(
                must=[FieldCondition(key="video_id", match=MatchText(text=video_id))]
            ),
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        cands = []
        for point in res:
            try:
                cand = _scored_point_to_candidate(point)
                if cand.video_id == video_id:
                    cands.append(cand)
            except Exception:
                pass
        return sorted(cands, key=lambda c: (c.timestamp_mid, c.frame_id))
    except Exception as err:
        logger.warning(f"Error fetching keyframes for video {video_id}: {err}. Falling back to Parquet metadata.")
        try:
            import pandas as pd
            from pathlib import Path
            data_dir = Path("data/unified")
            meta_path = data_dir / "scene_metadata.parquet"
            if not meta_path.exists():
                meta_path = Path("data/data/unified/scene_metadata.parquet")
            df_meta = pd.read_parquet(meta_path)
            sub = df_meta[df_meta["video_id"] == video_id].sort_values("timestamp_mid")
            cands = []
            for _, row in sub.head(limit).iterrows():
                scene_id = str(row["scene_id"])
                frame_id = int(row["frame_id"])
                ext = str(row.get("keyframe_path", "")).rsplit(".", 1)[-1] if "." in str(row.get("keyframe_path", "")) else "webp"
                cands.append(CandidateResult(
                    video_id=str(row["video_id"]),
                    scene_id=scene_id,
                    frame_id=frame_id,
                    timestamp_start=float(row.get("timestamp_start", 0.0)),
                    timestamp_end=float(row.get("timestamp_end", 0.0)),
                    timestamp_mid=float(row.get("timestamp_mid", 0.0)),
                    keyframe_path=f"unified/{scene_id}-{frame_id}.{ext}",
                    ocr_text="",
                    asr_text="",
                    score=1.0,
                ))
            return cands
        except Exception as e:
            logger.error(f"Fallback metadata search failed: {e}")
            return []

