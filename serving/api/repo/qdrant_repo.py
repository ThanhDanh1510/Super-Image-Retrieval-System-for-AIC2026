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

async def search_by_text_match(
    client: AsyncQdrantClient,
    field_name: str,
    query_text: str,
    limit: int = 200,
    collection_name: str = PRODUCTION_ALIAS,
) -> list[CandidateResult]:
    """
    Tìm kiếm keyframe bằng cách khớp từ khóa văn bản (MatchText filter) trực tiếp trên Qdrant.
    Không sử dụng embedding vector để đảm bảo khớp ký tự chuẩn xác 100%.
    """
    if not query_text or not query_text.strip():
        return []

    collection_name = await _resolve_collection(client, collection_name)

    # Sử dụng Qdrant scroll với bộ lọc MatchText để thực hiện tìm kiếm full-text
    res, _ = await client.scroll(
        collection_name=collection_name,
        scroll_filter=Filter(
            must=[
                FieldCondition(
                    key=field_name,
                    match=MatchText(text=query_text)
                )
            ]
        ),
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )

    cands = []
    # Gán score giảm dần theo thứ tự khớp để thuật toán RRF hoặc hiển thị hoạt động tốt
    for rank, point in enumerate(res, start=1):
        try:
            cand = _scored_point_to_candidate(point)
            cand.score = round(1.0 - (rank / (limit + 10)), 6)
            cands.append(cand)
        except Exception as e:
            logger.warning(f"Error parse point in text match: {e}")
            pass
    return cands


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
    query_text: str = "",
) -> list[CandidateResult]:
    """
    Tìm kiếm keyframe bằng text vectors và image vector qua Weighted RRF fusion.
    Nếu search_mode là 'ocr' hoặc 'asr', tìm kiếm khớp từ khóa văn bản thuần túy (MatchText).
    """
    import asyncio
    from shared.utils.config_loader import load_fusion_config

    collection_name = await _resolve_collection(client, collection_name)

    # 1. Chế độ tìm kiếm đơn nhánh OCR khớp từ khóa
    if search_mode == "ocr":
        logger.info(f"🔍 OCR-only search matching text: '{query_text}'")
        return await search_by_text_match(
            client=client,
            field_name="ocr_text",
            query_text=query_text,
            limit=top_k,
            collection_name=collection_name
        )

    # 2. Chế độ tìm kiếm đơn nhánh ASR khớp từ khóa
    if search_mode == "asr":
        logger.info(f"🔍 ASR-only search matching text: '{query_text}'")
        return await search_by_text_match(
            client=client,
            field_name="asr_text",
            query_text=query_text,
            limit=top_k,
            collection_name=collection_name
        )

    # 3. Chế độ Hybrid (all) hoặc chỉ Embedding
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

    # Nếu ở chế độ Hybrid (all), thay vì dùng dense vector embedding cho OCR/ASR,
    # chúng ta thực hiện tìm kiếm khớp từ khóa văn bản thuần túy (MatchText)
    if search_mode == "all" and query_text:
        # Nhánh OCR text match
        tasks.append(
            search_by_text_match(
                client=client,
                field_name="ocr_text",
                query_text=query_text,
                limit=_DEFAULT_PREFETCH_LIMIT,
                collection_name=collection_name
            )
        )
        task_weights.append(w_ocr_dense)

        # Nhánh ASR text match
        tasks.append(
            search_by_text_match(
                client=client,
                field_name="asr_text",
                query_text=query_text,
                limit=_DEFAULT_PREFETCH_LIMIT,
                collection_name=collection_name
            )
        )
        task_weights.append(w_asr_dense)
        
    elif search_mode == "embedding":
        # Chế độ chỉ dùng embedding: vẫn giữ dense query cho OCR/ASR nếu cần
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

    # Thực thi các nhánh
    branch_results = await asyncio.gather(*tasks)

    # Tích lũy điểm Weighted RRF
    rrf_scores: dict[str, float] = {}
    point_map: dict[str, any] = {}

    for branch_res, weight in zip(branch_results, task_weights):
        # Nếu là kết quả từ hàm search_by_text_match (dạng list)
        if isinstance(branch_res, list):
            points = branch_res
            for rank, cand in enumerate(points, start=1):
                pid = f"{cand.video_id}::{cand.scene_id}" # Tạo ID nhất quán cho map
                # Tái cấu trúc point giả lập để _scored_point_to_candidate hoạt động nếu cần,
                # hoặc lưu trực tiếp candidate vào point_map
                score_contrib = weight * (1.0 / (rrf_k + rank))
                rrf_scores[pid] = rrf_scores.get(pid, 0.0) + score_contrib
                if pid not in point_map:
                    point_map[pid] = cand
        else:
            # Nếu là kết quả từ client.query_points (ScoredPoint)
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
        item = point_map[pid]
        if isinstance(item, CandidateResult):
            cand = item
        else:
            cand = _scored_point_to_candidate(item)
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

_SCENE_DESC_MAP: dict[str, str] = {}

def _get_scene_desc(scene_id: str) -> str:
    global _SCENE_DESC_MAP
    if not _SCENE_DESC_MAP:
        try:
            import pandas as pd
            desc_df = pd.read_parquet("data/unified/scene_descriptions.parquet")
            _SCENE_DESC_MAP = dict(zip(desc_df["scene_id"], desc_df["description"].fillna("")))
        except Exception:
            pass
    return _SCENE_DESC_MAP.get(scene_id, "")

def _scored_point_to_candidate(scored_point: any) -> CandidateResult:
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

    # Lấy description từ payload hoặc tra cứu trực tiếp từ scene_descriptions.parquet
    desc = payload.get("description", "") or payload.get("scene_description", "")
    if not desc:
        desc = _get_scene_desc(scene_id)

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
        description=desc,
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
        logger.warning(f"Error fetching keyframes for video {video_id}: {err}")
        return []

