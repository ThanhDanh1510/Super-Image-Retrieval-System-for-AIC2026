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
    TextIndexParams,
    TokenizerType,
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
    """Tự động chuyển 'production' sang collection thực tế nếu alias chưa khởi tạo."""
    try:
        if not await client.collection_exists(collection_name):
            for candidate in ["aic2026_clean_v3", "aic2026_v1", "collection_v1"]:
                if await client.collection_exists(candidate):
                    logger.debug(f"Resolved collection '{collection_name}' -> '{candidate}'")
                    return candidate
    except Exception as e:
        logger.warning(f"Could not check collection existence for '{collection_name}': {e}")
    return collection_name


async def setup_full_text_indexes(
    client: AsyncQdrantClient,
    collection_name: str = PRODUCTION_ALIAS,
) -> None:
    """
    Khởi tạo Full-text index (MatchText) cho ocr_text và asr_text với replace_accent=True (bỏ dấu tiếng Việt)
    và lowercase=True để giải quyết triệt để vấn đề mismatch chữ hoa/thường hay có dấu/không dấu.
    """
    collection_name = await _resolve_collection(client, collection_name)
    try:
        info = await client.get_collection(collection_name)
        indexed_fields = info.payload_schema or {}
        
        for field in ["ocr_text", "asr_text"]:
            if field not in indexed_fields:
                logger.info(f"🏗️ Tự động tạo full-text index cho trường: '{field}' (bỏ dấu + viết thường)")
                await client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field,
                    field_schema=TextIndexParams(
                        type="text",
                        tokenizer=TokenizerType.WORD,
                        lowercase=True,
                        ascii_folding=True,
                    )
                )
                logger.info(f"✅ Đã tạo thành công full-text index cho: '{field}'")
    except Exception as e:
        logger.warning(f"Không thể tự động tạo full-text index cho Qdrant payload: {e}")


# =============================================================================
# Core Retrieval Functions
# =============================================================================

import re

def _normalize_text_for_scoring(text: str) -> str:
    """Chuẩn hóa chuỗi văn bản về dạng chữ thường không dấu để so khớp chính xác."""
    if not text:
        return ""
    text = text.lower()
    patterns = {
        '[àáảãạăằắẳẵặâầấẩẫậ]': 'a',
        '[èéẻẽẹêềếểễệ]': 'e',
        '[ìíỉĩị]': 'i',
        '[òóỏõọôồốổỗộơờớởỡợ]': 'o',
        '[ùúủũụưừứửữự]': 'u',
        '[ỳýỷỹỵ]': 'y',
        'đ': 'd'
    }
    for pattern, repl in patterns.items():
        text = re.sub(pattern, repl, text)
    # Loại bỏ ký tự đặc biệt
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return " ".join(text.split())


def _calculate_match_score(query: str, target_text: str) -> float:
    """Tính điểm khớp từ khóa (overlap score) giữa câu query và đoạn văn bản trong database."""
    if not target_text:
        return 0.0
    
    q_norm = _normalize_text_for_scoring(query)
    t_norm = _normalize_text_for_scoring(target_text)
    
    q_words = [w for w in q_norm.split() if len(w) > 1]
    if not q_words:
        return 0.0
        
    t_words = set(t_norm.split())
    
    # Tính tỉ lệ từ khóa trùng lặp
    matches = sum(1 for w in q_words if w in t_words)
    overlap_ratio = matches / len(q_words)
    
    # Cộng điểm bonus nếu khớp nguyên cả cụm từ liên tiếp (substring match)
    if q_norm in t_norm:
        overlap_ratio += 0.5
    else:
        # Check khớp các cụm 2 từ liên tiếp nếu có
        for i in range(len(q_words) - 1):
            bigram = f"{q_words[i]} {q_words[i+1]}"
            if bigram in t_norm:
                overlap_ratio += 0.15
                
    return round(overlap_ratio, 6)


async def search_by_text_match(
    client: AsyncQdrantClient,
    field_name: str,
    query_text: str,
    limit: int = 200,
    collection_name: str = PRODUCTION_ALIAS,
) -> list[CandidateResult]:
    """
    Tìm kiếm keyframe bằng cách khớp từ khóa văn bản (MatchText filter) trực tiếp trên Qdrant.
    Hỗ trợ tìm kiếm thả lỏng (OR matching) và sắp xếp kết quả theo điểm trùng khớp thực tế.
    """
    if not query_text or not query_text.strip():
        return []

    collection_name = await _resolve_collection(client, collection_name)

    # Phân tách từ khóa để tạo bộ lọc thả lỏng (should)
    # Giữ các từ có độ dài từ 2 ký tự trở lên để tăng tính chính xác
    clean_query = query_text.strip()
    words = [w for w in clean_query.split() if len(w) > 1]
    
    should_conditions = []
    
    # 1. Thêm cả cụm nguyên văn (Khớp đầy đủ cả cụm từ là tốt nhất)
    should_conditions.append(
        FieldCondition(key=field_name, match=MatchText(text=clean_query))
    )
    
    # 2. Tạo các cụm 2 từ liên tiếp (bigrams) nếu câu dài
    if len(words) >= 3:
        for i in range(len(words) - 1):
            bigram = f"{words[i]} {words[i+1]}"
            should_conditions.append(
                FieldCondition(key=field_name, match=MatchText(text=bigram))
            )
            
    # 3. Thêm các từ đơn lẻ (độ dài >= 3 để lọc từ quá ngắn)
    for word in words:
        if len(word) >= 3:
            should_conditions.append(
                FieldCondition(key=field_name, match=MatchText(text=word))
            )

    # Dự phòng: Nếu không có từ nào đủ dài, dùng tất cả các từ
    if not should_conditions:
        should_conditions = [
            FieldCondition(key=field_name, match=MatchText(text=w)) for w in clean_query.split()
        ]

    # Sử dụng Qdrant scroll với bộ lọc should để tìm kiếm thả lỏng
    res, _ = await client.scroll(
        collection_name=collection_name,
        scroll_filter=Filter(should=should_conditions),
        limit=limit * 2, # Lấy nhiều hơn một chút để lọc trùng khớp tốt ở python
        with_payload=True,
        with_vectors=False,
    )

    cands_with_scores = []
    for point in res:
        try:
            cand = _scored_point_to_candidate(point)
            # Lấy trường text tương ứng từ payload để chấm điểm
            db_text = cand.ocr_text if field_name == "ocr_text" else cand.asr_text
            
            # Tính điểm độ tương thích
            score = _calculate_match_score(clean_query, db_text)
            
            # Chỉ lấy các kết quả có độ trùng khớp tối thiểu (ví dụ: khớp ít nhất 1 từ khóa)
            if score > 0.0:
                cand.score = score
                cands_with_scores.append(cand)
        except Exception as e:
            logger.warning(f"Error parsing point in loose text match: {e}")
            pass

    # Sắp xếp danh sách candidates theo điểm số độ khớp giảm dần
    cands_with_scores.sort(key=lambda x: x.score, reverse=True)
    
    # Lấy đúng số lượng giới hạn limit
    return cands_with_scores[:limit]


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

