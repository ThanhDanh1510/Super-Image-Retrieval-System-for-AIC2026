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

import asyncio
import logging
import socket
from typing import Any, Optional
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


def _get_local_qdrant_path() -> str:
    """Xác định đường dẫn lưu trữ Local Embedded Qdrant chính xác trên đĩa."""
    from pathlib import Path
    for p in ["./data/qdrant_storage", "./data/qdrant_db"]:
        path_obj = Path(p)
        if path_obj.exists() and (path_obj / "collections").exists():
            return p
    return "./data/qdrant_storage" if Path("./data/qdrant_storage").exists() else "./data/qdrant_db"


def get_qdrant_client(url: str, api_key: str | None = None) -> QdrantClient:
    """
    Tạo synchronous Qdrant client. Tự động fallback sang Local Embedded Qdrant nếu remote server không kết nối được.
    """
    if (url.startswith("http://") or url.startswith("https://")) and _is_server_reachable(url):
        logger.info(f"Connected to Qdrant server (sync): {url}")
        return QdrantClient(url=url, api_key=api_key, timeout=30)
    
    if url.startswith("http://") or url.startswith("https://"):
        local_path = _get_local_qdrant_path()
        logger.warning(
            f"Không thể kết nối Qdrant server tại {url}. "
            f"Tự động chuyển sang Local Embedded Qdrant tại '{local_path}'..."
        )
        return QdrantClient(path=local_path)
    return QdrantClient(path=url)


def get_async_qdrant_client(url: str, api_key: str | None = None) -> AsyncQdrantClient:
    """
    Tạo async Qdrant client cho FastAPI async endpoints. Tự động fallback sang Local Embedded Qdrant nếu remote server không kết nối được.
    """
    if (url.startswith("http://") or url.startswith("https://")) and _is_server_reachable(url):
        logger.info(f"Connected to Qdrant server (async): {url}")
        return AsyncQdrantClient(url=url, api_key=api_key, timeout=30)

    if url.startswith("http://") or url.startswith("https://"):
        local_path = _get_local_qdrant_path()
        logger.warning(
            f"Không thể kết nối Qdrant server tại {url}. "
            f"Tự động chuyển sang Local Embedded Qdrant tại '{local_path}'..."
        )
        return AsyncQdrantClient(path=local_path)
    return AsyncQdrantClient(path=url)


_RESOLVED_COLLECTIONS: dict[str, str] = {}


async def _maybe_await(res):
    if hasattr(res, "__await__"):
        return await res
    return res


async def _resolve_collection(client: Any, collection_name: str) -> str:
    """Tự động chuyển 'production' sang collection thực tế nếu alias chưa khởi tạo hoặc dùng Local Embedded Qdrant."""
    if collection_name in _RESOLVED_COLLECTIONS:
        return _RESOLVED_COLLECTIONS[collection_name]

    try:
        all_cols = await _maybe_await(client.get_collections())
        col_names = [c.name for c in all_cols.collections] if all_cols and all_cols.collections else []

        # Nếu collection_name đã tồn tại trong danh sách
        if collection_name in col_names:
            _RESOLVED_COLLECTIONS[collection_name] = collection_name
            return collection_name

        # Thử các candidate thông dụng
        for candidate in ["aic2026_clean_v3", "aic2026_v1", "collection_v1"]:
            if candidate in col_names:
                logger.info(f"Resolved collection '{collection_name}' -> '{candidate}'")
                _RESOLVED_COLLECTIONS[collection_name] = candidate
                return candidate

        # Fallback collection đầu tiên nếu có
        if col_names:
            resolved = col_names[0]
            logger.info(f"Resolved fallback collection '{collection_name}' -> '{resolved}'")
            _RESOLVED_COLLECTIONS[collection_name] = resolved
            return resolved
    except Exception as e:
        logger.warning(f"Could not check collections for '{collection_name}': {e}")
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
    """
    Tính điểm khớp từ khóa (overlap score) giữa câu query và đoạn văn bản trong database.
    Ưu tiên tuyệt đối cho khớp chính xác 100% nguyên cụm (Exact Phrase Substring Match).
    """
    if not target_text:
        return 0.0
    
    q_norm = _normalize_text_for_scoring(query)
    t_norm = _normalize_text_for_scoring(target_text)
    
    q_words = [w for w in q_norm.split() if len(w) > 1]
    if not q_words:
        return 0.0
        
    t_words = set(t_norm.split())
    
    # 1. Tier 1: Khớp chính xác 100% nguyên cụm từ liên tiếp (Exact Phrase Match)
    if q_norm in t_norm:
        # Gán điểm vượt trội tuyệt đối (>= 2.0)
        return round(2.0 + min(len(q_words) * 0.1, 1.0), 6)
    
    # 2. Tính tỉ lệ từ khóa trùng lặp
    matches = sum(1 for w in q_words if w in t_words)
    overlap_ratio = matches / len(q_words)
    
    # 3. Tier 2: Khớp các cụm từ con liên tiếp (n-grams)
    longest_subphrase_len = 0
    for n in range(min(len(q_words), 5), 1, -1):
        for i in range(len(q_words) - n + 1):
            ngram = " ".join(q_words[i:i + n])
            if ngram in t_norm:
                longest_subphrase_len = max(longest_subphrase_len, n)
                break
        if longest_subphrase_len > 0:
            break

    if longest_subphrase_len >= 3:
        overlap_ratio += 0.4 * (longest_subphrase_len / len(q_words))
    elif longest_subphrase_len == 2:
        overlap_ratio += 0.15

    # 4. Phạt nếu thiếu nhiều từ khóa quan trọng trong câu query
    if overlap_ratio < 1.0:
        missing_count = len(q_words) - matches
        if missing_count >= 2:
            overlap_ratio *= 0.6  # Giảm 40% nếu thiếu từ 2 từ trở lên

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

    clean_query = query_text.strip()
    words = [w for w in clean_query.split() if len(w) > 1]
    collection_name = await _resolve_collection(client, collection_name)
    seen_pids = set()
    all_points = []

    async def _async_scroll(**kwargs):
        if hasattr(client, "scroll") and asyncio.iscoroutinefunction(client.scroll):
            return await client.scroll(**kwargs)
        else:
            return await asyncio.to_thread(client.scroll, **kwargs)

    # 1. Tier 1: Khớp đầy đủ toàn bộ từ khóa trong câu (Must MatchText)
    try:
        res = await _async_scroll(
            collection_name=collection_name,
            scroll_filter=Filter(must=[FieldCondition(key=field_name, match=MatchText(text=clean_query))]),
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        res_must = res[0] if isinstance(res, tuple) else getattr(res, "points", res)
        for p in res_must:
            seen_pids.add(str(p.id))
            all_points.append(p)
    except Exception as e:
        logger.warning(f"Error in must MatchText scroll: {e}")

    # 2. Tier 2: Khớp thả lỏng (OR matching) với các cụm n-grams dài trước, rồi mới đến từ đơn lẻ
    if len(all_points) < limit:
        remaining_limit = (limit - len(all_points)) * 2
        should_conditions = []
        
        # 2.1 Thêm cụm 3-grams hoặc 4-grams nếu có
        if len(words) >= 4:
            for i in range(len(words) - 2):
                trigram = f"{words[i]} {words[i+1]} {words[i+2]}"
                should_conditions.append(FieldCondition(key=field_name, match=MatchText(text=trigram)))
        
        # 2.2 Thêm cụm bigrams
        if len(words) >= 3:
            for i in range(len(words) - 1):
                bigram = f"{words[i]} {words[i+1]}"
                should_conditions.append(FieldCondition(key=field_name, match=MatchText(text=bigram)))
                
        # 2.3 Thêm từ đơn lẻ dài >= 3 ký tự
        for word in words:
            if len(word) >= 3:
                should_conditions.append(FieldCondition(key=field_name, match=MatchText(text=word)))

        if should_conditions:
            try:
                res = await _async_scroll(
                    collection_name=collection_name,
                    scroll_filter=Filter(should=should_conditions),
                    limit=remaining_limit,
                    with_payload=True,
                    with_vectors=False,
                )
                res_should = res[0] if isinstance(res, tuple) else getattr(res, "points", res)
                for p in res_should:
                    pid_str = str(p.id)
                    if pid_str not in seen_pids:
                        seen_pids.add(pid_str)
                        all_points.append(p)
            except Exception as e:
                logger.warning(f"Error in should MatchText scroll: {e}")

    cands_with_scores = []
    for point in all_points:
        try:
            cand = _scored_point_to_candidate(point)
            db_text = cand.ocr_text if field_name == "ocr_text" else cand.asr_text
            score = _calculate_match_score(clean_query, db_text)
            if score > 0.0:
                cand.score = score
                cands_with_scores.append(cand)
        except Exception as e:
            logger.warning(f"Error parsing point in loose text match: {e}")

    # Sắp xếp danh sách candidates theo điểm số độ khớp giảm dần
    cands_with_scores.sort(key=lambda x: x.score, reverse=True)
    return cands_with_scores[:limit]


async def search_by_text_vectors(
    client: AsyncQdrantClient,
    ocr_vector: list[float],
    asr_vector: list[float],
    ocr_sparse_vector: SparseVector | None,
    asr_sparse_vector: SparseVector | None,
    image_vector: list[float] | None = None,
    description_candidates: list[CandidateResult] | None = None,
    asr_candidates: list[CandidateResult] | None = None,
    ocr_candidates: list[CandidateResult] | None = None,
    top_k: int = 100,
    collection_name: str = PRODUCTION_ALIAS,
    score_threshold: float | None = None,
    search_mode: str = "all",
    query_text: str = "",
    custom_weights: dict[str, float] | None = None,
) -> list[CandidateResult]:
    """
    Tìm kiếm keyframe bằng text vectors, image vector, description vector, ASR và OCR qua Weighted RRF fusion.
    Ưu tiên sử dụng In-Memory BM25 ASR & OCR (~25ms) trước khi fallback về MatchText mạng.
    """
    import asyncio
    from shared.utils.config_loader import load_fusion_config

    collection_name = await _resolve_collection(client, collection_name)

    # 1. Chế độ tìm kiếm đơn nhánh OCR
    if search_mode == "ocr":
        if ocr_candidates:
            return ocr_candidates[:top_k]
        logger.info(f"🔍 OCR-only search matching text: '{query_text}'")
        return await search_by_text_match(
            client=client,
            field_name="ocr_text",
            query_text=query_text,
            limit=top_k,
            collection_name=collection_name
        )

    # 2. Chế độ tìm kiếm đơn nhánh ASR
    if search_mode == "asr":
        if asr_candidates:
            return asr_candidates[:top_k]
        logger.info(f"🔍 ASR-only search matching text: '{query_text}'")
        return await search_by_text_match(
            client=client,
            field_name="asr_text",
            query_text=query_text,
            limit=top_k,
            collection_name=collection_name
        )

    # 3. Chế độ tìm kiếm đơn nhánh Description vector
    if search_mode == "description":
        logger.info(f"🔍 Description-only vector search for: '{query_text}'")
        return (description_candidates or [])[:top_k]

    # 4. Chế độ Hybrid (all) hoặc chỉ Embedding
    rrf_k = 60.0
    if custom_weights:
        weights = custom_weights
    else:
        try:
            cfg = load_fusion_config()
            weights = cfg.get("weights", {})
            rrf_k = float(cfg.get("rrf", {}).get("k", 60))
        except Exception:
            weights = {"image_dense": 2.5, "description_dense": 2.0, "ocr_dense": 0.8, "asr_dense": 1.0, "ocr_sparse": 0.2, "asr_sparse": 0.2}
            rrf_k = 60.0

    w_image = float(weights.get("image_dense", 2.5))
    w_desc = float(weights.get("description_dense", 2.0))
    w_ocr_dense = float(weights.get("ocr_dense", 0.8))
    w_asr_dense = float(weights.get("asr_dense", 1.0))

    tasks = []
    task_weights = []

    async def _async_query_points(**kwargs):
        res = client.query_points(**kwargs)
        return await _maybe_await(res)

    async def _async_text_match(**kwargs):
        res = search_by_text_match(**kwargs)
        return await _maybe_await(res)

    # Nhánh 1: image_dense (Visual CLIP)
    if (search_mode in ["all", "embedding"]) and (image_vector is not None) and len(image_vector) > 0:
        tasks.append(
            _async_query_points(
                collection_name=collection_name,
                query=image_vector,
                using=IMAGE_DENSE.name,
                limit=_DEFAULT_PREFETCH_LIMIT,
                score_threshold=score_threshold,
                with_payload=True,
            )
        )
        task_weights.append(w_image)

    # Nhánh 2: Image Description (BGE-Large-EN VLM Dense Vector Search)
    if (search_mode in ["all", "embedding"]) and description_candidates:
        async def _wrap_desc(c):
            return c
        tasks.append(_wrap_desc(description_candidates))
        task_weights.append(w_desc)

    # Nhánh 3: ASR Stream (In-Memory BM25 hoặc Fallback MatchText)
    if (search_mode in ["all", "embedding"]):
        if asr_candidates:
            async def _wrap_asr(c):
                return c
            tasks.append(_wrap_asr(asr_candidates))
            task_weights.append(w_asr_dense)
        elif search_mode == "all" and query_text and w_asr_dense > 0.05:
            tasks.append(
                _async_text_match(
                    client=client,
                    field_name="asr_text",
                    query_text=query_text,
                    limit=_DEFAULT_PREFETCH_LIMIT,
                    collection_name=collection_name
                )
            )
            task_weights.append(w_asr_dense)

    # Nhánh 4: OCR Stream (In-Memory BM25 hoặc Fallback MatchText)
    if (search_mode in ["all", "embedding"]):
        if ocr_candidates:
            async def _wrap_ocr(c):
                return c
            tasks.append(_wrap_ocr(ocr_candidates))
            task_weights.append(w_ocr_dense)
        elif search_mode == "all" and query_text and w_ocr_dense > 0.05:
            tasks.append(
                _async_text_match(
                    client=client,
                    field_name="ocr_text",
                    query_text=query_text,
                    limit=_DEFAULT_PREFETCH_LIMIT,
                    collection_name=collection_name
                )
            )
            task_weights.append(w_ocr_dense)
        
    elif search_mode == "embedding":
        # Chế độ chỉ dùng embedding: nếu có ocr_vector/asr_vector thì vẫn query
        if ocr_vector:
            tasks.append(
                _async_query_points(
                    collection_name=collection_name,
                    query=ocr_vector,
                    using=OCR_DENSE.name,
                    limit=_DEFAULT_PREFETCH_LIMIT,
                    score_threshold=score_threshold,
                    with_payload=True,
                )
            )
            task_weights.append(w_ocr_dense)

        if asr_vector:
            tasks.append(
                _async_query_points(
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
                pid = f"{cand.video_id}_{cand.frame_id}"  # Tạo ID nhất quán theo video_id + frame_id
                # Nếu là exact substring match (cand.score >= 2.0), cho hệ số boost x3.0
                exact_mult = 3.0 if float(cand.score) >= 2.0 else (1.5 if float(cand.score) >= 1.0 else 1.0)
                score_contrib = weight * max(float(cand.score), 0.1) * exact_mult * (1.0 / (rrf_k + rank))
                rrf_scores[pid] = rrf_scores.get(pid, 0.0) + score_contrib
                if pid not in point_map:
                    point_map[pid] = cand
        else:
            # Nếu là kết quả từ client.query_points (ScoredPoint)
            points = getattr(branch_res, "points", branch_res)
            for rank, point in enumerate(points, start=1):
                payload = getattr(point, "payload", {}) or {}
                if "video_id" in payload and "frame_id" in payload:
                    pid = f"{payload['video_id']}_{payload['frame_id']}"
                else:
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
            from pathlib import Path
            desc_path = Path(__file__).resolve().parent.parent.parent.parent / "data" / "unified" / "scene_descriptions.parquet"
            if desc_path.exists():
                desc_df = pd.read_parquet(desc_path)
                _SCENE_DESC_MAP = dict(zip(desc_df["scene_id"].astype(str), desc_df["description"].fillna("")))
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

    # Ưu tiên lấy description mới nhất từ scene_descriptions.parquet, fallback payload
    desc = _get_scene_desc(scene_id) or payload.get("description", "") or payload.get("scene_description", "")

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
        score=max(0.0, min(1.0, float(getattr(scored_point, "score", 0.0) or 0.0))),
    )



async def get_keyframes_by_video_id(
    client: AsyncQdrantClient,
    video_id: str,
    limit: int = 500,
    collection_name: str = PRODUCTION_ALIAS,
) -> list[CandidateResult]:
    """
    Lấy tất cả keyframe thuộc 1 video cụ thể theo video_id.
    """
    collection_name = await _resolve_collection(client, collection_name)
    try:
        res, _ = await _maybe_await(client.scroll(
            collection_name=collection_name,
            scroll_filter=Filter(
                must=[FieldCondition(key="video_id", match=MatchValue(value=video_id))]
            ),
            limit=limit,
            with_payload=True,
            with_vectors=False,
        ))
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


async def get_keyframes_with_vectors_by_video_id(
    client: Any,
    video_id: str,
    limit: int = 500,
    vector_name: str = "image_dense",
    collection_name: str = PRODUCTION_ALIAS,
) -> tuple[list[CandidateResult], list[list[float]]]:
    """
    Lấy tất cả keyframe và vector tương ứng (mặc định image_dense) của 1 video.

    Returns:
        tuple (sorted_cands, sorted_vectors): danh sách candidate và vector tương ứng.
    """
    collection_name = await _resolve_collection(client, collection_name)
    try:
        res, _ = await _maybe_await(client.scroll(
            collection_name=collection_name,
            scroll_filter=Filter(
                must=[FieldCondition(key="video_id", match=MatchValue(value=video_id))]
            ),
            limit=limit,
            with_payload=True,
            with_vectors=[vector_name],
        ))
        items = []
        for point in res:
            try:
                cand = _scored_point_to_candidate(point)
                if cand.video_id == video_id:
                    vec = None
                    if isinstance(point.vector, dict):
                        vec = point.vector.get(vector_name)
                    elif isinstance(point.vector, list):
                        vec = point.vector
                    if vec is not None:
                        items.append((cand, vec))
            except Exception:
                pass

        # Sắp xếp theo thứ tự thời gian
        items.sort(key=lambda item: (item[0].timestamp_mid, item[0].frame_id))
        sorted_cands = [item[0] for item in items]
        sorted_vectors = [item[1] for item in items]
        return sorted_cands, sorted_vectors
    except Exception as err:
        logger.warning(f"Error fetching keyframe vectors for video {video_id}: {err}")
        return [], []



