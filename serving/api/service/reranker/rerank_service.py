"""
serving/api/service/reranker/rerank_service.py
================================================
Service layer duy nhất cho reranking — một interface đứng sau controller.

Hiện tại chỉ dùng ViRanker (text rerank) chạy trên GPU.
Jina M0 (crossmodal rerank) đã được loại bỏ vì quá nặng cho CPU.

CONSTRAINTS:
  - Lazy load model: viranker tự lazy-load model bên trong score().
  - Batch inference: xử lý toàn bộ candidates trong 1 forward pass.
  - Layering: module này chỉ gọi reranker skeletons + schema.
"""

from __future__ import annotations

import asyncio
import logging

from serving.api.schema.retrieval import CandidateResult
from serving.api.service.reranker.skeletons.text_branch import viranker
from serving.api.service.reranker.skeletons import fusion_score

logger = logging.getLogger(__name__)


async def rerank(
    query: str,
    candidates: list[CandidateResult],
    enhanced_query: str | None = None,
    keyframes_root_dir: str = "",
) -> list[CandidateResult]:
    """
    Rerank candidates bằng ViRanker (text) trên GPU.

    Args:
        query: Query gốc từ request.
        candidates: Danh sách candidates cần rerank.
        enhanced_query: Query đã dịch/enhance.
        keyframes_root_dir: Thư mục chứa ảnh keyframe (không dùng).

    Returns:
        Danh sách candidates đã rerank và sắp xếp theo score.
    """
    import time
    _t_start = time.time()
    if not candidates:
        return []

    query_for_rerank = enhanced_query or query

    # Chỉ rerank top 20 candidates để tăng tốc
    rerank_candidates = candidates[:20] if len(candidates) > 20 else candidates

    try:
        text_scores = await asyncio.to_thread(viranker.score, query_for_rerank, rerank_candidates)
        fused = fusion_score.fuse(text_scores, None, rerank_candidates)

        if len(candidates) > 20:
            return fused + candidates[20:]
        return fused
    except Exception as e:
        logger.warning(f"Reranking error: {e}")
        return candidates
