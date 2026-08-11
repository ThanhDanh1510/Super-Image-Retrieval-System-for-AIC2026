"""
serving/api/service/reranker/rerank_service.py
================================================
Service layer duy nhất cho reranking — một interface đứng sau controller.

- ViRanker (text): luôn bật, chạy GPU, rất nhẹ (~0.2s)
- Jina M0 (crossmodal): tắt mặc định, bật qua use_crossmodal_reranker=True

CONSTRAINTS:
  - Lazy load model: viranker/jina tự lazy-load model bên trong score().
  - Batch inference: xử lý toàn bộ candidates trong 1 forward pass.
  - Layering: module này chỉ gọi reranker skeletons + schema.
"""

from __future__ import annotations

import asyncio
import logging

from serving.api.schema.retrieval import CandidateResult
from serving.api.service.reranker.skeletons.text_branch import viranker
from serving.api.service.reranker.skeletons.crossmodal_branch import jina_reranker_m0
from serving.api.service.reranker.skeletons import fusion_score

logger = logging.getLogger(__name__)


async def rerank(
    query: str,
    candidates: list[CandidateResult],
    enhanced_query: str | None = None,
    keyframes_root_dir: str = "",
    use_crossmodal_reranker: bool = False,
) -> list[CandidateResult]:
    """
    Rerank candidates bằng ViRanker (text) + tùy chọn Jina M0 (crossmodal).

    Args:
        query: Query gốc từ request.
        candidates: Danh sách candidates cần rerank.
        enhanced_query: Query đã dịch/enhance.
        keyframes_root_dir: Thư mục chứa ảnh keyframe (cho Jina M0).
        use_crossmodal_reranker: Bật Jina M0 (mặc định False).

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
        if use_crossmodal_reranker:
            # Chạy song song cả hai nhánh
            text_task = asyncio.to_thread(viranker.score, query_for_rerank, rerank_candidates)
            visual_task = asyncio.to_thread(
                jina_reranker_m0.score,
                query_for_rerank,
                rerank_candidates,
                keyframes_root_dir,
            )
            results = await asyncio.gather(text_task, visual_task, return_exceptions=True)

            for res in results:
                if isinstance(res, Exception):
                    raise res

            text_scores, visual_scores = results
        else:
            # Chỉ chạy ViRanker (nhẹ, nhanh)
            text_scores = await asyncio.to_thread(viranker.score, query_for_rerank, rerank_candidates)
            visual_scores = None

        fused = fusion_score.fuse(text_scores, visual_scores, rerank_candidates)

        if len(candidates) > 20:
            return fused + candidates[20:]
        return fused
    except Exception as e:
        logger.warning(f"Reranking error: {e}")
        return candidates
