"""
serving/api/service/reranker/rerank_service.py
================================================
Service layer duy nhất cho reranking — một interface đứng sau controller/pipeline.

- ViRanker (text): luôn bật, chạy GPU, rất nhẹ (~0.2s)
- Jina M0 (crossmodal): tắt mặc định, bật qua use_visual_reranker=True
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from serving.api.schema.retrieval import CandidateResult
from serving.api.service.reranker.skeletons.text_branch import viranker
from serving.api.service.reranker.skeletons.crossmodal_branch import jina_reranker_m0
from serving.api.service.reranker.skeletons import fusion_score
from serving.api.core.models import ModelRegistry

logger = logging.getLogger(__name__)


async def rerank(
    query: str,
    candidates: list[CandidateResult],
    enhanced_query: str | None = None,
    keyframes_root_dir: str = "",
    use_visual_reranker: bool = False,
    model_registry: Optional[ModelRegistry] = None,
) -> list[CandidateResult]:
    """
    Rerank candidates bằng ViRanker (text) + tùy chọn Jina M0 (crossmodal).

    Args:
        query: Query gốc từ request.
        candidates: Danh sách candidates cần rerank.
        enhanced_query: Query đã dịch/enhance.
        keyframes_root_dir: Thư mục chứa ảnh keyframe (cho Jina M0).
        use_visual_reranker: Bật Jina M0 (mặc định False).
        model_registry: Optional ModelRegistry để tận dụng model caching.

    Returns:
        Danh sách candidates đã rerank và sắp xếp theo score.
    """
    import time

    _t_start = time.perf_counter()
    if not candidates:
        return []

    query_for_rerank = enhanced_query or query

    # Chỉ rerank top 20 candidates để tăng tốc
    rerank_candidates = candidates[:20] if len(candidates) > 20 else candidates

    try:
        viranker_bundle = None
        jina_bundle = None
        if model_registry:
            viranker_bundle = await model_registry.aget_viranker()
            if use_visual_reranker:
                jina_bundle = await model_registry.aget_jina_m0()

        if use_visual_reranker:
            _t_inference_start = time.perf_counter()
            text_task = asyncio.to_thread(
                viranker.score, query_for_rerank, rerank_candidates, viranker_bundle
            )
            visual_task = asyncio.to_thread(
                jina_reranker_m0.score,
                query_for_rerank,
                rerank_candidates,
                keyframes_root_dir,
                jina_bundle,
            )
            results = await asyncio.gather(text_task, visual_task, return_exceptions=True)

            for res in results:
                if isinstance(res, Exception):
                    raise res

            text_scores, visual_scores = results
            _t_inference_end = time.perf_counter()
            logger.info(
                f"[PROFILE] Dual Reranking inference (ViRanker + Jina M0) took {(_t_inference_end - _t_inference_start) * 1000:.2f}ms"
            )
        else:
            _t_inference_start = time.perf_counter()
            text_scores = await asyncio.to_thread(
                viranker.score, query_for_rerank, rerank_candidates, viranker_bundle
            )
            visual_scores = None
            _t_inference_end = time.perf_counter()
            logger.info(
                f"[PROFILE] ViRanker inference took {(_t_inference_end - _t_inference_start) * 1000:.2f}ms"
            )

        _t_fuse_start = time.perf_counter()
        fused = fusion_score.fuse(text_scores, visual_scores, rerank_candidates)
        _t_fuse_end = time.perf_counter()
        logger.info(
            f"[PROFILE] Score fusion & sorting took {(_t_fuse_end - _t_fuse_start) * 1000:.2f}ms"
        )

        _t_end = time.perf_counter()
        logger.info(
            f"[PROFILE] Total rerank process took {(_t_end - _t_start) * 1000:.2f}ms"
        )

        if len(candidates) > 20:
            return fused + candidates[20:]
        return fused
    except Exception as e:
        logger.warning(f"Reranking error: {e}")
        return candidates
