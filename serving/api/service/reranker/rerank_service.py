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
    use_viranker: bool = True,
    use_jina_m0: bool = False,
    use_visual_reranker: bool = False,
    model_registry: Optional[ModelRegistry] = None,
) -> list[CandidateResult]:
    """
    Rerank candidates bằng ViRanker (text) và/hoặc Jina M0 (crossmodal).

    Args:
        query: Query gốc từ request.
        candidates: Danh sách candidates cần rerank.
        enhanced_query: Query đã dịch/enhance.
        keyframes_root_dir: Thư mục chứa ảnh keyframe (cho Jina M0).
        use_viranker: Bật ViRanker text cross-encoder (mặc định True).
        use_jina_m0: Bật Jina M0 visual cross-encoder (mặc định False).
        use_visual_reranker: Tương thích ngược (alias cho use_jina_m0).
        model_registry: Optional ModelRegistry để tận dụng model caching.

    Returns:
        Danh sách candidates đã rerank và sắp xếp theo score.
    """
    import time

    # Tương thích ngược: nếu truyền use_visual_reranker=True -> use_jina_m0=True
    if use_visual_reranker:
        use_jina_m0 = True

    # Nếu cả 2 cờ đều tắt -> không rerank, trả về ngay lập tức (<1ms)
    if not use_viranker and not use_jina_m0:
        return candidates

    _t_start = time.perf_counter()
    if not candidates:
        return []

    viranker_query = query if query else (enhanced_query or "")
    jina_query = enhanced_query or query

    # Chỉ rerank top candidates để tăng tốc và đảm bảo độ bao phủ (ViRanker: 50, Jina M0: 25)
    rerank_limit = 25 if use_jina_m0 else 50
    rerank_candidates = candidates[:rerank_limit] if len(candidates) > rerank_limit else candidates

    try:
        viranker_bundle = None
        jina_bundle = None
        if model_registry:
            if use_viranker:
                viranker_bundle = await model_registry.aget_viranker()
            if use_jina_m0:
                jina_bundle = await model_registry.aget_jina_m0()

        text_scores = None
        visual_scores = None

        if use_viranker and use_jina_m0:
            _t_inference_start = time.perf_counter()
            text_task = asyncio.to_thread(
                viranker.score, viranker_query, rerank_candidates, viranker_bundle, enhanced_query
            )
            visual_task = asyncio.to_thread(
                jina_reranker_m0.score,
                jina_query,
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
        elif use_viranker:
            _t_inference_start = time.perf_counter()
            text_scores = await asyncio.to_thread(
                viranker.score, viranker_query, rerank_candidates, viranker_bundle, enhanced_query
            )
            _t_inference_end = time.perf_counter()
            logger.info(
                f"[PROFILE] ViRanker Text inference took {(_t_inference_end - _t_inference_start) * 1000:.2f}ms"
            )
        elif use_jina_m0:
            _t_inference_start = time.perf_counter()
            visual_scores = await asyncio.to_thread(
                jina_reranker_m0.score,
                jina_query,
                rerank_candidates,
                keyframes_root_dir,
                jina_bundle,
            )
            _t_inference_end = time.perf_counter()
            logger.info(
                f"[PROFILE] Jina M0 Visual inference took {(_t_inference_end - _t_inference_start) * 1000:.2f}ms"
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

        if len(candidates) > len(rerank_candidates):
            return fused + candidates[len(rerank_candidates):]
        return fused
    except Exception as e:
        logger.warning(f"Reranking error: {e}")
        return candidates
