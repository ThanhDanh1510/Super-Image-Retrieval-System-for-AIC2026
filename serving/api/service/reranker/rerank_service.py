"""
serving/api/service/reranker/rerank_service.py
================================================
Service layer duy nhất cho reranking — một interface đứng sau controller.

TRƯỚC (leakage): controller routes_kis.py import trực tiếp viranker,
jina_reranker_m0, fusion_score và duplicate cùng một block rerank ở cả
search_kis lẫn search_vqa — vi phạm layering controller → service → repo.

SAU (deep seam): controller gọi `rerank_service.rerank()`. Viranker (text)
và Jina M0 (crossmodal) chạy song song rồi fusion; nếu bất kỳ nhánh nào lỗi
thì fallback về candidates gốc (score RRF) — không bao giờ 500.

CONSTRAINTS:
  - Lazy load model: viranker/jina tự lazy-load model bên trong score().
    Import module không khởi tạo model (chỉ định nghĩa hàm).
  - Batch inference: xử lý toàn bộ candidates trong 1 forward pass mỗi nhánh.
  - Layering: module này chỉ gọi reranker skeletons + schema — không gọi repo.
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
    use_visual_reranker: bool = False,
) -> list[CandidateResult]:
    """
    Rerank candidates:
      - Luôn luôn chạy ViRanker (text) vì rất nhẹ.
      - Chạy thêm Jina M0 (crossmodal/visual) nếu use_visual_reranker = True.

    Args:
      - query: Query gốc từ request.
      - candidates: Danh sách candidates cần rerank.
      - enhanced_query: Query đã dịch/enhance.
      - keyframes_root_dir: Thư mục chứa ảnh keyframe.
      - use_visual_reranker: Bật/Tắt nhánh rerank hình ảnh Jina (mặc định False).
    """
    import time
    _t_start = time.time()
    if not candidates:
        return []

    query_for_rerank = enhanced_query or query

    # Chỉ rerank top 20 candidates để tăng tốc
    rerank_candidates = candidates[:20] if len(candidates) > 20 else candidates

    try:
        if use_visual_reranker:
            # Chạy song song cả hai nhánh với return_exceptions=True
            text_task = asyncio.to_thread(viranker.score, query_for_rerank, rerank_candidates)
            visual_task = asyncio.to_thread(
                jina_reranker_m0.score, 
                query_for_rerank, 
                rerank_candidates, 
                keyframes_root_dir
            )
            
            results = await asyncio.gather(text_task, visual_task, return_exceptions=True)
            
            # Kiểm tra nếu bất kỳ task nào trả về lỗi
            for res in results:
                if isinstance(res, Exception):
                    raise res
                    
            text_scores, visual_scores = results
        else:
            # Chỉ chạy nhánh ViRanker (text) rất nhẹ
            text_scores = await asyncio.to_thread(viranker.score, query_for_rerank, rerank_candidates)
            visual_scores = None
            
        _t1 = time.time()
        logger.info(f"[PROFILE] rerank (visual={use_visual_reranker}) took: {_t1 - _t_start:.2f}s")
        
        # Fusion scores
        fused = fusion_score.fuse(text_scores, visual_scores, rerank_candidates)
        
        if len(candidates) > 20:
            return fused + candidates[20:]
        return fused
    except Exception as e:
        logger.warning(f"Reranking error: {e}")
        return candidates
