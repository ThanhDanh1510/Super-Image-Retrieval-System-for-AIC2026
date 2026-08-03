"""
serving/api/controller/routes_kis.py
=====================================
FastAPI Router cho các endpoint KIS, VQA, và TRAKE.

Đây là điểm tiếp nhận HTTP request từ frontend, validate input,
gọi retrieval_pipeline, và trả về response chuẩn.

IMPORT RULES:
    - Gọi: serving/api/service/retrieval/retrieval_pipeline.py
    - Gọi: serving/api/core/config.py (lấy settings)
    - KHÔNG gọi qdrant_repo trực tiếp — đó là việc của service layer.

ENDPOINTS:
    POST /api/v1/kis    → Known-Item Search
    POST /api/v1/vqa    → Visual Question Answering
    POST /api/v1/trake  → Temporal Retrieval (multi-event)
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from serving.api.schema.retrieval import (
    InputType,
    TaskType,
    CandidateResult,
    KISRequest,
    KISResponse,
    VQARequest,
    VQAResponse,
    TRAKERequest,
    TRAKEResponse,
)
from serving.api.service.retrieval import retrieval_pipeline
from serving.api.core.dependencies import (
    get_qdrant_client,
    get_text_embedder,
    get_image_embedder,
    get_asr_model,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Retrieval"])


@router.post(
    "/kis",
    response_model=KISResponse,
    summary="Known-Item Search — tìm kiếm keyframe theo mô tả text/voice",
)
async def search_kis(
    request: KISRequest,
    qdrant_client = Depends(get_qdrant_client),
    text_embedder = Depends(get_text_embedder),
    image_embedder = Depends(get_image_embedder),
    asr_model = Depends(get_asr_model),
) -> KISResponse:
    """
    Endpoint KIS: nhận text query, trả về danh sách keyframe xếp hạng.

    Luồng xử lý:
        1. Validate KISRequest (Pydantic tự xử lý)
        2. Gọi retrieval_pipeline.run() với InputType tương ứng
        3. Build KISResponse và trả về
    """
    try:
        pipeline_res = await retrieval_pipeline.run(
            input_type=request.input_type,
            task_type=TaskType.KIS,
            request=request,
            qdrant_client=qdrant_client,
            text_embedder=text_embedder,
            image_embedder=image_embedder,
            asr_model=asr_model,
        )

        # Merge logic: Ghép các frame người dùng đã tự chọn bằng tay lên đầu
        final_results = []
        seen = set()

        # 1. Thêm các lựa chọn thủ công của người dùng trước
        for choice in request.user_choices:
            key = (choice.video_id, choice.frame_id)
            seen.add(key)
            final_results.append(
                CandidateResult(
                    video_id=choice.video_id,
                    frame_id=choice.frame_id,
                    scene_id=choice.scene_id,
                    timestamp_start=choice.timestamp_start,
                    timestamp_end=choice.timestamp_end,
                    timestamp_mid=choice.timestamp_mid,
                    keyframe_path=choice.keyframe_path,
                    ocr_text=choice.ocr_text,
                    asr_text=choice.asr_text,
                    score=1.0,  # Điểm số tối đa cho frame tự chọn
                )
            )

        # Rerank candidates nếu có
        candidates = pipeline_res.candidates
        if candidates:
            try:
                from serving.api.service.reranker.skeletons.text_branch import viranker
                from serving.api.service.reranker.skeletons.crossmodal_branch import jina_reranker_m0
                from serving.api.service.reranker.skeletons import fusion_score

                query_for_rerank = pipeline_res.enhanced_query or request.query
                text_scores = viranker.score(query_for_rerank, candidates)
                crossmodal_scores = jina_reranker_m0.score(query_for_rerank, candidates)
                candidates = fusion_score.fuse(text_scores, crossmodal_scores, candidates)
            except Exception as re_err:
                logger.warning(f"Reranking error (falling back to RRF scores): {re_err}")

        # Merge logic: Ghép các frame người dùng đã tự chọn bằng tay lên đầu
        final_results = []
        seen = set()

        # 1. Thêm các lựa chọn thủ công của người dùng trước
        for choice in request.user_choices:
            key = (choice.video_id, choice.frame_id)
            seen.add(key)
            final_results.append(
                CandidateResult(
                    video_id=choice.video_id,
                    frame_id=choice.frame_id,
                    scene_id=choice.scene_id,
                    timestamp_start=choice.timestamp_start,
                    timestamp_end=choice.timestamp_end,
                    timestamp_mid=choice.timestamp_mid,
                    keyframe_path=choice.keyframe_path,
                    ocr_text=choice.ocr_text,
                    asr_text=choice.asr_text,
                    score=1.0,  # Điểm số tối đa cho frame tự chọn
                )
            )

        # 2. Bổ sung nốt kết quả từ AI (Reranking) cho đủ top_k và tránh trùng lặp
        for cand in candidates:
            if len(final_results) >= request.top_k:
                break
            key = (cand.video_id, cand.frame_id)
            if key not in seen:
                seen.add(key)
                final_results.append(cand)

        return KISResponse(
            results=final_results,
            total=len(final_results),
            query_text=pipeline_res.enhanced_query or request.query,
        )
    except Exception as e:
        logger.error(f"Error in search_kis: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/vqa",
    response_model=VQAResponse,
    summary="Visual Question Answering — tìm cảnh và trả lời câu hỏi",
)
async def search_vqa(
    request: VQARequest,
    qdrant_client = Depends(get_qdrant_client),
    text_embedder = Depends(get_text_embedder),
    image_embedder = Depends(get_image_embedder),
    asr_model = Depends(get_asr_model),
) -> VQAResponse:
    """
    Endpoint VQA: nhận query + question, trả về danh sách kết quả kèm answer.

    Luồng xử lý:
        1. Gọi retrieval_pipeline.run() để tìm candidates
        2. Với mỗi candidate, sinh ra câu trả lời cho câu hỏi (VQA)
        3. Build VQAResponse với results + answers
    """
    try:
        pipeline_res = await retrieval_pipeline.run(
            input_type=request.input_type,
            task_type=TaskType.VQA,
            request=request,
            qdrant_client=qdrant_client,
            text_embedder=text_embedder,
            image_embedder=image_embedder,
            asr_model=asr_model,
        )

        candidates = pipeline_res.candidates
        if candidates:
            try:
                from serving.api.service.reranker.skeletons.text_branch import viranker
                from serving.api.service.reranker.skeletons.crossmodal_branch import jina_reranker_m0
                from serving.api.service.reranker.skeletons import fusion_score

                query_for_rerank = pipeline_res.enhanced_query or request.query
                text_scores = viranker.score(query_for_rerank, candidates)
                crossmodal_scores = jina_reranker_m0.score(query_for_rerank, candidates)
                candidates = fusion_score.fuse(text_scores, crossmodal_scores, candidates)
            except Exception as re_err:
                logger.warning(f"Reranking error in VQA: {re_err}")

        # Merge logic cho VQA: Ghép lựa chọn thủ công (kèm đáp án) lên đầu
        final_results = []
        answers = []
        seen = set()

        # 1. Thêm các lựa chọn thủ công của người dùng trước
        for choice in request.user_choices:
            key = (choice.video_id, choice.frame_id)
            seen.add(key)
            final_results.append(
                CandidateResult(
                    video_id=choice.video_id,
                    frame_id=choice.frame_id,
                    scene_id=choice.scene_id,
                    timestamp_start=choice.timestamp_start,
                    timestamp_end=choice.timestamp_end,
                    timestamp_mid=choice.timestamp_mid,
                    keyframe_path=choice.keyframe_path,
                    ocr_text=choice.ocr_text,
                    asr_text=choice.asr_text,
                    score=1.0,
                )
            )
            answers.append(choice.answer)

        # 2. Bổ sung nốt kết quả từ AI, gán câu trả lời trống ""
        for cand in candidates:
            if len(final_results) >= request.top_k:
                break
            key = (cand.video_id, cand.frame_id)
            if key not in seen:
                seen.add(key)
                final_results.append(cand)
                answers.append("")  # Để trống câu trả lời để người dùng tự gõ trên UI nếu cần

        return VQAResponse(
            results=final_results,
            answers=answers,
            total=len(final_results),
            query_text=pipeline_res.enhanced_query or request.query,
        )
    except Exception as e:
        logger.error(f"Error in search_vqa: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/trake",
    response_model=TRAKEResponse,
    summary="Temporal Retrieval — tìm chuỗi sự kiện theo thứ tự thời gian",
)
async def search_trake(
    request: TRAKERequest,
    qdrant_client = Depends(get_qdrant_client),
    text_embedder = Depends(get_text_embedder),
    image_embedder = Depends(get_image_embedder),
    asr_model = Depends(get_asr_model),
) -> TRAKEResponse:
    """
    Endpoint TRAKE: nhận mô tả chuỗi sự kiện, trả về N-tuple keyframes.

    Luồng xử lý:
        1. Gọi retrieval_pipeline.run() với TaskType.TRAKE
        2. Tách sub-events và retrieve candidates cho từng sub-event
        3. Căn chỉnh thứ tự thời gian bằng thuật toán DANTE DP (temporal_aligner)
        4. Build TRAKEResponse với danh sách chuỗi sự kiện
    """
    try:
        pipeline_res = await retrieval_pipeline.run(
            input_type=request.input_type,
            task_type=TaskType.TRAKE,
            request=request,
            qdrant_client=qdrant_client,
            text_embedder=text_embedder,
            image_embedder=image_embedder,
            asr_model=asr_model,
        )

        sub_events = pipeline_res.sub_events
        results = []

        if sub_events and len(sub_events) >= 2:
            # 1. Retrieve candidates cho từng sub-event
            from serving.api.service.retrieval import text_query
            from serving.api.service import temporal_aligner

            event_candidates_list = []
            for ev_text in sub_events:
                ev_input = text_query.TextQueryInput(
                    raw_query=ev_text,
                    task_type=TaskType.KIS,
                    top_k=request.top_k,
                )
                ev_output = await text_query.process(
                    query_input=ev_input,
                    qdrant_client=qdrant_client,
                    text_embedder=text_embedder,
                    image_embedder=image_embedder,
                )
                event_candidates_list.append(ev_output.candidates)

            # 2. Thuật toán DANTE Alignment
            aligned_chains = temporal_aligner.align(
                event_candidates=event_candidates_list,
                max_results=request.top_k,
            )
            results = aligned_chains
        
        if not results:
            # Fallback nếu không tách được sub-events hoặc DANTE không tìm thấy chuỗi
            candidates = pipeline_res.candidates
            n = request.num_events
            for i in range(0, len(candidates) - n + 1, n):
                results.append(candidates[i : i + n])
            if not results and candidates:
                results.append(candidates[:n])

        return TRAKEResponse(
            results=results,
            total=len(results),
            query_text=pipeline_res.enhanced_query or request.query,
            num_events=request.num_events,
        )
    except Exception as e:
        logger.error(f"Error in search_trake: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

