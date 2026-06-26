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

        return KISResponse(
            results=pipeline_res.candidates,
            total=len(pipeline_res.candidates),
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

        # Nhóm quyết định tự trả lời bằng tay trực tiếp trên UI (Human-in-the-loop).
        # Không dùng mô hình VQA tự động trên server, trả về danh sách chuỗi rỗng.
        answers = ["" for _ in pipeline_res.candidates]

        return VQAResponse(
            results=pipeline_res.candidates,
            answers=answers,
            total=len(pipeline_res.candidates),
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
        2. Sắp xếp candidates theo thứ tự thời gian (temporal_aligner)
        3. Build TRAKEResponse với danh sách chuỗi sự kiện
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

        # Mock gom nhóm candidates thành các chuỗi sự kiện cho Tuần 1.
        # Ở Tuần 3, Researcher B sẽ cài đặt thuật toán temporal_aligner thực tế.
        results = []
        candidates = pipeline_res.candidates
        n = request.num_events
        # Tạm thời gom nhóm tuần tự n phần tử làm 1 event chain
        for i in range(0, len(candidates) - n + 1, n):
            group = candidates[i : i + n]
            results.append(group)

        # Nếu không đủ candidates, đưa toàn bộ candidates thành 1 chain nếu có ít nhất 1
        if not results and candidates:
            results.append(candidates[:n])

        return TRAKEResponse(
            results=results,
            total=len(results),
            query_text=pipeline_res.enhanced_query or request.query,
            num_events=n,
        )
    except Exception as e:
        logger.error(f"Error in search_trake: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
