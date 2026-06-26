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

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Retrieval"])


@router.post(
    "/kis",
    response_model=KISResponse,
    summary="Known-Item Search — tìm kiếm keyframe theo mô tả text/voice",
)
async def search_kis(request: KISRequest) -> KISResponse:
    """
    Endpoint KIS: nhận text query, trả về danh sách keyframe xếp hạng.

    Luồng xử lý:
        1. Validate KISRequest (Pydantic tự xử lý)
        2. Gọi retrieval_pipeline.run() với InputType tương ứng
        3. (Tùy chọn) Gọi reranker để cải thiện thứ hạng
        4. Build KISResponse và trả về

    Args:
        request: KISRequest từ frontend.

    Returns:
        KISResponse với danh sách candidates và metadata.
    """
    raise NotImplementedError


@router.post(
    "/vqa",
    response_model=VQAResponse,
    summary="Visual Question Answering — tìm cảnh và trả lời câu hỏi",
)
async def search_vqa(request: VQARequest) -> VQAResponse:
    """
    Endpoint VQA: nhận query + question, trả về danh sách kết quả kèm answer.

    Luồng xử lý:
        1. Gọi retrieval_pipeline.run() để tìm candidates
        2. Với mỗi candidate, chạy VQA model để trả lời câu hỏi
        3. Build VQAResponse với results + answers

    Args:
        request: VQARequest từ frontend.

    Returns:
        VQAResponse với danh sách candidates và câu trả lời.
    """
    raise NotImplementedError


@router.post(
    "/trake",
    response_model=TRAKEResponse,
    summary="Temporal Retrieval — tìm chuỗi sự kiện theo thứ tự thời gian",
)
async def search_trake(request: TRAKERequest) -> TRAKEResponse:
    """
    Endpoint TRAKE: nhận mô tả chuỗi sự kiện, trả về N-tuple keyframes.

    Luồng xử lý:
        1. Gọi retrieval_pipeline.run() với TaskType.TRAKE
        2. Gọi temporal_aligner để sắp xếp candidates theo thứ tự thời gian
        3. Build TRAKEResponse với danh sách chuỗi sự kiện

    Args:
        request: TRAKERequest từ frontend.

    Returns:
        TRAKEResponse với danh sách chuỗi sự kiện.
    """
    raise NotImplementedError
