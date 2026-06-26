"""
serving/api/controller/routes_similar_image.py
===============================================
FastAPI Router cho Image-to-Image Search (tìm ảnh tương đồng).

ENDPOINTS:
    POST /api/v1/similar-image  → Tìm keyframe tương đồng với ảnh upload

IMPORT RULES:
    - Gọi: serving/api/service/retrieval/retrieval_pipeline.py
    - KHÔNG gọi qdrant_repo trực tiếp.
"""

from __future__ import annotations

import base64
import logging

from fastapi import APIRouter, File, UploadFile, HTTPException, Query

from serving.api.schema.retrieval import (
    InputType,
    TaskType,
    ImageSearchRequest,
    ImageSearchResponse,
    CandidateResult,
)
from serving.api.service.retrieval import retrieval_pipeline

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Similar Image"])

# Giới hạn kích thước ảnh upload (10MB)
_MAX_IMAGE_SIZE_BYTES = 10 * 1024 * 1024
_ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}


@router.post(
    "/similar-image",
    response_model=ImageSearchResponse,
    summary="Image-to-Image Search — tìm keyframe tương đồng với ảnh upload",
)
async def search_similar_image(
    file: UploadFile = File(..., description="Ảnh query (JPEG/PNG/WebP, tối đa 10MB)"),
    top_k: int = Query(default=100, ge=1, le=100, description="Số kết quả trả về"),
) -> ImageSearchResponse:
    """
    Endpoint Image Search: nhận ảnh upload, trả về danh sách keyframe tương đồng.

    Luồng xử lý:
        1. Validate file upload (content type, size)
        2. Đọc bytes → encode base64
        3. Tạo ImageSearchRequest và gọi retrieval_pipeline.run()
        4. Build ImageSearchResponse và trả về

    Args:
        file: Ảnh upload từ frontend (multipart/form-data).
        top_k: Số kết quả trả về. Tối đa 100.

    Returns:
        ImageSearchResponse với danh sách keyframe tương đồng.

    Raises:
        HTTPException 400: Nếu file không phải ảnh hoặc quá lớn.
        HTTPException 500: Nếu xử lý ảnh thất bại.
    """
    raise NotImplementedError
