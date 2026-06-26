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

from fastapi import APIRouter, File, UploadFile, HTTPException, Query, Depends

from serving.api.schema.retrieval import (
    InputType,
    TaskType,
    ImageSearchRequest,
    ImageSearchResponse,
    CandidateResult,
)
from serving.api.service.retrieval import retrieval_pipeline
from serving.api.core.dependencies import get_qdrant_client, get_image_embedder

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
    qdrant_client = Depends(get_qdrant_client),
    image_embedder = Depends(get_image_embedder),
) -> ImageSearchResponse:
    """
    Endpoint Image Search: nhận ảnh upload, trả về danh sách keyframe tương đồng.

    Luồng xử lý:
        1. Validate file upload (content type, size)
        2. Đọc bytes → encode base64
        3. Tạo ImageSearchRequest và gọi retrieval_pipeline.run()
        4. Build ImageSearchResponse và trả về
    """
    # 1. Validate content type
    if file.content_type not in _ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Định dạng file không được hỗ trợ. Chỉ cho phép các định dạng: {', '.join(_ALLOWED_CONTENT_TYPES)}",
        )

    try:
        # 2. Đọc bytes và validate kích thước
        contents = await file.read()
        if len(contents) > _MAX_IMAGE_SIZE_BYTES:
            raise HTTPException(
                status_code=400,
                detail=f"Kích thước ảnh vượt quá giới hạn cho phép ({_MAX_IMAGE_SIZE_BYTES / (1024*1024):.1f}MB).",
            )

        # Encode base64
        image_base64 = base64.b64encode(contents).decode("utf-8")

        # 3. Tạo request và gọi retrieval pipeline
        request = ImageSearchRequest(image_base64=image_base64, top_k=top_k)

        pipeline_res = await retrieval_pipeline.run(
            input_type=InputType.IMAGE,
            task_type=TaskType.KIS,  # Image search mặc định thuộc dạng KIS
            request=request,
            qdrant_client=qdrant_client,
            text_embedder=None,
            image_embedder=image_embedder,
            asr_model=None,
        )

        # 4. Trả về response
        return ImageSearchResponse(
            results=pipeline_res.candidates,
            total=len(pipeline_res.candidates),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in search_similar_image: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
