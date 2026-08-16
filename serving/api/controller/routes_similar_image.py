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

from pathlib import Path

_MAX_IMAGE_SIZE_BYTES = 15 * 1024 * 1024
_ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp", "application/octet-stream"}
_ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@router.post(
    "/similar-image",
    response_model=ImageSearchResponse,
    summary="Image-to-Image Search — tìm keyframe tương đồng với ảnh upload",
)
async def search_similar_image(
    file: UploadFile = File(..., description="Ảnh query (JPEG/PNG/WebP, tối đa 15MB)"),
    top_k: int = Query(default=100, ge=1, le=500, description="Số kết quả trả về"),
    use_reranker: bool = Query(default=False, description="Bật/tắt reranker"),
    qdrant_client = Depends(get_qdrant_client),
    image_embedder = Depends(get_image_embedder),
) -> ImageSearchResponse:
    """
    Endpoint Image Search: nhận ảnh upload, trả về danh sách keyframe tương đồng.

    Luồng xử lý:
        1. Validate file upload (content type hoặc extension, size)
        2. Đọc bytes → encode base64
        3. Tạo ImageSearchRequest và gọi retrieval_pipeline.run()
        4. Build ImageSearchResponse và trả về
    """
    ext = Path(file.filename or "").suffix.lower()
    is_valid_type = (file.content_type in _ALLOWED_CONTENT_TYPES) or (ext in _ALLOWED_EXTENSIONS)
    if not is_valid_type:
        raise HTTPException(
            status_code=400,
            detail=f"Định dạng file không được hỗ trợ ({file.content_type}). Chỉ cho phép JPEG, PNG, WebP, BMP.",
        )

    try:
        # 2. Đọc bytes và validate kích thước
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="File ảnh rỗng.")
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
            task_type=TaskType.KIS,
            request=request,
            qdrant_client=qdrant_client,
            text_embedder=None,
            image_embedder=image_embedder,
            asr_model=None,
        )

        candidates = pipeline_res.candidates
        return ImageSearchResponse(
            results=candidates,
            total=len(candidates),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in search_similar_image: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
