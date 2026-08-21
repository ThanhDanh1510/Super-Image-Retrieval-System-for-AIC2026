"""
serving/api/controller/routes_kis.py
======================================
Controller endpoints cho KIS, VQA, TRAKE, Hybrid Search, Video & Keyframe Image serving.
Thin controller pattern: Chỉ validate request, gọi retrieval_pipeline, và trả về Response schema.
"""

from __future__ import annotations

import asyncio
import logging
import mimetypes
from pathlib import Path

import cv2
from fastapi import APIRouter, Depends, HTTPException, Header, Query
from fastapi.responses import FileResponse, Response, StreamingResponse

from serving.api.core.config import get_settings
from serving.api.core.dependencies import (
    get_asr_model,
    get_image_embedder,
    get_model_registry,
    get_qdrant_client,
    get_text_embedder,
)
from serving.api.schema.retrieval import (
    CandidateResult,
    KISRequest,
    KISResponse,
    TaskType,
    TRAKERequest,
    TRAKEResponse,
    VQARequest,
    VQAResponse,
)
from serving.api.service.retrieval import retrieval_pipeline
from shared.utils.path_resolver import resolve_keyframe_path, resolve_video_path

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Retrieval"])


# =============================================================================
# Video & Image Serving Endpoints
# =============================================================================

@router.get("/video/{video_id}", summary="Lấy file video gốc kèm hỗ trợ HTTP Range")
async def get_video_stream(video_id: str, range: str | None = Header(None)):
    """
    Endpoint phục vụ file video gốc (.mp4, .mkv, .webm) kèm chuẩn HTTP 206 Partial Content.
    """
    video_path = resolve_video_path(video_id)
    if not video_path or not video_path.exists():
        raise HTTPException(status_code=404, detail=f"Không tìm thấy video file cho video_id={video_id}")

    file_size = video_path.stat().st_size
    mime_type, _ = mimetypes.guess_type(str(video_path))
    mime_type = mime_type or "video/mp4"

    if not range:
        return FileResponse(video_path, media_type=mime_type)

    try:
        range_val = range.strip().replace("bytes=", "")
        parts = range_val.split("-")
        start = int(parts[0]) if parts[0] else 0
        end = int(parts[1]) if len(parts) > 1 and parts[1] else file_size - 1
    except ValueError:
        raise HTTPException(status_code=416, detail="Invalid Range Header")

    if start >= file_size or end >= file_size or start > end:
        raise HTTPException(
            status_code=416,
            detail="Requested range not satisfiable",
            headers={"Content-Range": f"bytes */{file_size}"},
        )

    chunk_size = end - start + 1

    def iterfile():
        with open(video_path, mode="rb") as f:
            f.seek(start)
            bytes_left = chunk_size
            while bytes_left > 0:
                read_size = min(bytes_left, 1024 * 1024)
                data = f.read(read_size)
                if not data:
                    break
                bytes_left -= len(data)
                yield data

    headers = {
        "Content-Range": f"bytes {start}-{end}/{file_size}",
        "Accept-Ranges": "bytes",
        "Content-Length": str(chunk_size),
        "Content-Type": mime_type,
    }
    return StreamingResponse(iterfile(), status_code=206, headers=headers)


@router.get("/video/{video_id}/fps", summary="Lấy FPS thực tế của video gốc")
async def get_video_fps(video_id: str):
    """
    Trả về FPS thực tế của video gốc qua OpenCV.
    """
    video_path = resolve_video_path(video_id)
    if not video_path or not video_path.exists():
        return {"video_id": video_id, "fps": 25.0, "status": "default_fallback"}

    def _read_fps():
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.release()
        return fps

    fps = await asyncio.to_thread(_read_fps)
    return {"video_id": video_id, "fps": float(fps), "status": "ok"}


@router.get("/image/{video_id}/{frame_id}", summary="Lấy file ảnh keyframe thực tế")
async def get_keyframe_image(video_id: str, frame_id: int):
    """
    Endpoint phục vụ ảnh keyframe thực tế trên đĩa cứng cho Frontend/UI.
    """
    img_path = resolve_keyframe_path(video_id, frame_id)
    if not img_path or not img_path.exists():
        raise HTTPException(status_code=404, detail=f"Không tìm thấy ảnh cho video_id={video_id}, frame_id={frame_id}")
    return FileResponse(img_path)


# =============================================================================
# Core Retrieval Endpoints
# =============================================================================

@router.post(
    "/kis",
    response_model=KISResponse,
    summary="Known-Item Search — tìm kiếm keyframe theo mô tả text/voice",
)
async def search_kis(
    request: KISRequest,
    qdrant_client=Depends(get_qdrant_client),
    text_embedder=Depends(get_text_embedder),
    image_embedder=Depends(get_image_embedder),
    asr_model=Depends(get_asr_model),
    model_registry=Depends(get_model_registry),
) -> KISResponse:
    """
    Endpoint KIS: nhận text/voice query, trả về danh sách keyframe xếp hạng.
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
            model_registry=model_registry,
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
    qdrant_client=Depends(get_qdrant_client),
    text_embedder=Depends(get_text_embedder),
    image_embedder=Depends(get_image_embedder),
    asr_model=Depends(get_asr_model),
    model_registry=Depends(get_model_registry),
) -> VQAResponse:
    """
    Endpoint VQA: nhận text/voice question, trả về danh sách keyframe liên quan.
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
            model_registry=model_registry,
        )

        answers = [getattr(c, "answer", "") for c in pipeline_res.candidates]
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
    summary="TRAKE — Temporal Retrieval & Event Alignment",
)
async def search_trake(
    request: TRAKERequest,
    qdrant_client=Depends(get_qdrant_client),
    text_embedder=Depends(get_text_embedder),
    image_embedder=Depends(get_image_embedder),
    asr_model=Depends(get_asr_model),
    model_registry=Depends(get_model_registry),
) -> TRAKEResponse:
    """
    Endpoint TRAKE: nhận mô tả chuỗi sự kiện, trả về N-tuple keyframes theo đúng thứ tự thời gian.
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
            model_registry=model_registry,
        )

        chains = pipeline_res.chains or []
        return TRAKEResponse(
            results=chains,
            total=len(chains),
            query_text=pipeline_res.enhanced_query or request.query,
            num_events=request.num_events,
        )
    except Exception as e:
        logger.error(f"Error in search_trake: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/search/hybrid",
    response_model=KISResponse,
    summary="Hybrid Search (Weighted Convex + RRF Fusion)",
)
async def search_hybrid(
    query_vi: str = Query(..., description="Query tiếng Việt"),
    query_en: str | None = Query(None, description="Query tiếng Anh (tùy chọn)"),
    mode: str = Query("all", description="Branch: all, description, ocr, asr"),
    top_k: int = Query(36, ge=1, le=200),
    model_registry=Depends(get_model_registry),
) -> KISResponse:
    """
    Hybrid Search Endpoint hỗ trợ weighted convex + RRF fusion
    across description (Dense BGE), OCR (Sparse BM25), và ASR (Sparse BM25).
    """
    allowed_modes = {"all", "description", "ocr", "asr"}
    if mode.lower() not in allowed_modes:
        raise HTTPException(status_code=400, detail=f"Invalid mode '{mode}'. Allowed: {allowed_modes}")

    try:
        engine = await model_registry.aget_hybrid_engine()
        raw_results = await asyncio.to_thread(
            engine.search, query_vi=query_vi, query_en=query_en, mode=mode.lower(), top_k=top_k
        )

        candidates = [
            CandidateResult(
                video_id=r["video_id"],
                frame_id=int(r["frame_id"]),
                scene_id=r.get("scene_id", f"{r['video_id']}_{r['frame_id']}"),
                timestamp_mid=float(r.get("timestamp_mid", 0.0)),
                timestamp_start=float(r.get("timestamp_mid", 0.0)),
                timestamp_end=float(r.get("timestamp_mid", 0.0)),
                keyframe_path=r.get("keyframe_path", ""),
                ocr_text=r.get("ocr_text", ""),
                asr_text=r.get("asr_text", ""),
                description=r.get("description", ""),
                score=float(r.get("score", 0.0)),
            )
            for r in raw_results
        ]

        return KISResponse(
            results=candidates,
            total=len(candidates),
            query_text=query_vi,
        )
    except Exception as err:
        logger.error(f"Error in hybrid search: {err}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(err))


@router.post(
    "/models/warmup/jina",
    summary="Warmup ngầm Jina Reranker M0 lên GPU",
)
async def warmup_jina_model(
    model_registry=Depends(get_model_registry),
):
    """
    Nạp sẵn Jina Reranker M0 lên GPU trong background khi người dùng gạt toggle trên UI,
    giúp lần tìm kiếm KIS/VQA đầu tiên không bị delay do load weights.
    """
    try:
        await model_registry.aget_jina_m0()
        return {"status": "ok", "message": "Jina Reranker M0 đã được nạp sẵn lên GPU"}
    except Exception as e:
        logger.error(f"Error warming up Jina M0: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

