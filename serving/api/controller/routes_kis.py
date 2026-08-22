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
import os
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


@router.get("/frame/{video_id}/{frame_id}", summary="Lấy metadata chi tiết cho 1 frame (phục vụ kiểm tra đáp án nhanh)")
async def get_frame_metadata(video_id: str, frame_id: int):
    """
    Tra cứu metadata (description, OCR, ASR, timestamps) cho 1 frame cụ thể.
    Ưu tiên lookup từ VideoContextIndex (parquet unified), fallback về thông tin tối thiểu.

    Frontend Inspector sẽ gọi endpoint này để hiển thị đúng thông tin khi mở video bằng tay:
    Nhập "video_id,frame_id" → mở video ngay lập tức kèm metadata nếu có.
    """
    # 1. Thử lookup qua VideoContextIndex (in-memory parquet)
    try:
        from serving.api.service.retrieval.video_context_service import get_video_context_index

        idx = get_video_context_index()
        if idx.ensure_loaded():
            v_data = idx.video_contexts.get(video_id)
            if v_data:
                for sc in v_data.get("scenes", []):
                    try:
                        fid = int(sc.get("frame_id", -1))
                    except Exception:
                        continue
                    if fid == int(frame_id):
                        # Build CandidateResult JSON
                        return {
                            "video_id": str(sc.get("video_id", video_id)),
                            "scene_id": str(sc.get("scene_id", f"{video_id}_{frame_id}")),
                            "frame_id": int(sc.get("frame_id", frame_id)),
                            "timestamp_start": float(sc.get("timestamp_start", 0.0)),
                            "timestamp_end": float(sc.get("timestamp_end", 0.0)),
                            "timestamp_mid": float(sc.get("timestamp_mid", 0.0)),
                            "keyframe_path": str(sc.get("keyframe_path", "")),
                            "ocr_text": str(sc.get("ocr_text", "")),
                            "asr_text": str(sc.get("asr_text", "")),
                            "description": str(sc.get("description", "")),
                            "score": 1.0,
                            "exists_video": bool(resolve_video_path(video_id) is not None),
                            "exists_image": bool(resolve_keyframe_path(video_id, frame_id) is not None),
                        }
    except Exception as e:
        logger.debug(f"get_frame_metadata fallback lookup failed: {e}")

    # 2. Fallback: kiểm tra file tồn tại và trả về metadata tối thiểu
    img_path = resolve_keyframe_path(video_id, frame_id)
    vid_path = resolve_video_path(video_id)
    # Thử lấy fps để suy ra timestamp
    fps = 25.0
    if vid_path and vid_path.exists():
        try:
            cap = cv2.VideoCapture(str(vid_path))
            f = cap.get(cv2.CAP_PROP_FPS) or 25.0
            cap.release()
            if f and f > 0:
                fps = float(f)
        except Exception:
            pass
    ts = float(frame_id) / fps if fps > 0 else 0.0
    if not img_path and not vid_path:
        # Không tìm thấy cả video lẫn ảnh nhưng vẫn trả về synthetic để frontend vẫn mở được (sẽ báo lỗi trong player)
        logger.warning(f"get_frame_metadata: không tìm thấy file cho {video_id},{frame_id} — trả về synthetic")
    return {
        "video_id": video_id,
        "scene_id": f"{video_id}_{frame_id}",
        "frame_id": int(frame_id),
        "timestamp_start": max(0.0, ts - 0.5),
        "timestamp_end": ts + 0.5,
        "timestamp_mid": ts,
        "keyframe_path": "",
        "ocr_text": "",
        "asr_text": "",
        "description": "",
        "score": 1.0,
        "exists_video": bool(vid_path and vid_path.exists()),
        "exists_image": bool(img_path and img_path.exists()),
        "fps": fps,
    }


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


# =============================================================================
# Video Timeline & Smart QA Submission Endpoints
# =============================================================================

@router.get(
    "/videos/{video_id}/timeline",
    summary="Lấy toàn bộ dòng thời gian OCR & ASR cho video đang xem",
)
async def get_video_timeline(
    video_id: str,
    qdrant_client=Depends(get_qdrant_client),
):
    """
    Truy xuất toàn bộ danh sách keyframe, OCR text và ASR text theo thứ tự thời gian (timestamps)
    cho 1 video cụ thể, phục vụ tính năng OCR & ASR Inspector Drawer trên Web UI.
    """
    try:
        from qdrant_client.http import models as qmodels

        # Scroll points từ AsyncQdrantClient
        points, _ = await qdrant_client.scroll(
            collection_name="production",
            scroll_filter=qmodels.Filter(
                must=[qmodels.FieldCondition(key="video_id", match=qmodels.MatchValue(value=video_id))]
            ),
            limit=500,
            with_payload=True,
            with_vectors=False,
        )

        ocr_events = []
        asr_events = []
        keyframe_events = []

        seen_frames = set()
        for p in points:
            payload = p.payload or {}
            fid = int(payload.get("frame_id", 0))
            if fid in seen_frames:
                continue
            seen_frames.add(fid)

            ts_mid = float(payload.get("timestamp_mid", fid / 30.0))
            ocr_text = payload.get("ocr_text", "").strip()
            asr_text = payload.get("asr_text", "").strip()
            desc = payload.get("description", "").strip()
            kpath = payload.get("keyframe_path", "")

            keyframe_events.append({
                "frame_id": fid,
                "timestamp": ts_mid,
                "keyframe_path": kpath,
                "description": desc,
            })

            if ocr_text:
                ocr_events.append({
                    "frame_id": fid,
                    "timestamp": ts_mid,
                    "text": ocr_text,
                })

            if asr_text:
                asr_events.append({
                    "frame_id": fid,
                    "timestamp": ts_mid,
                    "text": asr_text,
                })

        # Sắp xếp tăng dần theo frame_id
        ocr_events.sort(key=lambda x: x["frame_id"])
        asr_events.sort(key=lambda x: x["frame_id"])
        keyframe_events.sort(key=lambda x: x["frame_id"])

        return {
            "video_id": video_id,
            "total_keyframes": len(keyframe_events),
            "ocr_events": ocr_events,
            "asr_events": asr_events,
            "keyframe_events": keyframe_events,
        }
    except Exception as err:
        logger.error(f"Error fetching timeline for video {video_id}: {err}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(err))


@router.post(
    "/submission/smart-save",
    summary="Lưu nhanh câu trả lời Q&A hoặc KIS kèm tự động sinh Top 30 frames lân cận",
)
async def smart_save_submission(
    payload: dict,
):
    """
    Nhận dữ liệu từ Smart QA / KIS Input Widget trên Web UI, tự động chuẩn hóa Unicode NFC,
    bọc dấu ngoặc kép chuẩn BTC, sinh mốc lân cận +-10, +-15, +-20... và cập nhật file CSV + ZIP.
    """
    import unicodedata
    import zipfile

    query_id = payload.get("query_id", "query-1-kis")
    task_type = payload.get("task_type", "kis").lower()
    video_id = payload.get("video_id", "").strip()
    frame_id = int(payload.get("frame_id", 0))
    raw_answer = payload.get("answer", "").strip()

    if not video_id or frame_id <= 0:
        raise HTTPException(status_code=400, detail="video_id và frame_id hợp lệ là bắt buộc")

    # Chuẩn hóa Unicode NFC
    clean_ans = unicodedata.normalize("NFC", raw_answer)
    if clean_ans.startswith('"') and clean_ans.endswith('"') and len(clean_ans) >= 2:
        clean_ans = clean_ans[1:-1].strip()

    # Bọc quote cho CSV
    quoted_ans = f'"{clean_ans}"'

    # Offsets lân cận (không có +-5, bắt đầu +-10)
    offsets = [10, -10, 15, -15, 20, -20, 25, -25, 30, -30, 40, -40, 50, -50, 60, -60, 75, -75, 90, -90, 100, -100, 120, -120, 150, -150]

    lines = []
    seen = set()

    if task_type == "qa" or task_type == "vqa":
        base_line = f"{video_id},{frame_id},{quoted_ans}"
        lines.append(base_line)
        seen.add(base_line)

        for off in offsets:
            if len(lines) >= 30:
                break
            cand_f = frame_id + off
            if cand_f > 0:
                cand_l = f"{video_id},{cand_f},{quoted_ans}"
                if cand_l not in seen:
                    seen.add(cand_l)
                    lines.append(cand_l)
    elif task_type == "trake":
        # TRAKE sequence
        e1 = int(payload.get("e1", frame_id))
        e2 = int(payload.get("e2", frame_id + 300))
        e3 = int(payload.get("e3", frame_id + 600))
        base_line = f"{video_id},{e1},{e2},{e3}"
        lines.append(base_line)
        seen.add(base_line)

        for off in [10, -10, 15, -15, 20, -20, 25, -25, 30, -30, 40, -40]:
            for var in [
                (e1 + off, e2, e3), (e1, e2 + off, e3), (e1, e2, e3 + off),
                (e1 + off, e2 + off, e3 + off), (e1 - off, e2 - off, e3 - off)
            ]:
                if len(lines) >= 30:
                    break
                if var[0] < var[1] < var[2] and var[0] > 0:
                    comb = f"{video_id},{var[0]},{var[1]},{var[2]}"
                    if comb not in seen:
                        seen.add(comb)
                        lines.append(comb)
    else:
        # KIS
        base_line = f"{video_id},{frame_id}"
        lines.append(base_line)
        seen.add(base_line)

        for off in offsets:
            if len(lines) >= 30:
                break
            cand_f = frame_id + off
            if cand_f > 0:
                cand_l = f"{video_id},{cand_f}"
                if cand_l not in seen:
                    seen.add(cand_l)
                    lines.append(cand_l)

    # Đảm bảo tên file chuẩn
    filename = f"{query_id}.csv" if not query_id.endswith(".csv") else query_id

    # Lưu vào cả D:\submission và D:\submission\submission nếu có
    saved_paths = []
    for base_dir in [r"D:\submission\submission", r"D:\submission"]:
        if os.path.exists(base_dir):
            target_p = os.path.join(base_dir, filename)
            with open(target_p, "w", encoding="utf-8", newline="\n") as fp:
                fp.write("\n".join(lines) + "\n")
            saved_paths.append(target_p)

    # Auto repack zip D:\submission.zip nếu thư mục D:\submission tồn tại
    zip_p = r"D:\submission.zip"
    src_dir = r"D:\submission\submission" if os.path.exists(r"D:\submission\submission") else r"D:\submission"
    if os.path.exists(src_dir):
        with zipfile.ZipFile(zip_p, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
            for f in sorted(os.listdir(src_dir)):
                f_full = os.path.join(src_dir, f)
                if os.path.isfile(f_full) and f.endswith(".csv"):
                    zipf.write(f_full, arcname=f"submission/{f}")

    return {
        "status": "success",
        "query_id": query_id,
        "filename": filename,
        "total_lines": len(lines),
        "top_line": lines[0] if lines else "",
        "saved_paths": saved_paths,
        "zip_updated": os.path.exists(zip_p),
    }


