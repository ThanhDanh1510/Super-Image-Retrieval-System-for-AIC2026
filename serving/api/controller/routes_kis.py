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
from serving.api.service.reranker import rerank_service
from serving.api.core.dependencies import (
    get_qdrant_client,
    get_text_embedder,
    get_image_embedder,
    get_asr_model,
)
from serving.api.core.config import get_settings

from pathlib import Path
from fastapi.responses import FileResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Retrieval"])


def _resolve_keyframe_path(video_id: str, frame_id: int, keyframe_path: str = "") -> Path | None:
    settings = get_settings()
    data_dir = Path(settings.DATA_DIR)
    frame_id_str = str(frame_id)

    parts = video_id.split("_")
    if len(parts) == 2:
        grp, vsub = parts[0], parts[1]
    else:
        grp, vsub = video_id, video_id

    candidates = [
        data_dir / "image" / grp / grp / vsub / f"{frame_id_str}.webp",
        data_dir / "image" / grp / grp / vsub / f"{frame_id_str}.jpg",
        data_dir / "image" / grp / f"{video_id}" / f"{frame_id_str}.webp",
        data_dir / "image" / f"{video_id}" / f"{frame_id_str}.webp",
        data_dir / "unified" / "keyframes" / keyframe_path if keyframe_path else data_dir,
        data_dir / keyframe_path if keyframe_path else data_dir,
    ]
    for p in candidates:
        if p and p.is_file() and p.exists():
            return p
    return None


@router.get("/image/{video_id}/{frame_id}", summary="Lấy file ảnh keyframe thực tế")
async def get_keyframe_image(video_id: str, frame_id: int):
    """
    Endpoint phục vụ ảnh keyframe thực tế trên đĩa cứng cho Frontend/UI.
    """
    img_path = _resolve_keyframe_path(video_id, frame_id)
    if not img_path or not img_path.exists():
        raise HTTPException(status_code=404, detail=f"Không tìm thấy ảnh cho video_id={video_id}, frame_id={frame_id}")
    return FileResponse(img_path)


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

        # Rerank candidates nếu có (qua seam rerank_service — controller không import reranker trực tiếp)
        candidates = pipeline_res.candidates
        if candidates:
            candidates = await rerank_service.rerank(
                query=request.query,
                candidates=candidates,
                enhanced_query=pipeline_res.enhanced_query,
                keyframes_root_dir=get_settings().DATA_DIR,
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
            candidates = await rerank_service.rerank(
                query=request.query,
                candidates=candidates,
                enhanced_query=pipeline_res.enhanced_query,
                keyframes_root_dir=get_settings().DATA_DIR,
            )

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
            # 1. Retrieve candidates song song cho từng sub-event
            import asyncio
            from serving.api.service.retrieval import text_query
            from serving.api.service import temporal_aligner
            from serving.api.repo import qdrant_repo

            # Ensure sub_events count matches request.num_events if specified
            n_events = request.num_events
            if len(sub_events) > n_events:
                sub_events = sub_events[:n_events]
            elif len(sub_events) < n_events:
                # Pad sub_events using the full query if sub_events is shorter
                while len(sub_events) < n_events:
                    sub_events.append(request.query)

            tasks = [
                text_query.process(
                    query_input=text_query.TextQueryInput(
                        raw_query=ev_text,
                        task_type=TaskType.KIS,
                        top_k=request.top_k,
                    ),
                    qdrant_client=qdrant_client,
                    text_embedder=text_embedder,
                    image_embedder=image_embedder,
                )
                for ev_text in sub_events
            ]
            outputs = await asyncio.gather(*tasks)
            event_candidates_list = [out.candidates for out in outputs]

            # Collect top candidate video_ids across all sub-events
            candidate_vids: set[str] = set()
            for cand_list in event_candidates_list:
                for c in cand_list[:20]:
                    candidate_vids.add(c.video_id)

            # Enrich keyframes for top candidate videos by querying Qdrant for full video timelines
            enrich_tasks = [
                qdrant_repo.get_keyframes_by_video_id(qdrant_client, v_id)
                for v_id in list(candidate_vids)[:15]
            ]
            enriched_results = await asyncio.gather(*enrich_tasks)
            for video_kfs in enriched_results:
                if video_kfs:
                    # Distribute enriched keyframes across event_candidates_list
                    for ev_cands in event_candidates_list:
                        existing_frames = {c.frame_id for c in ev_cands}
                        for kf in video_kfs:
                            if kf.frame_id not in existing_frames:
                                ev_cands.append(kf)

            # 2. Thuật toán DANTE Alignment
            aligned_chains = temporal_aligner.align(
                event_candidates=event_candidates_list,
                max_results=request.top_k,
            )
            results = aligned_chains
        
        if not results:
            # Fallback: Group candidates strictly by video_id
            candidates = pipeline_res.candidates
            n = request.num_events
            by_video: dict[str, list[CandidateResult]] = {}
            for cand in candidates:
                by_video.setdefault(cand.video_id, []).append(cand)

            for v_id, v_cands in by_video.items():
                # Deduplicate by timestamp_mid
                seen_ts: set[float] = set()
                uniq_v_cands = []
                for c in sorted(v_cands, key=lambda x: (x.timestamp_mid, x.frame_id)):
                    t_rounded = round(c.timestamp_mid, 2)
                    if t_rounded not in seen_ts:
                        seen_ts.add(t_rounded)
                        uniq_v_cands.append(c)

                if len(uniq_v_cands) >= n:
                    for i in range(0, len(uniq_v_cands) - n + 1):
                        results.append(uniq_v_cands[i : i + n])

        # 3. Final Guard: Enforce strict single video, exact length, and STRICTLY INCREASING timestamps (no exact duplicates)
        final_chains: list[list[CandidateResult]] = []
        n_req = request.num_events
        seen_chain_signatures: set[tuple[str, ...]] = set()

        for chain in results:
            if not chain or len(chain) != n_req:
                continue

            target_vid = chain[0].video_id
            # 1. Single video check
            if not all(c.video_id == target_vid for c in chain):
                continue

            # 2. Sort chronologically
            sorted_chain = sorted(chain, key=lambda c: (c.timestamp_mid, c.frame_id))

            # 3. Strictly increasing timestamp check (timestamps MUST NOT be exact duplicates)
            has_duplicates = False
            for i in range(len(sorted_chain) - 1):
                if round(sorted_chain[i].timestamp_mid, 2) >= round(sorted_chain[i + 1].timestamp_mid, 2):
                    has_duplicates = True
                    break

            if not has_duplicates:
                sig = tuple(f"{c.video_id}_{c.frame_id}" for c in sorted_chain)
                if sig not in seen_chain_signatures:
                    seen_chain_signatures.add(sig)
                    final_chains.append(sorted_chain)

        if len(final_chains) > request.top_k:
            final_chains = final_chains[:request.top_k]

        return TRAKEResponse(
            results=final_chains,
            total=len(final_chains),
            query_text=pipeline_res.enhanced_query or request.query,
            num_events=request.num_events,
        )
    except Exception as e:
        logger.error(f"Error in search_trake: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/search/hybrid", summary="Hybrid Search (Weighted Convex + RRF Fusion)")
async def search_hybrid(
    query_vi: str,
    query_en: str | None = None,
    mode: str = "all",
    top_k: int = 36
):
    """
    Hybrid Search Endpoint supporting weighted convex + RRF fusion
    across description (Dense BGE), OCR (Sparse BM25), and ASR (Sparse BM25).
    """
    allowed_modes = {"all", "description", "ocr", "asr"}
    if mode.lower() not in allowed_modes:
        raise HTTPException(status_code=400, detail=f"Invalid mode '{mode}'. Allowed: {allowed_modes}")
    if top_k < 1 or top_k > 200:
        raise HTTPException(status_code=400, detail="top_k must be between 1 and 200")

    try:
        from serving.api.service.retrieval.hybrid_search import get_hybrid_search_engine
        engine = get_hybrid_search_engine()
        results = engine.search(query_vi=query_vi, query_en=query_en, mode=mode.lower(), top_k=top_k)
        return {
            "mode": mode.lower(),
            "count": len(results),
            "data": results
        }
    except Exception as err:
        logger.error(f"Error in hybrid search: {err}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(err))

