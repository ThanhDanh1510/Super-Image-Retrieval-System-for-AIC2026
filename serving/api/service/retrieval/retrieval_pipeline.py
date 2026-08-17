"""
serving/api/service/retrieval/retrieval_pipeline.py
====================================================
Orchestrator trung tâm — nhận query từ controller, điều hướng sang đúng module,
thực hiện Reranking, TRAKE Temporal Alignment, và merge user choices.

TRÁCH NHIỆM CỦA FILE NÀY:
  - Nhận InputType (text/voice/image) và TaskType (KIS/VQA/TRAKE).
  - Điều phối: text_query / voice_query / image_query.
  - Tự động thực hiện reranking (ViRanker + tùy chọn Jina M0) qua rerank_service.
  - Tự động thực hiện TRAKE sub-event decomposition, timeline enrichment và DANTE alignment.
  - Hợp nhất và ưu tiên các lựa chọn thủ công của người dùng (user_choices).
  - Trả về kết quả cuối cùng hoàn chỉnh cho Controller.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Optional

from serving.api.schema.retrieval import (
    CandidateResult,
    InputType,
    TaskType,
    KISRequest,
    VQARequest,
    TRAKERequest,
    ImageSearchRequest,
    UserChoice,
)
from serving.api.service.retrieval import text_query, image_query, voice_query
from serving.api.service.retrieval.text_query import TextQueryInput, TextQueryOutput
from serving.api.service.retrieval.image_query import ImageQueryInput, ImageQueryOutput
from serving.api.service.retrieval.voice_query import VoiceQueryInput, VoiceQueryOutput
from serving.api.service.reranker import rerank_service
from serving.api.service import temporal_aligner
from serving.api.repo import qdrant_repo
from serving.api.core.config import get_settings

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class PipelineResult:
    """
    Kết quả trả về từ pipeline orchestrator.
    """
    candidates: list[CandidateResult]
    """Danh sách candidates xếp hạng (dùng cho KIS / VQA / Image search)."""

    task_type: TaskType
    """TaskType gốc."""

    enhanced_query: str | None = None
    """Query sau khi enhance/dịch."""

    transcript: str | None = None
    """Transcript ASR (với voice query)."""

    sub_events: list[str] | None = None
    """Sub-events đã tách (với TRAKE)."""

    chains: list[list[CandidateResult]] | None = None
    """Danh sách chuỗi sự kiện aligned (với TRAKE)."""


# =============================================================================
# Core Orchestrator Function
# =============================================================================

async def run(
    input_type: InputType,
    task_type: TaskType,
    request: KISRequest | VQARequest | TRAKERequest | ImageSearchRequest,
    qdrant_client,
    text_embedder,
    image_embedder,
    asr_model,
    model_registry=None,
) -> PipelineResult:
    """
    Entry point duy nhất cho toàn bộ retrieval pipeline.
    """
    logger.info(f"Pipeline run: input_type={input_type}, task_type={task_type}")

    if task_type == TaskType.TRAKE and isinstance(request, TRAKERequest):
        return await _run_trake(
            request=request,
            qdrant_client=qdrant_client,
            text_embedder=text_embedder,
            image_embedder=image_embedder,
            asr_model=asr_model,
            model_registry=model_registry,
        )

    # 1. Thu thập candidates thô theo input type
    if input_type == InputType.TEXT:
        pipeline_res = await _run_text(request, task_type, qdrant_client, text_embedder, image_embedder)
    elif input_type == InputType.VOICE:
        pipeline_res = await _run_voice(request, task_type, qdrant_client, text_embedder, image_embedder, asr_model)
    elif input_type == InputType.IMAGE:
        pipeline_res = await _run_image(request, qdrant_client, image_embedder)
    else:
        raise ValueError(f"Unsupported input_type: {input_type}")

    # 2. Thực hiện Rerank nếu có candidates, là input TEXT/VOICE có câu query rõ ràng
    candidates = pipeline_res.candidates
    if candidates and input_type in (InputType.TEXT, InputType.VOICE) and task_type in (TaskType.KIS, TaskType.VQA):
        raw_query = getattr(request, "query", "")
        if raw_query and raw_query.strip():
            use_viranker = getattr(request, "use_viranker", True)
            use_jina_m0 = getattr(request, "use_jina_m0", False)
            use_reranker = getattr(request, "use_reranker", None)
            if use_reranker is not None:
                if use_reranker is False:
                    use_viranker = False
                    use_jina_m0 = False
                elif use_reranker is True:
                    use_jina_m0 = True

            candidates = await rerank_service.rerank(
                query=raw_query,
                candidates=candidates,
                enhanced_query=pipeline_res.enhanced_query,
                keyframes_root_dir=str(get_settings().DATA_DIR),
                use_viranker=use_viranker,
                use_jina_m0=use_jina_m0,
                model_registry=model_registry,
            )

    # 3. Ghép user choices (nếu có) lên đầu danh sách
    user_choices = getattr(request, "user_choices", [])
    top_k = getattr(request, "top_k", 100)
    final_candidates = _merge_user_choices(user_choices, candidates, top_k)

    pipeline_res.candidates = final_candidates
    return pipeline_res


# =============================================================================
# Helper: User Choice Merging
# =============================================================================

def _merge_user_choices(
    user_choices: list[UserChoice],
    candidates: list[CandidateResult],
    top_k: int,
) -> list[CandidateResult]:
    """Hợp nhất các frame người dùng đã chọn bằng tay lên đầu danh sách kết quả."""
    final_results: list[CandidateResult] = []
    seen: set[tuple[str, int]] = set()

    for choice in user_choices:
        key = (choice.video_id, choice.frame_id)
        if key not in seen:
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

    for cand in candidates:
        if len(final_results) >= top_k:
            break
        key = (cand.video_id, cand.frame_id)
        if key not in seen:
            seen.add(key)
            final_results.append(cand)

    return final_results


# =============================================================================
# Private Routing Functions
# =============================================================================

async def _run_text(
    request,
    task_type: TaskType,
    qdrant_client,
    text_embedder,
    image_embedder,
) -> PipelineResult:
    raw_query = getattr(request, "query", "")
    top_k = getattr(request, "top_k", 100)
    search_mode = getattr(request, "search_mode", "all")
    retrieve_k = max(top_k * 3, 50)

    query_input = TextQueryInput(
        raw_query=raw_query,
        task_type=task_type,
        top_k=retrieve_k,
        search_mode=search_mode,
    )

    output: TextQueryOutput = await text_query.process(
        query_input=query_input,
        qdrant_client=qdrant_client,
        text_embedder=text_embedder,
        image_embedder=image_embedder,
    )

    return PipelineResult(
        candidates=output.candidates,
        task_type=task_type,
        enhanced_query=output.enhanced_query,
        sub_events=output.sub_events,
    )


async def _run_voice(
    request,
    task_type: TaskType,
    qdrant_client,
    text_embedder,
    image_embedder,
    asr_model,
) -> PipelineResult:
    import base64

    query_data = getattr(request, "query", b"")
    if isinstance(query_data, str):
        try:
            audio_bytes = base64.b64decode(query_data)
        except Exception:
            audio_bytes = query_data.encode("utf-8")
    elif isinstance(query_data, bytes):
        audio_bytes = query_data
    else:
        audio_bytes = bytes(query_data)

    audio_format = getattr(request, "audio_format", "wav")
    top_k = getattr(request, "top_k", 100)

    voice_input = VoiceQueryInput(
        audio_bytes=audio_bytes,
        audio_format=audio_format,
        task_type=task_type,
        top_k=top_k,
    )

    output: VoiceQueryOutput = await voice_query.process(
        query_input=voice_input,
        qdrant_client=qdrant_client,
        text_embedder=text_embedder,
        image_embedder=image_embedder,
        asr_model=asr_model,
    )

    return PipelineResult(
        candidates=output.candidates,
        task_type=task_type,
        enhanced_query=output.enhanced_query,
        transcript=output.transcript,
        sub_events=output.sub_events,
    )


async def _run_image(
    request: ImageSearchRequest,
    qdrant_client,
    image_embedder,
) -> PipelineResult:
    import base64

    image_base64 = getattr(request, "image_base64", "")
    if not image_base64:
        raise ValueError("image_base64 trong request không được rỗng")

    image_bytes = base64.b64decode(image_base64)
    top_k = getattr(request, "top_k", 100)

    image_input = ImageQueryInput(
        image_bytes=image_bytes,
        top_k=top_k,
    )

    output: ImageQueryOutput = await image_query.process(
        query_input=image_input,
        qdrant_client=qdrant_client,
        image_embedder=image_embedder,
    )

    return PipelineResult(
        candidates=output.candidates,
        task_type=TaskType.KIS,
        enhanced_query=None,
        transcript=None,
        sub_events=None,
    )


# =============================================================================
# TRAKE Orchestration
# =============================================================================

async def _run_trake(
    request: TRAKERequest,
    qdrant_client,
    text_embedder,
    image_embedder,
    asr_model,
    model_registry=None,
) -> PipelineResult:
    """
    Xử lý tìm kiếm chuỗi thời gian (TRAKE):
    1. Phân rã query thành sub-events
    2. Retrieve candidates song song cho từng sub-event
    3. Enrich video timeline
    4. DANTE Alignment thuật toán Dynamic Programming
    """
    if request.input_type == InputType.VOICE:
        base_res = await _run_voice(request, TaskType.TRAKE, qdrant_client, text_embedder, image_embedder, asr_model)
    else:
        base_res = await _run_text(request, TaskType.TRAKE, qdrant_client, text_embedder, image_embedder)

    sub_events = base_res.sub_events
    results: list[list[CandidateResult]] = []

    if sub_events and len(sub_events) >= 2:
        n_events = request.num_events
        if len(sub_events) > n_events:
            sub_events = sub_events[:n_events]
        elif len(sub_events) < n_events:
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

        # Thu thập top video_ids để enrich
        candidate_vids: set[str] = set()
        for cand_list in event_candidates_list:
            for c in cand_list[:20]:
                candidate_vids.add(c.video_id)

        enrich_tasks = [
            qdrant_repo.get_keyframes_by_video_id(qdrant_client, v_id)
            for v_id in list(candidate_vids)[:15]
        ]
        enriched_results = await asyncio.gather(*enrich_tasks)
        for video_kfs in enriched_results:
            if video_kfs:
                for ev_cands in event_candidates_list:
                    existing_frames = {c.frame_id for c in ev_cands}
                    for kf in video_kfs:
                        if kf.frame_id not in existing_frames:
                            ev_cands.append(kf)

        aligned_chains = temporal_aligner.align(
            event_candidates=event_candidates_list,
            max_results=request.top_k,
        )
        results = aligned_chains

    if not results:
        # Fallback grouping by video_id
        candidates = base_res.candidates
        n = request.num_events
        by_video: dict[str, list[CandidateResult]] = {}
        for cand in candidates:
            by_video.setdefault(cand.video_id, []).append(cand)

        for v_id, v_cands in by_video.items():
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

    # Final Guard: Enforce strict single video & monotonic timestamps
    final_chains: list[list[CandidateResult]] = []
    n_req = request.num_events
    seen_chain_signatures: set[tuple[str, ...]] = set()

    for chain in results:
        if not chain or len(chain) != n_req:
            continue

        target_vid = chain[0].video_id
        if not all(c.video_id == target_vid for c in chain):
            continue

        sorted_chain = sorted(chain, key=lambda c: (c.timestamp_mid, c.frame_id))
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

    return PipelineResult(
        candidates=[],
        task_type=TaskType.TRAKE,
        enhanced_query=base_res.enhanced_query,
        transcript=base_res.transcript,
        sub_events=sub_events,
        chains=final_chains,
    )
