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
from serving.api.service.retrieval import text_query, image_query, voice_query, sub_event_parser
from serving.api.service.retrieval.text_query import TextQueryInput, TextQueryOutput
from serving.api.service.retrieval.image_query import ImageQueryInput, ImageQueryOutput
from serving.api.service.retrieval.voice_query import VoiceQueryInput, VoiceQueryOutput
from serving.api.service.reranker import rerank_service
from serving.api.service import temporal_aligner, dense_similarity, dense_frame_extractor
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

    # 3. Áp dụng bộ lọc đa dạng hóa thời gian/slide (Temporal Diversity Filter)
    candidates = _apply_temporal_diversity_filter(candidates)

    # 4. Ghép user choices (nếu có) lên đầu danh sách
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
                    answer=getattr(choice, "answer", ""),
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


def _apply_temporal_diversity_filter(
    candidates: list[CandidateResult],
    max_frames_per_cluster: int = 2,
    time_window_sec: float = 6.0,
) -> list[CandidateResult]:
    """
    Giảm thiểu hiện tượng 'ngập lụt' các frame trùng lặp sát nhau trong cùng 1 video (ví dụ slide bài giảng lặp lại).
    Giữ tối đa `max_frames_per_cluster` frame có điểm cao nhất trong cùng cửa sổ `time_window_sec`,
    các frame còn lại được xếp về sau để đảm bảo kết quả tìm kiếm đa dạng nhiều video và bối cảnh.
    """
    if len(candidates) <= 1:
        return candidates

    selected: list[CandidateResult] = []
    overflow: list[CandidateResult] = []

    video_timestamps: dict[str, list[float]] = {}

    for cand in candidates:
        vid = cand.video_id
        t = cand.timestamp_mid if cand.timestamp_mid is not None else (cand.frame_id / 25.0)

        if vid not in video_timestamps:
            video_timestamps[vid] = [t]
            selected.append(cand)
        else:
            close_count = sum(1 for prev_t in video_timestamps[vid] if abs(prev_t - t) <= time_window_sec)
            if close_count < max_frames_per_cluster:
                video_timestamps[vid].append(t)
                selected.append(cand)
            else:
                overflow.append(cand)

    return selected + overflow


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
    Xử lý tìm kiếm chuỗi thời gian (TRAKE Precision Engine):
    1. Phân rã query thành N sub-events
    2. Retrieve candidates song song cho từng sub-event
    3. Rerank từng sub-event với ViRanker (Cross-Encoder) nếu được bật
    4. Dense Similarity Matrix: Tính True Cosine Similarity S[N, T] từ vector Qdrant
    5. Video-Level DANTE: Lấy mẫu khung hình dày đặc (0.5s/frame) cho top candidate videos có file gốc
    6. DANTE DP Monotonic Alignment với First-Occurrence Bias
    7. Strict Single Video & Monotonicity Final Guard
    """
    # 0. Tách sub-events từ raw Vietnamese query trước tiên
    if request.input_type == InputType.VOICE:
        base_res = await _run_voice(request, TaskType.TRAKE, qdrant_client, text_embedder, image_embedder, asr_model)
        raw_query = base_res.query_text or request.query
    else:
        raw_query = request.query

    sub_events = sub_event_parser.parse_sub_events(raw_query, target_num_events=request.num_events)
    if not sub_events or len(sub_events) < 2:
        try:
            sub_events = text_query._split_events(raw_query)
        except Exception:
            sub_events = [raw_query]

    results: list[list[CandidateResult]] = []

    if sub_events and len(sub_events) >= 2:
        n_events = request.num_events
        if len(sub_events) > n_events:
            sub_events = sub_events[:n_events]
        elif len(sub_events) < n_events:
            while len(sub_events) < n_events:
                sub_events.append(raw_query)

        # 1. Truy xuất song song candidates cho từng sub-event
        sub_event_top_k = max(request.top_k * 3, 150)
        tasks = [
            text_query.process(
                query_input=text_query.TextQueryInput(
                    raw_query=ev_text,
                    task_type=TaskType.KIS,
                    top_k=sub_event_top_k,
                ),
                qdrant_client=qdrant_client,
                text_embedder=text_embedder,
                image_embedder=image_embedder,
            )
            for ev_text in sub_events
        ]
        outputs = await asyncio.gather(*tasks)
        event_candidates_list = [out.candidates for out in outputs]
        sub_events_en = [out.enhanced_query for out in outputs]

        # 2. Sub-Event Reranking: Chạy ViRanker / Jina M0 nếu được bật
        use_viranker = getattr(request, "use_viranker", True)
        use_jina_m0 = getattr(request, "use_jina_m0", False)
        if (use_viranker or use_jina_m0) and event_candidates_list:
            try:
                rerank_tasks = [
                    rerank_service.rerank(
                        query=sub_events[i],
                        candidates=event_candidates_list[i],
                        enhanced_query=outputs[i].enhanced_query if i < len(outputs) else None,
                        keyframes_root_dir=str(get_settings().DATA_DIR),
                        use_viranker=use_viranker,
                        use_jina_m0=use_jina_m0,
                        model_registry=model_registry,
                    )
                    for i in range(len(sub_events))
                ]
                reranked_lists = await asyncio.gather(*rerank_tasks, return_exceptions=True)
                for i, r_res in enumerate(reranked_lists):
                    if isinstance(r_res, list):
                        event_candidates_list[i] = r_res
            except Exception as e:
                logger.warning(f"Sub-event reranking failed, proceeding with initial candidates: {e}")

        # 3. Thu thập danh sách candidate video_ids ưu tiên
        candidate_vids: list[str] = []
        seen_vids: set[str] = set()
        for cand_list in event_candidates_list:
            for c in cand_list[:30]:
                if c.video_id not in seen_vids:
                    seen_vids.add(c.video_id)
                    candidate_vids.append(c.video_id)

        # 4. Dense Similarity Matrix: Tính Cosine Similarity toàn phần từ Qdrant vectors
        max_dense_vids = max(request.top_k, 15)
        top_dense_vids = candidate_vids[:max_dense_vids]
        dense_matrices: dict[str, tuple[list[CandidateResult], np.ndarray]] = {}

        if image_embedder is not None and top_dense_vids:
            try:
                dense_matrices = await dense_similarity.build_dense_matrices_for_videos(
                    sub_events=sub_events,
                    video_ids=top_dense_vids,
                    qdrant_client=qdrant_client,
                    image_embedder=image_embedder,
                    limit_per_video=300,
                    sub_events_en=sub_events_en,
                )
            except Exception as e:
                logger.warning(f"Building dense similarity matrices failed: {e}")

        # 5. Video-Level DANTE: Trích xuất frames dày đặc cho top 2 video có sẵn file gốc
        if image_embedder is not None and top_dense_vids:
            try:
                sub_event_clip_vecs = await dense_similarity.encode_sub_events(
                    sub_events, image_embedder, sub_events_en=sub_events_en
                )
                if sub_event_clip_vecs.shape[0] > 0:
                    for vid in top_dense_vids[:2]:
                        dense_sample_res = await dense_frame_extractor.extract_and_dense_score_video(
                            video_id=vid,
                            sub_event_embeddings=sub_event_clip_vecs,
                            image_embedder=image_embedder,
                            interval_sec=1.0,
                            max_frames=150,
                        )
                        if dense_sample_res is not None:
                            dense_cands, dense_s_mat = dense_sample_res
                            dense_matrices[vid] = (dense_cands, dense_s_mat)
            except Exception as e:
                logger.warning(f"Video-level dense sampling failed: {e}")

        # 6. Timeline Enrichment cho các candidate còn lại (để không bị thiếu frame)
        enrich_vids = [v for v in candidate_vids[:max_dense_vids] if v not in dense_matrices]
        if enrich_vids:
            enrich_tasks = [
                qdrant_repo.get_keyframes_by_video_id(qdrant_client, v_id, limit=500)
                for v_id in enrich_vids
            ]
            enriched_results = await asyncio.gather(*enrich_tasks, return_exceptions=True)
            for video_kfs in enriched_results:
                if isinstance(video_kfs, list) and video_kfs:
                    for ev_cands in event_candidates_list:
                        existing_frames = {c.frame_id for c in ev_cands}
                        for kf in video_kfs:
                            if kf.frame_id not in existing_frames:
                                ev_cands.append(kf)

        # 7. DANTE DP Monotonic Alignment Engine
        aligned_chains = temporal_aligner.align(
            event_candidates=event_candidates_list,
            max_results=request.top_k * 2,
            dense_matrices=dense_matrices if dense_matrices else None,
            max_chains_per_video=5,
        )
        results = aligned_chains

    if not results or len(results) < request.top_k:
        # Fallback grouping by video_id to ensure top-K is fully populated
        all_pool: list[CandidateResult] = []
        if 'event_candidates_list' in locals() and event_candidates_list:
            for c_list in event_candidates_list:
                all_pool.extend(c_list)
        else:
            all_pool.extend(base_res.candidates)

        n = request.num_events
        by_video: dict[str, list[CandidateResult]] = {}
        for cand in all_pool:
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
                for i in range(0, min(len(uniq_v_cands) - n + 1, 5)):
                    results.append(uniq_v_cands[i : i + n])

    # 8. Final Guard: Enforce strict single video & monotonic timestamps
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
        enhanced_query=" ".join(sub_events_en) if "sub_events_en" in locals() and sub_events_en else raw_query,
        transcript=base_res.transcript if "base_res" in locals() and base_res else None,
        sub_events=sub_events,
        chains=final_chains,
    )
