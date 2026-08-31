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
import math
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
import time
from serving.api.service.retrieval import text_query, image_query, voice_query, sub_event_parser
from serving.api.service.retrieval.text_query import TextQueryInput, TextQueryOutput
from serving.api.service.retrieval.image_query import ImageQueryInput, ImageQueryOutput
from serving.api.service.retrieval.voice_query import VoiceQueryInput, VoiceQueryOutput
from serving.api.service.retrieval.dedup_filter import deduplicate_candidates
from serving.api.core.query_cache import get_query_cache
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

    stage_timings: dict[str, float] | None = None
    """Chi tiết thời gian các chặng (ms) từ text_query."""

    metrics: dict[str, Any] | None = None
    """Metrics hiệu năng trả về cho client / UI."""


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
    _t_start = time.perf_counter()
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

    # 0. Query Cache Lookup (chỉ áp dụng cho KIS/VQA text search khi không có user choices)
    user_choices = getattr(request, "user_choices", [])
    top_k = getattr(request, "top_k", 100)
    raw_query = getattr(request, "query", "")

    query_cache = None
    cache_key = None
    if not user_choices and input_type == InputType.TEXT and raw_query and raw_query.strip():
        query_cache = get_query_cache()
        cache_key = query_cache.make_key(
            query=raw_query,
            search_mode=getattr(request, "search_mode", "all"),
            top_k=top_k,
            use_viranker=getattr(request, "use_viranker", True),
            use_jina_m0=getattr(request, "use_jina_m0", False),
            task_type=task_type.value if hasattr(task_type, "value") else str(task_type),
        )
        cached_res = query_cache.get(cache_key)
        if cached_res is not None:
            _total_ms = (time.perf_counter() - _t_start) * 1000
            cached_res.metrics = {
                "cached": True,
                "total_ms": round(_total_ms, 1),
                "candidates_count": len(cached_res.candidates),
            }
            logger.info(f"⚡ [CACHE HIT] Query='{raw_query}' returned from QueryCache in {_total_ms:.2f}ms ({len(cached_res.candidates)} candidates)")
            return cached_res

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
    _rerank_ms = 0.0
    if candidates and input_type in (InputType.TEXT, InputType.VOICE) and task_type in (TaskType.KIS, TaskType.VQA):
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

            _t_rerank = time.perf_counter()
            candidates = await rerank_service.rerank(
                query=raw_query,
                candidates=candidates,
                enhanced_query=pipeline_res.enhanced_query,
                keyframes_root_dir=str(get_settings().DATA_DIR),
                use_viranker=use_viranker,
                use_jina_m0=use_jina_m0,
                model_registry=model_registry,
            )
            _rerank_ms = (time.perf_counter() - _t_rerank) * 1000

    # 2.05 Áp dụng Temporal Scene-Coherence Score Smoothing (Làm mịn điểm theo phân cảnh)
    try:
        from shared.utils.config_loader import load_fusion_config
        _kis_cfg = load_fusion_config().get("kis_accuracy", {})
    except Exception:
        _kis_cfg = {}

    if candidates and _kis_cfg.get("enable_temporal_smoothing", True):
        sigma = float(_kis_cfg.get("temporal_smooth_sigma", 8.0))
        weight = float(_kis_cfg.get("temporal_smooth_weight", 0.20))
        candidates = _apply_temporal_scene_smoothing(candidates, sigma_sec=sigma, weight=weight)

    # 2.1 Áp dụng Temporal Halo Boost nếu có user_choices từ phiên làm việc
    if user_choices and candidates:
        candidates = _apply_temporal_halo_boost(user_choices, candidates, halo_radius_sec=10.0, boost_factor=1.25)

    # 3. Áp dụng Deduplication toàn diện (Blacklist 14k frame rác, Temporal NMS, Diversity Capping)
    candidates = deduplicate_candidates(
        candidates=candidates,
        top_k=top_k,
        max_per_video=3,
        time_window_sec=5.0,
        is_trake=False,
    )

    # 4. Ghép user choices (nếu có) lên đầu danh sách
    final_candidates = _merge_user_choices(user_choices, candidates, top_k)

    pipeline_res.candidates = final_candidates

    # Ghi nhận phản hồi feedback không đồng bộ
    if user_choices:
        asyncio.create_task(asyncio.to_thread(_log_user_feedback_sync, raw_query, user_choices, final_candidates))

    _total_ms = (time.perf_counter() - _t_start) * 1000
    st = getattr(pipeline_res, "stage_timings", None) or {}
    pipeline_res.metrics = {
        "cached": False,
        "enhance_query_ms": st.get("enhance_query_ms", 0.0),
        "encode_desc_ms": st.get("encode_desc_ms", 0.0),
        "qdrant_search_ms": st.get("qdrant_search_ms", 0.0),
        "rerank_ms": round(_rerank_ms, 1),
        "total_ms": round(_total_ms, 1),
        "candidates_count": len(final_candidates),
    }

    # Lưu cache nếu thỏa điều kiện
    if query_cache is not None and cache_key and pipeline_res:
        query_cache.set(cache_key, pipeline_res)

    logger.info(
        f"[METRICS] Task={task_type} | Input={input_type} | Query='{raw_query[:40]}' | "
        f"Took={_total_ms:.1f}ms | Returned={len(final_candidates)} candidates"
    )

    return pipeline_res


def _apply_temporal_scene_smoothing(
    candidates: list[CandidateResult],
    sigma_sec: float = 8.0,
    weight: float = 0.20,
) -> list[CandidateResult]:
    """
    Làm mịn điểm theo phân cảnh video (1D Gaussian Temporal Kernel).
    Tăng điểm cho các khung hình nằm trong cụm phân cảnh có nhiều frame điểm cao,
    và giảm độ ưu tiên của các frame nhiễu đơn lẻ.
    """
    if not candidates or len(candidates) < 2:
        return candidates

    import numpy as np

    by_video: dict[str, list[CandidateResult]] = {}
    for c in candidates:
        by_video.setdefault(c.video_id, []).append(c)

    smoothed_candidates: list[CandidateResult] = []
    for vid, v_cands in by_video.items():
        if len(v_cands) == 1:
            smoothed_candidates.append(v_cands[0])
            continue

        times = np.array([c.timestamp_mid for c in v_cands], dtype=np.float32)
        scores = np.array([c.score for c in v_cands], dtype=np.float32)

        dt_matrix = np.abs(times[:, None] - times[None, :])
        kernel = np.exp(-dt_matrix / max(sigma_sec, 1.0))
        np.fill_diagonal(kernel, 0.0)

        row_sums = np.sum(kernel, axis=1) + 1e-5
        neighbor_support = np.dot(kernel, scores) / row_sums
        new_scores = scores + weight * neighbor_support

        for c, sc in zip(v_cands, new_scores):
            smoothed_candidates.append(c.model_copy(update={"score": round(float(sc), 4)}))

    smoothed_candidates.sort(key=lambda x: x.score, reverse=True)
    return smoothed_candidates


def _apply_temporal_halo_boost(
    user_choices: list[UserChoice],
    candidates: list[CandidateResult],
    halo_radius_sec: float = 10.0,
    boost_factor: float = 1.25,
) -> list[CandidateResult]:
    """
    Tăng điểm cho các frame nằm trong vùng lân cận thời gian [t - 10s, t + 10s]
    của cùng video mà người dùng đã click chọn (User Feedback Temporal Halo Boost).
    """
    if not user_choices or not candidates:
        return candidates

    halo_ranges: dict[str, list[tuple[float, float]]] = {}
    for choice in user_choices:
        vid = choice.video_id
        t = choice.timestamp_mid if choice.timestamp_mid is not None else (float(choice.frame_id) / 25.0)
        halo_ranges.setdefault(vid, []).append((t - halo_radius_sec, t + halo_radius_sec))

    boosted: list[CandidateResult] = []
    for cand in candidates:
        vid = cand.video_id
        if vid in halo_ranges:
            t = cand.timestamp_mid if cand.timestamp_mid is not None else (float(cand.frame_id) / 25.0)
            in_halo = any(start <= t <= end for start, end in halo_ranges[vid])
            if in_halo:
                boosted.append(cand.model_copy(update={"score": cand.score * boost_factor}))
                continue
        boosted.append(cand)

    boosted.sort(key=lambda x: x.score, reverse=True)
    return boosted


def _log_user_feedback_sync(query: str, user_choices: list[UserChoice], top_candidates: list[CandidateResult]):
    """Ghi log phản hồi người dùng xuống file JSONL để đánh giá offline."""
    if not user_choices:
        return
    try:
        import json
        from pathlib import Path
        log_file = Path("data/user_feedback.jsonl")
        log_file.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "query": query,
            "user_choices": [{"video_id": c.video_id, "frame_id": c.frame_id} for c in user_choices],
            "top_candidates": [{"video_id": c.video_id, "frame_id": c.frame_id, "score": c.score} for c in top_candidates[:10]],
        }
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass

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
        stage_timings=output.stage_timings,
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
    # 0. Tách preamble (bối cảnh tổng quan mở đầu video) và sub-events từ raw Vietnamese query
    if request.input_type == InputType.VOICE:
        base_res = await _run_voice(request, TaskType.TRAKE, qdrant_client, text_embedder, image_embedder, asr_model)
        raw_query = getattr(base_res, "transcript", None) or getattr(base_res, "enhanced_query", None) or request.query
    else:
        raw_query = request.query

    # Đọc cấu hình Temporal Aligner từ fusion_config.yaml
    temporal_cfg: dict[str, Any] = {}
    try:
        from shared.utils.config_loader import load_fusion_config
        fusion_cfg = load_fusion_config()
        temporal_cfg = fusion_cfg.get("temporal_aligner", {})
    except Exception:
        pass

    intro_fusion_weight = float(temporal_cfg.get("intro_fusion_weight", 0.35))
    preamble_time_cutoff = float(temporal_cfg.get("preamble_time_cutoff", 35.0))
    max_dense_videos = int(temporal_cfg.get("max_dense_videos", 5))
    dense_sampling_interval_sec = float(temporal_cfg.get("dense_sampling_interval_sec", 0.8))
    max_dense_frames = int(temporal_cfg.get("max_dense_frames", 60))
    top_video_hedging_count = int(temporal_cfg.get("top_video_hedging_count", 3))
    max_hedging_chains_per_video = int(temporal_cfg.get("max_hedging_chains_per_video", 3))
    hedging_min_temporal_gap = float(temporal_cfg.get("hedging_min_temporal_gap", 2.0))

    preamble, sub_events = sub_event_parser.extract_preamble_and_sub_events(raw_query, target_num_events=request.num_events)
    if not sub_events or len(sub_events) < 2:
        try:
            sub_events = text_query._split_events(raw_query)
        except Exception:
            sub_events = [raw_query]

    temporal_cues = sub_event_parser.extract_temporal_cues(sub_events, raw_query)

    results: list[list[CandidateResult]] = []

    if sub_events and len(sub_events) >= 2:
        n_events = request.num_events
        if len(sub_events) > n_events:
            sub_events = sub_events[:n_events]
        elif len(sub_events) < n_events:
            while len(sub_events) < n_events:
                sub_events.append(raw_query)

        # 1. Trích xuất Global Theme cốt lõi và truy xuất song song candidates cho từng sub-event
        clean_theme = sub_event_parser.extract_clean_theme(preamble) if preamble else ""
        sub_event_top_k = min(max(request.top_k * 3, 50), 100)
        query_items = list(sub_events)
        if clean_theme:
            query_items.append(clean_theme)

        # ⚡ Tối ưu Batch Enhance: Dịch toàn bộ query_items trong 1 forward pass GPU duy nhất
        # truyền context=clean_theme để làm giàu ngữ cảnh cho từng sub-event
        batch_enhanced = await text_query.batch_enhance_queries(query_items, context=clean_theme)

        tasks = [
            text_query.process(
                query_input=text_query.TextQueryInput(
                    raw_query=q_text,
                    task_type=TaskType.KIS,
                    top_k=sub_event_top_k,
                    context=clean_theme if q_text != clean_theme else None,
                    pre_enhanced_en=b_en,
                    pre_enhanced_vi=b_vi,
                ),
                qdrant_client=qdrant_client,
                text_embedder=text_embedder,
                image_embedder=image_embedder,
            )
            for q_text, (b_en, b_vi) in zip(query_items, batch_enhanced)
        ]
        outputs = await asyncio.gather(*tasks)
        event_candidates_list = [out.candidates for out in outputs[:len(sub_events)]]
        sub_events_en = [out.enhanced_query for out in outputs[:len(sub_events)]]
        theme_candidates = outputs[len(sub_events)].candidates if clean_theme and len(outputs) > len(sub_events) else []

        # 2. Sub-Event Reranking: Chạy Unified Batch ViRanker nếu được bật
        use_viranker = getattr(request, "use_viranker", True)
        use_jina_m0 = getattr(request, "use_jina_m0", False)
        if (use_viranker or use_jina_m0) and event_candidates_list:
            try:
                event_candidates_list = await rerank_service.batch_rerank_sub_events(
                    sub_events=sub_events,
                    event_candidates_list=event_candidates_list,
                    enhanced_queries=sub_events_en,
                    keyframes_root_dir=str(get_settings().DATA_DIR),
                    use_viranker=use_viranker,
                    use_jina_m0=use_jina_m0,
                    model_registry=model_registry,
                    limit_per_event=15,
                )
            except Exception as e:
                logger.warning(f"Sub-event batch reranking failed, proceeding with initial candidates: {e}")

        # 3. Video Identity (Tầng 1): RRF + Global Context Theme Matching
        # 3.1 Global Context Theme Score (Không áp dụng time decay - Preamble là chủ đề toàn cục)
        intro_score_by_vid: dict[str, float] = {}
        if theme_candidates:
            for c in theme_candidates:
                intro_score_by_vid[c.video_id] = max(intro_score_by_vid.get(c.video_id, 0.0), float(c.score))

        # 3.2 Reciprocal Rank Fusion (RRF) trên tất cả các nhánh (Mục 5)
        rrf_k = float(temporal_cfg.get("rrf_k", 60.0))
        rrf_scores_by_vid: dict[str, float] = {}

        all_branches: list[list[CandidateResult]] = []
        if theme_candidates:
            all_branches.append(theme_candidates)
        all_branches.extend(event_candidates_list)

        for branch_cands in all_branches:
            vid_best_sc: dict[str, float] = {}
            for c in branch_cands:
                vid_best_sc[c.video_id] = max(vid_best_sc.get(c.video_id, 0.0), float(c.score))

            sorted_branch_vids = sorted(vid_best_sc.keys(), key=lambda v: vid_best_sc[v], reverse=True)
            for rank_idx, v_id in enumerate(sorted_branch_vids):
                rrf_val = 1.0 / (rrf_k + float(rank_idx + 1))
                rrf_scores_by_vid[v_id] = rrf_scores_by_vid.get(v_id, 0.0) + rrf_val

        # ⚡ Multi-Event Consensus Boost: Tăng điểm mạnh mẽ cho các video bao phủ nhiều sub-events
        video_event_coverage: dict[str, set[int]] = {}
        for ev_idx, branch_cands in enumerate(event_candidates_list):
            for c in branch_cands[:80]:
                video_event_coverage.setdefault(c.video_id, set()).add(ev_idx)

        total_ev = max(1, len(event_candidates_list))
        for v_id, covered in video_event_coverage.items():
            coverage_ratio = len(covered) / total_ev
            if coverage_ratio >= 0.50:
                boost = 1.0 + (0.8 * coverage_ratio)  # Boost tối đa 1.8x nếu bao phủ 100% events
                rrf_scores_by_vid[v_id] = rrf_scores_by_vid.get(v_id, 0.0) * boost

        # Tổng hợp điểm: RRF + w_intro * IntroScore
        combined_vid_scores: list[tuple[str, float]] = []
        all_candidate_vids_set = set(rrf_scores_by_vid.keys()) | set(intro_score_by_vid.keys())
        for v_id in all_candidate_vids_set:
            tot_sc = rrf_scores_by_vid.get(v_id, 0.0) + (intro_fusion_weight * intro_score_by_vid.get(v_id, 0.0))
            combined_vid_scores.append((v_id, tot_sc))

        combined_vid_scores.sort(key=lambda x: x[1], reverse=True)
        candidate_vids: list[str] = [v for v, _ in combined_vid_scores]

        # 3.3 Adaptive Top-K Video Candidates via Elbow Detection
        min_vids = min(10, len(combined_vid_scores))
        max_limit = min(25, max(request.top_k + 5, 15), len(combined_vid_scores))
        elbow_drop_ratio = float(temporal_cfg.get("elbow_drop_ratio", 0.65))

        adaptive_k = max_limit
        for k_idx in range(min_vids, max_limit):
            prev_sc = combined_vid_scores[k_idx - 1][1]
            curr_sc = combined_vid_scores[k_idx][1]
            if prev_sc > 1e-6 and (curr_sc / prev_sc) < elbow_drop_ratio:
                adaptive_k = k_idx
                break

        max_dense_vids = max(min_vids, adaptive_k)
        top_dense_vids = candidate_vids[:max_dense_vids]
        dense_matrices: dict[str, tuple[list[CandidateResult], np.ndarray]] = {}

        _t_dense_start = time.perf_counter()
        if image_embedder is not None and top_dense_vids:
            try:
                dense_matrices = await dense_similarity.build_dense_matrices_for_videos(
                    sub_events=sub_events,
                    video_ids=top_dense_vids,
                    qdrant_client=qdrant_client,
                    image_embedder=image_embedder,
                    limit_per_video=150,
                    sub_events_en=sub_events_en,
                )
            except Exception as e:
                logger.warning(f"Building dense similarity matrices failed: {e}")
        _t_dense_end = time.perf_counter()
        logger.info(
            f"[TRAKE_PROFILE] Step 4 (build_dense_matrices for {len(top_dense_vids)} vids) took {(_t_dense_end - _t_dense_start) * 1000:.2f}ms"
        )

        # 5. Video-Level DANTE: Trích xuất frames dày đặc cho Top candidate videos (ROI window + CUDA batching)
        _t_sample_start = time.perf_counter()
        if image_embedder is not None and top_dense_vids:
            try:
                sub_event_clip_vecs = await dense_similarity.encode_sub_events(
                    sub_events, image_embedder, sub_events_en=sub_events_en
                )
                if sub_event_clip_vecs.shape[0] > 0:
                    vid_candidate_ts: dict[str, list[float]] = {}
                    for ev_cands in event_candidates_list:
                        for c in ev_cands:
                            vid_candidate_ts.setdefault(c.video_id, []).append(c.timestamp_mid)

                    async def _extract_one_video(vid: str):
                        _t_v_start = time.perf_counter()
                        try:
                            res = await dense_frame_extractor.extract_and_dense_score_video(
                                video_id=vid,
                                sub_event_embeddings=sub_event_clip_vecs,
                                image_embedder=image_embedder,
                                interval_sec=dense_sampling_interval_sec,
                                max_frames=max_dense_frames,
                                candidate_timestamps=vid_candidate_ts.get(vid, []),
                            )
                            _t_v_end = time.perf_counter()
                            if res is not None:
                                logger.info(
                                    f"[TRAKE_PROFILE] Step 5 dense_frame_extractor on {vid} ({len(res[0])} frames) took {(_t_v_end - _t_v_start) * 1000:.2f}ms"
                                )
                            return vid, res
                        except Exception as e:
                            logger.debug(f"Dense extractor failed for {vid}: {e}")
                            return vid, None

                    dense_tasks = [_extract_one_video(v) for v in top_dense_vids[:max_dense_videos]]
                    dense_results = await asyncio.gather(*dense_tasks, return_exceptions=True)
                    for item in dense_results:
                        if isinstance(item, tuple) and item[1] is not None:
                            vid, (dense_cands, dense_sim) = item
                            if vid in dense_matrices:
                                exist_cands, exist_sim = dense_matrices[vid]
                                all_cands = list(exist_cands)
                                exist_frames = {c.frame_id for c in exist_cands}
                                new_dense_cands = []
                                new_dense_indices = []
                                for d_idx, dc in enumerate(dense_cands):
                                    if dc.frame_id not in exist_frames:
                                        new_dense_cands.append(dc)
                                        new_dense_indices.append(d_idx)
                                if new_dense_indices:
                                    all_cands.extend(new_dense_cands)
                                    merged_sim = np.concatenate([exist_sim, dense_sim[:, new_dense_indices]], axis=1)
                                    sort_idx = np.argsort([c.timestamp_mid for c in all_cands])
                                    sorted_cands = [all_cands[i] for i in sort_idx]
                                    sorted_sim = merged_sim[:, sort_idx]
                                    dense_matrices[vid] = (sorted_cands, sorted_sim)
                            else:
                                dense_matrices[vid] = (dense_cands, dense_sim)
            except Exception as e:
                logger.warning(f"Video-level dense sampling failed: {e}")
        _t_sample_end = time.perf_counter()
        logger.info(f"[TRAKE_PROFILE] Step 5 total video-level extraction took {(_t_sample_end - _t_sample_start) * 1000:.2f}ms")

        # 6. Timeline Enrichment cho các candidate còn lại (để không bị thiếu frame)
        _t_enrich_start = time.perf_counter()
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
        _t_enrich_end = time.perf_counter()
        logger.info(f"[TRAKE_PROFILE] Step 6 (enrichment for {len(enrich_vids)} vids) took {(_t_enrich_end - _t_enrich_start) * 1000:.2f}ms")

        # 7. DANTE DP Monotonic Alignment Engine (Cho phép tối đa max_hedging_chains_per_video chuỗi/video)
        _t_dp_start = time.perf_counter()
        aligned_chains = temporal_aligner.align(
            event_candidates=event_candidates_list,
            max_results=request.top_k * 3,
            dense_matrices=dense_matrices if dense_matrices else None,
            max_chains_per_video=max(max_hedging_chains_per_video, 3),
            temporal_cues=temporal_cues,
        )
        _t_dp_end = time.perf_counter()
        logger.info(f"[TRAKE_PROFILE] Step 7 (DANTE DP aligner) took {(_t_dp_end - _t_dp_start) * 1000:.2f}ms, returned {len(aligned_chains)} chains")
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

            # Chỉ chấp nhận chuỗi nếu các frame liên tiếp thỏa mãn 0.2s <= gap <= 15.0s
            for start_c in range(len(uniq_v_cands) - n + 1):
                subseq = uniq_v_cands[start_c:start_c + n]
                if all(0.2 <= subseq[k + 1].timestamp_mid - subseq[k].timestamp_mid <= 15.0 for k in range(n - 1)):
                    results.append(subseq)
                    break

    # 8. Final Guard: Enforce strict single video, monotonic timestamps, Multi-Chain Hedging & Intro Fusion
    scored_chains: list[tuple[float, list[CandidateResult]]] = []
    n_req = request.num_events
    seen_chain_signatures: set[tuple[str, ...]] = set()

    for chain in results:
        if not chain or len(chain) != n_req:
            continue

        target_vid = chain[0].video_id
        if not all(c.video_id == target_vid for c in chain):
            continue

        sorted_chain = sorted(chain, key=lambda c: (c.timestamp_mid, c.frame_id))
        is_strictly_valid = True
        for i in range(len(sorted_chain) - 1):
            gap = sorted_chain[i + 1].timestamp_mid - sorted_chain[i].timestamp_mid
            # RÀNG BUỘC CỐT LÕI AIC2026: 0.2s <= khoảng cách giữa 2 sự kiện liên tiếp <= 15.0s
            if gap < 0.2 or gap > 15.0:
                is_strictly_valid = False
                break

        if is_strictly_valid:
            sig = tuple(f"{c.video_id}_{c.frame_id}" for c in sorted_chain)
            if sig not in seen_chain_signatures:
                seen_chain_signatures.add(sig)

                # Tính điểm tổng hợp: Fused DP similarity + Intro Match + Multi-Event Consensus Boost
                dp_avg_score = sum(c.score for c in sorted_chain) / float(len(sorted_chain))
                intro_score = intro_score_by_vid.get(target_vid, 0.0) if 'intro_score_by_vid' in locals() else 0.0

                if clean_theme and ('intro_score_by_vid' in locals()) and intro_score_by_vid:
                    dp_weight = 1.0 - intro_fusion_weight
                    base_rank_score = dp_weight * dp_avg_score + intro_fusion_weight * intro_score
                else:
                    base_rank_score = dp_avg_score

                # Multi-Event Consensus: Boost mạnh chuỗi từ video bao phủ đa số các sub-events
                coverage_cnt = len(video_event_coverage.get(target_vid, set())) if 'video_event_coverage' in locals() else 0
                cov_ratio = coverage_cnt / float(n_req) if n_req > 0 else 0.0
                consensus_multiplier = 1.0 + (0.5 * cov_ratio) if cov_ratio >= 0.50 else 1.0
                final_rank_score = base_rank_score * consensus_multiplier

                scored_chains.append((final_rank_score, sorted_chain))

    logger.info(f"[TRAKE_DEBUG] results count: {len(results)}, scored_chains count: {len(scored_chains)}")
    # Sắp xếp lại theo điểm kết hợp giảm dần
    scored_chains.sort(key=lambda x: x[0], reverse=True)

    # Multi-Chain Hedging & Temporal Overlap Suppression
    # Xác định các video ứng viên top đầu được phép hedging (nhiều chuỗi)
    hedging_vids = set(candidate_vids[:top_video_hedging_count]) if 'candidate_vids' in locals() and candidate_vids else set()

    accepted_chains_by_vid: dict[str, list[list[CandidateResult]]] = {}
    final_chains: list[list[CandidateResult]] = []

    for score, sorted_chain in scored_chains:
        target_vid = sorted_chain[0].video_id
        existing_chains = accepted_chains_by_vid.get(target_vid, [])

        max_allowed_chains = max_hedging_chains_per_video if target_vid in hedging_vids else 1
        if len(existing_chains) >= max_allowed_chains:
            continue

        if existing_chains:
            new_ts = [c.timestamp_mid for c in sorted_chain]
            new_frames = [c.frame_id for c in sorted_chain]
            is_distinct = True

            for old_chain in existing_chains:
                old_ts = [c.timestamp_mid for c in old_chain]
                old_frames = [c.frame_id for c in old_chain]

                # Nếu tất cả các frame trùng hệt nhau -> bỏ qua
                if new_frames == old_frames:
                    is_distinct = False
                    break

                # Tính độ lệch thời gian lớn nhất giữa các sự kiện
                max_time_diff = max(abs(new_t - old_t) for new_t, old_t in zip(new_ts, old_ts))
                if max_time_diff < hedging_min_temporal_gap:
                    is_distinct = False
                    break

            if not is_distinct:
                continue

        accepted_chains_by_vid.setdefault(target_vid, []).append(sorted_chain)
        final_chains.append(sorted_chain)
        if len(final_chains) >= request.top_k:
            break

    return PipelineResult(
        candidates=[],
        task_type=TaskType.TRAKE,
        enhanced_query=" ".join(sub_events_en) if "sub_events_en" in locals() and sub_events_en else raw_query,
        transcript=base_res.transcript if "base_res" in locals() and base_res else None,
        sub_events=sub_events,
        chains=final_chains,
    )
