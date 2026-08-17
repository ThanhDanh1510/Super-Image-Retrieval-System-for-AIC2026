"""
serving/api/service/retrieval/voice_query.py
=============================================
Service xử lý query dạng giọng nói (Voice Input).

LUỒNG XỬ LÝ:
  1. Nhận audio bytes từ controller
  2. Transcribe audio → text (dùng model ASR) trong worker thread không block event loop.
  3. Chuyển tiếp transcript sang text_query.process()
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from dataclasses import dataclass

from serving.api.schema.retrieval import CandidateResult, TaskType
from serving.api.service.retrieval import text_query
from serving.api.service.retrieval.text_query import TextQueryInput, TextQueryOutput

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class VoiceQueryInput:
    """Input chuẩn hóa cho voice_query.process()."""
    audio_bytes: bytes
    """Raw audio bytes (WAV/MP3/M4A). Controller đã đọc từ uploaded file."""

    audio_format: str
    """Format của audio: 'wav', 'mp3', 'webm', v.v."""

    task_type: TaskType
    """KIS / VQA / TRAKE — chuyển tiếp sang text_query."""

    top_k: int = 100


@dataclass
class VoiceQueryOutput:
    """Output trả về từ voice_query.process()."""
    candidates: list[CandidateResult]
    """Danh sách candidates thô — giống TextQueryOutput.candidates."""

    transcript: str
    """Text transcript từ ASR — hiển thị lên UI để user xác nhận."""

    enhanced_query: str
    """Query sau khi text_query đã enhance."""

    sub_events: list[str] | None = None
    """Sub-events (chỉ với TRAKE)."""


# =============================================================================
# Core Functions
# =============================================================================

async def process(
    query_input: VoiceQueryInput,
    qdrant_client,
    text_embedder,
    image_embedder,
    asr_model,
) -> VoiceQueryOutput:
    """
    Xử lý toàn bộ luồng voice query: transcribe → delegate sang text_query.
    """
    import time

    _t_start = time.perf_counter()
    transcript = await _transcribe(query_input.audio_bytes, query_input.audio_format, asr_model)
    _t_asr = time.perf_counter()
    logger.info(f"[PROFILE] ASR transcription: {(_t_asr - _t_start) * 1000:.2f}ms")
    logger.info(f"ASR transcript: {transcript[:100]}...")

    text_output: TextQueryOutput = await text_query.process(
        query_input=TextQueryInput(
            raw_query=transcript,
            task_type=query_input.task_type,
            top_k=query_input.top_k,
        ),
        qdrant_client=qdrant_client,
        text_embedder=text_embedder,
        image_embedder=image_embedder,
    )

    return VoiceQueryOutput(
        candidates=text_output.candidates,
        transcript=transcript,
        enhanced_query=text_output.enhanced_query,
        sub_events=text_output.sub_events,
    )


# =============================================================================
# Internal Helper Functions
# =============================================================================

async def _transcribe(audio_bytes: bytes, audio_format: str, asr_model) -> str:
    """
    Chuyển đổi audio → text bằng model ASR trên worker thread (non-blocking).
    """
    if not audio_bytes:
        raise ValueError("audio_bytes không được rỗng")

    if asr_model is None:
        logger.warning("ASR model là None, không thể transcribe audio. Trả về transcript fallback.")
        return "[VOICE_TRANSCRIPT_FALLBACK]"

    def _sync_transcribe() -> str:
        suffix = f".{audio_format.lstrip('.')}" if audio_format else ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            if callable(asr_model):
                result = asr_model(tmp_path)
                if isinstance(result, dict):
                    return str(result.get("text", "") or "").strip()
                elif isinstance(result, str):
                    return result.strip()
                else:
                    return str(result).strip()
            elif hasattr(asr_model, "transcribe"):
                result = asr_model.transcribe(tmp_path)
                if isinstance(result, tuple):
                    segments, _ = result
                    return " ".join([s.text for s in segments]).strip()
                elif isinstance(result, dict):
                    return str(result.get("text", "") or "").strip()
                else:
                    return str(result).strip()
            else:
                logger.warning(f"asr_model type {type(asr_model)} không supported callable/transcribe.")
                return "[VOICE_TRANSCRIPT_FALLBACK]"
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    return await asyncio.to_thread(_sync_transcribe)
