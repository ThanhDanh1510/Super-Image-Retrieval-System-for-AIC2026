"""
serving/api/service/retrieval/voice_query.py
=============================================
Service xử lý query dạng giọng nói (Voice Input).

LUỒNG XỬ LÝ:
  1. Nhận audio bytes từ controller
  2. Transcribe audio → text (dùng model ASR, giống PhoWhisper trong ingestion)
  3. Chuyển tiếp transcript sang text_query.process() — không duplicate logic

NGUYÊN TẮC THIẾT KẾ:
  Module này là adapter (bộ chuyển đổi), không phải retrieval engine.
  Sau khi có text, TOÀN BỘ logic encode và search đều delegate sang text_query.py.
  Không bao giờ gọi qdrant_repo trực tiếp từ module này.

IMPORT RULES:
  - Được gọi bởi: retrieval_pipeline.py
  - Gọi: voice_query._transcribe() rồi text_query.process()
  - Không gọi qdrant_repo trực tiếp.
"""

from __future__ import annotations

import logging
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

    Args:
        query_input: VoiceQueryInput chứa audio bytes và metadata.
        qdrant_client: AsyncQdrantClient đã inject.
        text_embedder: TextEmbeddingWrapper singleton.
        image_embedder: ImageEmbeddingWrapper singleton.
        asr_model: ASR model (PhoWhisper) singleton đã load.

    Returns:
        VoiceQueryOutput với candidates, transcript, và enhanced_query.
    """
    transcript = await _transcribe(query_input.audio_bytes, query_input.audio_format, asr_model)
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
# Internal Helper Functions (implement these)
# =============================================================================

async def _transcribe(audio_bytes: bytes, audio_format: str, asr_model) -> str:
    """
    Chuyển đổi audio → text bằng model ASR.

    Args:
        audio_bytes: Raw audio bytes.
        audio_format: Format của audio ('wav', 'mp3', 'webm', v.v.).
        asr_model: ASR model (PhoWhisper pipeline) singleton đã load.

    Returns:
        Transcript string.

    Raises:
        ValueError: Nếu audio_bytes rỗng.
        RuntimeError: Nếu ASR model không thể transcribe.
    """
    if not audio_bytes:
        raise ValueError("audio_bytes không được rỗng")

    if asr_model is None:
        logger.warning("ASR model là None, không thể transcribe audio real. Trả về transcript fallback.")
        return "[VOICE_TRANSCRIPT_FALLBACK]"

    import tempfile
    import os

    suffix = f".{audio_format.lstrip('.')}" if audio_format else ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        if callable(asr_model):
            result = asr_model(tmp_path)
            if isinstance(result, dict):
                transcript = str(result.get("text", "") or "").strip()
            elif isinstance(result, str):
                transcript = result.strip()
            else:
                transcript = str(result).strip()
            return transcript
        elif hasattr(asr_model, "transcribe"):
            # e.g., whisper model instance
            result = asr_model.transcribe(tmp_path)
            if isinstance(result, tuple):
                # e.g. faster-whisper returns (segments, info)
                segments, _ = result
                transcript = " ".join([s.text for s in segments]).strip()
            elif isinstance(result, dict):
                transcript = str(result.get("text", "") or "").strip()
            else:
                transcript = str(result).strip()
            return transcript
        else:
            logger.warning(f"asr_model type {type(asr_model)} không supported callable/transcribe.")
            return "[VOICE_TRANSCRIPT_FALLBACK]"
    except Exception as e:
        logger.error(f"Lỗi trong quá trình transcribe audio: {e}", exc_info=True)
        raise RuntimeError(f"Lỗi khi transcribe audio: {e}") from e
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
