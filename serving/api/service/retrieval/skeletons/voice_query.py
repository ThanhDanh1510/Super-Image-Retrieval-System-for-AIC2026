"""
serving/api/service/retrieval/skeletons/voice_query.py
======================================================
Service xử lý query dạng giọng nói (Voice Input) (Skeleton).

LUỒNG XỬ LÝ:
  1. Nhận audio bytes từ controller
  2. Transcribe audio → text (dùng model ASR)
  3. Chuyển tiếp transcript sang text_query.process() — không duplicate logic
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from serving.api.schema.retrieval import CandidateResult, TaskType

logger = logging.getLogger(__name__)

@dataclass
class VoiceQueryInput:
    audio_bytes: bytes
    audio_format: str
    task_type: TaskType
    top_k: int = 100

@dataclass
class VoiceQueryOutput:
    candidates: list[CandidateResult]
    transcript: str
    enhanced_query: str
    sub_events: list[str] | None = None

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
    raise NotImplementedError

async def _transcribe(audio_bytes: bytes, audio_format: str, asr_model) -> str:
    """
    Chuyển đổi audio → text bằng model ASR.
    """
    raise NotImplementedError
