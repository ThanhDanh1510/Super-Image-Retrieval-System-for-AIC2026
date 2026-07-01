"""
serving/api/service/retrieval/skeletons/retrieval_pipeline.py
=============================================================
Orchestrator trung tâm — nhận query từ controller, điều hướng sang đúng module (Skeleton).

LUỒNG XỬ LÝ:
  - Nhận InputType (text/voice/image) và TaskType (KIS/VQA/TRAKE).
  - Gọi đúng module: text_query / voice_query / image_query.
  - Trả về candidates thô cho controller.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from serving.api.schema.retrieval import (
    CandidateResult,
    InputType,
    TaskType,
)

logger = logging.getLogger(__name__)

@dataclass
class PipelineResult:
    candidates: list[CandidateResult]
    task_type: TaskType
    enhanced_query: str | None = None
    transcript: str | None = None
    sub_events: list[str] | None = None

async def run(
    input_type: InputType,
    task_type: TaskType,
    request,
    qdrant_client,
    text_embedder,
    image_embedder,
    asr_model,
) -> PipelineResult:
    """
    Entry point duy nhất cho toàn bộ retrieval pipeline.
    """
    logger.info(f"Pipeline run: input_type={input_type}, task_type={task_type}")
    if input_type == InputType.TEXT:
        return await _run_text(request, task_type, qdrant_client, text_embedder, image_embedder)
    elif input_type == InputType.VOICE:
        return await _run_voice(request, task_type, qdrant_client, text_embedder, image_embedder, asr_model)
    elif input_type == InputType.IMAGE:
        return await _run_image(request, qdrant_client, image_embedder)
    else:
        raise ValueError(f"Unsupported input_type: {input_type}")

async def _run_text(
    request,
    task_type: TaskType,
    qdrant_client,
    text_embedder,
    image_embedder,
) -> PipelineResult:
    """
    Route text query sang text_query.process() và wrap kết quả thành PipelineResult.
    """
    raise NotImplementedError

async def _run_voice(
    request,
    task_type: TaskType,
    qdrant_client,
    text_embedder,
    image_embedder,
    asr_model,
) -> PipelineResult:
    """
    Route voice query sang voice_query.process() và wrap kết quả.
    """
    raise NotImplementedError

async def _run_image(
    request,
    qdrant_client,
    image_embedder,
) -> PipelineResult:
    """
    Route image query sang image_query.process() và wrap kết quả.
    """
    raise NotImplementedError
