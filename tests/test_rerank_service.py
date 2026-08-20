"""
tests/test_rerank_service.py
=============================
Unit tests cho rerank_service — seam duy nhất đứng sau retrieval pipeline cho reranking.

Seam under test:
    serving/api/service/reranker/rerank_service.py::rerank
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from serving.api.schema.retrieval import CandidateResult
from serving.api.service.reranker import rerank_service


def _mk_candidate(video_id: str = "V001", frame_id: int = 1, score: float = 0.5) -> CandidateResult:
    return CandidateResult(
        video_id=video_id,
        scene_id=f"{video_id}_S{frame_id:03d}",
        frame_id=frame_id,
        timestamp_start=0.0,
        timestamp_end=1.0,
        timestamp_mid=0.5,
        keyframe_path=f"{video_id}/{frame_id}.jpg",
        ocr_text="",
        asr_text="",
        score=score,
    )


@pytest.fixture
def candidates() -> list[CandidateResult]:
    return [
        _mk_candidate(video_id="V001", frame_id=1, score=0.5),
        _mk_candidate(video_id="V002", frame_id=2, score=0.9),
    ]


@pytest.fixture
def mock_rerankers(monkeypatch):
    """Mock viranker/jina/fusion_score tại module rerank_service."""
    viranker = MagicMock()
    viranker.score.return_value = [0.9, 0.4]
    jina = MagicMock()
    jina.score.return_value = [0.5, 0.8]
    fuse_module = MagicMock()
    fuse_module.fuse.side_effect = lambda t, c, cands: cands

    monkeypatch.setattr(rerank_service, "viranker", viranker)
    monkeypatch.setattr(rerank_service, "jina_reranker_m0", jina)
    monkeypatch.setattr(rerank_service, "fusion_score", fuse_module)
    return viranker, jina, fuse_module.fuse


@pytest.mark.anyio
async def test_rerank_returns_fused_candidates(candidates, mock_rerankers):
    viranker, jina, fuse = mock_rerankers
    result = await rerank_service.rerank("query", candidates, use_visual_reranker=True)

    assert result == candidates
    viranker.score.assert_called_once_with("query", candidates, None, None)
    jina.score.assert_called_once_with("query", candidates, "", None)
    fuse.assert_called_once_with([0.9, 0.4], [0.5, 0.8], candidates)


@pytest.mark.anyio
async def test_rerank_only_runs_text_when_visual_is_disabled(candidates, mock_rerankers):
    viranker, jina, fuse = mock_rerankers
    result = await rerank_service.rerank("query", candidates, use_visual_reranker=False)

    assert result == candidates
    viranker.score.assert_called_once_with("query", candidates, None, None)
    jina.score.assert_not_called()
    fuse.assert_called_once_with([0.9, 0.4], None, candidates)


@pytest.mark.anyio
async def test_rerank_empty_candidates_returns_empty(mock_rerankers):
    viranker, jina, _ = mock_rerankers
    result = await rerank_service.rerank("query", [], use_visual_reranker=True)

    assert result == []
    viranker.score.assert_not_called()
    jina.score.assert_not_called()


@pytest.mark.anyio
async def test_rerank_falls_back_to_original_candidates_when_scorer_raises(candidates, monkeypatch):
    viranker = MagicMock()
    viranker.score.side_effect = RuntimeError("model load failed")
    monkeypatch.setattr(rerank_service, "viranker", viranker)

    jina = MagicMock()
    jina.score.return_value = [0.5, 0.8]
    monkeypatch.setattr(rerank_service, "jina_reranker_m0", jina)

    result = await rerank_service.rerank("query", candidates, use_visual_reranker=True)

    assert result == candidates
    jina.score.assert_called_once_with("query", candidates, "", None)


@pytest.mark.anyio
async def test_rerank_uses_enhanced_query_over_raw_query(candidates, mock_rerankers):
    viranker, jina, _ = mock_rerankers
    await rerank_service.rerank("câu hỏi tiếng Việt", candidates, enhanced_query="english query", use_visual_reranker=True)

    viranker.score.assert_called_once_with("câu hỏi tiếng Việt", candidates, None, "english query")
    jina.score.assert_called_once_with("english query", candidates, "", None)


# =============================================================================
# Wiring test: retrieval_pipeline runs rerank_service
# =============================================================================

@pytest.mark.anyio
async def test_search_kis_uses_rerank_service(monkeypatch):
    """retrieval_pipeline.run gọi rerank_service.rerank() khi task là KIS."""
    from serving.api.service.retrieval import retrieval_pipeline
    from serving.api.schema.retrieval import KISRequest, TaskType, InputType

    cands = [_mk_candidate()]
    output = MagicMock()
    output.candidates = cands
    output.enhanced_query = "enhanced english"
    output.sub_events = None

    text_process = AsyncMock(return_value=output)
    monkeypatch.setattr(retrieval_pipeline.text_query, "process", text_process)

    rerank_mock = AsyncMock(return_value=cands)
    monkeypatch.setattr(retrieval_pipeline.rerank_service, "rerank", rerank_mock)

    req = KISRequest(query="câu hỏi tiếng Việt", top_k=5, use_reranker=True)
    res = await retrieval_pipeline.run(
        input_type=InputType.TEXT,
        task_type=TaskType.KIS,
        request=req,
        qdrant_client=AsyncMock(),
        text_embedder=MagicMock(),
        image_embedder=MagicMock(),
        asr_model=MagicMock(),
    )

    rerank_mock.assert_awaited_once()
    assert res.candidates == cands


@pytest.mark.anyio
async def test_search_vqa_uses_rerank_service(monkeypatch):
    """retrieval_pipeline.run cũng đi qua rerank_service cho VQA."""
    from serving.api.service.retrieval import retrieval_pipeline
    from serving.api.schema.retrieval import VQARequest, TaskType, InputType

    cands = [_mk_candidate()]
    output = MagicMock()
    output.candidates = cands
    output.enhanced_query = "enhanced english"
    output.sub_events = None

    text_process = AsyncMock(return_value=output)
    monkeypatch.setattr(retrieval_pipeline.text_query, "process", text_process)

    rerank_mock = AsyncMock(return_value=cands)
    monkeypatch.setattr(retrieval_pipeline.rerank_service, "rerank", rerank_mock)

    req = VQARequest(query="câu hỏi tiếng Việt", question="Câu hỏi?", top_k=5, use_reranker=True)
    res = await retrieval_pipeline.run(
        input_type=InputType.TEXT,
        task_type=TaskType.VQA,
        request=req,
        qdrant_client=AsyncMock(),
        text_embedder=MagicMock(),
        image_embedder=MagicMock(),
        asr_model=MagicMock(),
    )

    rerank_mock.assert_awaited_once()
    assert res.candidates == cands
