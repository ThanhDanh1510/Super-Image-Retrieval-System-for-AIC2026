"""
tests/test_rerank_service.py
=============================
Unit tests cho rerank_service — seam duy nhất đứng sau controller cho reranking.

Sau refactor (Candidate 1: "deepen the rerank seam"), controller KHÔNG còn import
viranker/jina/fusion_score trực tiếp; tất cả chạy qua `rerank_service.rerank()`.

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
    # fusion_score là module trong rerank_service; mock cả module để .fuse() hoạt động
    fuse_module = MagicMock()
    fuse_module.fuse.side_effect = lambda t, c, cands: cands

    monkeypatch.setattr(rerank_service, "viranker", viranker)
    monkeypatch.setattr(rerank_service, "jina_reranker_m0", jina)
    monkeypatch.setattr(rerank_service, "fusion_score", fuse_module)
    return viranker, jina, fuse_module.fuse


@pytest.mark.anyio
async def test_rerank_returns_fused_candidates(candidates, mock_rerankers):
    viranker, jina, fuse = mock_rerankers
    result = await rerank_service.rerank("query", candidates)

    assert result == candidates
    # Cả hai nhánh được gọi song song với query gốc
    viranker.score.assert_called_once_with("query", candidates)
    jina.score.assert_called_once_with("query", candidates, "")
    # Fusion nhận điểm 2 nhánh
    fuse.assert_called_once_with([0.9, 0.4], [0.5, 0.8], candidates)


@pytest.mark.anyio
async def test_rerank_empty_candidates_returns_empty(mock_rerankers):
    viranker, jina, _ = mock_rerankers
    result = await rerank_service.rerank("query", [])

    assert result == []
    viranker.score.assert_not_called()
    jina.score.assert_not_called()


@pytest.mark.anyio
async def test_rerank_falls_back_to_original_candidates_when_scorer_raises(candidates, monkeypatch):
    """Khi một nhánh rerank lỗi, trả về candidates gốc (RRF scores) — không 500.

    QUAN TRỌNG: phải mock CẢ hai nhánh (viranker + jina). Nếu bỏ sót nhánh jina,
    `asyncio.gather` sẽ chạy thật `jina_reranker_m0._get_model()` (load model thật)
    trong thread nền — bị cancel nhưng không dừng được thread -> treo các test sau.
    """
    viranker = MagicMock()
    viranker.score.side_effect = RuntimeError("model load failed")
    monkeypatch.setattr(rerank_service, "viranker", viranker)

    jina = MagicMock()
    jina.score.return_value = [0.5, 0.8]
    monkeypatch.setattr(rerank_service, "jina_reranker_m0", jina)

    result = await rerank_service.rerank("query", candidates)

    assert result == candidates
    # Nhánh jina được gọi song song nhưng kết quả bị bỏ vì viranker lỗi → fallback
    jina.score.assert_called_once_with("query", candidates, "")


@pytest.mark.anyio
async def test_rerank_uses_enhanced_query_over_raw_query(candidates, mock_rerankers):
    viranker, jina, _ = mock_rerankers
    await rerank_service.rerank("câu hỏi tiếng Việt", candidates, enhanced_query="english query")

    viranker.score.assert_called_once_with("english query", candidates)
    jina.score.assert_called_once_with("english query", candidates, "")


# =============================================================================
# Wiring test: controller dùng seam (không import reranker trực tiếp)
# =============================================================================

@pytest.mark.anyio
async def test_search_kis_uses_rerank_service(monkeypatch):
    """search_kis gọi rerank_service.rerank() thay vì tự import viranker/jina."""
    from serving.api.controller import routes_kis
    from serving.api.schema.retrieval import KISRequest

    cands = [_mk_candidate()]
    pipeline_res = MagicMock()
    pipeline_res.candidates = cands
    pipeline_res.enhanced_query = "enhanced english"
    pipeline_res.task_type = MagicMock()

    run = AsyncMock(return_value=pipeline_res)
    monkeypatch.setattr(routes_kis.retrieval_pipeline, "run", run)

    rerank = AsyncMock(return_value=cands)
    monkeypatch.setattr(routes_kis.rerank_service, "rerank", rerank)

    req = KISRequest(query="câu hỏi tiếng Việt", top_k=5)
    resp = await routes_kis.search_kis(
        request=req,
        qdrant_client=AsyncMock(),
        text_embedder=MagicMock(),
        image_embedder=MagicMock(),
        asr_model=MagicMock(),
    )

    run.assert_awaited_once()
    rerank.assert_awaited_once()
    # enhanced_query được ưu tiên hơn query gốc
    assert rerank.await_args.kwargs["query"] == "câu hỏi tiếng Việt"
    assert rerank.await_args.kwargs["enhanced_query"] == "enhanced english"
    assert resp.total == 1
    assert resp.query_text == "enhanced english"


@pytest.mark.anyio
async def test_search_vqa_uses_rerank_service(monkeypatch):
    """search_vqa cũng đi qua rerank_service (cùng seam với KIS)."""
    from serving.api.controller import routes_kis
    from serving.api.schema.retrieval import VQARequest

    cands = [_mk_candidate()]
    pipeline_res = MagicMock()
    pipeline_res.candidates = cands
    pipeline_res.enhanced_query = "enhanced english"

    run = AsyncMock(return_value=pipeline_res)
    monkeypatch.setattr(routes_kis.retrieval_pipeline, "run", run)

    rerank = AsyncMock(return_value=cands)
    monkeypatch.setattr(routes_kis.rerank_service, "rerank", rerank)

    req = VQARequest(query="câu hỏi tiếng Việt", question="Câu hỏi?", top_k=5)
    resp = await routes_kis.search_vqa(
        request=req,
        qdrant_client=AsyncMock(),
        text_embedder=MagicMock(),
        image_embedder=MagicMock(),
        asr_model=MagicMock(),
    )

    rerank.assert_awaited_once()
    assert resp.total == 1
