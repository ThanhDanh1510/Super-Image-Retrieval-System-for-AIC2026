"""
tests/test_retrieval_serving.py
=================================
Unit tests cho Module Retrieval Serving (TASK-SRV-01 -> TASK-SRV-05).
"""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock
import numpy as np
import pytest

from serving.api.schema.retrieval import (
    CandidateResult,
    InputType,
    TaskType,
    KISRequest,
    VQARequest,
    TRAKERequest,
    ImageSearchRequest,
)
from serving.api.service.retrieval import (
    text_query,
    image_query,
    voice_query,
    retrieval_pipeline,
)
from shared.models.text_embedding_wrapper import TextEmbeddingWrapper
from shared.models.image_embedding_wrapper import ImageEmbeddingWrapper


@pytest.fixture
def mock_text_embedder():
    embedder = MagicMock(spec=TextEmbeddingWrapper)
    embedder.encode.return_value = np.zeros(1536, dtype=np.float32)
    return embedder


@pytest.fixture
def mock_image_embedder():
    embedder = MagicMock(spec=ImageEmbeddingWrapper)
    embedder.encode_text.return_value = np.zeros(1024, dtype=np.float32)
    embedder.encode_from_bytes.return_value = np.zeros(1024, dtype=np.float32)
    return embedder


@pytest.fixture
def mock_qdrant_client():
    client = AsyncMock()
    return client


# =============================================================================
# TASK-SRV-01 & TASK-SRV-05: Text Query & Event Splitting
# =============================================================================

@pytest.mark.anyio
async def test_enhance_query_english():
    en_query, _ = await text_query._enhance_query("a man walking a dog in the park")
    assert en_query == "a man walking a dog in the park"


def test_is_valid_translation_rejects_google_error_page():
    # Trang lỗi Google mà deep_translator trả về như một "bản dịch" thành công
    garbage = (
        "Error 500 (Server Error)!!1500.That’s an error.There was an error. "
        "Please try again later.That’s all we know."
    )
    assert text_query._is_valid_translation(garbage, "đoạn video mô tả cảnh trang trí bánh rán") is False


def test_is_valid_translation_rejects_empty_and_html():
    assert text_query._is_valid_translation("", "nguồn") is False
    assert text_query._is_valid_translation("   \t\n  ", "nguồn") is False
    assert text_query._is_valid_translation("<html><body>error</body></html>", "nguồn") is False
    assert text_query._is_valid_translation(",.!?@#$%%^", "nguồn") is False


def test_is_valid_translation_accepts_real_translation():
    assert text_query._is_valid_translation(
        "The video depicts the donut decoration scene.", "đoạn video mô tả cảnh trang trí bánh rán"
    ) is True


@pytest.mark.anyio
async def test_enhance_query_falls_back_when_translation_is_error_page(monkeypatch):
    """Khi Google trả về trang lỗi, enhanced_query phải giữ nguyên query gốc (không rò rỉ 'Error 500')."""
    async def fake_translate(query, target):
        return "Error 500 (Server Error)!!1500.That’s an error. Please try again later."

    monkeypatch.setattr(text_query, "_translate", fake_translate)
    en_query, _ = await text_query._enhance_query(
        "Đoạn video mô tả cảnh trang trí bánh rán"
    )
    # KHÔNG được chứa nội dung trang lỗi
    assert "Error 500" not in en_query
    assert "that's an error" not in en_query.lower()
    # Fallback: giữ nguyên query gốc đã làm sạch (unaccented hoặc vi)
    assert ("banh ran" in en_query.lower()) or ("bánh rán" in en_query.lower())


@pytest.mark.anyio
async def test_translate_retries_then_returns_none(monkeypatch):
    """Sau 2 lần thử đều trả về trang lỗi, _translate phải trả về None (không đổ rác ra)."""
    calls = {"n": 0}

    class FakeTranslator:
        def __init__(self, source, target):
            self.target = target

        def translate(self, query):
            calls["n"] += 1
            return "Error 500 (Server Error)!!1500.That’s an error."

    monkeypatch.setattr(text_query, "_get_qwen_bundle", lambda: None)
    monkeypatch.setattr("deep_translator.GoogleTranslator", FakeTranslator)

    result = await text_query._translate("đoạn video mô tả bánh rán", "en")
    assert result is None
    assert calls["n"] >= 2


@pytest.mark.anyio
async def test_translate_does_not_hang_when_translation_times_out(monkeypatch):
    """Translator treo (không timeout) phải bị asyncio.wait_for cắt, không chặn request vô hạn."""
    import asyncio

    class HangingTranslator:
        def __init__(self, source, target):
            self.target = target

        def translate(self, query):
            time.sleep(1.5)  # mô phỏng requests.get() treo lâu hơn timeout
            return "never reached"

    monkeypatch.setattr(text_query, "_get_qwen_bundle", lambda: None)
    monkeypatch.setattr("deep_translator.GoogleTranslator", HangingTranslator)
    monkeypatch.setattr(text_query, "TRANSLATE_TIMEOUT_S", 0.2)

    t0 = time.time()
    result = await text_query._translate("đoạn video mô tả bánh rán", "en")
    elapsed = time.time() - t0

    assert result is None
    assert elapsed < 5  # không được treo 1.5s+ mỗi attempt; 2 attempts * 0.2s timeout


def test_split_events_newline():
    query = "Event one description\nEvent two description\nEvent three description"
    events = text_query._split_events(query)
    assert len(events) == 3
    assert events[0] == "Event one description"
    assert events[1] == "Event two description"
    assert events[2] == "Event three description"


def test_split_events_connectors():
    query = "A man opens the door, then enters the room, after that sits on the chair"
    events = text_query._split_events(query)
    assert len(events) == 3
    assert events[0] == "A man opens the door"
    assert events[1] == "enters the room"
    assert events[2] == "sits on the chair"


def test_split_events_invalid():
    # Mock LLM fallback to return empty list so the ValueError is raised deterministically
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("serving.api.service.retrieval.text_query._split_events_llm", lambda x: [])
        with pytest.raises(ValueError):
            text_query._split_events("Single event without any connectors or newlines")


@pytest.mark.anyio
async def test_text_query_process(mock_qdrant_client, mock_text_embedder, mock_image_embedder):
    mock_candidate = CandidateResult(
        video_id="V001",
        scene_id="V001_S001",
        frame_id=10,
        timestamp_start=0.0,
        timestamp_end=5.0,
        timestamp_mid=2.5,
        keyframe_path="V001/V001_S001.jpg",
        score=0.95,
    )
    
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("serving.api.repo.qdrant_repo.search_by_text_vectors", AsyncMock(return_value=[mock_candidate]))
        
        inp = text_query.TextQueryInput(
            raw_query="A man riding a bicycle\nthen stopping at a red light",
            task_type=TaskType.TRAKE,
            top_k=10,
        )
        res = await text_query.process(inp, mock_qdrant_client, mock_text_embedder, mock_image_embedder)
        
        assert len(res.candidates) == 1
        assert res.candidates[0].video_id == "V001"
        assert res.sub_events == ["A man riding a bicycle", "stopping at a red light"]


# =============================================================================
# TASK-SRV-02: Image Query
# =============================================================================

@pytest.mark.anyio
async def test_image_query_process(mock_qdrant_client, mock_image_embedder):
    mock_candidate = CandidateResult(
        video_id="V002",
        scene_id="V002_S001",
        frame_id=5,
        timestamp_start=1.0,
        timestamp_end=4.0,
        timestamp_mid=2.5,
        keyframe_path="V002/V002_S001.jpg",
        score=0.88,
    )
    
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("serving.api.repo.qdrant_repo.search_by_image_vector", AsyncMock(return_value=[mock_candidate]))
        
        inp = image_query.ImageQueryInput(
            image_bytes=b"fake_image_bytes",
            top_k=5,
        )
        res = await image_query.process(inp, mock_qdrant_client, mock_image_embedder)
        assert len(res.candidates) == 1
        assert res.candidates[0].video_id == "V002"


# =============================================================================
# TASK-SRV-03: Voice Query
# =============================================================================

@pytest.mark.anyio
async def test_voice_query_transcribe():
    mock_asr = MagicMock(return_value={"text": "Người đàn ông đang chạy bộ"})
    transcript = await voice_query._transcribe(b"fake_audio", "wav", mock_asr)
    assert transcript == "Người đàn ông đang chạy bộ"


# =============================================================================
# TASK-SRV-04: Retrieval Pipeline
# =============================================================================

@pytest.mark.anyio
async def test_retrieval_pipeline_text(mock_qdrant_client, mock_text_embedder, mock_image_embedder):
    mock_candidate = CandidateResult(
        video_id="V003",
        scene_id="V003_S001",
        frame_id=1,
        timestamp_start=0.0,
        timestamp_end=2.0,
        timestamp_mid=1.0,
        keyframe_path="V003/V003_S001.jpg",
        score=0.9,
    )

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("serving.api.repo.qdrant_repo.search_by_text_vectors", AsyncMock(return_value=[mock_candidate]))

        req = KISRequest(query="a cat playing with ball", top_k=10)
        res = await retrieval_pipeline.run(
            input_type=InputType.TEXT,
            task_type=TaskType.KIS,
            request=req,
            qdrant_client=mock_qdrant_client,
            text_embedder=mock_text_embedder,
            image_embedder=mock_image_embedder,
            asr_model=None,
        )

        assert len(res.candidates) == 1
        assert res.candidates[0].video_id == "V003"
        assert res.task_type == TaskType.KIS
