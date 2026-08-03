"""
tests/test_retrieval_serving.py
=================================
Unit tests cho Module Retrieval Serving (TASK-SRV-01 -> TASK-SRV-05).
"""

from __future__ import annotations

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
    enhanced = await text_query._enhance_query("a man walking a dog in the park")
    assert enhanced == "a man walking a dog in the park"


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
