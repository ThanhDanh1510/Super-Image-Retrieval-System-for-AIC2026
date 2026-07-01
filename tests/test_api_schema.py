"""
tests/test_api_schema.py
=========================
Unit tests cho Pydantic schemas của serving API (TASK-TST-03).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from serving.api.schema.retrieval import (
    InputType,
    TaskType,
    CandidateResult,
    UserChoice,
    KISRequest,
    KISResponse,
)

def test_candidate_result_validation():
    """Kiểm tra việc parse và validate CandidateResult."""
    # Đúng định dạng
    data = {
        "video_id": "L01_V001",
        "scene_id": "L01_V001_S001",
        "frame_id": 150,
        "timestamp_start": 5.0,
        "timestamp_end": 7.5,
        "timestamp_mid": 6.25,
        "keyframe_path": "L01_V001/L01_V001_S001.jpg",
        "score": 0.85
    }
    candidate = CandidateResult(**data)
    assert candidate.video_id == "L01_V001"
    assert candidate.score == 0.85

    # Thiếu trường bắt buộc
    bad_data = {
        "video_id": "L01_V001",
        "frame_id": 150
    }
    with pytest.raises(ValidationError):
        CandidateResult(**bad_data)

def test_user_choice_validation():
    """Kiểm tra việc parse và validate UserChoice."""
    data = {
        "video_id": "L01_V001",
        "frame_id": 150
    }
    choice = UserChoice(**data)
    assert choice.video_id == "L01_V001"
    assert choice.frame_id == 150

def test_kis_request_validation():
    """Kiểm tra việc parse và validate KISRequest."""
    # Request hợp lệ không có user_choices
    data = {
        "input_type": "text",
        "query": "A man jumping over a fence",
        "top_k": 50
    }
    req = KISRequest(**data)
    assert req.input_type == InputType.TEXT
    assert req.top_k == 50
    assert req.user_choices == []

    # Request hợp lệ có user_choices
    data_with_choices = {
        "input_type": "text",
        "query": "A man jumping over a fence",
        "user_choices": [
            {"video_id": "L01_V001", "frame_id": 150}
        ]
    }
    req = KISRequest(**data_with_choices)
    assert len(req.user_choices) == 1
    assert req.user_choices[0].video_id == "L01_V001"

    # Sai top_k
    bad_top_k = {
        "input_type": "text",
        "query": "A man",
        "top_k": -5
    }
    with pytest.raises(ValidationError):
        KISRequest(**bad_top_k)

def test_kis_response_validation():
    """Kiểm tra việc validate KISResponse."""
    data = {
        "results": [
            {
                "video_id": "L01_V001",
                "scene_id": "L01_V001_S001",
                "frame_id": 150,
                "timestamp_start": 5.0,
                "timestamp_end": 7.5,
                "timestamp_mid": 6.25,
                "keyframe_path": "L01_V001/L01_V001_S001.jpg",
                "score": 0.85
            }
        ],
        "total": 1,
        "query_text": "A man jumping over a fence"
    }
    resp = KISResponse(**data)
    assert len(resp.results) == 1
    assert resp.total == 1
    assert resp.query_text == "A man jumping over a fence"
