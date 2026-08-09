import importlib
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

text_extractor = importlib.import_module("ingestion.skeletons.01b_text_extraction")


def test_run_ocr_file_not_found():
    mock_ocr = MagicMock()
    with pytest.raises(FileNotFoundError):
        text_extractor.run_ocr(mock_ocr, "/non/existent/image.jpg")


def test_run_ocr_mock_engine(tmp_path):
    # Tạo fake image file
    img_path = tmp_path / "test_frame.jpg"
    img_path.touch()

    # Mock PaddleOCR engine return structure: [ [ [box, (text, score)], ... ] ]
    mock_ocr = MagicMock()
    mock_ocr.ocr.return_value = [
        [
            [[[0, 0], [10, 0], [10, 10], [0, 10]], ("CỬA HÀNG", 0.98)],
            [[[0, 20], [10, 20], [10, 30], [0, 30]], ("TIỆN LỢI", 0.95)],
        ]
    ]

    result = text_extractor.run_ocr(mock_ocr, str(img_path))
    assert result == "CỬA HÀNG\nTIỆN LỢI"
    mock_ocr.ocr.assert_called_once_with(str(img_path), cls=True)


def test_run_ocr_quality_applies_conservative_correction(tmp_path):
    img_path = tmp_path / "attendance.jpg"
    img_path.touch()

    mock_ocr = MagicMock()
    mock_ocr.ocr.return_value = [
        [
            [[[0, 0], [10, 0], [10, 10], [0, 10]], ("Lớp 3 : 6 vq 1 (Nghĩa)", 0.61)]
        ]
    ]

    quality = text_extractor.run_ocr_quality(mock_ocr, str(img_path))

    assert quality.raw_text == "Lớp 3 : 6 vq 1 (Nghĩa)"
    assert quality.text == "Lớp 3 : 6 vắng 1 (Nghĩa)"
    assert quality.edits[0]["type"] == "expanded_class_attendance_vq_to_vang"


def test_run_ocr_empty_result(tmp_path):
    img_path = tmp_path / "blank.jpg"
    img_path.touch()

    mock_ocr = MagicMock()
    mock_ocr.ocr.return_value = [[]]

    result = text_extractor.run_ocr(mock_ocr, str(img_path))
    assert result == ""


def test_run_asr_file_not_found():
    mock_asr = MagicMock()
    with pytest.raises(FileNotFoundError):
        text_extractor.run_asr(mock_asr, "/non/existent/audio.wav")


def test_run_asr_mock_model(tmp_path):
    audio_path = tmp_path / "test_audio.wav"
    audio_path.touch()

    mock_asr = MagicMock()
    mock_asr.return_value = {"text": " Xin chào Việt Nam "}

    result = text_extractor.run_asr(mock_asr, str(audio_path))
    assert result == "Xin chào Việt Nam"


def test_extract_audio_segment_file_not_found():
    with pytest.raises(FileNotFoundError):
        text_extractor.extract_audio_segment("/non/existent/video.mp4", 0.0, 5.0)


def test_process_text_extraction_end_to_end(tmp_path, monkeypatch):
    # Setup dummy directory structure
    meta_dir = tmp_path / "input"
    meta_dir.mkdir()
    keyframe_dir = tmp_path / "keyframes" / "L01_V001"
    keyframe_dir.mkdir(parents=True)
    out_dir = tmp_path / "output"
    out_dir.mkdir()

    # Create dummy metadata parquet
    df_meta = pd.DataFrame([
        {
            "scene_id": "L01_V001_S001",
            "video_id": "L01_V001",
            "frame_id": 15,
            "timestamp_start": 0.0,
            "timestamp_end": 5.0,
            "timestamp_mid": 2.5,
            "fps": 25.0,
            "keyframe_path": "L01_V001/L01_V001_S001_000015.jpg",
        }
    ])
    meta_path = meta_dir / "scene_metadata.parquet"
    df_meta.to_parquet(meta_path)

    # Create keyframe file
    kf_file = keyframe_dir / "L01_V001_S001_000015.jpg"
    kf_file.touch()

    # Mock init_ocr_engine & init_asr_model
    mock_ocr = MagicMock()
    mock_ocr.ocr.return_value = [
        [
            [[[0, 0], [10, 0], [10, 10], [0, 10]], ("BIỂN BÁO", 0.99)]
        ]
    ]
    monkeypatch.setattr(text_extractor, "init_ocr_engine", lambda cfg: mock_ocr)
    monkeypatch.setattr(text_extractor, "init_asr_model", lambda cfg: None)

    ocr_df, asr_df = text_extractor.process_text_extraction(
        metadata_path=str(meta_path),
        keyframe_dir=str(tmp_path / "keyframes"),
        video_dir=str(tmp_path),
        output_dir=str(out_dir),
    )

    assert len(ocr_df) == 1
    assert ocr_df.iloc[0]["scene_id"] == "L01_V001_S001"
    assert ocr_df.iloc[0]["ocr_text"] == "BIỂN BÁO"

    assert len(asr_df) == 1
    assert asr_df.iloc[0]["scene_id"] == "L01_V001_S001"
    assert asr_df.iloc[0]["asr_text"] == ""

    # Verify saved files
    assert (out_dir / "ocr_text.parquet").exists()
    assert (out_dir / "asr_text.parquet").exists()


def test_process_text_extraction_writes_ocr_quality(tmp_path, monkeypatch):
    meta_dir = tmp_path / "input"
    meta_dir.mkdir()
    keyframe_dir = tmp_path / "keyframes" / "L01_V001"
    keyframe_dir.mkdir(parents=True)
    out_dir = tmp_path / "output"
    out_dir.mkdir()

    df_meta = pd.DataFrame([
        {
            "scene_id": "L01_V001_S001",
            "video_id": "L01_V001",
            "frame_id": 15,
            "timestamp_start": 0.0,
            "timestamp_end": 5.0,
            "timestamp_mid": 2.5,
            "fps": 25.0,
            "keyframe_path": "L01_V001/L01_V001_S001_000015.jpg",
        }
    ])
    meta_path = meta_dir / "scene_metadata.parquet"
    df_meta.to_parquet(meta_path)
    (keyframe_dir / "L01_V001_S001_000015.jpg").touch()

    mock_ocr = MagicMock()
    mock_ocr.ocr.return_value = [
        [
            [[[0, 0], [10, 0], [10, 10], [0, 10]], ("Lớp 3 : 6 vq 1 (Nghĩa)", 0.61)]
        ]
    ]
    monkeypatch.setattr(text_extractor, "init_ocr_engine", lambda cfg: mock_ocr)
    monkeypatch.setattr(text_extractor, "init_asr_model", lambda cfg: None)

    ocr_df, _ = text_extractor.process_text_extraction(
        metadata_path=str(meta_path),
        keyframe_dir=str(tmp_path / "keyframes"),
        video_dir=str(tmp_path),
        output_dir=str(out_dir),
        write_ocr_quality=True,
    )

    quality_df = pd.read_parquet(out_dir / "ocr_quality.parquet")
    assert ocr_df.iloc[0]["ocr_text"] == "Lớp 3 : 6 vắng 1 (Nghĩa)"
    assert quality_df.iloc[0]["ocr_raw_text"] == "Lớp 3 : 6 vq 1 (Nghĩa)"
    assert quality_df.iloc[0]["ocr_text"] == "Lớp 3 : 6 vắng 1 (Nghĩa)"
