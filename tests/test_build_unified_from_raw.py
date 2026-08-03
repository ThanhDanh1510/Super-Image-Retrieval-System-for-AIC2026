"""
tests/test_build_unified_from_raw.py
=====================================
Unit tests cho aggregation/build_unified_from_raw.py.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import pandas as pd
from PIL import Image

from aggregation.build_unified_from_raw import (
    parse_raw_images,
    build_scene_metadata,
    extract_ocr_for_metadata,
    extract_asr_for_metadata,
    run_build_unified,
)


@pytest.fixture
def mock_raw_data_dir(tmp_path):
    """
    Tạo cấu trúc thư mục dữ liệu giả lập để test.
    """
    img_dir = tmp_path / "image" / "L21" / "L21" / "V001"
    img_dir.mkdir(parents=True)

    # Tạo 2 ảnh giả lập webp
    img1 = Image.new("RGB", (10, 10), color="red")
    img1.save(img_dir / "100.webp")

    img2 = Image.new("RGB", (10, 10), color="blue")
    img2.save(img_dir / "200.webp")

    # Tạo OCR JSON giả lập
    ocr_dir = tmp_path / "OCR" / "L21"
    ocr_dir.mkdir(parents=True)
    ocr_data = {
        "100": "CHÀO MỪNG BẠN",
        "200": "TIN TỨC 24H",
    }
    with open(ocr_dir / "L21_V001.json", "w", encoding="utf-8") as f:
        json.dump(ocr_data, f, ensure_ascii=False)

    # Tạo ASR JSON giả lập
    asr_dir = tmp_path / "ASR"
    asr_dir.mkdir(parents=True)
    asr_data = [
        {
            "video_name": "L21_V001",
            "start_frame": 50,
            "end_frame": 150,
            "text": "xin chào quý vị và các bạn",
        },
        {
            "video_name": "L21_V001",
            "start_frame": 180,
            "end_frame": 250,
            "text": "chương trình thời sự hôm nay",
        },
    ]
    with open(asr_dir / "asr.json", "w", encoding="utf-8") as f:
        json.dump(asr_data, f, ensure_ascii=False)

    return {
        "image_dir": tmp_path / "image",
        "ocr_dir": tmp_path / "OCR",
        "asr_json": asr_dir / "asr.json",
        "output_dir": tmp_path / "unified_raw",
    }


def test_parse_raw_images(mock_raw_data_dir):
    raw_df = parse_raw_images(mock_raw_data_dir["image_dir"])
    assert len(raw_df) == 2
    assert set(raw_df["video_id"]) == {"L21_V001"}
    assert list(raw_df["frame_id"]) == [100, 200]


def test_build_scene_metadata(mock_raw_data_dir):
    raw_df = parse_raw_images(mock_raw_data_dir["image_dir"])
    meta_df = build_scene_metadata(raw_df, fps=25.0)

    assert len(meta_df) == 2
    assert list(meta_df["scene_id"]) == ["L21_V001_S000", "L21_V001_S001"]
    assert meta_df.loc[0, "timestamp_mid"] == 4.0  # 100 / 25.0
    assert meta_df.loc[1, "timestamp_mid"] == 8.0  # 200 / 25.0


def test_extract_ocr_and_asr(mock_raw_data_dir):
    raw_df = parse_raw_images(mock_raw_data_dir["image_dir"])
    meta_df = build_scene_metadata(raw_df, fps=25.0)

    ocr_df = extract_ocr_for_metadata(meta_df, mock_raw_data_dir["ocr_dir"])
    assert len(ocr_df) == 2
    assert ocr_df.loc[0, "ocr_text"] == "CHÀO MỪNG BẠN"
    assert ocr_df.loc[1, "ocr_text"] == "TIN TỨC 24H"

    asr_df = extract_asr_for_metadata(meta_df, mock_raw_data_dir["asr_json"])
    assert len(asr_df) == 2
    assert asr_df.loc[0, "asr_text"] == "xin chào quý vị và các bạn"
    assert asr_df.loc[1, "asr_text"] == "chương trình thời sự hôm nay"


def test_run_build_unified_end_to_end(mock_raw_data_dir):
    run_build_unified(
        image_dir=str(mock_raw_data_dir["image_dir"]),
        ocr_dir=str(mock_raw_data_dir["ocr_dir"]),
        asr_json=str(mock_raw_data_dir["asr_json"]),
        output_dir=str(mock_raw_data_dir["output_dir"]),
        copy_keyframes=True,
    )

    out_dir = mock_raw_data_dir["output_dir"]
    assert (out_dir / "scene_metadata.parquet").exists()
    assert (out_dir / "ocr_text.parquet").exists()
    assert (out_dir / "asr_text.parquet").exists()
    assert (out_dir / "keyframes" / "L21_V001" / "L21_V001_S000.webp").exists()
