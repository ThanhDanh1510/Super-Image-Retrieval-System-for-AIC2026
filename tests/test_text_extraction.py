import os
import json
import tempfile
import importlib
import unicodedata
import pandas as pd
import pytest

# Dynamic import vì module name bắt đầu bằng số (01b_text_extraction)
text_ext_mod = importlib.import_module("ingestion.skeletons.01b_text_extraction")

normalize_vietnamese_text = text_ext_mod.normalize_vietnamese_text
import_existing_ocr_json = text_ext_mod.import_existing_ocr_json
process_text_extraction = text_ext_mod.process_text_extraction


def test_normalize_vietnamese_text():
    # Test NFD string -> NFC normalization
    nfd_text = unicodedata.normalize("NFD", "Tiếng Việt có dấu")
    normalized = normalize_vietnamese_text(nfd_text)
    assert unicodedata.is_normalized("NFC", normalized)
    assert normalized == "Tiếng Việt có dấu"

    # Test whitespace strip
    raw_space = "   Hệ   thống   Truy  vấn    "
    assert normalize_vietnamese_text(raw_space) == "Hệ thống Truy vấn"


def test_import_existing_ocr_json(tmp_path):
    # Tạo fake metadata
    meta_df = pd.DataFrame([
        {"scene_id": "L21_V001_S001", "video_id": "L21_V001", "frame_id": 17},
        {"scene_id": "L21_V001_S002", "video_id": "L21_V001", "frame_id": 54},
    ])

    # Tạo fake OCR JSON với format key = frame_id
    ocr_dir = tmp_path / "OCR" / "L21"
    ocr_dir.mkdir(parents=True)
    ocr_json_path = ocr_dir / "L21_V001.json"
    
    ocr_data = {
        "17": unicodedata.normalize("NFD", "TIN CHÍNH 60 GIÂY"),
        "54": "TÌNH TRẠNG SỤT LÚN",
    }
    with open(ocr_json_path, "w", encoding="utf-8") as f:
        json.dump(ocr_data, f, ensure_ascii=False)

    ocr_map = import_existing_ocr_json(str(tmp_path / "OCR"), meta_df)
    
    assert "L21_V001_S001" in ocr_map
    assert ocr_map["L21_V001_S001"] == "TIN CHÍNH 60 GIÂY"
    assert unicodedata.is_normalized("NFC", ocr_map["L21_V001_S001"])
    assert ocr_map["L21_V001_S002"] == "TÌNH TRẠNG SỤT LÚN"


def test_process_text_extraction_skip(tmp_path):
    # Tạo fake scene_metadata.parquet
    meta_path = tmp_path / "scene_metadata.parquet"
    meta_df = pd.DataFrame([
        {
            "scene_id": "L21_V001_S001",
            "video_id": "L21_V001",
            "frame_id": 17,
            "timestamp_start": 0.0,
            "timestamp_end": 5.0,
            "timestamp_mid": 2.5,
            "fps": 25.0,
            "keyframe_path": "L21_V001/L21_V001_S001_000017.jpg",
        }
    ])
    meta_df.to_parquet(meta_path, index=False)

    kf_dir = tmp_path / "keyframes"
    kf_dir.mkdir()
    v_dir = tmp_path / "videos"
    v_dir.mkdir()
    out_dir = tmp_path / "output"

    ocr_df, asr_df = process_text_extraction(
        metadata_path=str(meta_path),
        keyframe_dir=str(kf_dir),
        video_dir=str(v_dir),
        output_dir=str(out_dir),
        skip_ocr=True,
        skip_asr=True,
    )

    assert set(ocr_df.columns) == {"scene_id", "video_id", "ocr_text"}
    assert set(asr_df.columns) == {"scene_id", "video_id", "asr_text"}
    assert len(ocr_df) == 1
    assert len(asr_df) == 1
    assert ocr_df.iloc[0]["ocr_text"] == ""
    assert asr_df.iloc[0]["asr_text"] == ""

    # Verify parquet files written to disk
    assert (out_dir / "ocr_text.parquet").exists()
    assert (out_dir / "asr_text.parquet").exists()
