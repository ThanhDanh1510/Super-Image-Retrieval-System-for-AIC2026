import importlib
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

export_module = importlib.import_module("ingestion.skeletons.02_export_shard_dataset")


def test_validate_shard_output_success():
    with tempfile.TemporaryDirectory() as tmpdir:
        base_path = Path(tmpdir)
        
        # 1. Tạo scene_metadata.parquet với 2 scenes
        df_meta = pd.DataFrame({
            "scene_id": ["L01_V001_S001", "L01_V001_S002"],
            "video_id": ["L01_V001", "L01_V001"],
            "frame_id": [150, 300],
            "timestamp_start": [0.0, 5.0],
            "timestamp_end": [5.0, 10.0],
            "timestamp_mid": [2.5, 7.5],
            "fps": [30.0, 30.0],
            "keyframe_path": ["L01_V001/L01_V001_S001_000150.jpg", "L01_V001/L01_V001_S002_000300.jpg"]
        })
        df_meta.to_parquet(base_path / "scene_metadata.parquet", index=False)
        
        # 2. Tạo ocr_text.parquet và asr_text.parquet
        pd.DataFrame({"scene_id": ["L01_V001_S001", "L01_V001_S002"], "ocr_text": ["text1", "text2"]}).to_parquet(base_path / "ocr_text.parquet", index=False)
        pd.DataFrame({"scene_id": ["L01_V001_S001", "L01_V001_S002"], "asr_text": ["audio1", "audio2"]}).to_parquet(base_path / "asr_text.parquet", index=False)

        # 3. Tạo scene_ids.txt
        with open(base_path / "scene_ids.txt", "w", encoding="utf-8") as f:
            f.write("L01_V001_S001\nL01_V001_S002\n")

        # 4. Tạo .npy vector files (N=2)
        np.save(base_path / "image_vectors.npy", np.zeros((2, 1024)))
        np.save(base_path / "ocr_vectors.npy", np.zeros((2, 1536)))
        np.save(base_path / "asr_vectors.npy", np.zeros((2, 1536)))

        # 5. Tạo thư mục keyframes và ảnh giả
        kf_dir = base_path / "keyframes" / "L01_V001"
        kf_dir.mkdir(parents=True)
        (kf_dir / "L01_V001_S001_000150.jpg").touch()

        # Validate
        is_valid, errors = export_module.validate_shard_output(str(base_path))
        assert is_valid is True
        assert len(errors) == 0


def test_validate_shard_output_missing_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        base_path = Path(tmpdir)
        
        # Chỉ tạo 1 vài file, thiếu scene_ids.txt và keyframes/
        np.save(base_path / "image_vectors.npy", np.zeros((2, 1024)))

        is_valid, errors = export_module.validate_shard_output(str(base_path))
        assert is_valid is False
        assert any("Thiếu file bắt buộc" in err for err in errors)


def test_package_dataset():
    with tempfile.TemporaryDirectory() as tmpdir:
        base_path = Path(tmpdir)
        dataset_name = "aic2026-shard-an-v1"
        
        output_dir = export_module.package_dataset(str(base_path), dataset_name)
        assert output_dir == str(base_path.resolve())
        
        meta_json = base_path / "dataset-metadata.json"
        assert meta_json.exists()
        
        with open(meta_json, "r", encoding="utf-8") as f:
            data = json.load(f)
            assert data["title"] == dataset_name
            assert dataset_name in data["id"]
