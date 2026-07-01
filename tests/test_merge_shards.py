"""
tests/test_merge_shards.py
===========================
Unit tests cho logic verify_shards và merge_datasets (TASK-TST-04).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aggregation.verify_shards import verify_shard_integrity, verify_all_shards
from aggregation.merge_datasets import merge_all

def create_mock_shard(
    shard_dir: Path,
    video_ids: list[str],
    scene_ids: list[str],
    num_samples: int,
):
    """Tạo shard giả lập phục vụ test."""
    shard_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Tạo parquets
    meta_df = pd.DataFrame({
        "scene_id": scene_ids,
        "video_id": [video_ids[0]] * num_samples,
        "frame_id": list(range(100, 100 + num_samples)),
        "timestamp_start": [0.0] * num_samples,
        "timestamp_end": [5.0] * num_samples,
        "timestamp_mid": [2.5] * num_samples,
        "fps": [25.0] * num_samples,
        "keyframe_path": [f"{video_ids[0]}/{sid}.jpg" for sid in scene_ids]
    })
    meta_df.to_parquet(shard_dir / "scene_metadata.parquet")
    
    ocr_df = pd.DataFrame({
        "scene_id": scene_ids,
        "video_id": [video_ids[0]] * num_samples,
        "ocr_text": ["hello"] * num_samples
    })
    ocr_df.to_parquet(shard_dir / "ocr_text.parquet")
    
    asr_df = pd.DataFrame({
        "scene_id": scene_ids,
        "video_id": [video_ids[0]] * num_samples,
        "asr_text": ["world"] * num_samples
    })
    asr_df.to_parquet(shard_dir / "asr_text.parquet")
    
    # 2. Tạo vectors
    np.save(shard_dir / "image_vectors.npy", np.random.rand(num_samples, 1024).astype(np.float32))
    np.save(shard_dir / "ocr_vectors.npy", np.random.rand(num_samples, 1536).astype(np.float32))
    np.save(shard_dir / "asr_vectors.npy", np.random.rand(num_samples, 1536).astype(np.float32))
    
    # 3. Tạo scene_ids.txt
    with open(shard_dir / "scene_ids.txt", "w", encoding="utf-8") as f:
        for sid in scene_ids:
            f.write(f"{sid}\n")
            
    # 4. Tạo keyframes folder giả lập
    keys_dir = shard_dir / "keyframes" / video_ids[0]
    keys_dir.mkdir(parents=True, exist_ok=True)
    for sid in scene_ids:
        (keys_dir / f"{sid}.jpg").touch()

def test_verify_shard_integrity_success(tmp_path):
    """Kiểm tra verify_shard_integrity thành công khi dữ liệu chuẩn."""
    shard_dir = tmp_path / "aic2026-shard-danh-v1"
    vids = ["L01_V001"]
    sids = ["L01_V001_S001", "L01_V001_S002"]
    
    create_mock_shard(shard_dir, vids, sids, 2)
    
    errors = verify_shard_integrity(shard_dir, set(vids))
    assert len(errors) == 0

def test_verify_shard_integrity_mismatch(tmp_path):
    """Kiểm tra verify_shard_integrity phát hiện lỗi lệch số lượng mẫu."""
    shard_dir = tmp_path / "aic2026-shard-danh-v1"
    vids = ["L01_V001"]
    sids = ["L01_V001_S001", "L01_V001_S002"]
    
    create_mock_shard(shard_dir, vids, sids, 2)
    
    # Ghi đè file vectors có 3 samples thay vì 2
    np.save(shard_dir / "image_vectors.npy", np.random.rand(3, 1024).astype(np.float32))
    
    errors = verify_shard_integrity(shard_dir, set(vids))
    assert any("image_vectors.npy" in err for err in errors)

def test_verify_shard_integrity_missing_file(tmp_path):
    """Kiểm tra verify_shard_integrity phát hiện thiếu file."""
    shard_dir = tmp_path / "aic2026-shard-danh-v1"
    vids = ["L01_V001"]
    sids = ["L01_V001_S001"]
    
    create_mock_shard(shard_dir, vids, sids, 1)
    
    # Xóa file scene_ids.txt
    (shard_dir / "scene_ids.txt").unlink()
    
    errors = verify_shard_integrity(shard_dir, set(vids))
    assert any("Thiếu file bắt buộc: scene_ids.txt" in err for err in errors)

def test_merge_all_success(tmp_path):
    """Kiểm tra merge_all gộp chính xác parquets và vectors."""
    shards_dir = tmp_path / "raw_shards"
    output_dir = tmp_path / "unified"
    keys_dir = tmp_path / "unified_keyframes"
    
    # Mock shard_assignment.yaml trong configs
    # Ta có thể mock load_shard_assignment hoặc tạo mock file configs/shard_assignment.yaml
    # ở workspace nếu configs_loader đọc từ đó.
    # Tuy nhiên, trong test_merge_all_success, configs_loader sẽ load từ workspace thực tế.
    # Để tránh sửa workspace configs thực tế trong pytest, ta có thể mock load_shard_assignment:
    import aggregation.merge_datasets as merge_datasets
    
    mock_config = {
        "members": {
            "Danh": {
                "dataset_name": "aic2026-shard-danh-v1",
                "video_ids": ["L01_V001"]
            },
            "Vu": {
                "dataset_name": "aic2026-shard-vu-v1",
                "video_ids": ["L01_V002"]
            }
        }
    }
    
    # Tạo mock shard dirs
    create_mock_shard(shards_dir / "aic2026-shard-danh-v1", ["L01_V001"], ["L01_V001_S001"], 1)
    create_mock_shard(shards_dir / "aic2026-shard-vu-v1", ["L01_V002"], ["L01_V002_S001", "L01_V002_S002"], 2)
    
    # Monkeypatch load_shard_assignment
    original_load = merge_datasets.load_shard_assignment
    merge_datasets.load_shard_assignment = lambda: mock_config
    
    try:
        success = merge_all(str(shards_dir), str(output_dir), str(keys_dir))
        assert success
        
        # Verify output parquets
        meta = pd.read_parquet(output_dir / "scene_metadata.parquet")
        assert len(meta) == 3
        assert list(meta["video_id"]) == ["L01_V001", "L01_V002", "L01_V002"]
        
        # Verify vectors shape
        img_vecs = np.load(output_dir / "image_vectors.npy")
        assert img_vecs.shape == (3, 1024)
        
        # Verify copied keyframes
        assert (keys_dir / "L01_V001" / "L01_V001_S001.jpg").exists()
        assert (keys_dir / "L01_V002" / "L01_V002_S001.jpg").exists()
        
    finally:
        # Restore
        merge_datasets.load_shard_assignment = original_load
