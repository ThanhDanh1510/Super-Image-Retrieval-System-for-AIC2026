import importlib
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ingestion.skeletons.checkpoint_manifest import CheckpointManifest

# Dùng importlib vì tên file bắt đầu bằng số '01a' không thể import trực tiếp bằng cú pháp thông thường
kf_extractor = importlib.import_module("ingestion.skeletons.01a_keyframe_extraction")


def test_checkpoint_manifest():
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "checkpoint.json"
        
        # 1. Khởi tạo manifest mới
        ckpt = CheckpointManifest(str(ckpt_path))
        all_scenes = ["L01_V001_S001", "L01_V001_S002", "L01_V001_S003"]
        
        # Check get_remaining ban đầu
        remaining = ckpt.get_remaining(all_scenes, step="keyframe")
        assert len(remaining) == 3
        assert "L01_V001_S001" in remaining
        
        # 2. Mark done và check
        ckpt.mark_done("L01_V001_S001", step="keyframe")
        assert ckpt.is_done("L01_V001_S001", step="keyframe") is True
        assert ckpt.is_done("L01_V001_S002", step="keyframe") is False
        
        # Check get_remaining sau khi mark
        remaining = ckpt.get_remaining(all_scenes, step="keyframe")
        assert len(remaining) == 2
        assert "L01_V001_S001" not in remaining
        
        # 3. Ghi ra đĩa
        ckpt.save()
        assert ckpt_path.exists()
        
        # 4. Load lại manifest và kiểm tra bảo toàn trạng thái
        ckpt_loaded = CheckpointManifest.load(str(ckpt_path))
        assert ckpt_loaded.is_done("L01_V001_S001", step="keyframe") is True
        assert ckpt_loaded.is_done("L01_V001_S002", step="keyframe") is False
        
        summary = ckpt_loaded.summary()
        assert summary["keyframe"] == 1
        assert summary["ocr"] == 0


def test_extract_keyframe_with_info():
    # Tạo fake video frames (T=5, H=10, W=10, C=3) chứa texture phong phú
    # (Độ lệch chuẩn cao để không bị lọc)
    frames = np.random.randint(0, 255, size=(5, 10, 10, 3), dtype=np.uint8)
    
    # Trích xuất scene từ frame 0 đến 4
    kf_img, frame_id = kf_extractor.extract_keyframe(frames, 0, 4)
    
    assert kf_img is not None
    assert frame_id == 2  # Mặc định lấy frame ở giữa là 2
    assert kf_img.shape == (10, 10, 3)


def test_extract_keyframe_monochrome_filtered():
    # Tạo fake video frames chứa cảnh đơn sắc hoàn toàn (độ lệch chuẩn = 0)
    # Ví dụ cảnh tối đen dài 5 frames
    frames = np.zeros((5, 10, 10, 3), dtype=np.uint8)
    
    kf_img, frame_id = kf_extractor.extract_keyframe(frames, 0, 4)
    
    # Phải bị lọc bỏ hoàn toàn và trả về None, -1
    assert kf_img is None
    assert frame_id == -1


def test_extract_keyframe_monochrome_fallback():
    # Tạo fake video frames trong đó frame giữa (idx=2) bị đen xì (đơn sắc),
    # nhưng các frame lân cận có thông tin phong phú (std > 12.0)
    frames = np.zeros((5, 10, 10, 3), dtype=np.uint8)
    # Frame 1 có texture phong phú
    frames[1] = np.random.randint(50, 200, size=(10, 10, 3), dtype=np.uint8)
    
    kf_img, frame_id = kf_extractor.extract_keyframe(frames, 0, 4)
    
    # Phải tự động quét xung quanh và tìm thấy frame 1 thay thế
    assert kf_img is not None
    assert frame_id == 1
    assert np.std(kf_img) > 12.0


def test_build_scene_metadata(tmp_path):
    # Setup giả lập các file keyframe đã lưu trên đĩa
    keyframe_dir = tmp_path / "keyframes" / "L01_V001"
    keyframe_dir.mkdir(parents=True)
    
    # Scene 1 lưu thành công
    (keyframe_dir / "L01_V001_S001_000002.jpg").touch()
    # Scene 2 bị lọc (không tạo file ảnh)
    
    scenes = [(0, 4), (5, 9)]
    df = kf_extractor.build_scene_metadata("L01_V001", scenes, fps=25.0, keyframe_dir=str(keyframe_dir))
    
    # Kiểm tra metadata chỉ chứa scene 1, bỏ qua scene 2
    assert len(df) == 1
    row = df.iloc[0]
    assert row["scene_id"] == "L01_V001_S001"
    assert row["video_id"] == "L01_V001"
    assert row["frame_id"] == 2
    assert row["timestamp_mid"] == 2 / 25.0
    assert row["keyframe_path"] == "L01_V001/L01_V001_S001_000002.jpg"
