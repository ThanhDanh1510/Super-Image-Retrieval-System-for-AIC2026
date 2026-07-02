"""
ingestion/skeletons/01a_keyframe_extraction.py
================================================
TASK-ING-01: Trích xuất Keyframe từ video.

SKELETON: Implement các hàm có `raise NotImplementedError`.
Paste nội dung file này vào Kaggle Notebook rồi dùng AI Agent fill implementation.

INPUT:
    - Thư mục chứa video gốc (MP4/AVI/MKV).
    - configs/model_versions.yaml (đọc qua shared.utils.config_loader).

OUTPUT:
    - Keyframe JPG files tại: output_dir/keyframes/{video_id}/{scene_id}.jpg
    - scene_metadata.parquet với schema:
        scene_id       (str)  — format: {video_id}_S{scene_idx:03d}
        video_id       (str)  — ví dụ: 'L01_V001'
        frame_id       (int)  — frame number của keyframe trong video gốc
        timestamp_start (float) — giây bắt đầu scene
        timestamp_end   (float) — giây kết thúc scene
        timestamp_mid   (float) — giây của keyframe (= frame_id / fps)
        fps            (float) — FPS của video gốc
        keyframe_path  (str)  — đường dẫn tương đối từ output_dir/keyframes/

CONSTRAINTS (KHÔNG ĐƯỢC THAY ĐỔI):
    - Model shot detection: TransNetV2 — version đọc từ configs/model_versions.yaml
    - Keyframe strategy: middle_frame — đọc từ configs/model_versions.yaml
    - Output path: /kaggle/working/shard_output/ (KHÔNG hardcode path khác)
    - Checkpoint granularity: scene-level (không phải video-level)
    - scene_id format bắt buộc: {video_id}_S{scene_idx:03d}  (3 chữ số)
    - Point ID: sha256(video_id + "_" + scene_id)[:16] — xem shared/schemas/named_vector_spec.py

CÁCH CHẠY (Kaggle Notebook cell):
    !python -m ingestion.skeletons.01a_keyframe_extraction \
        --video-dir /kaggle/input/aic2026-videos \
        --output-dir /kaggle/working/shard_output \
        --member an
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Load config — KHÔNG hardcode model version hay threshold
# from shared.utils.config_loader import load_model_versions
# Trên Kaggle, nếu chưa mount shared/, dùng yaml.safe_load trực tiếp:
#   import yaml
#   with open('/kaggle/input/aic2026-configs/model_versions.yaml') as f:
#       _cfg = yaml.safe_load(f)

logger = logging.getLogger(__name__)


# =============================================================================
# Helper: Point ID (KHÔNG THAY ĐỔI — phải nhất quán với serving/)
# =============================================================================

def make_point_id(video_id: str, scene_id: str) -> str:
    """
    Tạo Point ID deterministic cho mỗi keyframe.

    RÀNG BUỘC CỨNG:
        - Phải nhất quán 100% với shared/schemas/named_vector_spec.make_point_id()
        - Không dùng random — sẽ gây trùng ID khi upsert lại

    Args:
        video_id: ID video, ví dụ 'L01_V001'
        scene_id: ID scene, ví dụ 'L01_V001_S003'

    Returns:
        Chuỗi hex 16 ký tự — UUID-like deterministic ID.
    """
    key = f"{video_id}_{scene_id}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


# =============================================================================
# SKELETON FUNCTIONS — Implement các hàm dưới đây
# =============================================================================

def load_video(video_path: str) -> tuple[Any, float]:
    """
    Load video từ đường dẫn và trả về mảng frames và FPS.

    Gợi ý implementation:
        import cv2
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        return np.array(frames), fps

    Args:
        video_path: Đường dẫn tuyệt đối tới file video.

    Returns:
        Tuple (frames, fps):
            - frames: np.ndarray, shape (T, H, W, 3), dtype uint8, RGB.
            - fps: float — FPS của video gốc.

    Raises:
        FileNotFoundError: Nếu video_path không tồn tại.
        RuntimeError: Nếu không thể mở video.
    """
    raise NotImplementedError


def detect_scenes(
    frames: Any,
    threshold: float | None = None,
    model_cfg: dict | None = None,
) -> list[tuple[int, int]]:
    """
    Phát hiện ranh giới cảnh (shot boundaries) bằng TransNetV2.

    Threshold đọc từ configs/model_versions.yaml::shot_detection.threshold
    NẾU threshold không được truyền vào.

    Gợi ý implementation:
        - Load TransNetV2 weights từ Kaggle Dataset.
        - Chạy model inference theo batch (mỗi batch 100 frames).
        - threshold mặc định = model_cfg['shot_detection']['threshold'] = 0.5

    Args:
        frames: np.ndarray, shape (T, H, W, 3) — RGB frames từ load_video().
        threshold: Xác suất tối thiểu để xác nhận shot boundary. None → đọc từ config.
        model_cfg: Dict config từ load_model_versions(). None → tự load.

    Returns:
        List of (start_frame, end_frame) tuples (inclusive).
        Ví dụ: [(0, 49), (50, 120), (121, 199)]

    Raises:
        RuntimeError: Nếu model TransNetV2 không load được.
    """
    raise NotImplementedError


def extract_keyframe(
    frames: Any,
    start_frame: int,
    end_frame: int,
    strategy: str = "middle_frame",
) -> tuple[Any, int]:
    """
    Trích xuất keyframe đại diện từ một scene.

    Strategy đọc từ configs/model_versions.yaml::keyframe.strategy.
    Hiện tại chỉ hỗ trợ 'middle_frame' — lấy frame ở giữa scene.

    Gợi ý implementation (middle_frame):
        mid_idx = (start_frame + end_frame) // 2
        keyframe = frames[mid_idx]  # shape (H, W, 3)
        return keyframe, mid_idx

    Args:
        frames: np.ndarray, shape (T, H, W, 3).
        start_frame: Frame index bắt đầu scene (inclusive).
        end_frame: Frame index kết thúc scene (inclusive).
        strategy: 'middle_frame' | 'representative'. Chỉ implement middle_frame.

    Returns:
        Tuple (keyframe_image, frame_id):
            - keyframe_image: np.ndarray shape (H, W, 3), RGB.
            - frame_id: int — frame number trong video gốc (dùng để tính timestamp).

    Raises:
        ValueError: Nếu strategy không được hỗ trợ.
    """
    raise NotImplementedError


def build_scene_metadata(
    video_id: str,
    scenes: list[tuple[int, int]],
    fps: float,
    keyframe_dir: str,
) -> pd.DataFrame:
    """
    Xây dựng bảng metadata cho tất cả scene trong một video.

    Args:
        video_id: ID video, ví dụ 'L01_V001'.
        scenes: List (start_frame, end_frame) từ detect_scenes().
        fps: FPS của video gốc.
        keyframe_dir: Thư mục chứa file JPG keyframe đã lưu.
            Format tên file: {scene_id}.jpg

    Returns:
        pd.DataFrame với các cột bắt buộc:
            scene_id (str), video_id (str), frame_id (int),
            timestamp_start (float), timestamp_end (float),
            timestamp_mid (float), fps (float), keyframe_path (str)

        keyframe_path là đường dẫn tương đối từ root output_dir/keyframes/.
        Ví dụ: 'L01_V001/L01_V001_S003.jpg'

    Raises:
        ValueError: Nếu scenes rỗng.
    """
    raise NotImplementedError


def save_keyframe(
    keyframe_image: Any,
    output_path: str,
) -> None:
    """
    Lưu keyframe ra file JPG.

    Gợi ý:
        import cv2
        # Chuyển RGB → BGR trước khi lưu bằng cv2
        bgr = cv2.cvtColor(keyframe_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, bgr)

    Args:
        keyframe_image: np.ndarray shape (H, W, 3), RGB.
        output_path: Đường dẫn đầy đủ tới file JPG đích.
    """
    raise NotImplementedError


def run_keyframe_extraction(
    video_dir: str,
    output_dir: str,
    member: str,
    checkpoint_path: str | None = None,
) -> pd.DataFrame:
    """
    Entry point chính: Xử lý toàn bộ video trong video_dir.

    Luồng xử lý cho mỗi video:
        1. load_video() → frames, fps
        2. detect_scenes() → scenes list
        3. Với mỗi scene:
            a. extract_keyframe() → keyframe_image, frame_id
            b. save_keyframe() → lưu JPG
            c. checkpoint_manifest.mark_done() → ghi checkpoint
        4. build_scene_metadata() → metadata DataFrame
    5. Concatenate metadata của tất cả video
    6. Lưu ra output_dir/scene_metadata.parquet

    QUAN TRỌNG:
        - Dùng CheckpointManifest để resume nếu session Kaggle bị ngắt.
        - Skip video/scene đã có trong checkpoint.
        - Output path bắt buộc: /kaggle/working/shard_output/

    Args:
        video_dir: Thư mục chứa video gốc (flatten hoặc có sub-folder).
        output_dir: Thư mục output. Nên là '/kaggle/working/shard_output/'.
        member: Tên member đang chạy (ví dụ: 'an', 'binh'). Ghi vào metadata.
        checkpoint_path: Đường dẫn file checkpoint.json. None → mặc định.

    Returns:
        pd.DataFrame tổng hợp metadata của tất cả scene đã xử lý.

    Raises:
        FileNotFoundError: Nếu video_dir không tồn tại.
    """
    raise NotImplementedError


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="TASK-ING-01: Trích xuất Keyframe từ video.")
    parser.add_argument("--video-dir", required=True, help="Thư mục chứa video gốc.")
    parser.add_argument("--output-dir", default="/kaggle/working/shard_output", help="Thư mục lưu kết quả.")
    parser.add_argument("--member", required=True, help="Tên thành viên chạy pipeline.")
    parser.add_argument("--checkpoint-path", default=None, help="Đường dẫn file checkpoint.")
    
    args = parser.parse_args()
    
    run_keyframe_extraction(
        video_dir=args.video_dir,
        output_dir=args.output_dir,
        member=args.member,
        checkpoint_path=args.checkpoint_path,
    )
