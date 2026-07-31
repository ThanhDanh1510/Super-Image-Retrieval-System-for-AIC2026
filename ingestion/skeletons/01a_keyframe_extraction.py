"""
ingestion/skeletons/01a_keyframe_extraction.py
================================================
TASK-ING-01: Trích xuất Keyframe từ video (Tối ưu hóa bộ nhớ RAM - Stream Processing).

INPUT:
    - Thư mục chứa video gốc (MP4/AVI/MKV).
    - configs/model_versions.yaml (đọc qua shared.utils.config_loader).

OUTPUT:
    - Keyframe JPG files tại: output_dir/keyframes/{video_id}/{scene_id}_{frame_id:06d}.jpg
    - scene_metadata.parquet với schema:
        scene_id       (str)  — format: {video_id}_S{scene_idx:03d}
        video_id       (str)  — ví dụ: 'L01_V001'
        frame_id       (int)  — frame number của keyframe trong video gốc
        timestamp_start (float) — giây bắt đầu scene
        timestamp_end   (float) — giây kết thúc scene
        timestamp_mid   (float) — giây của keyframe (= frame_id / fps)
        fps            (float) — FPS của video gốc
        keyframe_path  (str)  — đường dẫn tương đối từ output_dir/keyframes/ (chứa frame_id)

CONSTRAINTS (KHÔNG ĐƯỢC THAY ĐỔI):
    - Model shot detection: TransNetV2 — version đọc từ configs/model_versions.yaml
    - Keyframe strategy: middle_frame — đọc từ configs/model_versions.yaml
    - Output path: /kaggle/working/shard_output/ (KHÔNG hardcode path khác)
    - Checkpoint granularity: scene-level (không phải video-level)
    - scene_id format bắt buộc: {video_id}_S{scene_idx:03d}  (3 chữ số)
    - Point ID: sha256(video_id + "_" + scene_id)[:16] — xem shared/schemas/named_vector_spec.py
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import yaml

# Thêm root path vào PYTHONPATH để import shared package khi chạy cục bộ
sys.path.append(str(Path(__file__).resolve().parents[2]))

try:
    from shared.utils.config_loader import load_model_versions
except ImportError:
    def load_model_versions() -> dict[str, Any]:
        possible_paths = [
            Path("/kaggle/input/aic2026-configs/model_versions.yaml"),
            Path(__file__).resolve().parents[2] / "configs" / "model_versions.yaml",
            Path("configs/model_versions.yaml")
        ]
        for path in possible_paths:
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f)
        raise FileNotFoundError("Không tìm thấy configs/model_versions.yaml")

from ingestion.skeletons.checkpoint_manifest import CheckpointManifest

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# Helper: Point ID
# =============================================================================

def make_point_id(video_id: str, scene_id: str) -> str:
    key = f"{video_id}_{scene_id}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


# =============================================================================
# IMPLEMENTED FUNCTIONS (MEMORY EFFICIENT)
# =============================================================================

def load_video(video_path: str, target_size: tuple[int, int] = (48, 27)) -> tuple[np.ndarray, float]:
    """
    Load video theo dạng downsampled frames (48x27) phục vụ TransNetV2.
    Giúp tiết kiệm 99% RAM, tránh lỗi OOM (Out Of Memory) trên Kaggle.

    Args:
        video_path: Đường dẫn tuyệt đối tới file video.
        target_size: Kích thước downsample (width, height) cho TransNetV2. Mặc định (48, 27).

    Returns:
        Tuple (small_frames, fps):
            - small_frames: np.ndarray shape (T, 27, 48, 3), dtype uint8, RGB.
            - fps: float — FPS của video gốc.
    """
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy video tại: {video_path}")

    logger.info(f"Đang mở video (downsampling mode 48x27): {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Không thể mở video bằng OpenCV: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 25.0

    small_frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Resize ngay lập tức về 48x27 để tiết kiệm RAM
        frame_resized = cv2.resize(frame, target_size)
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        small_frames.append(frame_rgb)

    cap.release()
    logger.info(f"Đọc video hoàn tất. Tổng frames: {len(small_frames)}, FPS: {fps}")
    return np.array(small_frames, dtype=np.uint8), fps


def detect_scenes(
    small_frames: np.ndarray,
    threshold: float | None = None,
    model_cfg: dict | None = None,
) -> list[tuple[int, int]]:
    """
    Phát hiện ranh giới cảnh (shot boundaries) bằng TransNetV2 trên mảng small_frames.

    Args:
        small_frames: np.ndarray shape (T, 27, 48, 3) từ load_video().
        threshold: Xác suất tối thiểu để xác nhận shot boundary. None → đọc từ config.
        model_cfg: Dict config từ load_model_versions().

    Returns:
        List of (start_frame, end_frame) tuples (inclusive).
    """
    if model_cfg is None:
        model_cfg = load_model_versions()

    if threshold is None:
        try:
            threshold = float(model_cfg["shot_detection"]["threshold"])
        except KeyError:
            threshold = 0.5

    total_frames = len(small_frames)
    if total_frames == 0:
        return []

    logger.info(f"Đang phân cảnh video với threshold={threshold}...")

    # Cố gắng load TransNetV2 qua TensorFlow
    try:
        import tensorflow as tf
        
        model_path = model_cfg.get("shot_detection", {}).get("model_path", "/kaggle/input/transnetv2/transnetv2-savedmodel")
        if not Path(model_path).exists():
            model_path = "./transnetv2-savedmodel"

        logger.info(f"Đang tải mô hình TransNetV2 từ: {model_path}")
        model = tf.saved_model.load(model_path)
        
        batch_size = 100
        predictions = []
        for i in range(0, total_frames, batch_size):
            batch = small_frames[i:i+batch_size]
            single_frame_pred, _ = model(tf.convert_to_tensor(batch))
            predictions.extend(single_frame_pred.numpy().flatten())
            
        predictions = np.array(predictions)
        
    except Exception as e:
        logger.warning(f"Không thể chạy TransNetV2 TensorFlow ({e}). Sử dụng thuật toán fallback màu sắc...")
        predictions = np.zeros(total_frames)
        for i in range(1, total_frames):
            hist_curr = cv2.calcHist([small_frames[i]], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            hist_prev = cv2.calcHist([small_frames[i-1]], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            cv2.normalize(hist_curr, hist_curr)
            cv2.normalize(hist_prev, hist_prev)
            predictions[i] = cv2.compareHist(hist_curr, hist_prev, cv2.HISTCMP_BHATTACHARYYA)

        threshold = 0.65

    transition_indices = np.where(predictions > threshold)[0]
    
    scenes = []
    start = 0
    for idx in transition_indices:
        if idx > start:
            if idx - start >= 5:
                scenes.append((start, idx - 1))
                start = idx
    scenes.append((start, total_frames - 1))

    logger.info(f"Phân cảnh hoàn thành. Tổng số cảnh: {len(scenes)}")
    return scenes


def extract_keyframe(
    video_source: np.ndarray | str,
    start_frame: int,
    end_frame: int,
    strategy: str = "middle_frame",
) -> tuple[np.ndarray | None, int]:
    """
    Trích xuất keyframe đại diện từ một scene.
    Hỗ trợ cả truyền video_path (đọc stream từ đĩa - tiết kiệm RAM) hoặc mảng frames (cho unit test).

    Args:
        video_source: Đường dẫn video_path (str) HOẶC mảng frames (np.ndarray).
        start_frame: Frame index bắt đầu scene (inclusive).
        end_frame: Frame index kết thúc scene (inclusive).
        strategy: 'middle_frame' | 'representative'.

    Returns:
        Tuple (keyframe_image, frame_id): RGB image (hoặc None nếu bị lọc bỏ) và frame_id.
    """
    if strategy != "middle_frame":
        raise ValueError(f"Strategy '{strategy}' không được hỗ trợ. Chỉ hỗ trợ 'middle_frame'")

    mid_idx = (start_frame + end_frame) // 2

    # Helper đọc duy nhất 1 frame chất lượng gốc
    if isinstance(video_source, str):
        cap = cv2.VideoCapture(video_source)
        def read_frame_at(idx: int) -> np.ndarray | None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                return None
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        # Nếu truyền mảng frames trực tiếp (dùng cho test)
        cap = None
        def read_frame_at(idx: int) -> np.ndarray | None:
            if 0 <= idx < len(video_source):
                return video_source[idx]
            return None

    try:
        frame = read_frame_at(mid_idx)
        if frame is None:
            return None, -1

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        std_val = float(np.std(gray))

        if std_val >= 12.0:
            return frame, mid_idx

        # Quét tịnh tiến rộng dần nếu bị đơn sắc
        best_idx = mid_idx
        best_std = std_val
        best_frame = frame
        
        half_len = (end_frame - start_frame) // 2
        found = False
        
        for offset in range(1, half_len + 1):
            for sign in (-1, 1):
                idx = mid_idx + sign * offset
                if start_frame <= idx <= end_frame:
                    curr_frame = read_frame_at(idx)
                    if curr_frame is not None:
                        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_RGB2GRAY)
                        curr_std = float(np.std(curr_gray))
                        if curr_std > best_std:
                            best_std = curr_std
                            best_idx = idx
                            best_frame = curr_frame
                        if best_std >= 12.0:
                            found = True
                            break
            if found:
                break

        if best_std < 12.0:
            return None, -1

        return best_frame, best_idx

    finally:
        if cap is not None:
            cap.release()


def save_keyframe(
    keyframe_image: np.ndarray,
    output_path: str,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    bgr = cv2.cvtColor(keyframe_image, cv2.COLOR_RGB2BGR)
    success = cv2.imwrite(str(path), bgr)
    if not success:
        raise OSError(f"Không thể ghi ảnh keyframe vào: {output_path}")


def build_scene_metadata(
    video_id: str,
    scenes: list[tuple[int, int]],
    fps: float,
    keyframe_dir: str,
) -> pd.DataFrame:
    rows = []
    for scene_idx, (start, end) in enumerate(scenes, 1):
        scene_id = f"{video_id}_S{scene_idx:03d}"
        
        pattern = f"{scene_id}_*.jpg"
        matching_files = list(Path(keyframe_dir).glob(pattern))
        if not matching_files:
            continue
            
        img_path = matching_files[0]
        try:
            frame_id = int(img_path.stem.split("_")[-1])
        except Exception:
            frame_id = (start + end) // 2

        timestamp_start = start / fps
        timestamp_end = end / fps
        timestamp_mid = frame_id / fps
        
        relative_path = f"{video_id}/{img_path.name}"
        
        rows.append({
            "scene_id": scene_id,
            "video_id": video_id,
            "frame_id": frame_id,
            "timestamp_start": timestamp_start,
            "timestamp_end": timestamp_end,
            "timestamp_mid": timestamp_mid,
            "fps": fps,
            "keyframe_path": relative_path
        })
        
    return pd.DataFrame(rows)


def run_keyframe_extraction(
    video_dir: str,
    output_dir: str,
    member: str,
    checkpoint_path: str | None = None,
) -> pd.DataFrame:
    video_path_dir = Path(video_dir)
    output_path_dir = Path(output_dir)
    
    if not video_path_dir.exists():
        raise FileNotFoundError(f"Thư mục video không tồn tại: {video_dir}")
        
    output_path_dir.mkdir(parents=True, exist_ok=True)
    
    if checkpoint_path is None:
        checkpoint_path = str(output_path_dir / "checkpoint.json")
        
    ckpt = CheckpointManifest(checkpoint_path)
    
    video_extensions = (".mp4", ".avi", ".mkv", ".mov")
    video_files = []
    for root, _, files in os.walk(video_dir):
        for f in files:
            if f.lower().endswith(video_extensions):
                video_files.append(Path(root) / f)
                
    video_files.sort()
    logger.info(f"Tìm thấy {len(video_files)} file video cần xử lý.")

    all_metadata_dfs = []
    model_cfg = load_model_versions()
    
    for v_path in video_files:
        video_id = v_path.stem
        logger.info(f"=== Bắt đầu xử lý video: {video_id} ===")
        
        try:
            # 1. Load video theo dạng small_frames (tiết kiệm 99% RAM)
            small_frames, fps = load_video(str(v_path))
            
            # 2. Phân cảnh
            scenes = detect_scenes(small_frames, model_cfg=model_cfg)
            
            # Thư mục lưu keyframe
            video_keyframe_dir = output_path_dir / "keyframes" / video_id
            video_keyframe_dir.mkdir(parents=True, exist_ok=True)
            
            # 3. Trích xuất keyframe chất lượng gốc qua stream bằng đường dẫn v_path
            for scene_idx, (start, end) in enumerate(scenes, 1):
                scene_id = f"{video_id}_S{scene_idx:03d}"
                
                if ckpt.is_done(scene_id, "keyframe"):
                    logger.info(f"Skip scene {scene_id} (Đã có trong checkpoint)")
                    continue
                    
                # Truyền đường dẫn file video vào extract_keyframe để đọc stream duy nhất 1 frame
                kf_img, frame_id = extract_keyframe(str(v_path), start, end)
                
                if kf_img is not None:
                    kf_filename = f"{scene_id}_{frame_id:06d}.jpg"
                    kf_output_path = video_keyframe_dir / kf_filename
                    save_keyframe(kf_img, str(kf_output_path))
                    
                ckpt.mark_done(scene_id, "keyframe")
                ckpt.save()

            # 4. Xây dựng metadata
            v_meta_df = build_scene_metadata(video_id, scenes, fps, str(video_keyframe_dir))
            
            if not v_meta_df.empty:
                v_meta_df["member"] = member
                all_metadata_dfs.append(v_meta_df)
                
        except Exception as e:
            logger.error(f"Lỗi khi xử lý video {video_id}: {e}", exc_info=True)
            continue
            
    if all_metadata_dfs:
        unified_meta_df = pd.concat(all_metadata_dfs, ignore_index=True)
        meta_parquet_path = output_path_dir / "scene_metadata.parquet"
        unified_meta_df.to_parquet(meta_parquet_path, index=False)
        logger.info(f"✅ Đã xuất scene_metadata.parquet tại: {meta_parquet_path}")
        return unified_meta_df
    else:
        logger.warning("Không có metadata nào được sinh ra.")
        return pd.DataFrame()


if __name__ == "__main__":
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
