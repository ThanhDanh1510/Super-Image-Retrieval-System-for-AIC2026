"""
serving/api/service/dense_frame_extractor.py
=============================================
Trích xuất khung hình dày đặc trực tiếp từ file video gốc (Video-Level DANTE).
Dùng khi keyframe trích xuất sẵn (~1 kf/scene) không đủ mật độ thời gian để bắt
các khoảnh khắc chính xác từng giây / nửa giây (như "khoảnh khắc đầu tiên dùi chạm kẻng").

Nguyên lý:
  1. Cho video ứng viên (top-K), mở video gốc bằng OpenCV.
  2. Lấy mẫu khung hình với bước nhảy đều đặn (mặc định delta = 0.5 giây).
  3. Encode batch các frame dày đặc bằng CLIP ViT-L/14 image encoder.
  4. Tính ma trận tương đồng Dense Similarity S[N, T_dense] với các sub-events.
  5. Trả về danh sách CandidateResult dày đặc kèm ma trận tương đồng để chạy DANTE DP.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Optional
import cv2
import numpy as np
from PIL import Image

from serving.api.schema.retrieval import CandidateResult
from shared.utils.path_resolver import resolve_video_path
from serving.api.service.dense_similarity import compute_dense_similarity

logger = logging.getLogger(__name__)


def _extract_frames_sync(
    video_path: Path,
    interval_sec: float = 0.5,
    max_frames: int = 600,
) -> tuple[list[tuple[int, float, Image.Image]], float]:
    """
    Trích xuất frames từ file video với bước nhảy interval_sec.

    Returns:
        tuple of (frames_data, fps)
        frames_data: list of (frame_idx, timestamp_sec, pil_image)
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning(f"Không thể mở file video: {video_path}")
        return [], 25.0

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_step = max(1, int(round(fps * interval_sec)))

    extracted: list[tuple[int, float, Image.Image]] = []
    frame_idx = 0

    while frame_idx < total_frames and len(extracted) < max_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame_bgr = cap.read()
        if not ret or frame_bgr is None:
            break

        timestamp_sec = float(frame_idx) / float(fps)
        # Chuyển BGR -> RGB -> PIL Image và resize nhỏ vừa đủ cho CLIP (224x224) để tiết kiệm RAM/tốc độ
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        
        extracted.append((frame_idx, timestamp_sec, pil_img))
        frame_idx += frame_step

    cap.release()
    return extracted, fps


async def extract_and_dense_score_video(
    video_id: str,
    sub_event_embeddings: np.ndarray,
    image_embedder,
    interval_sec: float = 0.5,
    max_frames: int = 600,
) -> Optional[tuple[list[CandidateResult], np.ndarray]]:
    """
    Trích xuất frames dày đặc từ video gốc và tính ma trận tương đồng Dense S[N, T_dense].

    Args:
        video_id: ID của video (ví dụ: L24_V024).
        sub_event_embeddings: Vector CLIP của N sub-events, shape (N, 1024).
        image_embedder: ImageEmbeddingWrapper singleton.
        interval_sec: Khoảng cách giữa các frame trích xuất (giây).
        max_frames: Số lượng frame tối đa trích xuất từ 1 video.

    Returns:
        tuple (dense_candidates, S_dense[N, T_dense]) hoặc None nếu không có video file.
    """
    video_path = resolve_video_path(video_id)
    if not video_path or not video_path.exists():
        return None

    try:
        # 1. Trích xuất frames trong background thread để không block event loop
        frames_data, fps = await asyncio.to_thread(
            _extract_frames_sync, video_path, interval_sec, max_frames
        )

        if not frames_data:
            return None

        # 2. Batch encode các frames bằng CLIP Image Encoder
        pil_images = [item[2] for item in frames_data]
        vectors = await asyncio.to_thread(
            image_embedder.encode, pil_images, batch_size=32
        )
        if hasattr(vectors, "numpy"):
            vectors = vectors.numpy().astype(np.float32)
        elif isinstance(vectors, list):
            vectors = np.array(vectors, dtype=np.float32)

        # 3. Tính Cosine Similarity Dense Matrix S[N, T_dense]
        sim_matrix = compute_dense_similarity(sub_event_embeddings, vectors)

        # 4. Tạo danh sách CandidateResult đại diện cho từng dense frame
        dense_candidates: list[CandidateResult] = []
        for i, (frame_idx, ts_sec, _) in enumerate(frames_data):
            # Điểm max qua các event để gán tạm vào score
            max_ev_score = float(np.max(sim_matrix[:, i])) if sim_matrix.shape[1] > i else 0.0
            dense_candidates.append(
                CandidateResult(
                    video_id=video_id,
                    scene_id=f"{video_id}_dense_{frame_idx:06d}",
                    frame_id=frame_idx,
                    timestamp_start=max(0.0, ts_sec - (interval_sec / 2.0)),
                    timestamp_end=ts_sec + (interval_sec / 2.0),
                    timestamp_mid=ts_sec,
                    keyframe_path=f"unified/{video_id}-{frame_idx}.webp",
                    ocr_text="",
                    asr_text="",
                    description=f"Dense sampled frame at {ts_sec:.2f}s (frame {frame_idx})",
                    score=round(max_ev_score, 4),
                )
            )

        return dense_candidates, sim_matrix

    except Exception as e:
        logger.warning(f"Dense frame extraction failed for video {video_id}: {e}")
        return None
