"""
ingestion/skeletons/01b_text_extraction.py
============================================
TASK-ING-02: Trích xuất text từ keyframes (OCR) và audio video (ASR).

SKELETON: Implement các hàm có `raise NotImplementedError`.

INPUT:
    - scene_metadata.parquet (từ TASK-ING-01)
    - Keyframe JPG files (từ TASK-ING-01)
    - Video gốc (để extract audio cho ASR)

OUTPUT:
    - ocr_text.parquet  — schema: (scene_id, video_id, ocr_text)
    - asr_text.parquet  — schema: (scene_id, video_id, asr_text)

CONSTRAINTS (KHÔNG ĐƯỢC THAY ĐỔI):
    - Model OCR: PaddleOCR v6 — version đọc từ configs/model_versions.yaml, KHÔNG hardcode
    - Model ASR: PhoWhisper-medium — version đọc từ configs/model_versions.yaml, KHÔNG hardcode
    - Nếu không nhận dạng được text/audio → lưu chuỗi rỗng '' (KHÔNG lưu None hay NaN)
    - Output encoding: UTF-8

CÁCH CHẠY (Kaggle Notebook cell):
    !python -m ingestion.skeletons.01b_text_extraction \
        --metadata-path /kaggle/working/shard_output/scene_metadata.parquet \
        --keyframe-dir /kaggle/working/shard_output/keyframes \
        --video-dir /kaggle/input/aic2026-videos \
        --output-dir /kaggle/working/shard_output
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

# Đọc config — KHÔNG hardcode model version
# from shared.utils.config_loader import load_model_versions
# Trên Kaggle: import yaml; cfg = yaml.safe_load(open('/kaggle/input/.../model_versions.yaml'))

logger = logging.getLogger(__name__)


# =============================================================================
# SKELETON FUNCTIONS — Implement các hàm dưới đây
# =============================================================================

def init_ocr_engine(model_cfg: dict) -> Any:
    """
    Khởi tạo PaddleOCR engine từ cấu hình trong model_versions.yaml.

    Version OCR đọc từ: model_cfg['ocr']['version'] == 'v6'
    Ngôn ngữ đọc từ:    model_cfg['ocr']['lang']    == 'vi'

    Gợi ý implementation:
        from paddleocr import PaddleOCR
        ocr = PaddleOCR(
            use_angle_cls=True,
            lang=model_cfg['ocr']['lang'],    # 'vi'
            use_gpu=model_cfg['ocr']['use_gpu'],
        )
        return ocr

    Args:
        model_cfg: Dict đọc từ load_model_versions().

    Returns:
        PaddleOCR instance đã khởi tạo.

    Raises:
        ImportError: Nếu paddleocr chưa được cài đặt.
        RuntimeError: Nếu không khởi tạo được engine.
    """
    raise NotImplementedError


def init_asr_model(model_cfg: dict) -> Any:
    """
    Khởi tạo ASR model (PhoWhisper) từ cấu hình trong model_versions.yaml.

    Model ID đọc từ: model_cfg['asr']['model'] == 'vinai/PhoWhisper-medium'
    Language đọc từ: model_cfg['asr']['language'] == 'vi'

    Gợi ý implementation:
        from transformers import pipeline
        asr = pipeline(
            'automatic-speech-recognition',
            model=model_cfg['asr']['model'],      # 'vinai/PhoWhisper-medium'
            device=0 if model_cfg['asr']['device'] == 'cuda' else -1,
        )
        return asr

    Args:
        model_cfg: Dict đọc từ load_model_versions().

    Returns:
        HuggingFace ASR pipeline instance đã load model.

    Raises:
        RuntimeError: Nếu không load được model.
    """
    raise NotImplementedError


def run_ocr(ocr_engine: Any, image_path: str) -> str:
    """
    Chạy OCR trên một ảnh keyframe và trả về text nhận dạng được.

    Gợi ý implementation:
        result = ocr_engine.ocr(image_path, cls=True)
        if not result or not result[0]:
            return ''
        texts = [line[1][0] for line in result[0] if line and line[1]]
        return ' '.join(texts).strip()

    Args:
        ocr_engine: PaddleOCR instance từ init_ocr_engine().
        image_path: Đường dẫn tuyệt đối tới file JPG keyframe.

    Returns:
        Chuỗi text đã nhận dạng. Trả về '' nếu không có text.

    Raises:
        FileNotFoundError: Nếu image_path không tồn tại.
    """
    raise NotImplementedError


def extract_audio_segment(
    video_path: str,
    start_sec: float,
    end_sec: float,
    tmp_dir: str = "/tmp",
) -> str:
    """
    Trích xuất đoạn audio từ video và lưu ra file WAV tạm thời.

    Gợi ý implementation:
        import subprocess, tempfile, os
        tmp_path = os.path.join(tmp_dir, f'audio_{start_sec:.2f}_{end_sec:.2f}.wav')
        duration = end_sec - start_sec
        cmd = [
            'ffmpeg', '-y', '-i', video_path,
            '-ss', str(start_sec), '-t', str(duration),
            '-vn', '-ar', '16000', '-ac', '1',
            tmp_path
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return tmp_path

    Args:
        video_path: Đường dẫn tuyệt đối tới file video.
        start_sec: Giây bắt đầu đoạn audio cần extract.
        end_sec: Giây kết thúc đoạn audio cần extract.
        tmp_dir: Thư mục tạm để lưu file WAV.

    Returns:
        Đường dẫn tuyệt đối tới file WAV tạm thời.

    Raises:
        FileNotFoundError: Nếu video_path không tồn tại.
        RuntimeError: Nếu ffmpeg không khả dụng hoặc extract thất bại.
    """
    raise NotImplementedError


def run_asr(asr_model: Any, audio_path: str) -> str:
    """
    Chạy ASR trên file audio và trả về transcript text.

    Gợi ý implementation:
        result = asr_model(audio_path)
        return result.get('text', '').strip()

    Args:
        asr_model: HuggingFace ASR pipeline từ init_asr_model().
        audio_path: Đường dẫn tới file WAV tạm thời (từ extract_audio_segment()).

    Returns:
        Transcript text (tiếng Việt). Trả về '' nếu không có giọng nói.

    Raises:
        FileNotFoundError: Nếu audio_path không tồn tại.
    """
    raise NotImplementedError


def process_text_extraction(
    metadata_path: str,
    keyframe_dir: str,
    video_dir: str,
    output_dir: str,
    checkpoint_path: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Entry point chính: Chạy OCR + ASR cho toàn bộ scenes trong metadata.

    Luồng xử lý:
        1. Load scene_metadata.parquet
        2. Init OCR engine và ASR model
        3. Với mỗi row trong metadata:
            a. OCR: run_ocr(ocr_engine, keyframe_path) → ocr_text
            b. ASR: extract_audio_segment() + run_asr() → asr_text
            c. Nếu có lỗi → ghi '' (KHÔNG bỏ qua row)
            d. checkpoint_manifest.mark_done(scene_id, step='text')
        4. Lưu ocr_text.parquet và asr_text.parquet

    QUAN TRỌNG:
        - Dùng CheckpointManifest để resume.
        - Xóa file WAV tạm sau mỗi scene để tránh đầy đĩa Kaggle.
        - Log progress mỗi 50 scene.

    Args:
        metadata_path: Đường dẫn tới scene_metadata.parquet.
        keyframe_dir: Thư mục chứa keyframe JPG.
        video_dir: Thư mục chứa video gốc (dùng cho ASR).
        output_dir: Thư mục output (/kaggle/working/shard_output/).
        checkpoint_path: Đường dẫn checkpoint.json. None → mặc định.

    Returns:
        Tuple (ocr_df, asr_df):
            - ocr_df: pd.DataFrame với cột (scene_id, video_id, ocr_text)
            - asr_df: pd.DataFrame với cột (scene_id, video_id, asr_text)

    Raises:
        FileNotFoundError: Nếu metadata_path không tồn tại.
    """
    raise NotImplementedError


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="TASK-ING-02: Trích xuất text từ keyframes (OCR) và audio video (ASR).")
    parser.add_argument("--metadata-path", required=True, help="Đường dẫn tới scene_metadata.parquet.")
    parser.add_argument("--keyframe-dir", required=True, help="Thư mục chứa keyframe JPG.")
    parser.add_argument("--video-dir", required=True, help="Thư mục chứa video gốc.")
    parser.add_argument("--output-dir", default="/kaggle/working/shard_output", help="Thư mục lưu kết quả.")
    parser.add_argument("--checkpoint-path", default=None, help="Đường dẫn file checkpoint.")
    
    args = parser.parse_args()
    
    process_text_extraction(
        metadata_path=args.metadata_path,
        keyframe_dir=args.keyframe_dir,
        video_dir=args.video_dir,
        output_dir=args.output_dir,
        checkpoint_path=args.checkpoint_path,
    )
