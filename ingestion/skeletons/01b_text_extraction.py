"""
ingestion/skeletons/01b_text_extraction.py
============================================
TASK-ING-02: Trích xuất text từ keyframes (OCR) và audio video (ASR).

INPUT:
    - scene_metadata.parquet (từ TASK-ING-01)
    - Keyframe JPG files (từ TASK-ING-01)
    - Video gốc (để extract audio cho ASR)

OUTPUT:
    - ocr_text.parquet  — schema: (scene_id, video_id, ocr_text)
    - asr_text.parquet  — schema: (scene_id, video_id, asr_text)

CONSTRAINTS:
    - Model OCR: PaddleOCR v6 — version đọc từ configs/model_versions.yaml, KHÔNG hardcode
    - Model ASR: PhoWhisper-medium — version đọc từ configs/model_versions.yaml, KHÔNG hardcode
    - Chuẩn hóa Unicode tiếng Việt NFC cho tất cả kết quả văn bản.
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

import argparse
import json
import logging
import os
import subprocess
import tempfile
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd

from ingestion.skeletons.checkpoint_manifest import CheckpointManifest
from shared.utils.config_loader import load_model_versions
from shared.utils.text_normalizer import normalize_nfc

logger = logging.getLogger(__name__)


def normalize_vietnamese_text(text: str) -> str:
    """Chuẩn hóa văn bản tiếng Việt sang dạng Unicode NFC và làm sạch khoảng trắng."""
    return normalize_nfc(text)


# =============================================================================
# IMPLEMENTED FUNCTIONS
# =============================================================================

def init_ocr_engine(model_cfg: dict) -> Any:
    """
    Khởi tạo PaddleOCR engine từ cấu hình trong model_versions.yaml.
    """
    ocr_cfg = model_cfg.get("ocr", {})
    lang = ocr_cfg.get("lang", "vi")
    use_gpu = ocr_cfg.get("use_gpu", True)

    try:
        from paddleocr import PaddleOCR
    except ImportError as e:
        raise ImportError(
            "Chưa cài đặt package 'paddleocr'. "
            "Vui lòng chạy: pip install paddlepaddle paddleocr"
        ) from e

    try:
        logger.info(f"Đang khởi tạo PaddleOCR (lang='{lang}', use_gpu={use_gpu})...")
        ocr_engine = PaddleOCR(
            use_angle_cls=True,
            lang=lang,
            use_gpu=use_gpu,
            show_log=False,
        )
        return ocr_engine
    except Exception as e:
        logger.warning(f"Không thể khởi tạo PaddleOCR GPU mode: {e}. Thử lại với use_gpu=False...")
        try:
            ocr_engine = PaddleOCR(
                use_angle_cls=True,
                lang=lang,
                use_gpu=False,
                show_log=False,
            )
            return ocr_engine
        except Exception as err:
            raise RuntimeError(f"Lỗi khởi tạo PaddleOCR engine: {err}") from err


def init_asr_model(model_cfg: dict) -> Any:
    """
    Khởi tạo ASR model (PhoWhisper) từ cấu hình trong model_versions.yaml.
    """
    asr_cfg = model_cfg.get("asr", {})
    model_name = asr_cfg.get("model", "vinai/PhoWhisper-medium")
    language = asr_cfg.get("language", "vi")
    device_cfg = asr_cfg.get("device", "cuda")

    try:
        import torch
        from transformers import pipeline
    except ImportError as e:
        raise ImportError(
            "Chưa cài đặt package 'transformers' hoặc 'torch'. "
            "Vui lòng chạy: pip install transformers torch soundfile"
        ) from e

    device_id = 0 if (device_cfg == "cuda" and torch.cuda.is_available()) else -1
    torch_dtype = torch.float16 if device_id >= 0 else torch.float32

    logger.info(f"Đang load ASR model '{model_name}' (device={device_id}, dtype={torch_dtype})...")
    try:
        asr_pipeline = pipeline(
            "automatic-speech-recognition",
            model=model_name,
            device=device_id,
            torch_dtype=torch_dtype,
            generate_kwargs={"language": language, "task": "transcribe"},
        )
        return asr_pipeline
    except Exception as e:
        raise RuntimeError(f"Không thể load ASR model '{model_name}': {e}") from e


def run_ocr(ocr_engine: Any, image_path: str) -> str:
    """
    Chạy OCR trên một ảnh keyframe và trả về text nhận dạng tiếng Việt.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"File ảnh keyframe không tồn tại: {image_path}")

    try:
        result = ocr_engine.ocr(str(path), cls=True)
        if not result or not result[0]:
            return ""

        extracted_lines = []
        for line in result[0]:
            if line and len(line) >= 2 and line[1]:
                text_content = line[1][0]
                if text_content:
                    extracted_lines.append(text_content)

        full_text = " ".join(extracted_lines)
        return normalize_vietnamese_text(full_text)
    except Exception as e:
        logger.warning(f"Lỗi khi chạy OCR trên ảnh {image_path}: {e}")
        return ""


def extract_audio_segment(
    video_path: str,
    start_sec: float,
    end_sec: float,
    tmp_dir: str = "/tmp",
) -> str:
    """
    Trích xuất đoạn audio từ video và lưu ra file WAV 16kHz Mono tạm thời.
    """
    v_path = Path(video_path)
    if not v_path.exists():
        raise FileNotFoundError(f"File video không tồn tại: {video_path}")

    os.makedirs(tmp_dir, exist_ok=True)

    # Đảm bảo start/end hợp lệ
    duration = max(0.1, end_sec - start_sec)
    
    # Tạo tên file WAV tạm duy nhất
    tmp_file = tempfile.NamedTemporaryFile(
        suffix=".wav",
        prefix=f"audio_{v_path.stem}_",
        dir=tmp_dir,
        delete=False,
    )
    tmp_path = tmp_file.name
    tmp_file.close()

    cmd = [
        "ffmpeg",
        "-y",
        "-ss", str(start_sec),
        "-t", str(duration),
        "-i", str(v_path),
        "-vn",
        "-ar", "16000",
        "-ac", "1",
        "-loglevel", "error",
        tmp_path,
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return tmp_path
    except subprocess.CalledProcessError as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise RuntimeError(f"Lỗi FFmpeg khi trích xuất audio từ {video_path}: {e.stderr.decode('utf-8', errors='ignore')}") from e
    except FileNotFoundError as e:
        raise RuntimeError("Hệ thống chưa cài đặt 'ffmpeg'. Vui lòng cài đặt ffmpeg trước khi chạy ASR.") from e


def run_asr(asr_model: Any, audio_path: str) -> str:
    """
    Chạy ASR trên file audio và trả về transcript text tiếng Việt.
    """
    a_path = Path(audio_path)
    if not a_path.exists():
        raise FileNotFoundError(f"File audio không tồn tại: {audio_path}")

    try:
        result = asr_model(str(a_path))
        text = ""
        if isinstance(result, dict):
            text = result.get("text", "")
        elif isinstance(result, str):
            text = result
        return normalize_vietnamese_text(text)
    except Exception as e:
        logger.warning(f"Lỗi khi chạy ASR trên {audio_path}: {e}")
        return ""


def import_existing_ocr_json(import_dir: str, metadata_df: pd.DataFrame) -> dict[str, str]:
    """
    Nạp dữ liệu OCR từ thư mục chứa các file JSON có sẵn (được đánh theo frame_id hoặc index).
    """
    ocr_map = {}
    import_path = Path(import_dir)
    if not import_path.exists():
        logger.warning(f"Thư mục import OCR không tồn tại: {import_dir}")
        return ocr_map

    logger.info(f"Đang nạp dữ liệu OCR có sẵn từ: {import_path}")

    # Đọc tất cả các file json trong import_path (ví dụ: L21_V001.json)
    json_files = list(import_path.rglob("*.json"))
    json_cache = {}
    for jf in json_files:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                json_cache[jf.stem] = json.load(f)
        except Exception as e:
            logger.warning(f"Không thể đọc file OCR JSON {jf}: {e}")

    for row in metadata_df.itertuples():
        scene_id = row.scene_id
        video_id = row.video_id
        frame_id = str(getattr(row, "frame_id", 0))

        ocr_text = ""
        if video_id in json_cache:
            v_json = json_cache[video_id]
            # Thử lấy theo frame_id (ví dụ "17", "54") hoặc theo scene_id
            if frame_id in v_json:
                ocr_text = str(v_json[frame_id])
            elif scene_id in v_json:
                ocr_text = str(v_json[scene_id])

        ocr_map[scene_id] = normalize_vietnamese_text(ocr_text)

    return ocr_map


def process_text_extraction(
    metadata_path: str,
    keyframe_dir: str,
    video_dir: str,
    output_dir: str,
    checkpoint_path: str | None = None,
    import_ocr_dir: str | None = None,
    import_asr_dir: str | None = None,
    skip_ocr: bool = False,
    skip_asr: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Entry point chính: Chạy OCR + ASR cho toàn bộ scenes trong metadata.
    """
    meta_p = Path(metadata_path)
    if not meta_p.exists():
        raise FileNotFoundError(f"File metadata không tồn tại: {metadata_path}")

    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    metadata_df = pd.read_parquet(meta_p)
    logger.info(f"Đã nạp {len(metadata_df)} scenes từ {metadata_path}")

    checkpoint_file = checkpoint_path or str(out_p / "checkpoint.json")
    manifest = CheckpointManifest(checkpoint_file)

    # 1. Nạp/Khởi tạo OCR
    ocr_map: dict[str, str] = {}
    ocr_engine = None
    if import_ocr_dir:
        ocr_map = import_existing_ocr_json(import_ocr_dir, metadata_df)
    elif not skip_ocr:
        try:
            cfg = load_model_versions()
            ocr_engine = init_ocr_engine(cfg)
        except Exception as e:
            logger.error(f"Không thể khởi tạo OCR Engine: {e}. Sẽ dùng chuỗi rỗng cho OCR.")

    # 2. Nạp/Khởi tạo ASR
    asr_map: dict[str, str] = {}
    asr_model = None
    if not skip_asr and not import_asr_dir:
        try:
            cfg = load_model_versions()
            asr_model = init_asr_model(cfg)
        except Exception as e:
            logger.error(f"Không thể khởi tạo ASR Model: {e}. Sẽ dùng chuỗi rỗng cho ASR.")

    # 3. Lặp qua từng scene
    ocr_rows = []
    asr_rows = []

    kf_dir_p = Path(keyframe_dir)
    v_dir_p = Path(video_dir)

    processed_count = 0
    total_scenes = len(metadata_df)

    for row in metadata_df.itertuples():
        scene_id = row.scene_id
        video_id = row.video_id
        rel_kf_path = getattr(row, "keyframe_path", "")
        start_sec = getattr(row, "timestamp_start", 0.0)
        end_sec = getattr(row, "timestamp_end", 0.0)

        # Trích xuất OCR
        if scene_id in ocr_map:
            ocr_text = ocr_map[scene_id]
        elif ocr_engine is not None:
            # Tìm file ảnh
            img_p = kf_dir_p / rel_kf_path
            if not img_p.exists():
                img_p = kf_dir_p / video_id / f"{scene_id}.jpg"

            if img_p.exists():
                ocr_text = run_ocr(ocr_engine, str(img_p))
            else:
                ocr_text = ""
        else:
            ocr_text = ""

        # Trích xuất ASR
        if scene_id in asr_map:
            asr_text = asr_map[scene_id]
        elif asr_model is not None:
            # Tìm file video
            v_p = v_dir_p / f"{video_id}.mp4"
            if not v_p.exists():
                v_p = v_dir_p / f"{video_id}.mkv"
                
            if v_p.exists():
                tmp_wav = None
                try:
                    tmp_wav = extract_audio_segment(str(v_p), start_sec, end_sec)
                    asr_text = run_asr(asr_model, tmp_wav)
                except Exception as e:
                    logger.warning(f"Lỗi extract/run ASR cho {scene_id}: {e}")
                    asr_text = ""
                finally:
                    if tmp_wav and os.path.exists(tmp_wav):
                        try:
                            os.remove(tmp_wav)
                        except Exception:
                            pass
            else:
                asr_text = ""
        else:
            asr_text = ""

        ocr_rows.append({"scene_id": scene_id, "video_id": video_id, "ocr_text": ocr_text})
        asr_rows.append({"scene_id": scene_id, "video_id": video_id, "asr_text": asr_text})

        manifest.mark_done(scene_id, step="ocr")
        manifest.mark_done(scene_id, step="asr")
        processed_count += 1

        if processed_count % 50 == 0 or processed_count == total_scenes:
            logger.info(f"Đã xử lý Text Extraction: {processed_count}/{total_scenes} scenes")
            manifest.save()

    ocr_df = pd.DataFrame(ocr_rows)
    asr_df = pd.DataFrame(asr_rows)

    # Đảm bảo string type và không chứa NaN/None
    ocr_df["ocr_text"] = ocr_df["ocr_text"].fillna("").astype(str)
    asr_df["asr_text"] = asr_df["asr_text"].fillna("").astype(str)

    # 4. Ghi file Parquet kết quả
    ocr_out = out_p / "ocr_text.parquet"
    asr_out = out_p / "asr_text.parquet"

    ocr_df.to_parquet(ocr_out, index=False)
    asr_df.to_parquet(asr_out, index=False)

    logger.info(f"✅ Hoàn tất TASK-ING-02! Đã lưu:")
    logger.info(f"  - {ocr_out} ({len(ocr_df)} rows)")
    logger.info(f"  - {asr_out} ({len(asr_df)} rows)")

    return ocr_df, asr_df


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="TASK-ING-02: Trích xuất text từ keyframes (OCR) và audio video (ASR).")
    parser.add_argument("--metadata-path", required=True, help="Đường dẫn tới scene_metadata.parquet.")
    parser.add_argument("--keyframe-dir", required=True, help="Thư mục chứa keyframe JPG.")
    parser.add_argument("--video-dir", required=True, help="Thư mục chứa video gốc.")
    parser.add_argument("--output-dir", default="/kaggle/working/shard_output", help="Thư mục lưu kết quả.")
    parser.add_argument("--checkpoint-path", default=None, help="Đường dẫn file checkpoint.")
    parser.add_argument("--import-ocr-dir", default=None, help="Thư mục chứa các file OCR JSON có sẵn để import nhanh.")
    parser.add_argument("--import-asr-dir", default=None, help="Thư mục chứa các file ASR JSON có sẵn để import nhanh.")
    parser.add_argument("--skip-ocr", action="store_true", help="Bỏ qua bước OCR (chỉ xuất chuỗi rỗng).")
    parser.add_argument("--skip-asr", action="store_true", help="Bỏ qua bước ASR (chỉ xuất chuỗi rỗng).")

    args = parser.parse_args()

    process_text_extraction(
        metadata_path=args.metadata_path,
        keyframe_dir=args.keyframe_dir,
        video_dir=args.video_dir,
        output_dir=args.output_dir,
        checkpoint_path=args.checkpoint_path,
        import_ocr_dir=args.import_ocr_dir,
        import_asr_dir=args.import_asr_dir,
        skip_ocr=args.skip_ocr,
        skip_asr=args.skip_asr,
    )
