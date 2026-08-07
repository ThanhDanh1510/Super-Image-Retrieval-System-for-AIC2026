"""
ingestion/skeletons/01b_text_extraction.py
============================================
TASK-ING-02: Trích xuất text từ keyframes (OCR) và audio video (ASR).

SKELETON IMPLEMENTATION: Trích xuất text bằng PaddleOCR v6 và PhoWhisper.

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

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

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
            Path("configs/model_versions.yaml"),
        ]
        for path in possible_paths:
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f)
        raise FileNotFoundError("Không tìm thấy configs/model_versions.yaml")

from shared.ocr.quality import build_quality_result
from ingestion.skeletons.checkpoint_manifest import CheckpointManifest

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# An toàn cho Paddle 3.x chạy CPU trên Windows: tắt PIR executor và MKLDNN (oneDNN)
# để tránh lỗi "ConvertPirAttribute2RuntimeAttribute ... onednn_instruction.cc".
# Phải set TRƯỚC khi import paddle/paddleocr (chỉ import bên trong init_ocr_engine).
os.environ.setdefault("FLAGS_enable_pir_api", "0")
os.environ.setdefault("FLAGS_use_mkldnn", "0")


# =============================================================================
# IMPLEMENTED FUNCTIONS
# =============================================================================

def init_ocr_engine(model_cfg: dict) -> Any:
    """
    Khởi tạo PaddleOCR engine từ cấu hình trong model_versions.yaml.

    Version OCR đọc từ: model_cfg['ocr']['version'] == 'v6'
    Ngôn ngữ đọc từ:    model_cfg['ocr']['lang']    == 'vi'

    Args:
        model_cfg: Dict đọc từ load_model_versions().

    Returns:
        PaddleOCR instance đã khởi tạo.

    Raises:
        ImportError: Nếu paddleocr chưa được cài đặt.
        RuntimeError: Nếu không khởi tạo được engine.
    """
    try:
        from paddleocr import PaddleOCR
    except ImportError as err:
        raise ImportError("PaddleOCR package chưa được cài đặt. Vui lòng install 'paddleocr' và 'paddlepaddle'.") from err

    ocr_cfg = model_cfg.get("ocr", {})
    lang = ocr_cfg.get("lang", "vi")
    use_gpu = ocr_cfg.get("use_gpu", True)
    det_model_dir = ocr_cfg.get("det_model_dir")
    rec_model_dir = ocr_cfg.get("rec_model_dir")

    # Quyết định device chạy. Thứ tự ưu tiên:
    #   1) Biến môi trường AIC_DEVICE (set trong .env, ví dụ AIC_DEVICE=cpu)
    #   2) Giá trị use_gpu trong config — nhưng tự kiểm tra CUDA khả dụng để tránh
    #      fallback GPU->CPU gây lỗi oneDNN/PIR trên máy không có GPU.
    env_device = os.environ.get("AIC_DEVICE", "").strip().lower()
    if env_device in ("cpu", "gpu"):
        device = env_device
    else:
        device = "cpu"
        if use_gpu:
            try:
                import paddle

                device = "gpu" if paddle.device.cuda.device_count() > 0 else "cpu"
            except Exception:
                device = "cpu"

    # Tắt MKLDNN (oneDNN) để tránh lỗi "ConvertPirAttribute2RuntimeAttribute not support ...
    # onednn_instruction.cc" của Paddle 3.x chạy CPU trên Windows. Không ảnh hưởng tới Kaggle/Linux.
    enable_mkldnn = bool(ocr_cfg.get("enable_mkldnn", False))

    kwargs: dict[str, Any] = {
        "lang": lang,
    }
    if det_model_dir:
        kwargs["det_model_dir"] = det_model_dir
    if rec_model_dir:
        kwargs["rec_model_dir"] = rec_model_dir

    last_err: Exception | None = None

    # 1) PaddleOCR 3.x style — dùng device + enable_mkldnn (ổn định trên Windows CPU/Paddle 3.x)
    try:
        ocr = PaddleOCR(device=device, enable_mkldnn=enable_mkldnn, **kwargs)
        return ocr
    except (TypeError, ValueError):
        pass
    except Exception as err:
        last_err = err

    # 2) PaddleOCR 2.x style — dùng use_gpu + use_angle_cls
    try:
        ocr = PaddleOCR(use_angle_cls=True, use_gpu=use_gpu, **kwargs)
        return ocr
    except (TypeError, ValueError):
        pass
    except Exception as err:
        last_err = err

    # 3) Base fallback — chỉ lang (và det/rec model dir nếu có)
    try:
        ocr = PaddleOCR(**kwargs)
        return ocr
    except Exception as err:
        last_err = err

    raise RuntimeError(f"Không thể khởi tạo PaddleOCR engine: {last_err}") from last_err


def init_asr_model(model_cfg: dict) -> Any:
    """
    Khởi tạo ASR model (PhoWhisper) từ cấu hình trong model_versions.yaml.

    Model ID đọc từ: model_cfg['asr']['model'] == 'vinai/PhoWhisper-medium'
    Language đọc từ: model_cfg['asr']['language'] == 'vi'

    Args:
        model_cfg: Dict đọc từ load_model_versions().

    Returns:
        HuggingFace ASR pipeline instance đã load model.

    Raises:
        RuntimeError: Nếu không load được model.
    """
    asr_cfg = model_cfg.get("asr", {})
    model_id = asr_cfg.get("model", "vinai/PhoWhisper-medium")
    device_setting = asr_cfg.get("device", "cuda")

    device = 0 if device_setting == "cuda" else -1

    try:
        from transformers import pipeline
        asr = pipeline(
            "automatic-speech-recognition",
            model=model_id,
            device=device,
        )
        return asr
    except Exception as err:
        raise RuntimeError(f"Không thể load ASR model '{model_id}': {err}") from err


def _collect_ocr_text(node: Any, out: list[str]) -> None:
    """
    Đệ quy trích xuất các chuỗi text nhận dạng từ output của PaddleOCR.

    Hỗ trợ nhiều cấu trúc kết quả:
      - 3.x predict:  { 'rec_texts': ['...'], 'rec_scores': [...] }  (có thể bọc trong list)
      - 2.x ocr:      [ [ [box, (text, score)], ... ] ]

    Args:
        node: Node bất kỳ của cấu trúc kết quả OCR (dict/list/tuple/str...).
        out:   List tích lũy các text đã trích xuất.
    """
    if isinstance(node, dict):
        # Ưu tiên key 'rec_texts' / 'rec_text' của PaddleOCR 3.x
        rec = node.get("rec_texts") or node.get("rec_text") or []
        if rec:
            for t in rec:
                if t:
                    out.append(str(t).strip())
            return
        for value in node.values():
            _collect_ocr_text(value, out)
    elif isinstance(node, (list, tuple)):
        # Line 2.x: [box, (text, score)] → phần tử thứ 2 là tuple (text, score)
        if (
            len(node) == 2
            and isinstance(node[0], (list, tuple))
            and isinstance(node[1], (list, tuple))
            and len(node[1]) >= 1
            and isinstance(node[1][0], str)
        ):
            out.append(node[1][0].strip())
        else:
            for item in node:
                _collect_ocr_text(item, out)


def run_ocr_raw(ocr_engine: Any, image_path: str) -> str:
    """
    Chạy OCR trên một ảnh keyframe và trả về text raw nhận dạng được.

    Args:
        ocr_engine: PaddleOCR instance từ init_ocr_engine().
        image_path: Đường dẫn tuyệt đối tới file JPG keyframe.

    Returns:
        Chuỗi text đã nhận dạng. Trả về '' nếu không có text.

    Raises:
        FileNotFoundError: Nếu image_path không tồn tại.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file ảnh keyframe: {image_path}")

    try:
        try:
            result = ocr_engine.ocr(str(path), cls=True)
        except (TypeError, ValueError):
            result = ocr_engine.ocr(str(path))
        if not result:
            return ""

        # Gom text từ output theo nhiều định dạng của PaddleOCR:
        #   - 3.x predict:  [ { 'rec_texts': [...], ... } ]  hoặc  { 'rec_texts': [...] }
        #   - 2.x ocr:      [ [ [box, (text, score)], ... ] ]
        # Hàm đệ quy dưới đây xử lý được tất cả các cấu trúc trên.
        texts: list[str] = []
        _collect_ocr_text(result, texts)

        return "\n".join(texts).strip()
    except FileNotFoundError:
        raise
    except Exception as err:
        logger.warning(f"Lỗi khi chạy OCR trên {image_path}: {err}")
        return ""


def run_ocr_quality(ocr_engine: Any, image_path: str):
    """Run OCR and return corrected text plus quality metadata."""
    return build_quality_result(run_ocr_raw(ocr_engine, image_path))


def run_ocr(ocr_engine: Any, image_path: str) -> str:
    """
    Chạy OCR trên một ảnh keyframe và trả về text đã chuẩn hóa/correct conservative.

    Giữ interface cũ cho downstream retrieval, nhưng nội bộ có quality layer để
    những lỗi chắc chắn theo ngữ cảnh như "vq" trong dòng điểm danh được sửa.
    """
    return run_ocr_quality(ocr_engine, image_path).text


def extract_audio_segment(
    video_path: str,
    start_sec: float,
    end_sec: float,
    tmp_dir: str = "/tmp",
) -> str:
    """
    Trích xuất đoạn audio từ video và lưu ra file WAV tạm thời.

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
    v_path = Path(video_path)
    if not v_path.exists():
        raise FileNotFoundError(f"Không tìm thấy video tại: {video_path}")

    os.makedirs(tmp_dir, exist_ok=True)
    duration = max(0.1, float(end_sec) - float(start_sec))

    filename = f"audio_{v_path.stem}_{start_sec:.2f}_{end_sec:.2f}_{os.urandom(4).hex()}.wav"
    tmp_path = os.path.join(tmp_dir, filename)

    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        str(start_sec),
        "-i",
        str(v_path),
        "-t",
        str(duration),
        "-vn",
        "-ar",
        "16000",
        "-ac",
        "1",
        tmp_path,
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return tmp_path
    except FileNotFoundError as err:
        raise RuntimeError("ffmpeg không được cài đặt hoặc không nằm trong PATH.") from err
    except subprocess.CalledProcessError as err:
        raise RuntimeError(f"Trích xuất audio ffmpeg thất bại: {err.stderr}") from err


def run_asr(asr_model: Any, audio_path: str) -> str:
    """
    Chạy ASR trên file audio và trả về transcript text.

    Args:
        asr_model: HuggingFace ASR pipeline từ init_asr_model().
        audio_path: Đường dẫn tới file WAV tạm thời (từ extract_audio_segment()).

    Returns:
        Transcript text (tiếng Việt). Trả về '' nếu không có giọng nói.

    Raises:
        FileNotFoundError: Nếu audio_path không tồn tại.
    """
    a_path = Path(audio_path)
    if not a_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file audio tại: {audio_path}")

    try:
        result = asr_model(str(a_path))
        if isinstance(result, dict):
            return str(result.get("text", "") or "").strip()
        elif isinstance(result, str):
            return result.strip()
        return ""
    except FileNotFoundError:
        raise
    except Exception as err:
        logger.warning(f"Lỗi khi chạy ASR trên {audio_path}: {err}")
        return ""


def process_text_extraction(
    metadata_path: str,
    keyframe_dir: str,
    video_dir: str,
    output_dir: str,
    checkpoint_path: str | None = None,
    write_ocr_quality: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Entry point chính: Chạy OCR + ASR cho toàn bộ scenes trong metadata.

    Args:
        metadata_path: Đường dẫn tới scene_metadata.parquet.
        keyframe_dir: Thư mục chứa keyframe JPG.
        video_dir: Thư mục chứa video gốc (dùng cho ASR).
        output_dir: Thư mục output (/kaggle/working/shard_output/).
        checkpoint_path: Đường dẫn checkpoint.json. None → mặc định.
        write_ocr_quality: Nếu True, lưu thêm ocr_quality.parquet với raw/corrected/review metadata.

    Returns:
        Tuple (ocr_df, asr_df):
            - ocr_df: pd.DataFrame với cột (scene_id, video_id, ocr_text)
            - asr_df: pd.DataFrame với cột (scene_id, video_id, asr_text)

    Raises:
        FileNotFoundError: Nếu metadata_path không tồn tại.
    """
    meta_path = Path(metadata_path)
    if not meta_path.exists():
        raise FileNotFoundError(f"Không tìm thấy scene_metadata.parquet tại: {metadata_path}")

    df_meta = pd.read_parquet(meta_path)
    if df_meta.empty:
        logger.warning("scene_metadata.parquet rỗng.")
        empty_ocr = pd.DataFrame(columns=["scene_id", "video_id", "ocr_text"])
        empty_asr = pd.DataFrame(columns=["scene_id", "video_id", "asr_text"])
        return empty_ocr, empty_asr

    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    if checkpoint_path is None:
        checkpoint_path = str(out_p / "checkpoint.json")

    manifest = CheckpointManifest(checkpoint_path)
    model_cfg = load_model_versions()

    # Load pre-existing parquet results if checkpoint resumption is active
    ocr_output_path = out_p / "ocr_text.parquet"
    asr_output_path = out_p / "asr_text.parquet"

    ocr_cache: dict[str, str] = {}
    if ocr_output_path.exists():
        try:
            cached_ocr_df = pd.read_parquet(ocr_output_path)
            ocr_cache = dict(zip(cached_ocr_df["scene_id"], cached_ocr_df["ocr_text"].fillna("")))
        except Exception as e:
            logger.warning(f"Không thể đọc cached ocr_text.parquet: {e}")

    asr_cache: dict[str, str] = {}
    if asr_output_path.exists():
        try:
            cached_asr_df = pd.read_parquet(asr_output_path)
            asr_cache = dict(zip(cached_asr_df["scene_id"], cached_asr_df["asr_text"].fillna("")))
        except Exception as e:
            logger.warning(f"Không thể đọc cached asr_text.parquet: {e}")

    # Init engines lazily / safely
    ocr_engine = None
    try:
        ocr_engine = init_ocr_engine(model_cfg)
    except Exception as e:
        logger.warning(f"Khởi tạo OCR engine thất bại ({e}). OCR text sẽ để rỗng.")

    asr_model = None
    try:
        asr_model = init_asr_model(model_cfg)
    except Exception as e:
        logger.warning(f"Khởi tạo ASR model thất bại ({e}). ASR text sẽ để rỗng.")

    # Cache video files in video_dir
    video_file_map: dict[str, Path] = {}
    v_dir = Path(video_dir) if video_dir else None
    if v_dir and v_dir.exists():
        for ext in ("*.mp4", "*.mkv", "*.avi", "*.webm"):
            for v_file in v_dir.rglob(ext):
                video_file_map[v_file.stem] = v_file

    ocr_records = []
    ocr_quality_records = []
    asr_records = []
    total_scenes = len(df_meta)

    for idx, row in df_meta.iterrows():
        scene_id = str(row["scene_id"])
        video_id = str(row["video_id"])
        keyframe_rel = str(row["keyframe_path"])
        start_sec = float(row.get("timestamp_start", 0.0))
        end_sec = float(row.get("timestamp_end", 0.0))

        # --- OCR Extraction ---
        if manifest.is_done(scene_id, step="ocr") and scene_id in ocr_cache:
            ocr_text = ocr_cache[scene_id]
        else:
            ocr_text = ""
            if ocr_engine:
                kf_path = Path(keyframe_dir) / keyframe_rel
                if not kf_path.exists():
                    kf_path = Path(keyframe_dir) / video_id / keyframe_rel
                if kf_path.exists():
                    try:
                        quality = run_ocr_quality(ocr_engine, str(kf_path))
                        ocr_text = quality.text
                        if write_ocr_quality:
                            ocr_quality_records.append({
                                "scene_id": scene_id,
                                "video_id": video_id,
                                "ocr_text": quality.text,
                                "ocr_raw_text": quality.raw_text,
                                "needs_review": quality.needs_review,
                                "review_reasons": quality.review_reasons,
                                "edits": quality.edits,
                                "lines": [line.__dict__ for line in quality.lines],
                            })
                    except Exception as err:
                        logger.warning(f"OCR error on scene {scene_id}: {err}")
                else:
                    logger.warning(f"Keyframe path not found: {kf_path}")

            manifest.mark_done(scene_id, step="ocr")
            ocr_cache[scene_id] = ocr_text

        if write_ocr_quality and not any(record["scene_id"] == scene_id for record in ocr_quality_records):
            quality = build_quality_result(ocr_text)
            ocr_quality_records.append({
                "scene_id": scene_id,
                "video_id": video_id,
                "ocr_text": quality.text,
                "ocr_raw_text": quality.raw_text,
                "needs_review": quality.needs_review,
                "review_reasons": quality.review_reasons,
                "edits": quality.edits,
                "lines": [line.__dict__ for line in quality.lines],
            })

        # --- ASR Extraction ---
        if manifest.is_done(scene_id, step="asr") and scene_id in asr_cache:
            asr_text = asr_cache[scene_id]
        else:
            asr_text = ""
            if asr_model and video_id in video_file_map:
                v_path = video_file_map[video_id]
                wav_path = None
                try:
                    wav_path = extract_audio_segment(
                        video_path=str(v_path),
                        start_sec=start_sec,
                        end_sec=end_sec,
                        tmp_dir=str(out_p / "tmp_audio"),
                    )
                    asr_text = run_asr(asr_model, wav_path)
                except Exception as err:
                    logger.warning(f"ASR error on scene {scene_id}: {err}")
                finally:
                    if wav_path and os.path.exists(wav_path):
                        try:
                            os.remove(wav_path)
                        except OSError:
                            pass

            manifest.mark_done(scene_id, step="asr")
            asr_cache[scene_id] = asr_text

        manifest.save()

        ocr_records.append({"scene_id": scene_id, "video_id": video_id, "ocr_text": ocr_text})
        asr_records.append({"scene_id": scene_id, "video_id": video_id, "asr_text": asr_text})

        if (idx + 1) % 50 == 0 or (idx + 1) == total_scenes:
            logger.info(f"Processed text extraction for {idx + 1}/{total_scenes} scenes.")

    ocr_df = pd.DataFrame(ocr_records)
    asr_df = pd.DataFrame(asr_records)

    # Fill any potential null values with empty string
    ocr_df["ocr_text"] = ocr_df["ocr_text"].fillna("")
    asr_df["asr_text"] = asr_df["asr_text"].fillna("")

    ocr_df.to_parquet(ocr_output_path, index=False)
    asr_df.to_parquet(asr_output_path, index=False)
    if write_ocr_quality:
        quality_output_path = out_p / "ocr_quality.parquet"
        pd.DataFrame(ocr_quality_records).to_parquet(quality_output_path, index=False)
        logger.info(f"Saved {quality_output_path}")
    logger.info(f"Saved {ocr_output_path} and {asr_output_path}")

    return ocr_df, asr_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TASK-ING-02: Trích xuất text từ keyframes (OCR) và audio video (ASR).")
    parser.add_argument("--metadata-path", required=True, help="Đường dẫn tới scene_metadata.parquet.")
    parser.add_argument("--keyframe-dir", required=True, help="Thư mục chứa keyframe JPG.")
    parser.add_argument("--video-dir", required=True, help="Thư mục chứa video gốc.")
    parser.add_argument("--output-dir", default="/kaggle/working/shard_output", help="Thư mục lưu kết quả.")
    parser.add_argument("--checkpoint-path", default=None, help="Đường dẫn file checkpoint.")
    parser.add_argument(
        "--write-ocr-quality",
        action="store_true",
        help="Lưu thêm ocr_quality.parquet với line-level metadata/review flags.",
    )

    args = parser.parse_args()

    process_text_extraction(
        metadata_path=args.metadata_path,
        keyframe_dir=args.keyframe_dir,
        video_dir=args.video_dir,
        output_dir=args.output_dir,
        checkpoint_path=args.checkpoint_path,
        write_ocr_quality=args.write_ocr_quality,
    )
