"""
aggregation/build_unified_from_raw.py
======================================
Script tổng hợp dữ liệu thô (Keyframes webp, OCR JSONs, ASR json) tại máy local.

CHỨC NĂNG:
  1. Quét đệ quy thư mục data/image/ tìm tất cả file ảnh (.webp, .jpg, .png).
     - Extract video_id (ví dụ 'L21_V001') và frame_id (ví dụ 10141).
     - Đánh mã scene_id = {video_id}_S{idx:03d}.
     - Ước tính timestamp_mid = frame_id / 25.0 (fps=25.0).
     - Xuất scene_metadata.parquet.

  2. Khớp OCR Text từ data/OCR/{group}/{video_id}.json theo frame_id.
     - Chuẩn hóa Unicode NFC.
     - Xuất ocr_text.parquet.

  3. Khớp ASR Transcript từ data/ASR/asr.json:
     - Khớp (video_name == video_id) và (start_frame <= frame_id <= end_frame).
     - Chuẩn hóa Unicode NFC.
     - Xuất asr_text.parquet.

  4. Tổ chức thư mục output (mặc định ./data/unified_raw/):
     - scene_metadata.parquet
     - ocr_text.parquet
     - asr_text.parquet
     - keyframes/{video_id}/{scene_id}.webp (copy hoặc symlink ảnh)
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
from pathlib import Path
import pandas as pd
import numpy as np

from shared.utils.text_normalizer import normalize_nfc, clean_whitespace

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_raw_images(image_dir: str | Path) -> pd.DataFrame:
    """
    Quét thư mục ảnh thô và tạo DataFrame metadata thô.

    Format đường dẫn kỳ vọng:
      .../<group>/<video_id>/<frame_id>.webp
      ví dụ: data/image/L21/L21/V001/10141.webp  -> video_id='L21_V001', frame_id=10141
      hoặc data/image/L21_V001/10141.jpg         -> video_id='L21_V001', frame_id=10141
    """
    image_dir = Path(image_dir)
    if not image_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục ảnh: {image_dir}")

    records = []
    # Tìm các file ảnh .webp, .jpg, .jpeg, .png
    img_extensions = {".webp", ".jpg", ".jpeg", ".png"}

    for file_path in image_dir.rglob("*"):
        if file_path.suffix.lower() not in img_extensions:
            continue

        # Extract frame_id từ stem (ví dụ "10141" -> 10141)
        stem = file_path.stem
        try:
            frame_id = int(stem)
        except ValueError:
            # Nếu stem chứa chữ, thử tìm số cuối cùng trong stem
            numbers = re.findall(r"\d+", stem)
            if not numbers:
                continue
            frame_id = int(numbers[-1])

        # Extract video_id từ các parent directories
        # Ví dụ: file_path = .../L21/L21/V001/10141.webp
        # parts: ..., 'L21', 'L21', 'V001', '10141.webp'
        parts = file_path.parts
        video_id = None

        # Thử ghép group + video: nếu parent folder dạng 'V001' và grand-parent dạng 'L21'
        for i in range(len(parts) - 1, 0, -1):
            part = parts[i]
            # Match dạng 'L21_V001' hoặc 'V001'
            if re.match(r"^L\d+_V\d+$", part, re.IGNORECASE):
                video_id = part.upper()
                break
            elif re.match(r"^V\d+$", part, re.IGNORECASE) and i >= 1:
                parent_part = parts[i - 1]
                if re.match(r"^L\d+$", parent_part, re.IGNORECASE):
                    video_id = f"{parent_part.upper()}_{part.upper()}"
                    break

        if not video_id:
            # Fallback nếu folder là 'V001' mà không tìm thấy 'L21'
            v_match = re.search(r"(L\d+_V\d+|V\d+)", file_path.as_posix(), re.IGNORECASE)
            if v_match:
                video_id = v_match.group(1).upper()
            else:
                continue

        records.append({
            "video_id": video_id,
            "frame_id": frame_id,
            "raw_path": str(file_path),
            "ext": file_path.suffix.lower(),
        })

    if not records:
        raise ValueError(f"Không tìm thấy ảnh keyframe hợp lệ nào trong {image_dir}")

    df = pd.DataFrame(records)
    # Sắp xếp theo video_id và frame_id tăng dần
    df.sort_values(by=["video_id", "frame_id"], ascending=[True, True], inplace=True)
    df.reset_index(drop=True, inplace=True)
    logger.info(f"Đã quét được {len(df)} keyframes từ {df['video_id'].nunique()} videos.")
    return df


def build_scene_metadata(raw_df: pd.DataFrame, fps: float = 25.0) -> pd.DataFrame:
    """
    Đánh mã scene_id và tính toán timestamps.
    """
    metadata_rows = []

    for video_id, group in raw_df.groupby("video_id", sort=False):
        group = group.sort_values(by="frame_id").reset_index(drop=True)
        n = len(group)

        for idx, row in group.iterrows():
            scene_id = f"{video_id}_S{idx:03d}"
            f_id = int(row["frame_id"])
            t_mid = round(f_id / fps, 3)

            # Ước tính start/end timestamp
            if idx == 0:
                t_start = max(0.0, round(t_mid - 1.0, 3))
            else:
                prev_mid = round(int(group.loc[idx - 1, "frame_id"]) / fps, 3)
                t_start = round((prev_mid + t_mid) / 2.0, 3)

            if idx == n - 1:
                t_end = round(t_mid + 1.0, 3)
            else:
                next_mid = round(int(group.loc[idx + 1, "frame_id"]) / fps, 3)
                t_end = round((t_mid + next_mid) / 2.0, 3)

            # Keyframe output path chuẩn: {video_id}/{scene_id}.webp
            ext = row["ext"]
            keyframe_path = f"{video_id}/{scene_id}{ext}"

            metadata_rows.append({
                "scene_id": scene_id,
                "video_id": video_id,
                "frame_id": f_id,
                "timestamp_start": t_start,
                "timestamp_end": t_end,
                "timestamp_mid": t_mid,
                "fps": fps,
                "keyframe_path": keyframe_path,
                "raw_path": row["raw_path"],
            })

    meta_df = pd.DataFrame(metadata_rows)
    return meta_df


def extract_ocr_for_metadata(meta_df: pd.DataFrame, ocr_dir: str | Path) -> pd.DataFrame:
    """
    Tra cứu text OCR từ data/OCR/<group>/<video_id>.json sử dụng video_id và frame_id.
    """
    ocr_dir = Path(ocr_dir)
    ocr_cache: dict[str, dict[str, str]] = {}

    if ocr_dir.exists():
        for json_file in ocr_dir.glob("**/*.json"):
            try:
                v_name = json_file.stem  # Ví dụ: "L21_V001"
                with open(json_file, "r", encoding="utf-8") as f:
                    ocr_cache[v_name] = json.load(f)
            except Exception as e:
                logger.warning(f"Không thể đọc OCR JSON {json_file}: {e}")

    ocr_rows = []
    for _, row in meta_df.iterrows():
        video_id = row["video_id"]
        frame_id_str = str(row["frame_id"])

        v_ocr_dict = ocr_cache.get(video_id, {})

        # Tra cứu trực tiếp theo frame_id
        raw_ocr = v_ocr_dict.get(frame_id_str, "")

        clean_ocr = clean_whitespace(normalize_nfc(raw_ocr)) if raw_ocr else ""

        ocr_rows.append({
            "scene_id": row["scene_id"],
            "video_id": row["video_id"],
            "ocr_text": clean_ocr,
        })

    return pd.DataFrame(ocr_rows)


def extract_asr_for_metadata(meta_df: pd.DataFrame, asr_json_path: str | Path) -> pd.DataFrame:
    """
    Tra cứu text ASR từ data/ASR/asr.json bằng cách match video_name và khoảng [start_frame, end_frame].
    """
    asr_json_path = Path(asr_json_path)
    asr_by_video: dict[str, list[dict]] = {}

    if asr_json_path.exists():
        try:
            logger.info(f"Loading ASR JSON từ {asr_json_path}...")
            with open(asr_json_path, "r", encoding="utf-8") as f:
                asr_data = json.load(f)

            for item in asr_data:
                v_name = item.get("video_name", "").upper()
                if v_name:
                    if v_name not in asr_by_video:
                        asr_by_video[v_name] = []
                    asr_by_video[v_name].append(item)
        except Exception as e:
            logger.warning(f"Không thể đọc ASR JSON {asr_json_path}: {e}")

    asr_rows = []
    for _, row in meta_df.iterrows():
        video_id = row["video_id"]
        frame_id = row["frame_id"]

        matched_texts = []
        if video_id in asr_by_video:
            for item in asr_by_video[video_id]:
                s_frame = item.get("start_frame", 0)
                e_frame = item.get("end_frame", float("inf"))
                if s_frame <= frame_id <= e_frame:
                    t_text = item.get("text", "").strip()
                    if t_text:
                        matched_texts.append(t_text)

        combined_asr = " ".join(matched_texts) if matched_texts else ""
        clean_asr = clean_whitespace(normalize_nfc(combined_asr)) if combined_asr else ""

        asr_rows.append({
            "scene_id": row["scene_id"],
            "video_id": row["video_id"],
            "asr_text": clean_asr,
        })

    return pd.DataFrame(asr_rows)


def run_build_unified(
    image_dir: str,
    ocr_dir: str,
    asr_json: str,
    output_dir: str,
    copy_keyframes: bool = True,
    fps: float = 25.0,
):
    """
    Hàm chính điều phối quy trình tổng hợp dữ liệu thô.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    keyframes_out_dir = output_path / "keyframes"
    keyframes_out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Bước 1: Quét ảnh thô và tạo scene_metadata...")
    raw_df = parse_raw_images(image_dir)
    meta_df = build_scene_metadata(raw_df, fps=fps)

    logger.info("Bước 2: Khớp OCR text từ JSON...")
    ocr_df = extract_ocr_for_metadata(meta_df, ocr_dir)

    logger.info("Bước 3: Khớp ASR transcript từ asr.json...")
    asr_df = extract_asr_for_metadata(meta_df, asr_json)

    # Đưa ảnh về thư mục output chuẩn keyframes/{video_id}/{scene_id}.ext nếu copy_keyframes=True
    if copy_keyframes:
        logger.info("Bước 4: Đưa các file ảnh keyframe về thư mục chuẩn...")
        for _, row in meta_df.iterrows():
            src = Path(row["raw_path"])
            dest = keyframes_out_dir / row["keyframe_path"]
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                shutil.copy2(src, dest)

    # Loại bỏ cột tạm raw_path trước khi lưu scene_metadata.parquet
    final_meta_df = meta_df.drop(columns=["raw_path"])

    # Save parquets
    final_meta_df.to_parquet(output_path / "scene_metadata.parquet", index=False)
    ocr_df.to_parquet(output_path / "ocr_text.parquet", index=False)
    asr_df.to_parquet(output_path / "asr_text.parquet", index=False)

    logger.info(f"✅ HOÀN THÀNH TỔNG HỢP DỮ LIỆU THÔ!")
    logger.info(f"📁 Output dir: {output_path.absolute()}")
    logger.info(f"📊 Metadata: {len(final_meta_df)} rows")
    logger.info(f"📝 OCR non-empty: {(ocr_df['ocr_text'] != '').sum()} rows")
    logger.info(f"🎙️ ASR non-empty: {(asr_df['asr_text'] != '').sum()} rows")


def main():
    parser = argparse.ArgumentParser(description="Tổng hợp dữ liệu thô thành data/unified_raw")
    parser.add_argument("--image-dir", default="./data/image", help="Thư mục chứa ảnh keyframe thô")
    parser.add_argument("--ocr-dir", default="./data/OCR", help="Thư mục chứa OCR json")
    parser.add_argument("--asr-json", default="./data/ASR/asr.json", help="Đường dẫn file ASR json")
    parser.add_argument("--output-dir", default="./data/unified_raw", help="Thư mục xuất kết quả hợp nhất")
    parser.add_argument("--fps", type=float, default=25.0, help="Frames per second chuẩn của video")
    parser.add_argument("--skip-copy", action="store_true", help="Bỏ qua copy ảnh keyframes")

    args = parser.parse_args()
    run_build_unified(
        image_dir=args.image_dir,
        ocr_dir=args.ocr_dir,
        asr_json=args.asr_json,
        output_dir=args.output_dir,
        copy_keyframes=not args.skip_copy,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
