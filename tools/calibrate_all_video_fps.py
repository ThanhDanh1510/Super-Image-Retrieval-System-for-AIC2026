"""
tools/calibrate_all_video_fps.py
================================
Chuẩn hóa lại timestamp_mid, timestamp_start, timestamp_end cho toàn bộ keyframes
theo FPS thực tế của từng file video trong data/video/.
"""

import cv2
import pandas as pd
from pathlib import Path

def main():
    print("=== AIC2026: Chuẩn Hóa Timestamp Khớp FPS Video Thực Tế Cho Toàn Bộ Dataset ===")
    meta_path = Path("data/unified/scene_metadata.parquet")
    ocr_path = Path("data/unified/ocr_text.parquet")
    asr_path = Path("data/unified/asr_text.parquet")
    desc_path = Path("data/unified/scene_descriptions.parquet")

    meta_df = pd.read_parquet(meta_path)
    ocr_df = pd.read_parquet(ocr_path)
    asr_df = pd.read_parquet(asr_path)
    desc_df = pd.read_parquet(desc_path)

    video_dir = Path("data/video")
    v_files = sorted(list(video_dir.rglob("*.mp4")) + list(video_dir.rglob("*.mkv")))

    fps_map = {}
    dur_map = {}
    for vf in v_files:
        vid = vf.stem
        cap = cv2.VideoCapture(str(vf))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps_map[vid] = fps
        dur_map[vid] = count / fps if fps > 0 else 0.0
        cap.release()

    updated = 0
    for idx in range(len(meta_df)):
        vid = meta_df.at[idx, "video_id"]
        if vid in fps_map:
            fps = fps_map[vid]
            fid = meta_df.at[idx, "frame_id"]
            correct_ts = round(float(fid) / fps, 3)
            if abs(meta_df.at[idx, "timestamp_mid"] - correct_ts) > 0.05:
                meta_df.at[idx, "fps"] = fps
                meta_df.at[idx, "timestamp_mid"] = correct_ts
                meta_df.at[idx, "timestamp_start"] = max(0.0, round(correct_ts - 2.0, 3))
                meta_df.at[idx, "timestamp_end"] = round(correct_ts + 2.0, 3)
                ocr_df.at[idx, "timestamp_mid"] = correct_ts
                asr_df.at[idx, "timestamp_mid"] = correct_ts
                desc_df.at[idx, "timestamp_mid"] = correct_ts
                updated += 1

    print(f"Đã chuẩn hóa lại timestamp cho {updated} / {len(meta_df)} keyframes!")
    meta_df.to_parquet(meta_path, index=False)
    ocr_df.to_parquet(ocr_path, index=False)
    asr_df.to_parquet(asr_path, index=False)
    desc_df.to_parquet(desc_path, index=False)
    print("✅ Đã lưu Unified Dataset với timestamp chuẩn xác 100%!")

if __name__ == "__main__":
    main()
