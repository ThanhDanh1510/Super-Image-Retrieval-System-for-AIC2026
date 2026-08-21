"""
tools/data_prep/verify_qa_frames.py
===================================
Kiểm tra chi tiết từng dòng trong các file Q&A nộp bài:
- Video ID, Frame ID, Timestamp
- File ảnh keyframe thực tế trên đĩa
- Đáp án nộp bài (Answer)
- OCR Text trên ảnh
- Gemini Scene Description
"""

import csv
import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SUB_DIR = PROJECT_ROOT / "team_submission_1787240406787" / "submission"
UNIFIED_DIR = PROJECT_ROOT / "data" / "unified"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.utils.path_resolver import resolve_keyframe_path, resolve_video_path

def main():
    qa_files = [
        "query-p1-15-qa.csv",
        "query-p1-19-qa.csv",
        "query-p1-22-qa.csv",
    ]

    meta_df = pd.read_parquet(UNIFIED_DIR / "scene_metadata.parquet")
    ocr_df = pd.read_parquet(UNIFIED_DIR / "ocr_text.parquet") if (UNIFIED_DIR / "ocr_text.parquet").exists() else pd.DataFrame()
    desc_df = pd.read_parquet(UNIFIED_DIR / "scene_descriptions.parquet") if (UNIFIED_DIR / "scene_descriptions.parquet").exists() else pd.DataFrame()

    merged = meta_df.copy()
    if not ocr_df.empty and "ocr_text" in ocr_df.columns:
        merged["ocr_text"] = ocr_df["ocr_text"]
    if not desc_df.empty and "description" in desc_df.columns:
        merged["description"] = desc_df["description"]

    for qf in qa_files:
        fpath = SUB_DIR / qf
        print("\n" + "=" * 80)
        print(f"📋 KIỂM TRA FILE QA: {qf}")
        print("=" * 80)

        with open(fpath, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            for row_idx, row in enumerate(reader, start=1):
                if len(row) < 3:
                    continue
                vid, fid_str, ans = row[0].strip(), row[1].strip(), row[2].strip()
                fid = int(fid_str)

                img_path = resolve_keyframe_path(vid, fid)
                exists = (img_path is not None and img_path.exists())

                match_row = merged[(merged["video_id"] == vid) & (merged["frame_id"] == fid)]
                ocr = match_row.iloc[0].get("ocr_text", "") if not match_row.empty else "N/A"
                desc = match_row.iloc[0].get("description", "") if not match_row.empty else "N/A"
                t_mid = match_row.iloc[0].get("timestamp_mid", 0.0) if not match_row.empty else 0.0

                # Check if frame exists in raw video
                vpath = resolve_video_path(vid)
                video_ok = False
                total_video_frames = 0
                if vpath and vpath.exists():
                    import cv2
                    cap = cv2.VideoCapture(str(vpath))
                    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    video_ok = (0 <= fid < total_video_frames)
                    cap.release()

                print(f"🔹 Dòng #{row_idx}: {vid} | Frame: {fid} ({int(t_mid//60):02d}:{int(t_mid%60):02d})")
                print(f"   • Đáp án nộp  : \"{ans}\"")
                print(f"   • File ảnh    : {'✅ ' + str(img_path.relative_to(PROJECT_ROOT)) if exists else '⚠️ Không có file .webp trong thư mục ảnh'}")
                print(f"   • Video gốc   : {'✅ Frame ' + str(fid) + ' hợp lệ trong video (' + str(total_video_frames) + ' frames)' if video_ok else '❌ Ngoài phạm vi video'}")
                print(f"   • OCR Text    : {ocr}")
                print(f"   • Mô tả Scene : {desc}")
                print("-" * 80)

if __name__ == "__main__":
    main()
