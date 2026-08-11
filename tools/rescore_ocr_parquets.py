import os
import glob
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.ocr.ocr_local import select_best_candidate, should_filter_text

ocr_dir = os.path.join(PROJECT_ROOT, "data", "ocr_output")
parquet_files = sorted(glob.glob(os.path.join(ocr_dir, "L*_ocr.parquet")))

print("=" * 80)
print("🔄 RE-SCORING OCR PARQUET FILES WITH ENHANCED VLM PRIORITY SELECTION")
print("=" * 80)

total_updated_frames = 0
total_processed_frames = 0

for pf in parquet_files:
    fname = os.path.basename(pf)
    df = pd.read_parquet(pf)
    updated_in_file = 0

    new_ocr_texts = []
    for idx, row in df.iterrows():
        cands = {}
        if "local_raw" in row and pd.notna(row["local_raw"]):
            cands["local"] = str(row["local_raw"])
        if "vlm_raw" in row and pd.notna(row["vlm_raw"]):
            cands["vlm"] = str(row["vlm_raw"])

        current_ocr = str(row.get("ocr_text", ""))
        new_ocr = select_best_candidate(cands, filter_spam=True)

        if new_ocr != current_ocr:
            updated_in_file += 1
        new_ocr_texts.append(new_ocr)

    df["ocr_text"] = new_ocr_texts
    df.to_parquet(pf, index=False)
    
    total_processed_frames += len(df)
    total_updated_frames += updated_in_file
    print(f" 💾 {fname:15s}: Đã cập nhật {updated_in_file:>5,d} / {len(df):>7,d} frames với kết quả VLM chuẩn hóa!")

# Synchronize data/unified/ocr_text.parquet
print("\n🔄 Đang hợp nhất lại dữ liệu mới vào data/unified/ocr_text.parquet...")
all_dfs = []
for pf in sorted(glob.glob(os.path.join(ocr_dir, "L*_ocr.parquet"))):
    all_dfs.append(pd.read_parquet(pf)[['scene_id', 'video_id', 'ocr_text']])

df_unified = pd.concat(all_dfs, ignore_index=True)
df_unified.to_parquet(os.path.join(PROJECT_ROOT, "data", "unified", "ocr_text.parquet"), index=False)

print("\n" + "=" * 80)
print("🎉 HOÀN THÀNH CẬP NHẬT RE-SCORE CHO TOÀN BỘ HỆ THỐNG!")
print(f"📊 Tổng số frames cập nhật chất lượng cao hơn: {total_updated_frames:,d} / {total_processed_frames:,d} frames")
print(f"💾 File System Parquet: data/unified/ocr_text.parquet")
print("=" * 80)
