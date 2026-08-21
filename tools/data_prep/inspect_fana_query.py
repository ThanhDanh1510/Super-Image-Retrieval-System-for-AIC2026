"""
tools/data_prep/inspect_fana_query.py
=====================================
Điều tra chi tiết câu hỏi về FANA & Khánh Hòa:
"""

import sys
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
UNIFIED_DIR = PROJECT_ROOT / "data" / "unified"

def main():
    meta = pd.read_parquet(UNIFIED_DIR / "scene_metadata.parquet")
    ocr = pd.read_parquet(UNIFIED_DIR / "ocr_text.parquet")
    asr = pd.read_parquet(UNIFIED_DIR / "asr_text.parquet")
    desc = pd.read_parquet(UNIFIED_DIR / "scene_descriptions.parquet")

    merged = meta.copy()
    merged["ocr_text"] = ocr["ocr_text"]
    merged["asr_text"] = asr["asr_text"]
    merged["description"] = desc["description"]

    # Search for FANA
    fana_mask = merged["asr_text"].str.contains("fana|ép ai en|ep ai en", case=False, na=False) | merged["ocr_text"].str.contains("fana", case=False, na=False)
    fana = merged[fana_mask]
    print(f"🔍 Tìm thấy {len(fana)} scenes chứa từ khóa FANA:")
    for idx, r in fana.iterrows():
        print(f"  • {r['video_id']} frame {r['frame_id']} ({r['scene_id']}):")
        print(f"    - ASR: {r['asr_text']}")
        print(f"    - OCR: {r['ocr_text']}")

    vids = sorted(list(set(fana["video_id"])))
    print(f"\n🎥 Danh sách video có FANA: {vids}")

    for vid in vids:
        v_df = merged[merged["video_id"] == vid]
        print(f"\n=======================================================")
        print(f"📹 Toàn bộ {len(v_df)} scenes của video {vid}:")
        print(f"=======================================================")
        for idx, r in v_df.iterrows():
            asr_t = r["asr_text"].lower()
            ocr_t = r["ocr_text"].lower()
            has_kh = "khánh" in asr_t or "khánh" in ocr_t or "khanh" in asr_t
            has_hoa = "hòa" in asr_t or "hoà" in asr_t or "hòa" in ocr_t or "hoà" in ocr_t or "hoa" in asr_t
            tag = " ⭐" if (has_kh or has_hoa) else ""
            print(f"  Frame {r['frame_id']:>5}{tag}:")
            if r["ocr_text"]:
                print(f"    [OCR]: {r['ocr_text']}")
            if r["asr_text"]:
                print(f"    [ASR]: {r['asr_text']}")

if __name__ == "__main__":
    main()
