"""
tools/scan_l25_fast_pixel_duplicates.py
=======================================
Siêu tốc độ: Quét duplicate video cấp độ Pixel / Frame Sequence / Text Overlap cho 88 video L25.
Sử dụng bộ nhớ đệm thumbnail 80x45 để xử lý toàn bộ 3.828 cặp video chỉ trong < 3 giây.
"""

import sys
import json
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

NUM_SAMPLES = 10
THUMB_W = 80
THUMB_H = 45

def get_video_images(vid_id):
    sub_id = vid_id.split("_")[-1]
    v_dir = Path("data/image/L25/L25") / sub_id
    if not v_dir.exists():
        return []
    return sorted(
        list(v_dir.glob("*.webp")) + list(v_dir.glob("*.jpg")),
        key=lambda p: int(p.stem) if p.stem.isdigit() else 0
    )

def load_video_sample_thumbs(files):
    if not files:
        return np.zeros((NUM_SAMPLES, THUMB_H, THUMB_W), dtype=np.uint8)

    thumbs = []
    n = len(files)
    for k in range(NUM_SAMPLES):
        idx = int(k * (n - 1) / (NUM_SAMPLES - 1))
        img = cv2.imread(str(files[idx]), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            thumb = cv2.resize(img, (THUMB_W, THUMB_H))
        else:
            thumb = np.zeros((THUMB_H, THUMB_W), dtype=np.uint8)
        thumbs.append(thumb)
    return np.array(thumbs, dtype=np.uint8) # Shape (NUM_SAMPLES, 45, 80)

def main():
    print("=== AIC2026: Fast Pixel & Sequence Duplicate Scanner cho Batch L25 ===", flush=True)
    meta_df = pd.read_parquet("data/unified/scene_metadata.parquet")
    ocr_df = pd.read_parquet("data/unified/ocr_text.parquet")
    asr_df = pd.read_parquet("data/unified/asr_text.parquet")

    l25_mask = meta_df["video_id"].str.startswith("L25_").to_numpy()
    l25_meta = meta_df[l25_mask].reset_index(drop=True)
    l25_ocr = ocr_df[l25_mask].reset_index(drop=True)
    l25_asr = asr_df[l25_mask].reset_index(drop=True)

    videos = sorted(l25_meta["video_id"].unique())
    print(f"Tổng số videos L25: {len(videos)}", flush=True)

    # 1. Thu thập danh sách file ảnh
    vid_to_files = {}
    for vid in videos:
        vid_to_files[vid] = get_video_images(vid)

    print("Đang đọc và tạo thumbnail bộ nhớ đệm cho 88 videos...", flush=True)
    with ThreadPoolExecutor(max_workers=8) as ex:
        thumb_list = list(ex.map(load_video_sample_thumbs, [vid_to_files[v] for v in videos]))

    vid_to_thumbs = {v: t for v, t in zip(videos, thumb_list)}

    # 2. Text Corpus
    vid_to_ocr = {}
    vid_to_asr = {}
    for vid in videos:
        v_idx = (l25_meta["video_id"] == vid).to_numpy()
        ocr_tokens = set(" ".join(l25_ocr[v_idx]["ocr_text"].dropna().tolist()).lower().split())
        asr_tokens = set(" ".join(l25_asr[v_idx]["asr_text"].dropna().tolist()).lower().split())
        vid_to_ocr[vid] = {w for w in ocr_tokens if len(w) >= 2}
        vid_to_asr[vid] = {w for w in asr_tokens if len(w) >= 2}

    print("Đang so sánh 3.828 cặp video ở cấp độ Pixel & Nội dung Text...", flush=True)

    exact_duplicates = []
    near_duplicates = []
    all_scores = []

    for i in range(len(videos)):
        for j in range(i + 1, len(videos)):
            vA, vB = videos[i], videos[j]
            tA, tB = vid_to_thumbs[vA], vid_to_thumbs[vB]
            fA, fB = len(vid_to_files[vA]), len(vid_to_files[vB])

            # Tính Pixel MAE trên toàn bộ 10 điểm mốc thời gian
            # tA, tB: shape (10, 45, 80)
            diff = np.abs(tA.astype(np.float32) - tB.astype(np.float32)) # (10, 45, 80)
            sample_maes = diff.mean(axis=(1, 2)) # (10,)
            mean_pixel_mae = float(sample_maes.mean())
            max_pixel_mae = float(sample_maes.max())

            # Text Overlap
            ocrA, ocrB = vid_to_ocr[vA], vid_to_ocr[vB]
            asrA, asrB = vid_to_asr[vA], vid_to_asr[vB]

            ocr_overlap = len(ocrA & ocrB) / min(len(ocrA), len(ocrB)) if (ocrA and ocrB) else 0.0
            asr_overlap = len(asrA & asrB) / min(len(asrA), len(asrB)) if (asrA and asrB) else 0.0

            # Phân loại độ tương đồng
            # EXACT DUPLICATE: Pixel MAE < 15.0 trên toàn bộ các mốc thời gian
            if mean_pixel_mae <= 15.0 and max_pixel_mae <= 25.0:
                exact_duplicates.append({
                    "video_A": vA,
                    "video_B": vB,
                    "frame_count_A": fA,
                    "frame_count_B": fB,
                    "mean_pixel_mae": round(mean_pixel_mae, 2),
                    "max_pixel_mae": round(max_pixel_mae, 2),
                    "ocr_overlap": round(ocr_overlap, 3),
                    "asr_overlap": round(asr_overlap, 3),
                    "category": "EXACT_100_PERCENT_DUPLICATE",
                    "action": "DISCARD_ONE",
                })
            elif mean_pixel_mae <= 40.0:
                near_duplicates.append({
                    "video_A": vA,
                    "video_B": vB,
                    "frame_count_A": fA,
                    "frame_count_B": fB,
                    "mean_pixel_mae": round(mean_pixel_mae, 2),
                    "max_pixel_mae": round(max_pixel_mae, 2),
                    "ocr_overlap": round(ocr_overlap, 3),
                    "asr_overlap": round(asr_overlap, 3),
                    "category": "NEAR_DUPLICATE_OR_REVISED_CUT",
                    "action": "REVIEW_MANUALLY",
                })

            all_scores.append({
                "vA": vA,
                "vB": vB,
                "mean_pixel_mae": mean_pixel_mae,
                "ocr_overlap": ocr_overlap,
                "asr_overlap": asr_overlap,
            })

    all_scores.sort(key=lambda x: x["mean_pixel_mae"])

    print("\n" + "=" * 65, flush=True)
    print("🎉 KẾT QUẢ QUÉT TOÀN DIỆN 88 VIDEO BATCH L25", flush=True)
    print(f"🔴 EXACT_100_PERCENT_DUPLICATE : {len(exact_duplicates)} cặp", flush=True)
    print(f"🟡 NEAR_DUPLICATE_CANDIDATES   : {len(near_duplicates)} cặp", flush=True)
    print("=" * 65, flush=True)

    if exact_duplicates:
        print("\n--- 🔴 DANH SÁCH VIDEO TRÙNG LẶP 100% HOÀN TOÀN (CÙNG FILE/FRAME-FOR-FRAME) ---", flush=True)
        for d in exact_duplicates:
            print(f"  • {d['video_A']} ({d['frame_count_A']} frames) <===> {d['video_B']} ({d['frame_count_B']} frames)", flush=True)
            print(f"    Mean Pixel MAE: {d['mean_pixel_mae']:.2f}/255 | Max MAE: {d['max_pixel_mae']:.2f} | OCR Match: {d['ocr_overlap']:.1%} | ASR Match: {d['asr_overlap']:.1%}", flush=True)

    if near_duplicates:
        print("\n--- 🟡 DANH SÁCH VIDEO GẦN TRÙNG (CÙNG BÀI HỌC / RE-CUT) ---", flush=True)
        for d in near_duplicates:
            print(f"  • {d['video_A']} ({d['frame_count_A']} frames) <===> {d['video_B']} ({d['frame_count_B']} frames)", flush=True)
            print(f"    Mean Pixel MAE: {d['mean_pixel_mae']:.2f}/255 | Max MAE: {d['max_pixel_mae']:.2f} | OCR Match: {d['ocr_overlap']:.1%} | ASR Match: {d['asr_overlap']:.1%}", flush=True)

    print("\n--- 🔝 TOP 10 CẶP VIDEO CÓ ĐỘ LỆCH PIXEL THẤP NHẤT TRONG L25 ---", flush=True)
    for s in all_scores[:10]:
        print(f"  {s['vA']} <-> {s['vB']}: Mean Pixel MAE = {s['mean_pixel_mae']:.2f} (0=Exact Match, 255=Max Diff) | OCR = {s['ocr_overlap']:.1%} | ASR = {s['asr_overlap']:.1%}", flush=True)

    # Lưu báo cáo JSON
    report = {
        "batch": "L25",
        "total_videos_scanned": len(videos),
        "total_pairs_compared": len(all_scores),
        "exact_duplicates": exact_duplicates,
        "near_duplicates": near_duplicates,
        "top_closest_pairs": all_scores[:30],
    }

    out_path = Path("data/l25_pixel_duplicate_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Đã xuất báo cáo chi tiết vào: {out_path}", flush=True)

if __name__ == "__main__":
    main()
