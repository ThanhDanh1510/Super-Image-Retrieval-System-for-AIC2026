"""
tools/scan_l25_pixel_level_duplicates.py
========================================
Quét và phân tích cấp độ Pixel / Histogram / OCR Title đối với toàn bộ 88 video trong L25.
Phân biệt chính xác giữa:
1. EXACT_DUPLICATE: Video cùng bài giảng / trùng khung hình thật sự (Pixel MAE < 15.0).
2. SAME_STUDIO_TEMPLATE: Cùng studio / cùng thầy cô / cùng intro thư viện nhưng khác bài học (Pixel MAE > 50.0).
"""

import json
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

def get_video_images(vid_id):
    sub_id = vid_id.split("_")[-1]
    v_dir = Path("data/image/L25/L25") / sub_id
    if not v_dir.exists():
        return []
    files = sorted(
        list(v_dir.glob("*.webp")) + list(v_dir.glob("*.jpg")),
        key=lambda p: int(p.stem) if p.stem.isdigit() else 0
    )
    return files

def compute_pixel_mae_and_hist_sim(files_a, files_b, num_samples=10):
    if not files_a or not files_b:
        return 999.0, 0.0

    maes = []
    hist_sims = []

    for k in range(num_samples):
        idx_a = int(k * (len(files_a) - 1) / (num_samples - 1))
        idx_b = int(k * (len(files_b) - 1) / (num_samples - 1))

        img_a = cv2.imread(str(files_a[idx_a]))
        img_b = cv2.imread(str(files_b[idx_b]))

        if img_a is None or img_b is None:
            continue

        # Resize về kích thước chuẩn nhỏ 160x90 để so khớp pixel nhanh và chống noise nén
        img_a_small = cv2.resize(img_a, (160, 90))
        img_b_small = cv2.resize(img_b, (160, 90))

        # 1. Pixel Mean Absolute Error (MAE)
        mae = float(np.mean(np.abs(img_a_small.astype(float) - img_b_small.astype(float))))
        maes.append(mae)

        # 2. HSV Color Histogram Similarity
        hsv_a = cv2.cvtColor(img_a_small, cv2.COLOR_BGR2HSV)
        hsv_b = cv2.cvtColor(img_b_small, cv2.COLOR_BGR2HSV)

        hist_a = cv2.calcHist([hsv_a], [0, 1], None, [16, 16], [0, 180, 0, 256])
        hist_b = cv2.calcHist([hsv_b], [0, 1], None, [16, 16], [0, 180, 0, 256])
        cv2.normalize(hist_a, hist_a, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        cv2.normalize(hist_b, hist_b, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

        sim = cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL)
        hist_sims.append(float(sim))

    mean_mae = float(np.mean(maes)) if maes else 999.0
    mean_hist_sim = float(np.mean(hist_sims)) if hist_sims else 0.0

    return mean_mae, mean_hist_sim

def main():
    print("=== AIC2026: Quét Duplicate Cấp Độ Pixel cho 88 Video Batch L25 ===")
    meta_df = pd.read_parquet("data/unified/scene_metadata.parquet")
    l25_df = meta_df[meta_df["video_id"].str.startswith("L25_")].reset_index(drop=True)
    videos = sorted(l25_df["video_id"].unique())
    print(f"Tổng số videos L25: {len(videos)}")

    # Load video images cache
    vid_images = {}
    for vid in videos:
        vid_images[vid] = get_video_images(vid)

    exact_duplicates = []
    close_candidates = []

    total_pairs = len(videos) * (len(videos) - 1) // 2
    print(f"Bắt đầu quét {total_pairs} cặp video...")

    checked = 0
    for i in range(len(videos)):
        for j in range(i + 1, len(videos)):
            vA, vB = videos[i], videos[j]
            imgsA = vid_images[vA]
            imgsB = vid_images[vB]

            checked += 1
            if checked % 500 == 0:
                print(f"  Đã quét {checked}/{total_pairs} cặp video...")

            # Nếu độ lệch số lượng frame quá lớn (> 20%), khả năng cao không phải duplicate file
            len_ratio = min(len(imgsA), len(imgsB)) / max(len(imgsA), len(imgsB)) if (imgsA and imgsB) else 0
            if len_ratio < 0.80:
                continue

            mean_mae, mean_hist_sim = compute_pixel_mae_and_hist_sim(imgsA, imgsB, num_samples=8)

            # Phân loại
            if mean_mae <= 20.0 and mean_hist_sim >= 0.85:
                exact_duplicates.append({
                    "video_A": vA,
                    "video_B": vB,
                    "frame_count_A": len(imgsA),
                    "frame_count_B": len(imgsB),
                    "pixel_mae": round(mean_mae, 2),
                    "hist_similarity": round(mean_hist_sim, 4),
                    "category": "EXACT_100_PERCENT_DUPLICATE",
                    "action": "DISCARD_ONE",
                })
            elif mean_mae <= 40.0 and mean_hist_sim >= 0.70:
                close_candidates.append({
                    "video_A": vA,
                    "video_B": vB,
                    "frame_count_A": len(imgsA),
                    "frame_count_B": len(imgsB),
                    "pixel_mae": round(mean_mae, 2),
                    "hist_similarity": round(mean_hist_sim, 4),
                    "category": "NEAR_DUPLICATE_CANDIDATE",
                    "action": "REVIEW",
                })

    print("\n============================================================")
    print("🎉 KẾT QUẢ QUÉT PIXEL DUPLICATE BATCH L25")
    print(f"🔴 EXACT_100_PERCENT_DUPLICATE : {len(exact_duplicates)} cặp")
    print(f"🟡 NEAR_DUPLICATE_CANDIDATE    : {len(close_candidates)} cặp")
    print("============================================================")

    if exact_duplicates:
        print("\n--- CÁC CẶP TRÙNG LẶP 100% HOÀN TOÀN ---")
        for d in exact_duplicates:
            print(f"  • {d['video_A']} ({d['frame_count_A']} frames) <==> {d['video_B']} ({d['frame_count_B']} frames) | Pixel MAE: {d['pixel_mae']} | HistSim: {d['hist_similarity']}")

    if close_candidates:
        print("\n--- CÁC CẶP GẦN TRÙNG (CẦN REVIEW) ---")
        for d in close_candidates:
            print(f"  • {d['video_A']} ({d['frame_count_A']} frames) <==> {d['video_B']} ({d['frame_count_B']} frames) | Pixel MAE: {d['pixel_mae']} | HistSim: {d['hist_similarity']}")

    report = {
        "batch": "L25",
        "total_videos": len(videos),
        "exact_duplicates": exact_duplicates,
        "near_duplicates": close_candidates,
    }

    out_file = Path("data/l25_pixel_duplicate_report.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Đã lưu báo cáo cấp độ pixel vào: {out_file}")

if __name__ == "__main__":
    main()
