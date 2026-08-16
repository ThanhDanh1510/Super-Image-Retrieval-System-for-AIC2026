"""
tools/filter_l25_repeated_tvc.py
================================
Tự động lọc bỏ các đoạn TVC Quảng cáo nhà tài trợ (SIU, Quốc tế Á Châu)
và các bản sao lặp lại của đoạn Intro/Outro/Bumper xuất hiện trên nhiều video L25.
Ghi nhận toàn bộ vào data/filtered_keyframes.json và chạy apply_filter để xóa ảnh thực tế.
"""

import json
import datetime
import re
import numpy as np
import pandas as pd
from pathlib import Path

def main():
    print("=== AIC2026: Lọc Bỏ TVC Quảng Cáo & Intro Lặp Lại cho Batch L25 ===")
    meta_df = pd.read_parquet("data/unified/scene_metadata.parquet")
    ocr_df = pd.read_parquet("data/unified/ocr_text.parquet")
    img_vecs = np.load("data/unified/image_vectors.npy")

    l25_mask = meta_df.video_id.str.startswith("L25_").to_numpy()
    l25_meta = meta_df[l25_mask].reset_index(drop=True)
    l25_ocr = ocr_df[l25_mask].reset_index(drop=True)
    l25_vecs = img_vecs[l25_mask]
    l25_vecs = l25_vecs / np.linalg.norm(l25_vecs, axis=1, keepdims=True)

    print(f"Tổng số keyframes L25 trước khi lọc: {len(l25_meta)}")

    sim_matrix = np.dot(l25_vecs, l25_vecs.T)
    video_ids = l25_meta.video_id.to_numpy()
    diff_video_mask = (video_ids[:, None] != video_ids[None, :])
    cross_counts = ((sim_matrix >= 0.98) & diff_video_mask).sum(axis=1)

    tvc_explicit_kw = [
        "saigon international university",
        "quốc tế á châu",
        "asian international school",
        "siu",
    ]

    discard_map = {} # idx -> reason

    # 1. Các frame quảng cáo TVC rõ ràng
    for idx in range(len(l25_meta)):
        ocr_raw = str(l25_ocr.iloc[idx]["ocr_text"] or "").lower()
        if any(kw in ocr_raw for kw in tvc_explicit_kw):
            discard_map[idx] = "tvc_sponsor_ad_siu_asian_school"

    # 2. Các cụm Intro/Outro/Bumper lặp lại trên >= 20 videos khác nhau
    high_dup_indices = np.where((cross_counts >= 20) & (~np.isin(np.arange(len(l25_meta)), list(discard_map.keys()))))[0]

    visited = set()
    kept_reps = set()

    for idx in high_dup_indices:
        if idx in visited:
            continue
        # Giữ 1 frame đại diện đầu tiên
        kept_reps.add(idx)
        visited.add(idx)

        # Các frame bản sao ở các video khác
        dup_neighbors = np.where((sim_matrix[idx] >= 0.98) & (cross_counts >= 20))[0]
        rep_video = l25_meta.iloc[idx]["video_id"]
        rep_frame = l25_meta.iloc[idx]["frame_id"]

        for n in dup_neighbors:
            if n != idx and n not in visited:
                discard_map[n] = f"repeated_bumper_dup_of_{rep_video}_f{rep_frame}"
                visited.add(n)

    print(f"Tổng số frame TVC & Bumper lặp lại cần loại bỏ: {len(discard_map)} ({len(discard_map)/len(l25_meta):.1%})")
    print(f"Tổng số frame bài giảng độc lập & đại diện được giữ lại: {len(l25_meta) - len(discard_map)}")

    # 3. Cập nhật vào data/filtered_keyframes.json
    fk_path = Path("data/filtered_keyframes.json")
    if fk_path.exists():
        with open(fk_path, "r", encoding="utf-8") as f:
            fk_data = json.load(f)
    else:
        fk_data = {"filtered_images": []}

    existing_items = fk_data.get("filtered_images", [])
    existing_keys = {item.get("keyframe_path"): item for item in existing_items if item.get("keyframe_path")}

    now_iso = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    added_count = 0
    for idx, reason in discard_map.items():
        row = l25_meta.iloc[idx]
        b = "L25"
        full_vid = row["video_id"]
        sub_vid = full_vid.split("_")[-1] if "_" in full_vid else full_vid
        fid = int(row["frame_id"])
        fname = f"{fid}.webp"
        k = f"{b}/{sub_vid}/{fname}"

        if k not in existing_keys:
            item = {
                "batch": b,
                "video_id": sub_vid,
                "full_video_id": full_vid,
                "filename": fname,
                "frame_id": fid,
                "keyframe_path": k,
                "reason": reason,
                "added_at": now_iso,
            }
            existing_keys[k] = item
            existing_items.append(item)
            added_count += 1

    summary_by_batch = {}
    for item in existing_items:
        b = item.get("batch", "UNKNOWN")
        summary_by_batch[b] = summary_by_batch.get(b, 0) + 1

    output_data = {
        "total_filtered_count": len(existing_items),
        "updated_at": now_iso,
        "summary_by_batch": dict(sorted(summary_by_batch.items())),
        "filtered_images": existing_items,
    }

    with open(fk_path, "w", encoding="utf-8") as out_f:
        json.dump(output_data, out_f, ensure_ascii=False, indent=2)

    print(f"\n✅ Đã thêm {added_count} frame vào {fk_path}")
    print(f"Tổng số frame đã lọc trong filtered_keyframes.json: {len(existing_items)}")
    print(f"Thống kê theo batch: {dict(sorted(summary_by_batch.items()))}")

if __name__ == "__main__":
    main()
