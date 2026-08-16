"""
tools/detect_l25_video_duplicates.py
====================================
Quét và phát hiện Video trùng lặp (Duplicate Videos) cho Batch L25
Sử dụng chiến lược Đa tầng (Visual Vectors + Sequence Alignment + OCR/ASR Overlap).
"""

import json
import re
from pathlib import Path
import numpy as np
import pandas as pd

def normalize_text(text: str) -> str:
    if not text or text == "None" or text == "-":
        return ""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())

def get_tokens(text: str) -> set:
    norm = normalize_text(text)
    return {w for w in norm.split() if len(w) >= 2}

def jaccard_similarity(tokens_a: set, tokens_b: set) -> float:
    if not tokens_a and not tokens_b:
        return 0.0
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    return intersection / union if union > 0 else 0.0

def token_overlap_ratio(tokens_a: set, tokens_b: set) -> float:
    """Tỷ lệ token của video ngắn hơn nằm trong video dài hơn."""
    if not tokens_a or not tokens_b:
        return 0.0
    min_len = min(len(tokens_a), len(tokens_b))
    if min_len == 0:
        return 0.0
    return len(tokens_a & tokens_b) / min_len

def main():
    print("=== AIC2026: Quét Duplicate Video Toàn Diện cho Batch L25 ===")
    
    # 1. Load Unified Dataset
    meta_df = pd.read_parquet("data/unified/scene_metadata.parquet")
    ocr_df = pd.read_parquet("data/unified/ocr_text.parquet")
    asr_df = pd.read_parquet("data/unified/asr_text.parquet")
    desc_df = pd.read_parquet("data/unified/scene_descriptions.parquet")
    img_vecs = np.load("data/unified/image_vectors.npy")

    # Lọc riêng L25
    l25_mask = meta_df["video_id"].str.startswith("L25_").to_numpy()
    l25_meta = meta_df[l25_mask].reset_index(drop=True)
    l25_ocr = ocr_df[l25_mask].reset_index(drop=True)
    l25_asr = asr_df[l25_mask].reset_index(drop=True)
    l25_desc = desc_df[l25_mask].reset_index(drop=True)
    l25_vecs = img_vecs[l25_mask]

    # Normalize vectors
    norms = np.linalg.norm(l25_vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    l25_vecs = l25_vecs / norms

    videos = sorted(l25_meta["video_id"].unique())
    print(f"Tổng số videos trong L25: {len(videos)}")
    print(f"Tổng số keyframes trong L25: {len(l25_meta)}")

    # 2. Xây dựng Video Profiles
    video_profiles = {}
    for vid in videos:
        v_idx = (l25_meta["video_id"] == vid).to_numpy()
        sub_meta = l25_meta[v_idx]
        sub_ocr = l25_ocr[v_idx]
        sub_asr = l25_asr[v_idx]
        sub_desc = l25_desc[v_idx]
        sub_vecs = l25_vecs[v_idx]

        # Aggregate texts
        full_ocr = " ".join(sub_ocr["ocr_text"].dropna().tolist())
        full_asr = " ".join(sub_asr["asr_text"].dropna().tolist())
        full_desc = " ".join(sub_desc["description"].dropna().tolist())

        ocr_tokens = get_tokens(full_ocr)
        asr_tokens = get_tokens(full_asr)
        
        timestamps = sub_meta["timestamp_mid"].to_numpy()
        duration = float(timestamps.max() - timestamps.min()) if len(timestamps) > 0 else 0.0

        mean_vec = sub_vecs.mean(axis=0)
        mean_vec = mean_vec / np.linalg.norm(mean_vec)

        video_profiles[vid] = {
            "video_id": vid,
            "frame_count": len(sub_meta),
            "frame_ids": sub_meta["frame_id"].tolist(),
            "timestamps": timestamps.tolist(),
            "duration": duration,
            "vecs": sub_vecs,
            "mean_vec": mean_vec,
            "full_ocr": full_ocr,
            "full_asr": full_asr,
            "ocr_tokens": ocr_tokens,
            "asr_tokens": asr_tokens,
            "desc_sample": full_desc[:200] if full_desc else "",
        }

    # 3. So khớp tất cả các cặp Video (Pairwise Comparison)
    results = []
    num_videos = len(videos)

    for i in range(num_videos):
        for j in range(i + 1, num_videos):
            vA_id, vB_id = videos[i], videos[j]
            pA = video_profiles[vA_id]
            pB = video_profiles[vB_id]

            # 3.1 Mean vector similarity (Global Filter)
            mean_sim = float(np.dot(pA["mean_vec"], pB["mean_vec"]))

            # 3.2 Keyframe-to-keyframe Cross Similarity Matrix
            cross_sim = np.dot(pA["vecs"], pB["vecs"].T) # shape (nA, nB)
            max_A_in_B = cross_sim.max(axis=1) # (nA,)
            max_B_in_A = cross_sim.max(axis=0) # (nB,)

            score_A_in_B = float(max_A_in_B.mean())
            score_B_in_A = float(max_B_in_A.mean())
            bidir_visual_score = 0.5 * (score_A_in_B + score_B_in_A)

            # Tỷ lệ frame giống gần như tuyệt đối (cos >= 0.96)
            high_match_ratio_A = float((max_A_in_B >= 0.96).mean())
            high_match_ratio_B = float((max_B_in_A >= 0.96).mean())
            high_match_score = 0.5 * (high_match_ratio_A + high_match_ratio_B)

            # 3.3 Text Similarities (OCR & ASR)
            ocr_jaccard = jaccard_similarity(pA["ocr_tokens"], pB["ocr_tokens"])
            ocr_overlap = token_overlap_ratio(pA["ocr_tokens"], pB["ocr_tokens"])

            asr_jaccard = jaccard_similarity(pA["asr_tokens"], pB["asr_tokens"])
            asr_overlap = token_overlap_ratio(pA["asr_tokens"], pB["asr_tokens"])

            # 3.4 Duration & Frame Count Differences
            dur_diff = abs(pA["duration"] - pB["duration"])
            frame_ratio = min(pA["frame_count"], pB["frame_count"]) / max(pA["frame_count"], pB["frame_count"])

            # 3.5 Classification
            # EXACT_DUPLICATE: Khung hình khớp cực cao VÀ nội dung OCR/ASR trùng lặp cao
            is_exact_dup = (
                bidir_visual_score >= 0.95
                and high_match_score >= 0.85
                and (ocr_overlap >= 0.70 or asr_overlap >= 0.70)
            )

            # PARTIAL_DUPLICATE: Trùng một phần nội dung bài giảng
            is_partial_dup = (
                not is_exact_dup
                and bidir_visual_score >= 0.92
                and (ocr_overlap >= 0.40 or asr_overlap >= 0.40)
            )

            # SAME_TEMPLATE: Cùng studio / cùng thầy cô / cùng intro thư viện nhưng khác bài học
            is_same_template = (
                bidir_visual_score >= 0.90
                and ocr_overlap < 0.30
                and asr_overlap < 0.30
            )

            if is_exact_dup:
                category = "EXACT_100_PERCENT_DUPLICATE"
                action_recommendation = "DISCARD_ONE"
            elif is_partial_dup:
                category = "PARTIAL_NEAR_DUPLICATE"
                action_recommendation = "REVIEW_REQUIRED"
            elif is_same_template:
                category = "SAME_TEMPLATE_DIFFERENT_CONTENT"
                action_recommendation = "KEEP_BOTH"
            else:
                category = "DIFFERENT_VIDEOS"
                action_recommendation = "KEEP_BOTH"

            # Chỉ lưu lại các cặp có visual similarity cao hoặc text similarity đáng chú ý
            if bidir_visual_score >= 0.85 or ocr_overlap >= 0.30 or asr_overlap >= 0.30:
                results.append({
                    "video_A": vA_id,
                    "video_B": vB_id,
                    "frame_count_A": pA["frame_count"],
                    "frame_count_B": pB["frame_count"],
                    "duration_A_sec": round(pA["duration"], 1),
                    "duration_B_sec": round(pB["duration"], 1),
                    "visual_bidirectional_score": round(bidir_visual_score, 4),
                    "high_match_score": round(high_match_score, 4),
                    "ocr_overlap": round(ocr_overlap, 4),
                    "ocr_jaccard": round(ocr_jaccard, 4),
                    "asr_overlap": round(asr_overlap, 4),
                    "asr_jaccard": round(asr_jaccard, 4),
                    "category": category,
                    "action_recommendation": action_recommendation,
                })

    # Sort results
    results.sort(key=lambda x: (
        1 if x["category"] == "EXACT_100_PERCENT_DUPLICATE" else (
            2 if x["category"] == "PARTIAL_NEAR_DUPLICATE" else (
                3 if x["category"] == "SAME_TEMPLATE_DIFFERENT_CONTENT" else 4
            )
        ),
        -x["visual_bidirectional_score"]
    ))

    # Summary
    exact_dups = [r for r in results if r["category"] == "EXACT_100_PERCENT_DUPLICATE"]
    partial_dups = [r for r in results if r["category"] == "PARTIAL_NEAR_DUPLICATE"]
    same_templates = [r for r in results if r["category"] == "SAME_TEMPLATE_DIFFERENT_CONTENT"]

    print("\n=== KẾT QUẢ PHÂN TÍCH DUPLICATE BATCH L25 ===")
    print(f"🔴 EXACT_100_PERCENT_DUPLICATE : {len(exact_dups)} cặp")
    print(f"🟡 PARTIAL_NEAR_DUPLICATE     : {len(partial_dups)} cặp")
    print(f"🟢 SAME_TEMPLATE_DIFFERENT_CONTENT : {len(same_templates)} cặp")

    if exact_dups:
        print("\n--- CHI TIẾT CÁC CẶP TRÙNG LẶP 100% (EXACT DUPLICATE) ---")
        for r in exact_dups:
            print(f"  • {r['video_A']} ({r['frame_count_A']} frames) <==> {r['video_B']} ({r['frame_count_B']} frames)")
            print(f"    Visual={r['visual_bidirectional_score']:.3f} | HighMatch={r['high_match_score']:.1%} | OCR_Overlap={r['ocr_overlap']:.1%} | ASR_Overlap={r['asr_overlap']:.1%}")

    if partial_dups:
        print("\n--- CHI TIẾT CÁC CẶP TRÙNG LẶP MỘT PHẦN (PARTIAL DUPLICATE) ---")
        for r in partial_dups:
            print(f"  • {r['video_A']} ({r['frame_count_A']} frames) <==> {r['video_B']} ({r['frame_count_B']} frames)")
            print(f"    Visual={r['visual_bidirectional_score']:.3f} | HighMatch={r['high_match_score']:.1%} | OCR_Overlap={r['ocr_overlap']:.1%} | ASR_Overlap={r['asr_overlap']:.1%}")

    # Export report JSON
    output_report = {
        "batch": "L25",
        "total_videos_scanned": len(videos),
        "exact_duplicate_pairs_count": len(exact_dups),
        "partial_duplicate_pairs_count": len(partial_dups),
        "same_template_pairs_count": len(same_templates),
        "exact_duplicate_pairs": exact_dups,
        "partial_duplicate_pairs": partial_dups,
        "all_pairs": results,
    }

    report_path = Path("data/l25_duplicate_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(output_report, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Đã lưu toàn bộ báo cáo chi tiết vào: {report_path}")

if __name__ == "__main__":
    main()
