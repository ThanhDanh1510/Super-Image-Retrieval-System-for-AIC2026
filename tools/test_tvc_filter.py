import re
import numpy as np
import pandas as pd

def main():
    df = pd.read_parquet("data/unified/scene_metadata.parquet")
    ocr_df = pd.read_parquet("data/unified/ocr_text.parquet")
    desc_df = pd.read_parquet("data/unified/scene_descriptions.parquet")
    img_vecs = np.load("data/unified/image_vectors.npy")

    l25_mask = df.video_id.str.startswith("L25_").to_numpy()
    l25_df = df[l25_mask].reset_index(drop=True)
    l25_ocr = ocr_df[l25_mask].reset_index(drop=True)
    l25_desc = desc_df[l25_mask].reset_index(drop=True)
    l25_vecs = img_vecs[l25_mask]
    l25_vecs = l25_vecs / np.linalg.norm(l25_vecs, axis=1, keepdims=True)

    sim_matrix = np.dot(l25_vecs, l25_vecs.T)
    video_ids = l25_df.video_id.to_numpy()
    diff_video_mask = (video_ids[:, None] != video_ids[None, :])
    cross_counts = ((sim_matrix >= 0.975) & diff_video_mask).sum(axis=1)

    # Từ khóa quảng cáo TVC / Bumper
    tvc_keywords = [
        "saigon international university", "quốc tế á châu", "siu",
        "thư viện", "báo thanh niên", "chương trình tư vấn mùa thi",
        "tổng đài", "học phí", "international school", "international university",
        "thực chiến luyện đề", "đỗ đức đức anh"
    ]

    discard_candidates = []
    for idx in range(len(l25_df)):
        row = l25_df.iloc[idx]
        ocr_raw = str(l25_ocr.iloc[idx]["ocr_text"] or "").lower()
        desc_raw = str(l25_desc.iloc[idx]["description"] or "").lower()

        has_tvc_kw = any(kw in ocr_raw for kw in tvc_keywords)
        is_super_repeated = (cross_counts[idx] >= 30)
        has_substantive_ocr = len(re.sub(r"[^a-zA-Z0-9\s]", "", ocr_raw).split()) > 15

        if has_tvc_kw or (is_super_repeated and not has_substantive_ocr):
            discard_candidates.append(idx)

    print(f"Total L25 keyframes: {len(l25_df)}")
    print(f"Identified TVC/Bumper/Generic Intro frames to discard: {len(discard_candidates)} ({len(discard_candidates)/len(l25_df):.1%})")
    print(f"L25 keyframes preserved (real lessons & slides): {len(l25_df) - len(discard_candidates)}")

    print("\n--- MẪU 5 FRAME BỊ LỌC (TVC/INTRO) ---")
    for idx in discard_candidates[:5]:
        row = l25_df.iloc[idx]
        ocr_snippet = str(l25_ocr.iloc[idx]["ocr_text"] or "")[:60].replace("\n", " ")
        print(f"{row.video_id} frame {row.frame_id} (ts={row.timestamp_mid:.1f}s) | Cross-dup={cross_counts[idx]} | OCR: {ocr_snippet}")

    preserved = [i for i in range(len(l25_df)) if i not in discard_candidates]
    print("\n--- MẪU 5 FRAME ĐƯỢC BẢO TOÀN (BÀI GIẢNG / ĐỀ THI) ---")
    for idx in preserved[:5]:
        row = l25_df.iloc[idx]
        ocr_snippet = str(l25_ocr.iloc[idx]["ocr_text"] or "")[:60].replace("\n", " ")
        print(f"{row.video_id} frame {row.frame_id} (ts={row.timestamp_mid:.1f}s) | Cross-dup={cross_counts[idx]} | OCR: {ocr_snippet}")

if __name__ == "__main__":
    main()
