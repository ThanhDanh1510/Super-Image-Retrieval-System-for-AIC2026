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
    cross_counts = ((sim_matrix >= 0.98) & diff_video_mask).sum(axis=1)

    tvc_explicit_kw = [
        "saigon international university",
        "quốc tế á châu",
        "asian international school",
        "siu",
    ]

    # Cluster duplicate frames
    # 1. Any frame with explicit sponsor TVC ad
    discard_indices = set()
    for idx in range(len(l25_df)):
        ocr_raw = str(l25_ocr.iloc[idx]["ocr_text"] or "").lower()
        if any(kw in ocr_raw for kw in tvc_explicit_kw):
            discard_indices.add(idx)

    # 2. Repeated intro / outro / bumper clusters appearing in >= 20 videos
    high_dup_indices = np.where((cross_counts >= 20) & (~np.isin(np.arange(len(l25_df)), list(discard_indices))))[0]
    
    # Sort by video id so we keep representative in V001
    visited = set()
    kept_reps = set()
    
    for idx in high_dup_indices:
        if idx in visited:
            continue
        # Frame này là đại diện đầu tiên -> Giữ lại
        kept_reps.add(idx)
        visited.add(idx)

        # Các frame ở các video khác có độ tương đồng cực cao (cos >= 0.98) -> Bỏ bản sao
        dup_neighbors = np.where((sim_matrix[idx] >= 0.98) & (cross_counts >= 20))[0]
        for n in dup_neighbors:
            if n != idx and n not in visited:
                discard_indices.add(n)
                visited.add(n)

    print(f"Total L25 keyframes: {len(l25_df)}")
    print(f"Explicit Sponsor TVC frames to discard: {len([i for i in discard_indices if i not in high_dup_indices])}")
    print(f"Cross-video bumper duplicate copies to discard: {len([i for i in discard_indices if i in high_dup_indices])}")
    print(f"Total frames to discard: {len(discard_indices)} ({len(discard_indices)/len(l25_df):.1%})")
    print(f"L25 keyframes preserved (clean unique lessons & 1 rep per bumper): {len(l25_df) - len(discard_indices)}")

if __name__ == "__main__":
    main()
