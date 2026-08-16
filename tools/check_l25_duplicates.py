import numpy as np
import pandas as pd

def main():
    df = pd.read_parquet("data/unified/scene_metadata.parquet")
    img_vecs = np.load("data/unified/image_vectors.npy")

    # Lọc L25
    l25_mask = df["video_id"].str.startswith("L25_").to_numpy()
    l25_df = df[l25_mask].reset_index(drop=True)
    l25_vecs = img_vecs[l25_mask]

    # Normalize vectors
    norms = np.linalg.norm(l25_vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    l25_vecs = l25_vecs / norms

    videos = sorted(l25_df["video_id"].unique())
    print(f"Total L25 videos: {len(videos)}, total keyframes: {len(l25_df)}")

    vid_to_vecs = {}
    vid_to_frames = {}
    for vid in videos:
        v_mask = (l25_df["video_id"] == vid).to_numpy()
        vid_to_vecs[vid] = l25_vecs[v_mask]
        vid_to_frames[vid] = l25_df[v_mask]["frame_id"].to_numpy()

    # 1. Mean video vectors
    vid_means = np.array([vid_to_vecs[v].mean(axis=0) for v in videos])
    vid_means = vid_means / np.linalg.norm(vid_means, axis=1, keepdims=True)
    sim_matrix = np.dot(vid_means, vid_means.T)

    high_sim_pairs = []
    for i in range(len(videos)):
        for j in range(i + 1, len(videos)):
            vA, vB = videos[i], videos[j]
            mean_sim = sim_matrix[i, j]
            if mean_sim >= 0.70:
                vecsA = vid_to_vecs[vA]
                vecsB = vid_to_vecs[vB]
                cross_sim = np.dot(vecsA, vecsB.T)

                score_A_in_B = cross_sim.max(axis=1).mean()
                score_B_in_A = cross_sim.max(axis=0).mean()
                bidirectional_score = 0.5 * (score_A_in_B + score_B_in_A)

                high_match_ratio_A = (cross_sim.max(axis=1) >= 0.95).mean()
                high_match_ratio_B = (cross_sim.max(axis=0) >= 0.95).mean()

                high_sim_pairs.append({
                    "vA": vA,
                    "vB": vB,
                    "lenA": len(vecsA),
                    "lenB": len(vecsB),
                    "mean_sim": float(mean_sim),
                    "bidirectional_score": float(bidirectional_score),
                    "high_match_A": float(high_match_ratio_A),
                    "high_match_B": float(high_match_ratio_B),
                })

    high_sim_pairs.sort(key=lambda x: x["bidirectional_score"], reverse=True)
    print(f"\nFound {len(high_sim_pairs)} video pairs with mean_sim >= 0.70:")
    for p in high_sim_pairs[:25]:
        vA, vB = p["vA"], p["vB"]
        lenA, lenB = p["lenA"], p["lenB"]
        bs = p["bidirectional_score"]
        ms = p["mean_sim"]
        hmA = p["high_match_A"]
        hmB = p["high_match_B"]
        print(f"Pair {vA} ({lenA} frames) <-> {vB} ({lenB} frames): BidirScore={bs:.4f} | MeanSim={ms:.4f} | HighMatchA={hmA:.1%} | HighMatchB={hmB:.1%}")

if __name__ == "__main__":
    main()
