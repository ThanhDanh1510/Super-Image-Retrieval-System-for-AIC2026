import cv2
import numpy as np
import pandas as pd
from pathlib import Path

def main():
    video_dir = Path("data/video")
    l25_videos = sorted(list(video_dir.glob("L25_*.mp4")) + list(video_dir.glob("L25_*.mkv")))
    print(f"Found {len(l25_videos)} video files in data/video for L25")

    video_infos = {}
    for vp in l25_videos:
        vid = vp.stem
        cap = cv2.VideoCapture(str(vp))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        fcount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        dur = fcount / fps if fps > 0 else 0
        size = vp.stat().st_size
        cap.release()
        video_infos[vid] = {
            "path": str(vp),
            "fps": fps,
            "frame_count": fcount,
            "duration": dur,
            "size": size,
        }

    potential_pairs = []
    v_list = list(video_infos.keys())
    for i in range(len(v_list)):
        for j in range(i + 1, len(v_list)):
            vA, vB = v_list[i], v_list[j]
            infoA, infoB = video_infos[vA], video_infos[vB]

            dur_diff = abs(infoA["duration"] - infoB["duration"])
            size_ratio = min(infoA["size"], infoB["size"]) / max(infoA["size"], infoB["size"])

            if dur_diff <= 5.0 and size_ratio >= 0.85:
                potential_pairs.append((vA, vB, dur_diff, size_ratio))

    print(f"\nFound {len(potential_pairs)} potential duplicate video pairs based on video duration & file size:")
    for vA, vB, d_diff, s_rat in potential_pairs:
        durA = video_infos[vA]["duration"]
        durB = video_infos[vB]["duration"]
        print(f"Candidate: {vA} <-> {vB} | DurDiff={d_diff:.1f}s | SizeRatio={s_rat:.2f} | DurA={durA:.1f}s | DurB={durB:.1f}s")

if __name__ == "__main__":
    main()
