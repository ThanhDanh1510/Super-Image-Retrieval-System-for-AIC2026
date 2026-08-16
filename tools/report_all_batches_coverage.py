import cv2
import pandas as pd
import numpy as np
from pathlib import Path

def main():
    df = pd.read_parquet("data/unified/scene_metadata.parquet")
    video_dir = Path("data/video")
    v_files = sorted(list(video_dir.rglob("*.mp4")) + list(video_dir.rglob("*.mkv")))

    batches = ["L21", "L22", "L23", "L24", "L25", "L26", "L27", "L28", "L29", "L30"]
    print(f"=== AIC2026: TỔNG KẾT ĐỘ PHỦ TIMELINE SAU KHI BỔ SUNG KEYFRAME ({len(v_files)} VIDEOS) ===")

    summary = {}
    for b in batches:
        b_vids = [vf for vf in v_files if vf.stem.split("_")[0] == b]
        if not b_vids:
            continue
        
        gaps = []
        for vf in b_vids:
            vid = vf.stem
            cap = cv2.VideoCapture(str(vf))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            dur = total_f / fps if fps > 0 else 0.0
            cap.release()
            
            v_df = df[df.video_id == vid].sort_values("timestamp_mid")
            ts = v_df["timestamp_mid"].tolist()
            
            if not ts:
                max_gap = dur
            else:
                start_gap = ts[0]
                end_gap = max(0, dur - ts[-1])
                inter_gaps = [ts[i+1] - ts[i] for i in range(len(ts)-1)] if len(ts) > 1 else [0]
                max_inter = max(inter_gaps) if inter_gaps else 0
                max_gap = max(start_gap, max_inter, end_gap)
                
            status = "PASS" if max_gap <= 30.0 else ("WARNING" if max_gap <= 60.0 else "FAIL")
            gaps.append((vid, dur, len(ts), max_gap, status))
            
        n_pass = sum(1 for x in gaps if x[4] == "PASS")
        n_warn = sum(1 for x in gaps if x[4] == "WARNING")
        n_fail = sum(1 for x in gaps if x[4] == "FAIL")
        mean_gap = float(np.mean([x[3] for x in gaps]))
        summary[b] = (len(b_vids), n_pass, n_warn, n_fail, mean_gap)

    print(f"{'Batch':<8} | {'Tổng Video':<12} | {'PASS (<=30s)':<14} | {'WARNING':<10} | {'FAIL':<8} | {'Max Gap TB (s)':<15}")
    print("-" * 75)
    for b, (tot, p, w, f, mg) in sorted(summary.items()):
        print(f"{b:<8} | {tot:<12} | {p:<14} | {w:<10} | {f:<8} | {mg:<15.2f}")

    print("-" * 75)
    total_v = sum(x[0] for x in summary.values())
    total_p = sum(x[1] for x in summary.values())
    total_w = sum(x[2] for x in summary.values())
    total_f = sum(x[3] for x in summary.values())
    overall_mean = float(np.mean([x[4] for x in summary.values()]))
    print(f"{'TỔNG':<8} | {total_v:<12} | {total_p:<14} | {total_w:<10} | {total_f:<8} | {overall_mean:<15.2f}")
    print(f"\n🎉 TỶ LỆ PASS ĐỘ PHỦ TIMELINE: {total_p}/{total_v} ({total_p/total_v:.1%})")
    print(f"Tổng số keyframes hiện có trong Unified Dataset: {len(df)}")

if __name__ == "__main__":
    main()
