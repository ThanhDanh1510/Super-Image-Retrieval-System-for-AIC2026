"""
tools/merge_additional_filters.py
=================================
Script merge dữ liệu từ:
- data/filtered_noise_frames.json (221 noise frames)
- data/duplicate_and_discarded_videos.json (3 duplicate videos to discard)
vào data/filtered_keyframes.json
"""

import json
import datetime
from pathlib import Path
import pandas as pd

def main():
    fk_path = Path("data/filtered_keyframes.json")
    if fk_path.exists():
        with open(fk_path, "r", encoding="utf-8") as f:
            fk_data = json.load(f)
    else:
        fk_data = {"filtered_images": []}

    existing_items = fk_data.get("filtered_images", [])
    existing_keys = {item.get("keyframe_path"): item for item in existing_items if item.get("keyframe_path")}
    print(f"Existing filtered keyframes: {len(existing_items)}")

    now_iso = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # 1. Merge noise frames
    noise_path = Path("data/filtered_noise_frames.json")
    added_noise_count = 0
    if noise_path.exists():
        with open(noise_path, "r", encoding="utf-8") as f:
            noise_data = json.load(f)
        noise_frames = noise_data.get("filtered_noise_frames", [])

        for n in noise_frames:
            b = n.get("group", "L25")
            full_vid = n.get("video_id", "")
            sub_vid = full_vid.split("_")[-1] if "_" in full_vid else full_vid
            fid = int(n.get("frame_id"))
            fname = n.get("filename", f"{fid}.webp")
            k = f"{b}/{sub_vid}/{fname}"
            reason = f"noise_{n.get('issue_type', 'LOW_VARIANCE_BLANK')}"

            if k not in existing_keys:
                item = {
                    "batch": b,
                    "video_id": sub_vid,
                    "full_video_id": f"{b}_{sub_vid}",
                    "filename": fname,
                    "frame_id": fid,
                    "keyframe_path": k,
                    "reason": reason,
                    "added_at": now_iso,
                }
                existing_keys[k] = item
                existing_items.append(item)
                added_noise_count += 1

    print(f"Merged {added_noise_count} new noise frames from filtered_noise_frames.json")

    # 2. Merge duplicate video pairs
    dup_path = Path("data/duplicate_and_discarded_videos.json")
    added_dup_count = 0
    if dup_path.exists():
        with open(dup_path, "r", encoding="utf-8") as f:
            dup_data = json.load(f)

        meta_path = Path("data/unified/scene_metadata.parquet")
        df = pd.read_parquet(meta_path) if meta_path.exists() else pd.DataFrame()

        for p in dup_data.get("duplicate_video_pairs", []):
            primary_vid = p.get("primary_video", {}).get("video_id", "")
            d_vid = p.get("duplicate_video", {})
            if d_vid.get("action") == "DISCARD_CANDIDATE":
                grp = p.get("group", "")
                full_vid = d_vid.get("video_id", "")
                sub_vid = full_vid.split("_")[-1] if "_" in full_vid else full_vid

                # Tìm các frame trên đĩa hoặc trong metadata
                matches = df[df.video_id == full_vid] if not df.empty and "video_id" in df.columns else pd.DataFrame()

                v_dir_nested = Path("data/image") / grp / grp / sub_vid
                v_dir_single = Path("data/image") / grp / sub_vid
                v_dir = v_dir_nested if v_dir_nested.exists() else v_dir_single

                frame_files = set()
                if v_dir.exists():
                    for f in list(v_dir.glob("*.webp")) + list(v_dir.glob("*.jpg")):
                        try:
                            frame_files.add((int(f.stem), f.name))
                        except ValueError:
                            pass
                for _, row in matches.iterrows():
                    frame_files.add((int(row.frame_id), f"{row.frame_id}.webp"))

                for fid, fname in sorted(frame_files):
                    k = f"{grp}/{sub_vid}/{fname}"
                    if k not in existing_keys:
                        item = {
                            "batch": grp,
                            "video_id": sub_vid,
                            "full_video_id": full_vid,
                            "filename": fname,
                            "frame_id": fid,
                            "keyframe_path": k,
                            "reason": f"duplicate_video_of_{primary_vid}",
                            "added_at": now_iso,
                        }
                        existing_keys[k] = item
                        existing_items.append(item)
                        added_dup_count += 1

    print(f"Merged {added_dup_count} frames from duplicate discarded videos")

    # 3. Build summary by batch and save
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

    print("=== HOÀN TẤT MERGE ===")
    print(f"Total filtered count: {len(existing_items)}")
    print(f"Summary by batch: {dict(sorted(summary_by_batch.items()))}")

if __name__ == "__main__":
    main()
