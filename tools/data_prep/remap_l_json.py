"""
tools/remap_l_json.py
=====================
Rebuild L.json → scene_id mapping using sequential CLIP alignment.

Problem:
  L.json has 108,779 entries (descriptions for ALL original images, before filtering).
  scene_metadata.parquet has 108,413 scenes (after monochrome filter removed ~366 images).
  data/image_filtered/ no longer exists, so we can't replay the full original list.

Solution:
  Both L.json and current images are sorted by (group L21→L30, video V001→V999, frame_id ASC).
  The current images are a subsequence of the original (366 entries removed).
  We use sequential CLIP alignment to detect gap positions and build exact mapping.

Algorithm:
  1. Scan data/image/ sorted by (group, video, frame_id) → 108,413 images
  2. Encode all images + all L.json descriptions with CLIP
  3. Two-pointer alignment: image[i] should match L.json[i + offset]
     - If CLIP similarity is highest at expected position → match, advance both
     - If similarity is higher at a later position → gap detected (filtered image), skip
  4. Output scene_descriptions.parquet with exact 1:1 mapping

Usage:
    python -m tools.remap_l_json
    python -m tools.remap_l_json --model openai/clip-vit-base-patch32 --batch-size 64
"""

from __future__ import annotations

import os
import sys
import json
import time
import argparse
from pathlib import Path

import pandas as pd
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# 1. Image scanning (matching L.json generation order)
# ---------------------------------------------------------------------------

def scan_images_sorted(image_dir: Path) -> list[dict]:
    """
    Scan all keyframe images sorted by (group L21→L30, video V001→V999, frame_id ASC).
    This matches the exact order used to generate L.json.
    """
    records = []

    for group_dir in sorted(image_dir.iterdir()):
        if not group_dir.is_dir():
            continue
        group_name = group_dir.name  # "L21"

        # Handle nested structure: data/image/L21/L21/V001/
        inner_dir = group_dir / group_name
        if not inner_dir.exists():
            continue

        for video_dir in sorted(inner_dir.iterdir()):
            if not video_dir.is_dir():
                continue
            video_name = video_dir.name  # "V001"
            video_id = f"{group_name}_{video_name}"

            # Collect all webp images, sorted by frame_id
            img_files = sorted(
                video_dir.glob("*.webp"),
                key=lambda p: int(p.stem),
            )

            for img_path in img_files:
                frame_id = int(img_path.stem)
                records.append({
                    "video_id": video_id,
                    "frame_id": frame_id,
                    "image_path": str(img_path),
                })

    return records


# ---------------------------------------------------------------------------
# 2. CLIP batch encoding
# ---------------------------------------------------------------------------

def encode_images_clip(
    model: CLIPModel,
    processor: CLIPProcessor,
    image_paths: list[str],
    device: str,
    batch_size: int = 64,
) -> torch.Tensor:
    """Encode images using CLIP in batches. Returns L2-normalized embeddings."""
    all_embeds = []

    with torch.no_grad():
        for i in tqdm(range(0, len(image_paths), batch_size), desc="🖼️  Encoding images"):
            batch_paths = image_paths[i : i + batch_size]
            batch_imgs = []
            for p in batch_paths:
                try:
                    img = Image.open(p).convert("RGB")
                    batch_imgs.append(img)
                except Exception:
                    batch_imgs.append(Image.new("RGB", (224, 224)))

            inputs = processor(images=batch_imgs, return_tensors="pt").to(device)
            out = model.get_image_features(**inputs)
            # Handle both tensor and BaseModelOutput return types
            if hasattr(out, "pooler_output"):
                embeds = out.pooler_output
            elif hasattr(out, "last_hidden_state"):
                embeds = out.last_hidden_state[:, 0]
            elif isinstance(out, torch.Tensor):
                embeds = out
            else:
                embeds = out[0] if isinstance(out, (tuple, list)) else out
            embeds = embeds / embeds.norm(dim=-1, keepdim=True)
            all_embeds.append(embeds.cpu())

    return torch.cat(all_embeds, dim=0)


def encode_texts_clip(
    model: CLIPModel,
    processor: CLIPProcessor,
    texts: list[str],
    device: str,
    batch_size: int = 256,
) -> torch.Tensor:
    """Encode text descriptions using CLIP in batches. Returns L2-normalized embeddings."""
    all_embeds = []

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="🔤 Encoding texts"):
            batch_texts = texts[i : i + batch_size]
            inputs = processor(
                text=batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77,
            ).to(device)
            out = model.get_text_features(**inputs)
            # Handle both tensor and BaseModelOutput return types
            if hasattr(out, "pooler_output"):
                embeds = out.pooler_output
            elif hasattr(out, "last_hidden_state"):
                embeds = out.last_hidden_state[:, 0]
            elif isinstance(out, torch.Tensor):
                embeds = out
            else:
                embeds = out[0] if isinstance(out, (tuple, list)) else out
            embeds = embeds / embeds.norm(dim=-1, keepdim=True)
            all_embeds.append(embeds.cpu())

    return torch.cat(all_embeds, dim=0)


# ---------------------------------------------------------------------------
# 3. Sequential alignment algorithm
# ---------------------------------------------------------------------------

def sequential_align(
    image_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    max_gap: int = 100,
) -> list[tuple[int, int, float]]:
    """
    Globally optimal alignment using Dynamic Programming.
    
    Finds a strictly monotonic mapping from N images to M texts (N <= M)
    that maximizes the sum of CLIP similarities, limiting the search space
    to at most `max_gap` cumulative skips at any point.
    """
    n_images = len(image_embeds)
    n_texts = len(text_embeds)
    total_gaps = n_texts - n_images

    # DP state: dp[i][g] = max similarity for mapping first `i` images with exactly `g` gaps.
    # We restrict g to [0, total_gaps].
    # dp table shape: (n_images, total_gaps + 1)
    dp = np.full((n_images, total_gaps + 1), -np.inf, dtype=np.float32)
    parents = np.zeros((n_images, total_gaps + 1), dtype=np.int32)

    # Convert embeddings to CPU numpy for memory efficiency and fast indexed lookups
    img_np = image_embeds.numpy()
    txt_np = text_embeds.numpy()

    print(f"\n🔗 Dynamic Programming Global Alignment")
    print(f"   Images (N)     : {n_images:,d}")
    print(f"   Texts  (M)     : {n_texts:,d}")
    print(f"   Total gaps     : {total_gaps}")
    print(f"   DP Table size  : {n_images} x {total_gaps + 1}")

    # Base case: first image (i = 0)
    # We can match image 0 to any text index from T_0 to T_{total_gaps} (representing g gaps before image 0)
    for g in range(total_gaps + 1):
        dp[0, g] = np.dot(img_np[0], txt_np[g])
        parents[0, g] = -1

    # Fill DP Table
    for i in tqdm(range(1, n_images), desc="🧠 Building DP Table"):
        # Parallel computation optimization using matrix multiplication for similarities
        # At step i, image i matches text at index (i + g) for each g in [0, total_gaps]
        # Text indices to compare: [i, i + 1, ..., i + total_gaps]
        txt_candidates = txt_np[i : i + total_gaps + 1]  # Shape: (total_gaps + 1, D)
        sims = np.dot(txt_candidates, img_np[i])  # Shape: (total_gaps + 1,)

        for g in range(total_gaps + 1):
            # We must transition from a previous state (i-1, prev_g) where prev_g <= g
            # To limit large jumps at a single step, we can restrict transition step size if needed,
            # but standard DP allows any prev_g <= g.
            # search_start = max(0, g - max_gap) # option to speed up
            prev_g_candidates = dp[i - 1, : g + 1]
            if len(prev_g_candidates) == 0 or np.all(prev_g_candidates == -np.inf):
                continue
                
            best_prev_g = np.argmax(prev_g_candidates)
            if prev_g_candidates[best_prev_g] != -np.inf:
                dp[i, g] = prev_g_candidates[best_prev_g] + sims[g]
                parents[i, g] = best_prev_g

    # Backtracking
    print("\n🔍 Backtracking optimal path...")
    alignments = []
    
    # Start backtrack from the last image at the exact expected total_gaps offset
    current_g = total_gaps
    
    # If the last state has no path, find the best reachable final state
    if dp[n_images - 1, current_g] == -np.inf:
        print("   ⚠️  Warning: Exact gap target state unreachable. Picking best final state.")
        current_g = int(np.argmax(dp[n_images - 1]))

    for i in range(n_images - 1, -1, -1):
        text_idx = i + current_g
        # Compute exact cosine similarity for recording
        sim = float(np.dot(img_np[i], txt_np[text_idx]))
        alignments.append((i, text_idx, sim))
        current_g = int(parents[i, current_g])

    # Reverse alignments to be in ascending order
    alignments.reverse()

    final_offset = alignments[-1][1] - (n_images - 1)
    print(f"   ✅ Final cumulative offset : {final_offset}")
    print(f"   ✅ Expected gaps           : {total_gaps}")
    if final_offset != total_gaps:
        print(f"   ⚠️  Offset mismatch by {abs(final_offset - total_gaps)}!")
    else:
        print(f"   🎯 Perfect match!")

    return alignments


# ---------------------------------------------------------------------------
# 4. Main pipeline
# ---------------------------------------------------------------------------

def main():
    # --- Paths ---
    image_dir = PROJECT_ROOT / "data" / "image"
    l_json_path = PROJECT_ROOT / "data" / "L.json"
    if not l_json_path.exists():
        l_json_path = PROJECT_ROOT / "L.json"
    meta_path = PROJECT_ROOT / "data" / "unified" / "scene_metadata.parquet"
    output_parquet = PROJECT_ROOT / "data" / "unified" / "scene_descriptions.parquet"
    output_json = PROJECT_ROOT / "L_by_scene_id.json"
    filtered_json_path = PROJECT_ROOT / "data" / "filtered_images_366.json"

    print("=" * 80)
    print("🚀 DETERMINISTIC L.JSON REMAP ENGINE (JSON-BASED)")
    print(f"   Using filter log: {filtered_json_path}")
    print("=" * 80)

    # 1. Load L.json
    print("\n📄 Step 1: Loading L.json...")
    with open(l_json_path, "r", encoding="utf-8") as f:
        l_json = json.load(f)
    l_keys = sorted(l_json.keys(), key=lambda k: int(k))
    l_descriptions = [l_json[k] for k in l_keys]
    print(f"   Total L.json entries: {len(l_descriptions):,d}")

    # 2. Scan current images (sorted)
    print("\n📷 Step 2: Scanning current images from data/image...")
    current_images = scan_images_sorted(image_dir)
    print(f"   Total current images: {len(current_images):,d}")

    # 3. Load filtered images from JSON
    print("\n🗑️  Step 3: Loading filtered images from JSON...")
    if not filtered_json_path.exists():
        raise FileNotFoundError(f"Missing filtered images log at: {filtered_json_path}")
    
    with open(filtered_json_path, "r", encoding="utf-8") as f:
        filtered_data = json.load(f)
    
    filtered_records = []
    for item in filtered_data["filtered_images"]:
        filtered_records.append({
            "video_id": str(item["full_video_id"]),
            "frame_id": int(item["frame_id"]),
            "is_filtered": True
        })
    print(f"   Loaded {len(filtered_records):,d} filtered image records.")

    # Mark current images as not filtered
    for img in current_images:
        img["is_filtered"] = False

    # 4. Merge and sort all images to reconstruct the original dataset order
    print("\n🔀 Step 4: Reconstructing original sequence...")
    # Helper to parse video group and ID for sorting
    # e.g., "L21_V001" -> group=L21, video=V001
    def get_sort_key(item):
        v_id = item["video_id"]
        parts = v_id.split('_')
        group_num = int(parts[0][1:]) if len(parts) > 0 and parts[0][1:].isdigit() else 0
        video_num = int(parts[1][1:]) if len(parts) > 1 and parts[1][1:].isdigit() else 0
        return (group_num, video_num, item["frame_id"])

    all_images = current_images + filtered_records
    all_images.sort(key=get_sort_key)
    print(f"   Reconstructed sequence size: {len(all_images):,d}")

    # Check match with L.json size
    assert len(all_images) == len(l_descriptions), (
        f"Reconstructed sequence size ({len(all_images)}) does not match L.json size ({len(l_descriptions)})."
    )

    # 5. Map 1-to-1 with L.json indices
    print("\n🎯 Step 5: Mapping descriptions to non-filtered scenes...")
    meta_df = pd.read_parquet(meta_path)
    print(f"   Total scenes in metadata: {len(meta_df):,d}")

    # Build (video_id, frame_id) -> scene_id lookup
    meta_lookup: dict[tuple[str, int], str] = {}
    for _, row in meta_df.iterrows():
        key = (str(row["video_id"]), int(row["frame_id"]))
        meta_lookup[key] = str(row["scene_id"])

    records = []
    mapped_count = 0
    
    for idx, img in enumerate(all_images):
        if img["is_filtered"]:
            continue
            
        vid = img["video_id"]
        fid = img["frame_id"]
        scene_id = meta_lookup.get((vid, fid))
        
        if scene_id is None:
            # Fallback check
            print(f"   ⚠️ Warning: Scene ID not found in metadata for {vid} frame {fid}")
            continue

        records.append({
            "scene_id": scene_id,
            "video_id": vid,
            "description": l_descriptions[idx],
            "l_json_key": str(idx),
            "clip_confidence": 1.0  # Deterministic mapping has maximum confidence
        })
        mapped_count += 1

    print(f"   Successfully mapped {mapped_count:,d} scenes.")

    # 6. Save results
    df_out = pd.DataFrame(records)
    df_out.to_parquet(str(output_parquet), index=False)
    print(f"\n💾 Saved Parquet: {output_parquet}")

    l_by_scene = {r["scene_id"]: r["description"] for r in records}
    with open(str(output_json), "w", encoding="utf-8") as f:
        json.dump(l_by_scene, f, ensure_ascii=False, indent=2)
    print(f"💾 Saved JSON: {output_json}")

    print("\n" + "=" * 80)
    print("🎉 DETERMINISTIC REMAP COMPLETE!")
    print(f"   Mapped Scenes  : {len(df_out):,d}")
    print(f"   Unique Keys    : {df_out['l_json_key'].nunique():,d}")
    print(f"   Duplicate Keys : {df_out['l_json_key'].duplicated().sum()}")
    print("=" * 80)


if __name__ == "__main__":
    main()
