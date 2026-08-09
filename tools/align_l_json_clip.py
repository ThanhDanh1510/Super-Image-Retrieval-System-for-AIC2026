import os
import sys
import json
import time
import argparse
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from transformers import CLIPModel, CLIPProcessor

def align_l_json_with_clip(model_name: str = "openai/clip-vit-base-patch32", batch_size: int = 128, window_size: int = 500):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 80)
    print("🚀 KHỞI CHẠY CLIP SEMANTIC ALIGNMENT ENGINE")
    print(f"⚡ Device        : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"📦 CLIP Model   : {model_name}")
    print(f"🪟 Search Window: ±{window_size} keys")
    print("=" * 80)

    json_path = os.path.join(PROJECT_ROOT, "L.json")
    if not os.path.exists(json_path):
        json_path = os.path.join(PROJECT_ROOT, "L_backup.json")

    meta_path = os.path.join(PROJECT_ROOT, "data", "unified", "scene_metadata.parquet")
    output_json_path = os.path.join(PROJECT_ROOT, "L_by_scene_id.json")
    output_parquet_path = os.path.join(PROJECT_ROOT, "data", "unified", "scene_descriptions.parquet")

    with open(json_path, 'r', encoding='utf-8') as f:
        l_json = json.load(f)

    df_meta = pd.read_parquet(meta_path)
    print(f"📄 Nạp {len(l_json):,d} mô tả từ L.json")
    print(f"📹 Nạp {len(df_meta):,d} scenes từ scene_metadata.parquet")

    # Load CLIP model & processor
    print("\n📦 Đang nạp mô hình CLIP...")
    model = CLIPModel.from_pretrained(model_name).to(device).eval()
    processor = CLIPProcessor.from_pretrained(model_name)
    print("✅ CLIP Model ready!")

    # 1. Encode all L.json text descriptions into CLIP text embeddings
    print("\n🔤 Đang mã hóa 108,779 câu mô tả trong L.json thành Text Embeddings...")
    l_keys = sorted(list(l_json.keys()), key=lambda k: int(k))
    l_descriptions = [l_json[k] for k in l_keys]

    text_embeds_list = []
    text_batch_size = 256
    with torch.no_grad():
        for i in tqdm(range(0, len(l_descriptions), text_batch_size), desc="Mã hóa L.json Text"):
            batch_texts = l_descriptions[i:i + text_batch_size]
            inputs = processor(text=batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=77).to(device)
            out = model.get_text_features(**inputs)
            embeds = out.pooler_output if hasattr(out, 'pooler_output') else (out.text_embeds if hasattr(out, 'text_embeds') else out)
            embeds = embeds / embeds.norm(dim=-1, keepdim=True)
            text_embeds_list.append(embeds.cpu())

    all_text_embeds = torch.cat(text_embeds_list, dim=0)
    print(f"✅ Đã tạo xong Text Embeddings Tensor: {all_text_embeds.shape}")

    # 2. Iterate image keyframes in GPU batches, encode images, and find max similarity match in window
    print("\n📷 Đang nạp & mã hóa Keyframe Images và tìm kiếm khớp nối nữ nghĩa...")
    aligned_records = []
    total_start = time.time()
    current_search_idx = 0

    def resolve_img_path(row):
        v_id = str(row['video_id'])
        f_id = str(row['frame_id'])
        parts = v_id.split('_')
        prefix = parts[0]
        v_folder = parts[1] if len(parts) > 1 else parts[0]

        candidate = os.path.join(PROJECT_ROOT, 'data', 'image', prefix, prefix, v_folder, f'{f_id}.webp')
        if os.path.exists(candidate):
            return candidate

        for ext in ['.jpg', '.png', '.jpeg', '.webp']:
            c = os.path.join(PROJECT_ROOT, 'data', 'image', prefix, prefix, v_folder, f'{f_id}{ext}')
            if os.path.exists(c):
                return c
            c2 = os.path.join(PROJECT_ROOT, 'data', 'image', prefix, v_folder, f'{f_id}{ext}')
            if os.path.exists(c2):
                return c2
        return None

    img_batch_size = 64
    rows = [r for _, r in df_meta.iterrows()]

    with torch.no_grad():
        for b_i in tqdm(range(0, len(rows), img_batch_size), desc="CLIP Semantic Alignment"):
            batch_rows = rows[b_i:b_i + img_batch_size]

            batch_pil_imgs = []
            valid_indices = []

            for r_idx, r in enumerate(batch_rows):
                img_path = resolve_img_path(r)
                if img_path and os.path.exists(img_path):
                    try:
                        pil_img = Image.open(img_path).convert("RGB")
                        batch_pil_imgs.append(pil_img)
                        valid_indices.append(r_idx)
                    except Exception:
                        pass

            if batch_pil_imgs:
                inputs = processor(images=batch_pil_imgs, return_tensors="pt").to(device)
                out = model.get_image_features(**inputs)
                img_embeds = out.pooler_output if hasattr(out, 'pooler_output') else (out.image_embeds if hasattr(out, 'image_embeds') else out)
                img_embeds = img_embeds / img_embeds.norm(dim=-1, keepdim=True)

            valid_pos = 0
            for r_idx, r in enumerate(batch_rows):
                s_id = r['scene_id']
                v_id = r['video_id']

                if r_idx in valid_indices and batch_pil_imgs:
                    img_embed = img_embeds[valid_pos:valid_pos + 1]
                    valid_pos += 1

                    win_start = max(0, current_search_idx - 10)
                    win_end = min(len(l_descriptions), current_search_idx + window_size)

                    candidate_text_embeds = all_text_embeds[win_start:win_end].to(device)
                    sims = (img_embed @ candidate_text_embeds.T).squeeze(0)

                    best_relative_idx = torch.argmax(sims).item()
                    best_idx = win_start + best_relative_idx
                    sim_score = sims[best_relative_idx].item()
                    best_desc = l_descriptions[best_idx]
                    current_search_idx = best_idx
                else:
                    best_idx = min(current_search_idx, len(l_descriptions) - 1)
                    best_desc = l_descriptions[best_idx]
                    sim_score = 0.0

                aligned_records.append({
                    "scene_id": s_id,
                    "video_id": v_id,
                    "description": best_desc,
                    "l_json_key": str(best_idx),
                    "clip_confidence": round(float(sim_score), 4)
                })

    # Save outputs
    l_by_scene_id = {rec["scene_id"]: rec["description"] for rec in aligned_records}
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(l_by_scene_id, f, ensure_ascii=False, indent=2)

    df_out = pd.DataFrame(aligned_records)
    df_out.to_parquet(output_parquet_path, index=False)

    elapsed = time.time() - total_start
    avg_conf = df_out["clip_confidence"].mean()

    print("\n" + "=" * 80)
    print("🎉 HOÀN THÀNH GHÉP NỐI NGHĨA 100% CLIP SEMANTIC ALIGNMENT!")
    print(f"📊 Tổng số scenes        : {len(df_out):,d} scenes")
    print(f"⏱️ Tổng thời gian         : {elapsed:.1f}s ({elapsed/len(df_out):.3f}s/scene)")
    print(f"⭐ Độ tương đồng trung bình: {avg_conf:.4f}")
    print(f"💾 File Mapping JSON     : {output_json_path}")
    print(f"💾 File Unified Parquet  : {output_parquet_path}")
    print("=" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="openai/clip-vit-base-patch32", help="Mô hình CLIP")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--window", type=int, default=500, help="Kích thước cửa sổ trượt")
    args = parser.parse_args()

    align_l_json_with_clip(model_name=args.model, batch_size=args.batch_size, window_size=args.window)
