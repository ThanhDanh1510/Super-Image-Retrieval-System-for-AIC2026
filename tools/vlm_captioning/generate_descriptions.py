"""
tools/vlm_captioning/generate_descriptions.py
=============================================
High-Speed GPU + CPU VLM Video Keyframe Description Engine.
Optimized for NVIDIA RTX 5060 Ti (16GB VRAM) & Multi-Core CPU.
Features:
- Fast Perceptual Hash (dHash) Temporal Deduplication (skips 40-50% redundant frames)
- Multi-Threaded CPU Image I/O Prefetching
- Left-Padding GPU Batching (4-bit NF4 pinned strictly on cuda:0)
- Event-Focused Dense Output (No TV logos / OCR)
- Resumable Parquet Sharding (Auto-recovers progress if interrupted)
"""

import argparse
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from PIL import Image

# Ensure project root in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Enable TF32 & high-performance CUDA kernels for Blackwell / Ada / Ampere
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
)
from qwen_vl_utils import process_vision_info
from shared.utils.path_resolver import resolve_keyframe_path
from tools.vlm_captioning.config import (
    BASE_DIR,
    DEFAULT_BATCH_SIZE,
    DEFAULT_MODEL_ID,
    MAX_IMAGE_PIXELS,
    MAX_NEW_TOKENS,
    METADATA_PARQUET,
    MIN_IMAGE_PIXELS,
    OUTPUT_SHARDS_DIR,
    RETRIEVAL_SYSTEM_PROMPT,
    SHARD_SAVE_INTERVAL_VIDEOS,
    TEMPERATURE,
    TOP_P,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("VLM-Captioner")


import numpy as np


def compute_dhash(img: Image.Image) -> int:
    """Tính 64-bit difference hash siêu nhanh (0.03ms) bằng NumPy."""
    resized = img.resize((9, 8), Image.Resampling.BILINEAR).convert("L")
    arr = np.asarray(resized, dtype=np.int16)
    diff = arr[:, :-1] > arr[:, 1:]
    val = 0
    for bit in diff.flatten():
        val = (val << 1) | int(bit)
    return val


def hamming_distance(h1: int, h2: int) -> int:
    """Đo khoảng cách bit giữa 2 dHash."""
    return bin(h1 ^ h2).count("1")


def get_completed_scene_ids(shards_dir: Path) -> set[str]:
    """Quét tất cả các file shard đã có để lấy danh sách scene_id đã xử lý."""
    completed = set()
    if not shards_dir.exists():
        shards_dir.mkdir(parents=True, exist_ok=True)
        return completed

    shard_files = list(shards_dir.glob("*.parquet"))
    for sf in shard_files:
        try:
            df = pd.read_parquet(sf, columns=["scene_id"])
            completed.update(df["scene_id"].astype(str).tolist())
        except Exception as e:
            logger.warning(f"Không thể đọc shard {sf.name}: {e}")
    return completed


def load_vlm_model(
    model_id: str,
    quantization: str = "4bit",
    device: str = "cuda:0",
):
    """Khởi tạo processor và model Qwen2.5-VL tối ưu hóa trên GPU."""
    logger.info(f"🚀 Đang tải mô hình {model_id} (Quantization: {quantization}, Device: {device})...")
    t0 = time.time()

    processor = AutoProcessor.from_pretrained(
        model_id,
        min_pixels=MIN_IMAGE_PIXELS,
        max_pixels=MAX_IMAGE_PIXELS,
        padding_side="left",
    )
    if hasattr(processor, "tokenizer") and processor.tokenizer:
        processor.tokenizer.padding_side = "left"

    if quantization == "4bit":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map={"": device},
            attn_implementation="sdpa",
        )
    elif quantization == "8bit":
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map={"": device},
            attn_implementation="sdpa",
        )
    else:  # Full / BF16
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map={"": device},
            attn_implementation="sdpa",
        )

    model.eval()
    logger.info(f"✅ Tải mô hình thành công trong {time.time() - t0:.2f}s!")
    if torch.cuda.is_available():
        vram_gb = torch.cuda.memory_allocated() / (1024**3)
        logger.info(f"📊 GPU VRAM: {vram_gb:.2f} GB Allocated / 16 GB")

    return processor, model


def _read_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def generate_batch_descriptions(
    model: Qwen2_5_VLForConditionalGeneration,
    processor: AutoProcessor,
    batch_items: list[dict],
    system_prompt: str,
    device: str = "cuda:0",
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> list[str]:
    """Sinh mô tả song song cho một batch ảnh với multi-threaded preloading."""
    paths = [item["resolved_path"] for item in batch_items]
    with ThreadPoolExecutor(max_workers=min(16, len(paths))) as pool:
        pil_images = list(pool.map(_read_image, paths))

    messages_list = []
    for img in pil_images:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": system_prompt},
                ],
            }
        ]
        messages_list.append(messages)

    texts = [
        processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in messages_list
    ]

    image_inputs, video_inputs = process_vision_info(messages_list)
    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            do_sample=False,
            use_cache=True,
        )

    # Trim input prompt tokens
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    descriptions = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return descriptions


def main():
    parser = argparse.ArgumentParser(description="Generate accelerated VLM descriptions for AIC2026")
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID, help="HuggingFace model ID")
    parser.add_argument("--quantization", type=str, default="4bit", choices=["bf16", "4bit", "8bit"], help="Model precision")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Inference batch size")
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS, help="Max output tokens per image")
    parser.add_argument("--limit-videos", type=int, default=None, help="Chỉ xử lý N videos đầu tiên (dùng để test)")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device target")
    args = parser.parse_args()

    base_dir = BASE_DIR
    metadata_path = METADATA_PARQUET
    shards_dir = OUTPUT_SHARDS_DIR
    shards_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"📖 Đang đọc {metadata_path}...")
    df_meta = pd.read_parquet(metadata_path)
    total_scenes_dataset = len(df_meta)
    video_list = df_meta["video_id"].unique().tolist()
    logger.info(f"Tổng số scenes: {total_scenes_dataset:,} scenes ({len(video_list)} videos).")

    if args.limit_videos:
        video_list = video_list[:args.limit_videos]
        df_meta = df_meta[df_meta["video_id"].isin(video_list)].copy()
        logger.info(f"Đã giới hạn {len(video_list)} videos đầu tiên ({len(df_meta):,} scenes).")

    # Kiểm tra shard đã xử lý
    completed_scene_ids = get_completed_scene_ids(shards_dir)
    logger.info(f"🔍 Đã tìm thấy {len(completed_scene_ids):,} scenes đã xử lý trong shards.")

    df_pending = df_meta[~df_meta["scene_id"].astype(str).isin(completed_scene_ids)].copy()
    logger.info(f"⏳ Cần xử lý: {len(df_pending):,} / {len(df_meta):,} scenes còn lại.")

    if df_pending.empty:
        logger.info("🎉 Tất cả scenes đã được xử lý xong!")
        return

    # Tải mô hình
    processor, model = load_vlm_model(
        model_id=args.model_id,
        quantization=args.quantization,
        device=args.device,
    )

    grouped = df_pending.groupby("video_id", sort=False)
    total_videos = len(grouped)
    logger.info(f"🎬 Bắt đầu xử lý {total_videos} videos với Batch Size = {args.batch_size} (Max Tokens = {args.max_new_tokens})...")

    buffer_results: list[dict] = []
    videos_processed = 0
    total_scenes_done = 0
    total_scenes_inferred = 0
    t_start_all = time.time()

    for video_id, group_df in grouped:
        v_start = time.time()
        video_items = []

        for _, row in group_df.iterrows():
            scene_id = str(row["scene_id"])
            frame_id = int(row["frame_id"])
            kf_path_rel = str(row.get("keyframe_path", ""))
            resolved_p = resolve_keyframe_path(str(video_id), frame_id, kf_path_rel, data_dir=base_dir / "data")

            if resolved_p is None or not resolved_p.exists():
                logger.warning(f"⚠️ Không tìm thấy ảnh cho scene {scene_id} ({kf_path_rel}), bỏ qua.")
                continue

            video_items.append({
                "scene_id": scene_id,
                "video_id": str(video_id),
                "frame_id": frame_id,
                "timestamp_mid": float(row.get("timestamp_mid", 0.0)),
                "resolved_path": resolved_p,
            })

        if not video_items:
            continue

        # ======================================================================
        # 1. Fast Perceptual Temporal Deduplication (Lọc frame tĩnh trùng lặp)
        # ======================================================================
        unique_items = []
        dedup_map = {}  # index_in_video_items -> index_in_unique_items
        prev_hash = None
        prev_master_idx = None

        for idx, item in enumerate(video_items):
            try:
                img = Image.open(item["resolved_path"]).convert("RGB")
                h = compute_dhash(img)
                # Hamming distance <= 5 tương đương độ tương đồng > 92-95%
                if prev_hash is not None and hamming_distance(h, prev_hash) <= 5:
                    dedup_map[idx] = prev_master_idx
                else:
                    master_idx = len(unique_items)
                    unique_items.append(item)
                    dedup_map[idx] = master_idx
                    prev_hash = h
                    prev_master_idx = master_idx
            except Exception:
                master_idx = len(unique_items)
                unique_items.append(item)
                dedup_map[idx] = master_idx

        num_skipped = len(video_items) - len(unique_items)
        skip_pct = (num_skipped / len(video_items)) * 100 if video_items else 0.0

        # ======================================================================
        # 2. GPU Inference Trên Các Unique Frames
        # ======================================================================
        unique_descriptions = []
        num_batches = (len(unique_items) + args.batch_size - 1) // args.batch_size

        for batch_idx, i in enumerate(range(0, len(unique_items), args.batch_size)):
            batch = unique_items[i:i + args.batch_size]
            _tb0 = time.time()
            descs = generate_batch_descriptions(
                model,
                processor,
                batch,
                RETRIEVAL_SYSTEM_PROMPT,
                device=args.device,
                max_new_tokens=args.max_new_tokens,
            )
            _tb1 = time.time()
            b_dt = max(0.01, _tb1 - _tb0)
            unique_descriptions.extend(descs)

            done_unique = min(i + len(batch), len(unique_items))
            if (batch_idx + 1) % 2 == 0 or done_unique == len(unique_items):
                logger.info(
                    f"  🎬 [{video_id}] Batch {batch_idx+1}/{num_batches} "
                    f"({done_unique}/{len(unique_items)} unique frames) | {b_dt:.2f}s ({len(batch)/b_dt:.1f} fps)"
                )

        # ======================================================================
        # 3. Gán Mô Tả Lại Cho Toàn Bộ Scenes Trong Video
        # ======================================================================
        for idx, item in enumerate(video_items):
            m_idx = dedup_map.get(idx, 0)
            desc_text = unique_descriptions[m_idx] if m_idx < len(unique_descriptions) else ""
            buffer_results.append({
                "scene_id": item["scene_id"],
                "video_id": item["video_id"],
                "frame_id": item["frame_id"],
                "timestamp_mid": item["timestamp_mid"],
                "description": desc_text.strip(),
            })

        videos_processed += 1
        total_scenes_done += len(video_items)
        total_scenes_inferred += len(unique_items)
        v_time = time.time() - v_start
        effective_fps = len(video_items) / max(0.01, v_time)
        overall_time = time.time() - t_start_all
        overall_effective_fps = total_scenes_done / max(0.01, overall_time)
        remaining_scenes = len(df_pending) - total_scenes_done
        eta_seconds = remaining_scenes / max(0.1, overall_effective_fps)
        eta_hours = eta_seconds / 3600.0

        logger.info(
            f"✅ [{videos_processed}/{total_videos}] Video {video_id} ({len(video_items)} frames, "
            f"lọc {num_skipped} frames trùng {skip_pct:.1f}%) | "
            f"Tốc độ hiệu dụng: {effective_fps:.1f} fps | Đã xong: {total_scenes_done:,} frames | "
            f"ETA: {eta_hours:.2f} giờ"
        )

        # Lưu Shard sau mỗi SHARD_SAVE_INTERVAL_VIDEOS hoặc ở video cuối cùng
        if videos_processed % SHARD_SAVE_INTERVAL_VIDEOS == 0 or videos_processed == total_videos:
            if buffer_results:
                shard_fn = shards_dir / f"shard_{int(time.time())}_{videos_processed}.parquet"
                df_shard = pd.DataFrame(buffer_results)
                df_shard.to_parquet(shard_fn, index=False)
                logger.info(f"💾 ĐÃ LƯU SHARD AN TOÀN: {shard_fn.name} ({len(buffer_results):,} scenes).")
                buffer_results.clear()

    logger.info(
        f"🏁 HOÀN TẤT TOÀN BỘ QUÁ TRÌNH TRONG {(time.time() - t_start_all)/3600:.2f} GIỜ! "
        f"(Tổng {total_scenes_done:,} scenes, GPU suy luận {total_scenes_inferred:,} unique frames)."
    )


if __name__ == "__main__":
    main()
