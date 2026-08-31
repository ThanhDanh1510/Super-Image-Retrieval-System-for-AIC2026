"""
tools/vlm_captioning/config.py
==============================
Configuration settings for high-density visual captioning on RTX 5060 Ti.
"""

from pathlib import Path

# Paths
BASE_DIR = Path("d:/AIC2026")
DATA_DIR = BASE_DIR / "data"
UNIFIED_DIR = DATA_DIR / "unified"
METADATA_PARQUET = UNIFIED_DIR / "scene_metadata.parquet"
OUTPUT_SHARDS_DIR = UNIFIED_DIR / "vlm_shards"
FINAL_DESCRIPTIONS_PARQUET = UNIFIED_DIR / "scene_descriptions.parquet"
FINAL_EMBEDDINGS_NPY = UNIFIED_DIR / "description_embeddings.npy"

# Model configuration (Qwen2.5-VL-7B-Instruct with 4-bit NF4 quantization)
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

# Image processing limits (384x384 yields ~267 vision tokens per image, optimal for GPU throughput)
MAX_IMAGE_PIXELS = 384 * 384
MIN_IMAGE_PIXELS = 256 * 256

# Generation parameters (Concise dense bullet points, zero fluff)
MAX_NEW_TOKENS = 85
TEMPERATURE = 0.01
TOP_P = 0.9

# Batch & Shard settings
DEFAULT_BATCH_SIZE = 6
SHARD_SAVE_INTERVAL_VIDEOS = 5
SIMILARITY_DEDUP_THRESHOLD = 0.94

# Retrieval-Optimized Generalized Event-Focused System Prompt (No TV logos / OCR)
RETRIEVAL_SYSTEM_PROMPT = (
    "Extract dense video search metadata. "
    "DO NOT mention TV logos, watermarks, subtitles, channel bugs, or corner icons. "
    "Focus purely on the physical event and real-world scene:\n"
    "- ACTION: physical actions, body postures, gestures, movements\n"
    "- OBJECTS: salient items, clothing, colors, counts, quantities\n"
    "- SCENE: real-world setting, environment, camera perspective\n"
    "- TAGS (VI & EN): 8-10 dense search keywords separated by commas\n"
    "Be direct and factual. No filler."
)
