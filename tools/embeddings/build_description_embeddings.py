import os
import sys
import time
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "unified")
DESCRIPTIONS_PARQUET = os.path.join(DATA_DIR, "scene_descriptions.parquet")
OUTPUT_EMBEDDINGS = os.path.join(DATA_DIR, "description_embeddings.pt")

MODEL_ID = "BAAI/bge-large-en-v1.5"
BATCH_SIZE = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    print("=" * 80)
    print("🚀 BUILDING DESCRIPTION VECTOR EMBEDDINGS (BGE-LARGE-EN-V1.5)")
    print("=" * 80)
    print(f"📁 Input Parquet : {DESCRIPTIONS_PARQUET}")
    print(f"💾 Output Tensor : {OUTPUT_EMBEDDINGS}")
    print(f"⚡ Device        : {DEVICE} (FP16 Batch Processing)")
    
    if not os.path.exists(DESCRIPTIONS_PARQUET):
        raise FileNotFoundError(f"Input file not found: {DESCRIPTIONS_PARQUET}")
        
    df = pd.read_parquet(DESCRIPTIONS_PARQUET)
    descriptions = df["description"].fillna("").tolist()
    num_items = len(descriptions)
    print(f"📊 Total scenes to embed: {num_items:,d} rows")
    
    print(f"📥 Loading model '{MODEL_ID}'...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID).to(DEVICE)
    model.eval()
    if DEVICE == "cuda":
        model = model.half() # Use FP16 for RTX 5060 Ti hardware acceleration
        
    embeddings_list = []
    start_time = time.time()
    
    print("⚡ Starting batch embedding inference...", flush=True)
    for i in tqdm(range(0, num_items, BATCH_SIZE), desc="Embedding Descriptions", unit="batch"):
        batch_texts = descriptions[i : i + BATCH_SIZE]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        ).to(DEVICE)

        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16) if DEVICE == "cuda" else torch.no_grad():
                model_output = model(**encoded)
                sentence_embeddings = model_output[0][:, 0]
                sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
                embeddings_list.append(sentence_embeddings.cpu())

        if (i // BATCH_SIZE) % 10 == 0 or (i + BATCH_SIZE) >= num_items:
            processed = min(i + BATCH_SIZE, num_items)
            elapsed = time.time() - start_time
            speed = processed / elapsed if elapsed > 0 else 0
            print(f"  ⏳ Processed {processed:>7,d} / {num_items:>7,d} scenes ({processed/num_items*100:.1f}%) - {speed:.1f} scenes/sec", flush=True)

    all_embeddings = torch.cat(embeddings_list, dim=0)
    assert all_embeddings.shape[0] == num_items, f"Mismatch: {all_embeddings.shape[0]} != {num_items}"
    
    torch.save(all_embeddings, OUTPUT_EMBEDDINGS)
    np_path = os.path.join(DATA_DIR, "description_embeddings.npy")
    np.save(np_path, all_embeddings.cpu().numpy().astype(np.float32))
    print("\n" + "=" * 80)
    print(f"🎉 SUCCESS! Saved description embeddings shape {list(all_embeddings.shape)} to:")
    print(f"   - PyTorch Tensor: {OUTPUT_EMBEDDINGS}")
    print(f"   - NumPy Array   : {np_path}")
    print(f"⏱️ Total time elapsed: {time.time() - start_time:.1f}s")
    print("=" * 80)

if __name__ == "__main__":
    main()
