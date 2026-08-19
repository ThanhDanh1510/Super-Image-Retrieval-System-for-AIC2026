"""
tools/data_prep/fast_sync_l25_qdrant.py
=======================================
Đồng bộ tốc độ cao payload và ocr_dense vector cho 8,268 điểm L25 vào Qdrant Server.
"""

import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, PointVectors
from shared.schemas.named_vector_spec import make_point_id

def main():
    print("🚀 Bắt đầu cập nhật Qdrant collection 'aic2026_clean_v3' cho L25...")
    client = QdrantClient(url="http://localhost:6333", timeout=60)
    
    meta_df = pd.read_parquet("data/unified/scene_metadata.parquet")
    ocr_df = pd.read_parquet("data/unified/ocr_text.parquet")
    desc_df = pd.read_parquet("data/unified/scene_descriptions.parquet")
    ocr_vecs = np.load("data/unified/ocr_vectors.npy")
    scene_ids = [line.strip() for line in open("data/unified/scene_ids.txt") if line.strip()]
    sid_to_idx = {sid: i for i, sid in enumerate(scene_ids)}

    l25_meta = meta_df[meta_df["video_id"].str.startswith("L25_")].copy()
    total_l25 = len(l25_meta)
    print(f"📊 Tổng số points L25: {total_l25:,d}")

    ocr_map = dict(zip(ocr_df["scene_id"], ocr_df["ocr_text"]))
    desc_map = dict(zip(desc_df["scene_id"], desc_df["description"]))

    batch_size = 200
    t0 = time.time()

    for b_start in range(0, total_l25, batch_size):
        batch = l25_meta.iloc[b_start : b_start + batch_size]
        pids = []
        updates = []
        for _, row in batch.iterrows():
            sid = row["scene_id"]
            vid = row["video_id"]
            pid = make_point_id(vid, sid)
            pids.append((pid, sid, vid))

        # Retrieve current points in batch
        retrieved = client.retrieve(
            collection_name="aic2026_clean_v3",
            ids=[p[0] for p in pids],
            with_payload=True,
            with_vectors=True,
        )
        
        ret_map = {p.id: p for p in retrieved}
        points_to_upsert = []

        for pid, sid, vid in pids:
            if pid in ret_map:
                p_curr = ret_map[pid]
                pl = dict(p_curr.payload) if p_curr.payload else {}
                pl["ocr_text"] = ocr_map.get(sid, "")
                pl["description"] = desc_map.get(sid, "")

                vecs = dict(p_curr.vector) if isinstance(p_curr.vector, dict) else {}
                if sid in sid_to_idx:
                    vecs["ocr_dense"] = ocr_vecs[sid_to_idx[sid]].tolist()

                points_to_upsert.append(
                    PointStruct(
                        id=pid,
                        vector=vecs,
                        payload=pl,
                    )
                )

        if points_to_upsert:
            client.upsert(
                collection_name="aic2026_clean_v3",
                points=points_to_upsert,
                wait=False,
            )

        processed = min(b_start + batch_size, total_l25)
        elapsed = time.time() - t0
        speed = processed / elapsed if elapsed > 0 else 0
        print(f"  ⏳ Đã cập nhật {processed:>5,d} / {total_l25:,d} points ({processed/total_l25*100:.1f}%) - {speed:.1f} pts/sec", flush=True)

    print(f"\n🎉 CẬP NHẬT QDRANT HOÀN TẤT TRONG {time.time() - t0:.2f} GIÂY!")

if __name__ == "__main__":
    main()
