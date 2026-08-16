import sys
from qdrant_client import QdrantClient
import numpy as np

def main():
    print("=== AIC2026: KIỂM TRA TRẠNG THÁI COLLECTION QDRANT ===")
    client = QdrantClient(path="./data/qdrant_db")
    
    collections = client.get_collections().collections
    print("Danh sách Collections:", [c.name for c in collections])
    
    for c in collections:
        info = client.get_collection(c.name)
        print(f"\nCollection: {c.name}")
        print(f"  • Số lượng points: {info.points_count:,}")
        print(f"  • Trạng thái     : {info.status}")
    
    # Test thử query 3 kết quả từ image vector
    print("\n🔍 Test thử truy vấn tìm kiếm Top-3 bằng vector ngẫu nhiên:")
    vec = np.random.randn(1024).astype(np.float32)
    vec /= np.linalg.norm(vec)
    
    hits = client.query_points(
        collection_name="aic2026_clean_v3",
        query=vec.tolist(),
        using="image_dense",
        limit=3,
    )
    
    for idx, hit in enumerate(hits.points, 1):
        vid = hit.payload.get("video_id")
        sid = hit.payload.get("scene_id")
        fid = hit.payload.get("frame_id")
        ts = hit.payload.get("timestamp_mid")
        print(f"  #{idx}: Point ID={hit.id} | Score={hit.score:.4f} | Video={vid} | Scene={sid} | Frame={fid} | Time={ts}s")
        
    print("\n🎉 QDRANT ĐÃ SẴN SÀNG HOÀN TOÀN CHO PHỤC VỤ TÌM KIẾM!")

if __name__ == "__main__":
    main()
