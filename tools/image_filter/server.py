"""
tools/image_filter/server.py
=============================
FastAPI backend phục vụ công cụ lọc ảnh keyframe & tìm kiếm ảnh trùng lặp AIC2026.
Chạy độc lập trên cổng 8501:
    python -m tools.image_filter.server
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ImageFilterServer")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
IMAGE_DIR = DATA_DIR / "image"
UNIFIED_DIR = DATA_DIR / "unified"
FILTER_JSON_PATH = DATA_DIR / "filtered_keyframes.json"
OLD_FILTER_JSON_PATH = DATA_DIR / "filtered_images_366.json"
WEB_DIR = Path(__file__).resolve().parent

app = FastAPI(title="AIC2026 Keyframe Image Filter Tool", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# In-Memory State & Cache
# =============================================================================

class AppState:
    image_vectors: np.ndarray | None = None
    scene_ids: list[str] = []
    scene_id_to_idx: dict[str, int] = {}
    metadata_df: pd.DataFrame | None = None
    # Lookup: (batch, video_id, frame_id) -> scene_id
    video_frame_to_scene: dict[tuple[str, str, int], dict[str, Any]] = {}
    # Filter state: dict keyed by unique key "L21/V001/0.webp"
    filtered_items: dict[str, dict[str, Any]] = {}


state = AppState()


def load_or_migrate_filter_json() -> None:
    """Tải file filtered_keyframes.json hoặc migrate từ filtered_images_366.json."""
    if FILTER_JSON_PATH.exists():
        try:
            with open(FILTER_JSON_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            items = data.get("filtered_images", [])
            for item in items:
                key = item.get("keyframe_path") or f"{item.get('batch')}/{item.get('video_id')}/{item.get('filename')}"
                state.filtered_items[key] = item
            logger.info(f"Đã load {len(state.filtered_items)} ảnh đã lọc từ {FILTER_JSON_PATH.name}")
            return
        except Exception as e:
            logger.error(f"Lỗi đọc {FILTER_JSON_PATH}: {e}")

    # Fallback / migrate từ file 366 cũ nếu có
    if OLD_FILTER_JSON_PATH.exists():
        try:
            with open(OLD_FILTER_JSON_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            items = data.get("filtered_images", [])
            for item in items:
                batch = item.get("batch", "")
                vid = item.get("video_id", "")
                fname = item.get("filename", "")
                key = item.get("keyframe_path") or f"{batch}/{vid}/{fname}"
                item["reason"] = item.get("reason", "auto_monochrome")
                state.filtered_items[key] = item
            logger.info(f"Đã migrate {len(state.filtered_items)} ảnh từ {OLD_FILTER_JSON_PATH.name}")
            save_filter_json()
        except Exception as e:
            logger.error(f"Lỗi migrate {OLD_FILTER_JSON_PATH}: {e}")


def save_filter_json() -> None:
    """Ghi đè danh sách lọc vào filtered_keyframes.json."""
    summary_by_batch: dict[str, int] = {}
    for item in state.filtered_items.values():
        b = item.get("batch", "unknown")
        summary_by_batch[b] = summary_by_batch.get(b, 0) + 1

    payload = {
        "total_filtered_count": len(state.filtered_items),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "summary_by_batch": dict(sorted(summary_by_batch.items())),
        "filtered_images": list(state.filtered_items.values()),
    }
    with open(FILTER_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved {len(state.filtered_items)} filtered images to {FILTER_JSON_PATH}")


def init_dataset_index() -> None:
    """Load metadata parquet và vector embeddings để phục vụ tìm kiếm tương đồng."""
    load_or_migrate_filter_json()

    meta_path = UNIFIED_DIR / "scene_metadata.parquet"
    if meta_path.exists():
        try:
            df = pd.read_parquet(meta_path)
            state.metadata_df = df
            for _, row in df.iterrows():
                vid = str(row["video_id"])
                # Lấy batch và video ngắn (L21_V001 -> batch: L21, video: V001)
                parts = vid.split("_")
                batch = parts[0] if len(parts) == 2 else vid
                sub_vid = parts[1] if len(parts) == 2 else vid
                frame_id = int(row["frame_id"])
                state.video_frame_to_scene[(batch, sub_vid, frame_id)] = {
                    "scene_id": str(row["scene_id"]),
                    "timestamp_start": float(row.get("timestamp_start", 0.0)),
                    "timestamp_mid": float(row.get("timestamp_mid", 0.0)),
                    "timestamp_end": float(row.get("timestamp_end", 0.0)),
                    "keyframe_path": str(row.get("keyframe_path", "")),
                }
            logger.info(f"Đã index {len(state.video_frame_to_scene)} scenes từ scene_metadata.parquet")
        except Exception as e:
            logger.error(f"Lỗi load metadata parquet: {e}")

    # Load scene_ids.txt
    scene_txt_path = UNIFIED_DIR / "scene_ids.txt"
    if scene_txt_path.exists():
        try:
            with open(scene_txt_path, "r", encoding="utf-8") as f:
                state.scene_ids = [line.strip() for line in f if line.strip()]
            state.scene_id_to_idx = {sid: i for i, sid in enumerate(state.scene_ids)}
            logger.info(f"Đã load {len(state.scene_ids)} scene_ids")
        except Exception as e:
            logger.error(f"Lỗi load scene_ids.txt: {e}")

    # Load image_vectors.npy
    vec_path = UNIFIED_DIR / "image_vectors.npy"
    if vec_path.exists():
        try:
            logger.info("Đang nạp image_vectors.npy vào RAM để tìm kiếm ảnh tương tự...")
            state.image_vectors = np.load(vec_path)  # (N, 1024)
            # Chuẩn hóa L2 nếu chưa chuẩn hóa
            norms = np.linalg.norm(state.image_vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            state.image_vectors = state.image_vectors / norms
            logger.info(f"Nạp thành công image_vectors.npy: shape={state.image_vectors.shape}")
        except Exception as e:
            logger.error(f"Lỗi load image_vectors.npy: {e}")


@app.on_event("startup")
async def on_startup():
    init_dataset_index()


# =============================================================================
# API Schemas & Endpoints
# =============================================================================

class FilterToggleRequest(BaseModel):
    action: str  # "add" | "remove"
    items: list[dict[str, Any]]
    reason: str = "manual"


class SimilarSearchRequest(BaseModel):
    scene_id: str | None = None
    batch: str | None = None
    video_id: str | None = None
    frame_id: int | None = None
    top_k: int = 1000
    min_similarity: float = 0.70


@app.get("/api/batches")
def get_batches():
    """Lấy danh sách batches kèm thống kê số video và ảnh."""
    batches_data = []
    if not IMAGE_DIR.exists():
        return {"batches": []}

    batch_dirs = sorted([d for d in IMAGE_DIR.iterdir() if d.is_dir() and d.name.startswith("L")])
    for b_dir in batch_dirs:
        b_name = b_dir.name
        nested_b = b_dir / b_name
        target_dir = nested_b if nested_b.exists() and nested_b.is_dir() else b_dir

        vids = sorted([v for v in target_dir.iterdir() if v.is_dir()])
        filtered_count = sum(
            1 for item in state.filtered_items.values() if item.get("batch") == b_name
        )
        batches_data.append({
            "batch": b_name,
            "video_count": len(vids),
            "filtered_count": filtered_count,
        })

    return {
        "batches": batches_data,
        "total_filtered": len(state.filtered_items),
    }


@app.get("/api/videos/{batch}")
def get_videos(batch: str):
    """Lấy danh sách videos trong batch."""
    b_dir = IMAGE_DIR / batch
    nested_b = b_dir / batch
    target_dir = nested_b if nested_b.exists() and nested_b.is_dir() else b_dir

    if not target_dir.exists():
        raise HTTPException(status_code=404, detail=f"Batch {batch} không tồn tại")

    vids = sorted([v for v in target_dir.iterdir() if v.is_dir()])
    video_list = []
    for v in vids:
        v_name = v.name
        images = list(v.glob("*.webp")) + list(v.glob("*.jpg"))
        f_count = sum(
            1
            for item in state.filtered_items.values()
            if item.get("batch") == batch and (item.get("video_id") == v_name or item.get("full_video_id") == f"{batch}_{v_name}")
        )
        video_list.append({
            "video_id": v_name,
            "full_video_id": f"{batch}_{v_name}",
            "image_count": len(images),
            "filtered_count": f_count,
        })

    return {"batch": batch, "videos": video_list}


@app.get("/api/images/{batch}/{video_id}")
def get_images(batch: str, video_id: str):
    """Lấy danh sách toàn bộ ảnh keyframe trong video."""
    b_dir = IMAGE_DIR / batch
    nested_b = b_dir / batch
    v_dir = (nested_b if nested_b.exists() and nested_b.is_dir() else b_dir) / video_id

    if not v_dir.exists():
        raise HTTPException(status_code=404, detail=f"Video {batch}/{video_id} không tồn tại")

    files = list(v_dir.glob("*.webp")) + list(v_dir.glob("*.jpg"))

    def parse_frame(p: Path) -> int:
        try:
            return int(p.stem)
        except ValueError:
            return 999999

    files.sort(key=parse_frame)

    result_images = []
    for f in files:
        fid = parse_frame(f)
        key = f"{batch}/{video_id}/{f.name}"
        is_filtered = key in state.filtered_items

        scene_info = state.video_frame_to_scene.get((batch, video_id, fid), {})
        scene_id = scene_info.get("scene_id", f"{batch}_{video_id}_S{fid:03d}")

        rel_image_path = f"{batch}/{batch}/{video_id}/{f.name}" if nested_b.exists() else f"{batch}/{video_id}/{f.name}"

        result_images.append({
            "filename": f.name,
            "frame_id": fid,
            "scene_id": scene_id,
            "keyframe_path": key,
            "rel_path": rel_image_path,
            "image_url": f"/api/image-file/{rel_image_path}",
            "is_filtered": is_filtered,
            "filter_reason": state.filtered_items.get(key, {}).get("reason", "") if is_filtered else "",
            "timestamp_mid": scene_info.get("timestamp_mid", 0.0),
        })

    return {
        "batch": batch,
        "video_id": video_id,
        "total_images": len(result_images),
        "filtered_count": sum(1 for img in result_images if img["is_filtered"]),
        "images": result_images,
    }


@app.get("/api/image-file/{file_path:path}")
def get_image_file(file_path: str):
    """Serve file ảnh keyframe từ disk."""
    full_path = IMAGE_DIR / file_path
    if not full_path.exists() or not full_path.is_file():
        raise HTTPException(status_code=404, detail="Ảnh không tồn tại")
    return FileResponse(full_path)


@app.post("/api/similar")
def search_similar_keyframes(req: SimilarSearchRequest):
    """
    Tìm kiếm ảnh tương đồng (Duplicate / Near-duplicate Search)
    sử dụng in-memory cosine similarity trên image_vectors.npy (< 5ms).
    """
    if state.image_vectors is None or not state.scene_ids:
        raise HTTPException(status_code=500, detail="Vector embeddings chưa được load sẵn sàng")

    scene_id = req.scene_id
    if not scene_id and req.batch and req.video_id is not None and req.frame_id is not None:
        info = state.video_frame_to_scene.get((req.batch, req.video_id, req.frame_id))
        if info:
            scene_id = info.get("scene_id")

    if not scene_id or scene_id not in state.scene_id_to_idx:
        raise HTTPException(status_code=404, detail=f"Không tìm thấy vector cho scene_id={scene_id}")

    query_idx = state.scene_id_to_idx[scene_id]
    query_vec = state.image_vectors[query_idx]

    scores = state.image_vectors @ query_vec

    # Lấy tất cả vector đạt ngưỡng tối thiểu
    match_mask = scores >= req.min_similarity
    match_indices = np.where(match_mask)[0]

    # Sắp xếp giảm dần theo điểm tương đồng
    sorted_order = np.argsort(scores[match_indices])[::-1][: req.top_k]
    top_indices = match_indices[sorted_order]

    results = []
    for idx in top_indices:
        sim = float(scores[idx])
        if sim < req.min_similarity:
            continue

        target_sid = state.scene_ids[idx]
        parts = target_sid.split("_")
        batch = parts[0] if len(parts) >= 1 else ""
        sub_vid = parts[1] if len(parts) >= 2 else ""

        meta_row = None
        if state.metadata_df is not None:
            matches = state.metadata_df[state.metadata_df["scene_id"] == target_sid]
            if not matches.empty:
                meta_row = matches.iloc[0]

        frame_id = int(meta_row["frame_id"]) if meta_row is not None else 0
        fname = f"{frame_id}.webp"
        key = f"{batch}/{sub_vid}/{fname}"
        is_filtered = key in state.filtered_items

        nested = IMAGE_DIR / batch / batch / sub_vid / fname
        rel_p = f"{batch}/{batch}/{sub_vid}/{fname}" if nested.exists() else f"{batch}/{sub_vid}/{fname}"

        results.append({
            "scene_id": target_sid,
            "batch": batch,
            "video_id": sub_vid,
            "full_video_id": f"{batch}_{sub_vid}",
            "frame_id": frame_id,
            "filename": fname,
            "keyframe_path": key,
            "rel_path": rel_p,
            "image_url": f"/api/image-file/{rel_p}",
            "similarity": round(sim, 4),
            "similarity_percent": round(sim * 100, 1),
            "is_query_image": (target_sid == scene_id),
            "is_filtered": is_filtered,
        })

    return {
        "query_scene_id": scene_id,
        "total_found": len(results),
        "results": results,
    }


@app.get("/api/filter")
def get_filter_list():
    """Lấy toàn bộ danh sách ảnh đã đánh dấu lọc."""
    return {
        "total_filtered_count": len(state.filtered_items),
        "filtered_images": list(state.filtered_items.values()),
    }


@app.post("/api/filter")
def update_filter_list(req: FilterToggleRequest):
    """
    Cập nhật danh sách lọc (thêm hoặc xóa).
    Auto-save vào file filtered_keyframes.json ngay lập tức.
    """
    modified_count = 0
    for item in req.items:
        batch = item.get("batch", "")
        vid = item.get("video_id", "")
        fname = item.get("filename", "")
        fid = item.get("frame_id", 0)
        key = item.get("keyframe_path") or f"{batch}/{vid}/{fname}"

        if req.action == "add":
            state.filtered_items[key] = {
                "batch": batch,
                "video_id": vid,
                "full_video_id": f"{batch}_{vid}",
                "filename": fname,
                "frame_id": fid,
                "keyframe_path": key,
                "reason": item.get("reason", req.reason),
                "added_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            modified_count += 1
        elif req.action == "remove":
            if key in state.filtered_items:
                del state.filtered_items[key]
                modified_count += 1

    save_filter_json()

    return {
        "status": "ok",
        "action": req.action,
        "modified_count": modified_count,
        "total_filtered": len(state.filtered_items),
    }


@app.get("/api/filter/stats")
def get_filter_stats():
    """Thống kê chi tiết số lượng ảnh đã lọc theo batch."""
    summary_by_batch: dict[str, int] = {}
    for item in state.filtered_items.values():
        b = item.get("batch", "unknown")
        summary_by_batch[b] = summary_by_batch.get(b, 0) + 1

    return {
        "total_filtered": len(state.filtered_items),
        "by_batch": dict(sorted(summary_by_batch.items())),
        "json_path": str(FILTER_JSON_PATH.resolve()),
    }


@app.get("/api/filter/export")
def export_filter_file():
    """Tải file JSON đã lọc."""
    save_filter_json()
    return FileResponse(
        FILTER_JSON_PATH,
        filename="filtered_keyframes.json",
        media_type="application/json",
    )


# Mount UI static
if WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="ui")


def main():
    print("=" * 60)
    print("🕷️  AIC2026 Keyframe Image Filter Tool")
    print("=" * 60)
    print("Server running at: http://localhost:8501")
    print(f"Filter JSON file : {FILTER_JSON_PATH}")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8501, log_level="info")


if __name__ == "__main__":
    main()
