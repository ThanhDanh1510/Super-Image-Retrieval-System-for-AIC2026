"""
serving/api/service/dense_similarity.py
========================================
Module tính toán ma trận tương đồng dày đặc (Dense Similarity Matrix) giữa N sub-events và T keyframes.
Thay thế hoàn toàn ma trận RRF score thưa (chứa nhiều số 0.0 giả) bằng True Cosine Similarity.

Nguyên lý:
  1. Với N sub-events, encode thành N vector U (shape: [N, 1024], L2-normalized) qua CLIP Text Encoder.
  2. Với video có T keyframes, lấy T vector V (shape: [T, 1024], L2-normalized) từ Qdrant (named vector 'image_dense').
  3. Tính ma trận tương đồng toàn phần: S = U @ V.T (shape: [N, T]).
  4. Mọi keyframe t đều nhận được điểm tương đồng ngữ nghĩa thực sự cho mọi sub-event i,
     loại bỏ hoàn toàn hiện tượng 0.0 điểm cho các frame được enrich.
"""

from __future__ import annotations

import asyncio
import logging
import numpy as np
from typing import Optional

from serving.api.schema.retrieval import CandidateResult
from serving.api.repo import qdrant_repo

logger = logging.getLogger(__name__)


def compute_dense_similarity(
    query_embeddings: np.ndarray,
    keyframe_vectors: np.ndarray,
) -> np.ndarray:
    """
    Tính ma trận cosine similarity giữa query embeddings (N, D) và keyframe vectors (T, D).

    Args:
        query_embeddings: numpy array shape (N, D), L2-normalized.
        keyframe_vectors: numpy array shape (T, D), L2-normalized.

    Returns:
        numpy array shape (N, T) với giá trị cosine similarity >= 0.0.
    """
    if query_embeddings.size == 0 or keyframe_vectors.size == 0:
        return np.zeros((len(query_embeddings), len(keyframe_vectors)), dtype=np.float32)

    # Đảm bảo 2D arrays
    if query_embeddings.ndim == 1:
        query_embeddings = query_embeddings[np.newaxis, :]
    if keyframe_vectors.ndim == 1:
        keyframe_vectors = keyframe_vectors[np.newaxis, :]

    # Normalize để an toàn tuyệt đối
    q_norm = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
    q_norm[q_norm == 0] = 1.0
    u = query_embeddings / q_norm

    k_norm = np.linalg.norm(keyframe_vectors, axis=1, keepdims=True)
    k_norm[k_norm == 0] = 1.0
    v = keyframe_vectors / k_norm

    # Dot product (Cosine Similarity vì đã L2-normalized)
    sim_matrix = np.matmul(u, v.T)  # Shape: (N, T)

    # Clip về [0.0, 1.0] để phù hợp với thang điểm DP
    sim_matrix = np.clip(sim_matrix, 0.0, 1.0)
    return sim_matrix.astype(np.float32)


async def encode_sub_events(
    sub_events: list[str],
    image_embedder,
) -> np.ndarray:
    """
    Encode N sub-events thành N vector CLIP text (shape: [N, 1024], L2-normalized).
    """
    from serving.api.service.retrieval.text_query import _translate

    if not sub_events:
        return np.zeros((0, 1024), dtype=np.float32)

    # Dịch song ngữ sang tiếng Anh đồng thời cho CLIP text encoder
    async def _trans(text: str) -> str:
        try:
            res = await _translate(text, target="en")
            return res if res else text
        except Exception:
            return text

    en_sub_events = list(await asyncio.gather(*[_trans(ev) for ev in sub_events]))

    # Encode bằng image_embedder.encode_text
    vecs = await asyncio.to_thread(image_embedder.encode_text, en_sub_events)
    if isinstance(vecs, list):
        vecs = np.array(vecs, dtype=np.float32)
    elif hasattr(vecs, "numpy"):
        vecs = vecs.numpy().astype(np.float32)
    elif not isinstance(vecs, np.ndarray):
        vecs = np.array(vecs, dtype=np.float32)

    if vecs.ndim == 1:
        vecs = vecs[np.newaxis, :]

    return vecs


async def build_dense_matrices_for_videos(
    sub_events: list[str],
    video_ids: list[str],
    qdrant_client,
    image_embedder,
    limit_per_video: int = 500,
) -> dict[str, tuple[list[CandidateResult], np.ndarray]]:
    """
    Lấy keyframes và vector từ Qdrant cho danh sách video_ids và tính ma trận S_dense[N, T].

    Returns:
        dict mapping video_id -> (sorted_candidates, S_dense[N, T])
    """
    if not video_ids or not sub_events:
        return {}

    # 1. Encode N sub-events sang vector CLIP
    sub_event_vecs = await encode_sub_events(sub_events, image_embedder)
    if sub_event_vecs.shape[0] == 0:
        return {}

    # 2. Lấy keyframe vectors song song cho tất cả videos
    async def _fetch_and_score(vid: str):
        cands, vecs = await qdrant_repo.get_keyframes_with_vectors_by_video_id(
            client=qdrant_client,
            video_id=vid,
            limit=limit_per_video,
            vector_name="image_dense",
        )
        if not cands or not vecs:
            return vid, (cands, np.zeros((len(sub_events), 0), dtype=np.float32))

        vecs_np = np.array(vecs, dtype=np.float32)
        sim_mat = compute_dense_similarity(sub_event_vecs, vecs_np)
        return vid, (cands, sim_mat)

    tasks = [_fetch_and_score(v_id) for v_id in video_ids]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    dense_map: dict[str, tuple[list[CandidateResult], np.ndarray]] = {}
    for res in results:
        if isinstance(res, tuple) and len(res) == 2:
            vid, (cands, s_mat) = res
            if cands and s_mat.shape[1] > 0:
                dense_map[vid] = (cands, s_mat)

    return dense_map
