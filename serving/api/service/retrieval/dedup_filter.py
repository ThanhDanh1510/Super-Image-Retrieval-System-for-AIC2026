"""
serving/api/service/retrieval/dedup_filter.py
=============================================
Module khử trùng lặp (Deduplication) và lọc khung hình rác cho KIS & TRAKE.

Bao gồm:
1. Blacklist Filter: Tự động loại bỏ 14,547 frame ảnh đơn sắc/intro/outro rác từ `data/filtered_keyframes.json`.
2. Temporal NMS: Lọc triệt để các frame quá gần nhau (< 5s) từ cùng một cảnh quay trong cùng video.
3. Diversity Capping: Giới hạn số lượng frame tối đa từ một video trong top kết quả để tăng độ đa dạng.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from serving.api.schema.retrieval import CandidateResult

logger = logging.getLogger(__name__)

# Cache danh sách đen (Blacklist) trong RAM để tra cứu O(1)
_BLACKLIST_LOADED = False
_BLACKLIST_KEYS: set[tuple[str, int]] = set()  # (video_id, frame_id)
_BLACKLIST_PATHS: set[str] = set()


def load_blacklist(filter_json_path: Optional[str | Path] = None) -> tuple[set[tuple[str, int]], set[str]]:
    """
    Nạp danh sách các frame rác/monochrome cần loại bỏ từ filtered_keyframes.json.
    Chỉ nạp 1 lần vào bộ nhớ RAM (Singleton).
    """
    global _BLACKLIST_LOADED, _BLACKLIST_KEYS, _BLACKLIST_PATHS
    if _BLACKLIST_LOADED:
        return _BLACKLIST_KEYS, _BLACKLIST_PATHS

    possible_paths = []
    if filter_json_path:
        possible_paths.append(Path(filter_json_path))

    try:
        from serving.api.core.config import get_settings
        possible_paths.append(get_settings().DATA_DIR / "filtered_keyframes.json")
    except Exception:
        pass

    possible_paths.extend([
        Path("data/filtered_keyframes.json"),
        Path("./data/filtered_keyframes.json"),
        Path("../data/filtered_keyframes.json"),
    ])

    target_file = None
    for p in possible_paths:
        if p and p.exists():
            target_file = p
            break

    if not target_file:
        logger.info("ℹ️ Không tìm thấy filtered_keyframes.json, bỏ qua blacklist filter.")
        _BLACKLIST_LOADED = True
        return _BLACKLIST_KEYS, _BLACKLIST_PATHS

    try:
        with open(target_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        filtered_list = data.get("filtered_images", [])
        for item in filtered_list:
            vid = str(item.get("full_video_id") or item.get("video_id") or "")
            fid = item.get("frame_id")
            if vid and fid is not None:
                _BLACKLIST_KEYS.add((vid, int(fid)))
                # Thêm cả short video id (ví dụ V001 nếu vid là L21_V001)
                if "_" in vid:
                    _BLACKLIST_KEYS.add((vid.split("_")[-1], int(fid)))

            kp = item.get("keyframe_path")
            if kp:
                _BLACKLIST_PATHS.add(str(kp).replace("\\", "/"))

        _BLACKLIST_LOADED = True
        logger.info(f"✅ Đã nạp thành công {len(_BLACKLIST_KEYS)} blacklist frames từ {target_file}")
    except Exception as e:
        logger.warning(f"Lỗi khi đọc blacklist {target_file}: {e}")
        _BLACKLIST_LOADED = True

    return _BLACKLIST_KEYS, _BLACKLIST_PATHS


def is_blacklisted(
    cand: CandidateResult,
    blacklist_keys: set[tuple[str, int]],
    blacklist_paths: set[str],
) -> bool:
    """Kiểm tra một candidate có nằm trong danh sách đen không."""
    if not blacklist_keys and not blacklist_paths:
        return False

    vid = cand.video_id
    fid = cand.frame_id
    if (vid, fid) in blacklist_keys:
        return True
    if "_" in vid and (vid.split("_")[-1], fid) in blacklist_keys:
        return True

    kp = cand.keyframe_path
    if kp:
        norm_kp = kp.replace("\\", "/")
        if norm_kp in blacklist_paths:
            return True

    return False


def deduplicate_candidates(
    candidates: list[CandidateResult],
    top_k: int = 100,
    max_per_video: int = 3,
    time_window_sec: float = 5.0,
    is_trake: bool = False,
    filter_json_path: Optional[str | Path] = None,
) -> list[CandidateResult]:
    """
    Áp dụng toàn diện các tầng khử trùng lặp cho danh sách kết quả:
    1. Blacklist Filtering: Loại bỏ các frame lỗi, đen, intro từ filtered_keyframes.json.
    2. Temporal NMS: Không cho phép các frame trong cùng video cách nhau <= time_window_sec lặp lại.
    3. Diversity Capping: Giới hạn số frame tối đa từ cùng 1 video trong nhóm hàng đầu (trừ khi là TRAKE).

    Args:
        candidates: Danh sách kết quả thô hoặc sau reranking.
        top_k: Số lượng kết quả cuối cùng cần trả về.
        max_per_video: Số frame tối đa được phép xuất hiện từ cùng 1 video trong top.
        time_window_sec: Cửa sổ thời gian (giây) coi là trùng lặp cảnh trong cùng 1 video.
        is_trake: True nếu là chế độ TRAKE (không giới hạn max_per_video để giữ timeline).
        filter_json_path: Đường dẫn tùy chọn tới file filtered_keyframes.json.

    Returns:
        Danh sách CandidateResult đã khử trùng lặp và đa dạng hóa.
    """
    if not candidates:
        return []

    # 1. Blacklist Filtering
    bl_keys, bl_paths = load_blacklist(filter_json_path)
    clean_candidates = [
        cand for cand in candidates
        if not is_blacklisted(cand, bl_keys, bl_paths)
    ]

    if not clean_candidates:
        clean_candidates = candidates

    # Nếu là TRAKE mode, chỉ áp dụng Blacklist, để Temporal Aligner tự xử lý chuỗi thời gian
    if is_trake:
        return clean_candidates[:top_k]

    # 2. Temporal NMS + Diversity Capping kết hợp
    selected: list[CandidateResult] = []
    overflow: list[CandidateResult] = []

    video_timestamps: dict[str, list[float]] = {}
    video_counts: dict[str, int] = {}

    for cand in clean_candidates:
        vid = cand.video_id
        t = cand.timestamp_mid if cand.timestamp_mid is not None else (float(cand.frame_id) / 25.0)

        # Kiểm tra giới hạn max_per_video
        curr_count = video_counts.get(vid, 0)
        if curr_count >= max_per_video:
            overflow.append(cand)
            continue

        # Kiểm tra Temporal NMS trong cùng video
        if vid not in video_timestamps:
            video_timestamps[vid] = [t]
            video_counts[vid] = 1
            selected.append(cand)
        else:
            is_close = any(abs(prev_t - t) <= time_window_sec for prev_t in video_timestamps[vid])
            if not is_close:
                video_timestamps[vid].append(t)
                video_counts[vid] = curr_count + 1
                selected.append(cand)
            else:
                overflow.append(cand)

    # Ghép phần overflow về sau để đảm bảo luôn đủ top_k nếu có thể
    final_list = selected + overflow
    return final_list[:top_k]
