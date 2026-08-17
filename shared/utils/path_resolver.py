"""
shared/utils/path_resolver.py
==============================
Single source of truth for resolving keyframe image and video file paths on disk.
"""

from pathlib import Path
from typing import Optional
from serving.api.core.config import get_settings


def resolve_keyframe_path(
    video_id: str,
    frame_id: int,
    keyframe_path: Optional[str] = None,
    data_dir: Optional[Path] = None,
) -> Optional[Path]:
    """
    Tìm file ảnh keyframe thực tế trên đĩa cứng.
    Hỗ trợ cấu trúc lồng `data/image/{Group}/{Group}/{Vsub}/{frame_id}.webp|jpg`
    hoặc phẳng `data/image/{Group}/{video_id}/{frame_id}.webp`.
    """
    if data_dir is None:
        data_dir = get_settings().DATA_DIR
    data_dir = Path(data_dir)

    frame_id_str = f"{frame_id:05d}"
    parts = video_id.split("_")
    if len(parts) >= 2:
        grp, vsub = parts[0], parts[1]
    else:
        grp, vsub = video_id, video_id

    candidates = [
        data_dir / "image" / grp / grp / vsub / f"{frame_id_str}.webp",
        data_dir / "image" / grp / grp / vsub / f"{frame_id_str}.jpg",
        data_dir / "image" / grp / f"{video_id}" / f"{frame_id_str}.webp",
        data_dir / "image" / grp / f"{video_id}" / f"{frame_id_str}.jpg",
        data_dir / "image" / f"{video_id}" / f"{frame_id_str}.webp",
        data_dir / "image" / f"{video_id}" / f"{frame_id_str}.jpg",
        data_dir / "unified" / "keyframes" / keyframe_path if keyframe_path else None,
        data_dir / keyframe_path if keyframe_path else None,
    ]
    for p in candidates:
        if p and p.is_file() and p.exists():
            return p
    return None


def resolve_video_path(video_id: str, data_dir: Optional[Path | str] = None) -> Optional[Path]:
    """
    Tìm file video gốc trên đĩa cứng (.mp4, .mkv, .webm).
    """
    if data_dir is None:
        data_dir = get_settings().DATA_DIR
    data_dir = Path(data_dir)

    parts = video_id.split("_")
    grp = parts[0] if len(parts) >= 2 else video_id

    candidates = [
        data_dir / "video" / grp / f"{video_id}.mp4",
        data_dir / "video" / grp / f"{video_id}.mkv",
        data_dir / "video" / f"{video_id}.mp4",
        data_dir / "video" / f"{video_id}.mkv",
        data_dir / "video" / grp / f"{video_id}.webm",
        data_dir / "video" / f"{video_id}.webm",
    ]
    for p in candidates:
        if p and p.is_file() and p.exists():
            return p
    return None
