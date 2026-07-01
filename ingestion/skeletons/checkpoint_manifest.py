"""
ingestion/skeletons/checkpoint_manifest.py
============================================
TASK-ING-05: Checkpoint để resume pipeline khi Kaggle session bị ngắt.

SKELETON: Implement class CheckpointManifest.

MỤC ĐÍCH:
    Kaggle Notebook session có thể bị ngắt bất kỳ lúc nào (timeout, disconnect...).
    CheckpointManifest giúp pipeline biết đã xử lý đến đâu để resume mà không cần chạy lại từ đầu.

GRANULARITY: Scene-level (KHÔNG phải video-level).
    → Nếu chạy xử lý đến giữa video, khi resume chỉ cần skip những scene đã xong.

FORMAT file checkpoint.json:
    {
        "version": "1.0",
        "created_at": "2026-06-01T10:00:00",
        "updated_at": "2026-06-01T11:30:00",
        "steps": {
            "keyframe": ["L01_V001_S001", "L01_V001_S002", ...],
            "ocr": ["L01_V001_S001", ...],
            "asr": ["L01_V001_S001", ...],
            "image_embed": ["L01_V001_S001", ...],
            "text_embed": ["L01_V001_S001", ...]
        }
    }

CÁCH DÙNG (trong notebook cell):
    >>> from ingestion.skeletons.checkpoint_manifest import CheckpointManifest
    >>> ckpt = CheckpointManifest('/kaggle/working/checkpoint.json')
    >>> remaining = ckpt.get_remaining(all_scene_ids, step='keyframe')
    >>> for scene_id in remaining:
    ...     # Xử lý scene
    ...     ckpt.mark_done(scene_id, step='keyframe')
    ...     ckpt.save()  # Lưu sau mỗi scene
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# Các bước pipeline hợp lệ
VALID_STEPS = ("keyframe", "ocr", "asr", "image_embed", "text_embed")


class CheckpointManifest:
    """
    Ghi/đọc checkpoint để resume Kaggle pipeline khi session bị ngắt.

    Granularity: Scene-level — track từng scene_id đã hoàn thành ở từng step.

    Usage:
        ckpt = CheckpointManifest('/kaggle/working/checkpoint.json')
        remaining = ckpt.get_remaining(all_scene_ids, step='keyframe')
        for scene_id in remaining:
            # ... xử lý scene ...
            ckpt.mark_done(scene_id, step='keyframe')
            ckpt.save()  # Ghi ra file ngay sau mỗi scene
    """

    def __init__(self, checkpoint_path: str = "/kaggle/working/checkpoint.json") -> None:
        """
        Khởi tạo CheckpointManifest. Load từ file nếu đã tồn tại.

        Gợi ý implementation:
            self.checkpoint_path = Path(checkpoint_path)
            self._data = {
                'version': '1.0',
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat(),
                'steps': {step: [] for step in VALID_STEPS},
            }
            if self.checkpoint_path.exists():
                with open(self.checkpoint_path, 'r', encoding='utf-8') as f:
                    self._data = json.load(f)
                logger.info(f'Loaded checkpoint: {checkpoint_path}')
            else:
                logger.info(f'New checkpoint: {checkpoint_path}')

        Args:
            checkpoint_path: Đường dẫn tới file checkpoint.json.
        """
        raise NotImplementedError

    def mark_done(self, scene_id: str, step: str) -> None:
        """
        Đánh dấu một scene đã hoàn thành ở một step nhất định.

        Gợi ý implementation:
            if step not in VALID_STEPS:
                raise ValueError(f'Invalid step: {step}. Valid: {VALID_STEPS}')
            if scene_id not in self._data['steps'][step]:
                self._data['steps'][step].append(scene_id)
                self._data['updated_at'] = datetime.now().isoformat()

        Args:
            scene_id: ID scene đã hoàn thành (ví dụ: 'L01_V001_S003').
            step: Tên step ('keyframe', 'ocr', 'asr', 'image_embed', 'text_embed').

        Raises:
            ValueError: Nếu step không hợp lệ.
        """
        raise NotImplementedError

    def is_done(self, scene_id: str, step: str) -> bool:
        """
        Kiểm tra xem một scene đã hoàn thành step chỉ định chưa.

        Gợi ý implementation:
            if step not in VALID_STEPS:
                raise ValueError(f'Invalid step: {step}')
            return scene_id in self._data['steps'][step]

        Args:
            scene_id: ID scene cần kiểm tra.
            step: Tên step.

        Returns:
            True nếu scene_id đã được mark_done cho step này.

        Raises:
            ValueError: Nếu step không hợp lệ.
        """
        raise NotImplementedError

    def get_remaining(self, all_scene_ids: list[str], step: str) -> list[str]:
        """
        Trả về danh sách scene_id chưa hoàn thành ở step chỉ định.

        Gợi ý implementation:
            done = set(self._data['steps'].get(step, []))
            remaining = [sid for sid in all_scene_ids if sid not in done]
            total = len(all_scene_ids)
            logger.info(f'Step [{step}]: {total - len(remaining)}/{total} done, {len(remaining)} remaining')
            return remaining

        Args:
            all_scene_ids: Toàn bộ scene_id cần xử lý trong shard.
            step: Tên step.

        Returns:
            List scene_id chưa hoàn thành, giữ nguyên thứ tự từ all_scene_ids.

        Raises:
            ValueError: Nếu step không hợp lệ.
        """
        raise NotImplementedError

    def save(self) -> None:
        """
        Ghi checkpoint hiện tại ra file JSON.

        Nên gọi sau mỗi scene để đảm bảo không mất tiến độ.

        Gợi ý implementation:
            self._data['updated_at'] = datetime.now().isoformat()
            self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.checkpoint_path, 'w', encoding='utf-8') as f:
                json.dump(self._data, f, ensure_ascii=False, indent=2)

        Raises:
            OSError: Nếu không ghi được file (hết đĩa, không có quyền...).
        """
        raise NotImplementedError

    @classmethod
    def load(cls, checkpoint_path: str) -> "CheckpointManifest":
        """
        Class method để load CheckpointManifest từ file đã tồn tại.

        Gợi ý implementation:
            return cls(checkpoint_path)

        Args:
            checkpoint_path: Đường dẫn tới file checkpoint.json.

        Returns:
            CheckpointManifest instance đã load dữ liệu từ file.

        Raises:
            FileNotFoundError: Nếu checkpoint_path không tồn tại.
        """
        raise NotImplementedError

    def summary(self) -> dict[str, int]:
        """
        Trả về thống kê số scene đã hoàn thành ở mỗi step.

        Returns:
            Dict: {step_name: count_done}
            Ví dụ: {'keyframe': 500, 'ocr': 480, 'asr': 450, ...}
        """
        raise NotImplementedError
