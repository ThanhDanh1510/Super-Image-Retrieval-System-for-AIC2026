"""
serving/api/core/translation_cache.py
=====================================
Bộ nhớ đệm dịch thuật (Translation Cache) hai chiều (VI<->EN).

Dung lượng 10,000 mục, lưu trữ trong RAM và tự động nạp/lưu xuống đĩa
(`data/translation_cache.json`) để phản hồi tức thì (<0.01ms), loại bỏ
chi phí GPU và sự phụ thuộc vào internet.
"""

from __future__ import annotations

import json
import logging
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class TranslationCache:
    """
    Thread-Safe LRU Cache cho các bản dịch kèm tính năng lưu bền vững vào đĩa.
    """

    def __init__(
        self,
        max_size: int = 10000,
        cache_file_path: Optional[str | Path] = None,
    ) -> None:
        self.max_size = max_size
        self._cache: OrderedDict[str, str] = OrderedDict()
        self._lock = threading.Lock()
        self._dirty_count = 0  # Số lượt thay đổi chưa lưu xuống đĩa
        self._save_threshold = 20  # Lưu mỗi 20 lượt ghi mới

        if cache_file_path is not None:
            self.cache_file = Path(cache_file_path)
        else:
            try:
                from serving.api.core.config import get_settings
                self.cache_file = get_settings().DATA_DIR / "translation_cache.json"
            except Exception:
                self.cache_file = Path("data/translation_cache.json")

        self.load_from_disk()

    @staticmethod
    def _make_key(text: str, target: str) -> str:
        return f"{target.strip().lower()}:{text.strip().lower()}"

    def get(self, text: str, target: str) -> str | None:
        """Lấy bản dịch từ cache. Trả về None nếu chưa có."""
        if not text or not text.strip():
            return ""

        key = self._make_key(text, target)
        with self._lock:
            if key not in self._cache:
                return None
            val = self._cache[key]
            self._cache.move_to_end(key)
            return val

    def set(self, text: str, target: str, translated: str) -> None:
        """Lưu bản dịch vào cache và tự động lưu đĩa định kỳ."""
        if not text or not text.strip() or not translated or not translated.strip():
            return

        key = self._make_key(text, target)
        clean_val = translated.strip()

        with self._lock:
            if key in self._cache:
                self._cache[key] = clean_val
                self._cache.move_to_end(key)
                return

            if len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)

            self._cache[key] = clean_val
            self._dirty_count += 1

            if self._dirty_count >= self._save_threshold:
                self._save_to_disk_unlocked()
                self._dirty_count = 0

    def load_from_disk(self) -> None:
        """Đọc danh sách bản dịch đã lưu từ đĩa."""
        if not self.cache_file or not self.cache_file.exists():
            return

        try:
            with open(self.cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, dict):
                with self._lock:
                    for k, v in data.items():
                        if isinstance(k, str) and isinstance(v, str):
                            self._cache[k] = v
                logger.info(f"✅ Đã nạp {len(self._cache)} mục dịch từ {self.cache_file}")
        except Exception as e:
            logger.warning(f"Không thể đọc file translation cache {self.cache_file}: {e}")

    def save_to_disk(self) -> None:
        """Lưu toàn bộ cache hiện tại xuống file JSON."""
        with self._lock:
            self._save_to_disk_unlocked()

    def _save_to_disk_unlocked(self) -> None:
        if not self.cache_file:
            return
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, ensure_ascii=False, indent=2)
            self._dirty_count = 0
        except Exception as e:
            logger.warning(f"Lỗi khi lưu translation cache xuống đĩa: {e}")

    def clear(self) -> None:
        """Xóa sạch cache."""
        with self._lock:
            self._cache.clear()
            self._dirty_count = 0

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)


# Global Singleton Instance
_global_translation_cache: TranslationCache | None = None


def get_translation_cache() -> TranslationCache:
    global _global_translation_cache
    if _global_translation_cache is None:
        _global_translation_cache = TranslationCache(max_size=10000)
    return _global_translation_cache
