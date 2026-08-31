"""
serving/api/core/query_cache.py
===============================
In-Memory Thread-Safe LRU Cache có Time-To-Live (TTL) cho kết quả tìm kiếm KIS.

Giúp hệ thống phản hồi tức thì (<1ms) cho các câu truy vấn lặp lại hoặc
khi người dùng thao tác chuyển đổi trang/chế độ xem trên Web UI.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from collections import OrderedDict
from typing import Any, Optional

logger = logging.getLogger(__name__)


_FILLER_PHRASES = (
    "trong đoạn video có",
    "trong đoạn video",
    "trong cảnh quay có",
    "trong cảnh quay",
    "trong khung hình có",
    "trong khung hình",
    "đoạn video quay cảnh",
    "video quay cảnh",
    "video quay",
    "hình ảnh có",
    "cảnh quay có",
    "khung hình có",
    "hãy tìm kiếm",
    "hãy tìm",
    "có cảnh",
    "có hình ảnh",
    "cảnh quay",
    "hình ảnh",
    "có ",
)


def normalize_query_text(text: str) -> str:
    """Loại bỏ các từ đệm, từ dừng chỉ định cảnh quay thường gặp trong đề thi AIC."""
    import re
    cleaned = text.strip().lower()
    changed = True
    while changed:
        changed = False
        for phrase in _FILLER_PHRASES:
            if cleaned.startswith(phrase):
                cleaned = cleaned[len(phrase):].strip()
                changed = True
                break
    cleaned = re.sub(r'\s+', ' ', cleaned).strip(' .,;:!?\"\'`~')
    return cleaned if cleaned else text.strip().lower()


class QueryCache:
    """
    LRU Cache với Time-To-Live (TTL) cho kết quả PipelineResult.
    """

    def __init__(self, max_size: int = 500, default_ttl_sec: float = 300.0) -> None:
        self.max_size = max_size
        self.default_ttl = default_ttl_sec
        self._cache: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._lock = threading.Lock()

    def _is_expired(self, expiry_time: float) -> bool:
        return time.time() > expiry_time

    def get(self, key: str) -> Any | None:
        """Lấy kết quả từ cache. Trả về None nếu không tồn tại hoặc đã hết hạn."""
        with self._lock:
            if key not in self._cache:
                return None

            value, expiry = self._cache[key]
            if self._is_expired(expiry):
                del self._cache[key]
                return None

            # Đưa lên đầu danh sách LRU
            self._cache.move_to_end(key)
            return value

    def set(self, key: str, value: Any, ttl_sec: Optional[float] = None) -> None:
        """Lưu kết quả vào cache kèm thời gian hết hạn."""
        ttl = ttl_sec if ttl_sec is not None else self.default_ttl
        expiry = time.time() + ttl

        with self._lock:
            # Nếu key đã tồn tại -> cập nhật và đưa lên đầu
            if key in self._cache:
                self._cache[key] = (value, expiry)
                self._cache.move_to_end(key)
                return

            # Nếu cache đã đầy -> dọn dẹp các mục hết hạn trước
            if len(self._cache) >= self.max_size:
                now = time.time()
                expired_keys = [k for k, (_, exp) in self._cache.items() if now > exp]
                for k in expired_keys:
                    del self._cache[k]

            # Nếu vẫn đầy -> xóa mục cũ nhất (FIFO/LRU)
            if len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)

            self._cache[key] = (value, expiry)

    def clear(self) -> None:
        """Xóa toàn bộ cache."""
        with self._lock:
            self._cache.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)

    @staticmethod
    def make_key(
        query: str,
        search_mode: str = "all",
        top_k: int = 100,
        use_viranker: bool = True,
        use_jina_m0: bool = False,
        task_type: str = "kis",
        **extra,
    ) -> str:
        """Tạo MD5 hash key xác định duy nhất một request tìm kiếm (đã qua chuẩn hóa từ đệm)."""
        norm_q = normalize_query_text(query)
        raw_str = f"{task_type}:{search_mode}:{top_k}:{use_viranker}:{use_jina_m0}:{norm_q}"
        if extra:
            extra_sorted = json.dumps(extra, sort_keys=True)
            raw_str += f":{extra_sorted}"
        return hashlib.md5(raw_str.encode("utf-8")).hexdigest()


# Singleton Instance
_global_query_cache: QueryCache | None = None


def get_query_cache() -> QueryCache:
    """Lấy thể hiện duy nhất của QueryCache."""
    global _global_query_cache
    if _global_query_cache is None:
        _global_query_cache = QueryCache(max_size=500, default_ttl_sec=300.0)
    return _global_query_cache
