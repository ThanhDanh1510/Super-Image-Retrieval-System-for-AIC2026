"""
tests/test_translation_cache.py
===============================
Unit tests cho TranslationCache (RAM + Persistent Disk storage).
"""

import tempfile
from pathlib import Path
from serving.api.core.translation_cache import TranslationCache


def test_translation_cache_get_set():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_file = Path(tmpdir) / "test_cache.json"
        cache = TranslationCache(max_size=10, cache_file_path=tmp_file)
        
        assert cache.get("con lân", "en") is None
        cache.set("con lân", "en", "lion dance")
        assert cache.get("con lân", "en") == "lion dance"
        assert cache.get("CON LÂN", "en") == "lion dance" # case-insensitive key


def test_translation_cache_persistence():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_file = Path(tmpdir) / "test_cache.json"
        cache1 = TranslationCache(max_size=10, cache_file_path=tmp_file)
        cache1.set("áo dài", "en", "Vietnamese ao dai")
        cache1.save_to_disk()
        assert tmp_file.exists()

        # Tạo cache mới nạp từ file đã lưu
        cache2 = TranslationCache(max_size=10, cache_file_path=tmp_file)
        assert cache2.get("áo dài", "en") == "Vietnamese ao dai"
