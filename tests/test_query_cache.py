import pytest
from serving.api.core.query_cache import QueryCache, normalize_query_text


def test_normalize_query_text():
    assert normalize_query_text("trong cảnh quay có 4 tài xế xe ôm") == "4 tài xế xe ôm"
    assert normalize_query_text("video quay cảnh hai con lân") == "hai con lân"
    assert normalize_query_text("hình ảnh có người phụ nữ") == "người phụ nữ"
    assert normalize_query_text("hãy tìm kiếm xe cứu hỏa") == "xe cứu hỏa"
    assert normalize_query_text("xe máy màu đỏ") == "xe máy màu đỏ"


def test_query_cache_normalized_key_hit():
    cache = QueryCache(max_size=10, default_ttl_sec=60.0)
    k1 = QueryCache.make_key("trong cảnh quay có 4 tài xế xe ôm")
    k2 = QueryCache.make_key("4 tài xế xe ôm")
    assert k1 == k2
    cache.set(k1, "sample_result")
    assert cache.get(k2) == "sample_result"
