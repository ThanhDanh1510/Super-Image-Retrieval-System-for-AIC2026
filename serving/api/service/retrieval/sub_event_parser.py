"""
serving/api/service/retrieval/sub_event_parser.py
=================================================
Bo phan ra cau truy van da su kien (Multi-Event Sub-Query Parser) cho bai toan TRAKE.
"""

from __future__ import annotations

import re
import logging

logger = logging.getLogger(__name__)

# Regex cho cac mau danh so
_NUMBERED_PATTERN = re.compile(
    r"(?:^|\s+)(?:(?:(?:\d+|[a-zA-Z])[\.\)\:\-])|(?:\[\d+\])|(?:Bước\s+\d+\:?)|(?:Step\s+\d+\:?)|(?:Event\s+\d+\:?))\s+",
    re.IGNORECASE,
)

# Regex cho tu noi thoi gian tieng Viet & tieng Anh
_TEMPORAL_MARKERS = [
    r"đầu tiên", r"trước tiên", r"trước hết", r"ban đầu",
    r"sau đó", r"tiếp theo", r"tiếp tục", r"kế tiếp", r"liền sau đó", r"ngay sau đó", r"tiếp đó",
    r"cuối cùng", r"sau cùng", r"kết thúc", r"cuối cuộc", r"kết lại",
    r"first", r"initially", r"firstly",
    r"then", r"next", r"after that", r"afterwards", r"subsequently",
    r"finally", r"lastly", r"at last", r"in the end",
]
_TEMPORAL_SPLIT_REGEX = re.compile(r"(?:\b(?:" + "|".join(_TEMPORAL_MARKERS) + r")\b)\s*[:,\\-]?\s*", re.IGNORECASE)


def parse_sub_events(query: str, target_num_events: int | None = None) -> list[str]:
    """
    Phan tach query thanh danh sach cac sub-events doc lap theo thu tu thoi gian.

    Args:
        query: Chuoi truy van da su kien.
        target_num_events: So luong su kien ky vong (2-10).

    Returns:
        list[str]: Danh sach cac chuoi con mo ta tung su kien.
    """
    if not query or not query.strip():
        return []

    text = query.strip()

    # 1. Xuong dong
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) >= 2:
        cleaned = [_clean_sub_event(l) for l in lines if _clean_sub_event(l)]
        if len(cleaned) >= 2:
            return _adjust_count(cleaned, target_num_events, text)

    # 2. Danh so (1. 2. 3.)
    numbered = _NUMBERED_PATTERN.split(text)
    if len(numbered) >= 2:
        cleaned = [_clean_sub_event(p) for p in numbered if _clean_sub_event(p)]
        if len(cleaned) >= 2:
            return _adjust_count(cleaned, target_num_events, text)

    # 3. Mui ten / Cham phay (->, =>, ;, |)
    for sep in [r"\s*-->\s*", r"\s*->\s*", r"\s*=>\s*", r"\s*;\s*", r"\s*\|\s*"]:
        parts = re.split(sep, text)
        if len(parts) >= 2:
            cleaned = [_clean_sub_event(p) for p in parts if _clean_sub_event(p)]
            if len(cleaned) >= 2:
                return _adjust_count(cleaned, target_num_events, text)

    # 4. Tu noi thoi gian
    temp_parts = _TEMPORAL_SPLIT_REGEX.split(text)
    if len(temp_parts) >= 2:
        cleaned = [_clean_sub_event(p) for p in temp_parts if _clean_sub_event(p)]
        if len(cleaned) >= 2:
            return _adjust_count(cleaned, target_num_events, text)

    # 5. Sentence split
    sents = [s.strip() for s in re.split(r"[.?!]\s+", text) if s.strip()]
    if len(sents) >= 2:
        cleaned = [_clean_sub_event(p) for p in sents if _clean_sub_event(p)]
        if len(cleaned) >= 2:
            return _adjust_count(cleaned, target_num_events, text)

    single = _clean_sub_event(text)
    if target_num_events and target_num_events > 1:
        return [single] * target_num_events
    return [single]


def _clean_sub_event(s: str) -> str:
    cleaned = re.sub(r"^[,\-\:\.\s]+", "", s)
    cleaned = re.sub(r"[,\-\:\.\s]+$", "", cleaned)
    return cleaned.strip()


def _adjust_count(events: list[str], target_num_events: int | None, fallback_query: str) -> list[str]:
    if not target_num_events or target_num_events <= 0:
        return events
    if len(events) == target_num_events:
        return events
    if len(events) > target_num_events:
        return events[:target_num_events]
    res = list(events)
    while len(res) < target_num_events:
        res.append(fallback_query)
    return res
