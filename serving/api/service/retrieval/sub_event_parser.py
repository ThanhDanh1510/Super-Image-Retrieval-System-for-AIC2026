"""
serving/api/service/retrieval/sub_event_parser.py
=================================================
Bộ phân rã câu truy vấn đa sự kiện (Multi-Event Sub-Query Parser) cho bài toán TRAKE.

Hỗ trợ mọi định dạng:
  - Explicit event tags (E1, E2, E3, EV1, Event 1, Bước 1, Step 1, Phase 1...) dù có hay không có dấu chấm/hai chấm.
  - Xuống dòng (newlines).
  - Đánh số thứ tự (1. 2. 3. hoặc [1] [2] [3]).
  - Mũi tên và ký tự phân cách (->, =>, ;, |).
  - Từ nối thời gian tiếng Việt & tiếng Anh (đầu tiên, sau đó, tiếp theo, then, next, after that...).
  - Tách câu tự nhiên (sentence split).
"""

from __future__ import annotations

import re
import logging

logger = logging.getLogger(__name__)

# Regex tìm các nhãn sự kiện rõ ràng (E1, E2, Step 1, Bước 1...)
_EXPLICIT_MARKER_REGEX = re.compile(
    r"(?:^|\s+)(E\d+|EV\d+|Event\s*\d+|Bước\s*\d+|Step\s*\d+|Phase\s*\d+|\b\d+[\.\)\:\-]\s*|\bE\d+[\.\)\:\-]?)\s+",
    re.IGNORECASE,
)

# Regex cho các mẫu đánh số phân đoạn
_NUMBERED_PATTERN = re.compile(
    r"(?:^|\s+)(?:(?:(?:\d+|[a-zA-Z])[\.\)\:\-])|(?:\[\d+\])|(?:Bước\s+\d+\:?)|(?:Step\s+\d+\:?)|(?:Event\s+\d+\:?)|(?:E\d+[\.\)\:\-]?))\s+",
    re.IGNORECASE,
)

# Regex cho từ nối thời gian tiếng Việt & tiếng Anh
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
    Phân tách query thành danh sách các sub-events độc lập theo thứ tự thời gian.

    Args:
        query: Chuỗi truy vấn đa sự kiện.
        target_num_events: Số lượng sự kiện kỳ vọng (2-10).

    Returns:
        list[str]: Danh sách các chuỗi con mô tả từng sự kiện.
    """
    if not query or not query.strip():
        return []

    text = query.strip()

    # 1. Khớp theo Explicit Event Markers (E1, E2, E3, Event 1, Bước 1...)
    matches = list(_EXPLICIT_MARKER_REGEX.finditer(text))
    if len(matches) >= 2:
        extracted = []
        for i in range(len(matches)):
            start = matches[i].end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            seg = _clean_sub_event(text[start:end])
            if seg:
                extracted.append(seg)
        if len(extracted) >= 2:
            return _adjust_count(extracted, target_num_events, text)

    # 2. Xuống dòng (Newlines)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) >= 2:
        cleaned = [_clean_sub_event(l) for l in lines if _clean_sub_event(l)]
        if len(cleaned) >= 2:
            return _adjust_count(cleaned, target_num_events, text)

    # 3. Đánh số phân đoạn (1. 2. 3.)
    numbered = _NUMBERED_PATTERN.split(text)
    if len(numbered) >= 2:
        cleaned = [_clean_sub_event(p) for p in numbered if _clean_sub_event(p)]
        # Nếu phần tử đầu tiên là preamble giới thiệu video trước E1/1., bỏ qua nếu số phần tử còn lại >= 2
        if target_num_events and len(cleaned) == target_num_events + 1:
            cleaned = cleaned[1:]
        if len(cleaned) >= 2:
            return _adjust_count(cleaned, target_num_events, text)

    # 4. Mũi tên / Chấm phẩy (->, =>, ;, |)
    for sep in [r"\s*-->\s*", r"\s*->\s*", r"\s*=>\s*", r"\s*;\s*", r"\s*\|\s*"]:
        parts = re.split(sep, text)
        if len(parts) >= 2:
            cleaned = [_clean_sub_event(p) for p in parts if _clean_sub_event(p)]
            if len(cleaned) >= 2:
                return _adjust_count(cleaned, target_num_events, text)

    # 5. Từ nối thời gian (Temporal Connectors)
    temp_parts = _TEMPORAL_SPLIT_REGEX.split(text)
    if len(temp_parts) >= 2:
        cleaned = [_clean_sub_event(p) for p in temp_parts if _clean_sub_event(p)]
        if len(cleaned) >= 2:
            return _adjust_count(cleaned, target_num_events, text)

    # 6. Sentence split theo dấu câu (.?!)
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
    # Loại bỏ nhãn E1:, E2:, Bước 1: nếu còn dính ở đầu câu
    cleaned = re.sub(r"^(?:E\d+|EV\d+|Event\s*\d+|Bước\s*\d+|Step\s*\d+|Phase\s*\d+)[\:\.\-\s]+", "", cleaned, flags=re.IGNORECASE)
    # Loại bỏ từ nối thời gian ở đầu câu
    cleaned = re.sub(r"^(?:then|sau đó|tiếp theo|next|after that|followed by|và|and|rồi)\s+", "", cleaned, flags=re.IGNORECASE)
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
