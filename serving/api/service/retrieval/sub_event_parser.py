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

# Regex tìm các nhãn sự kiện rõ ràng (E1, E2, Cảnh 1, Cảnh 2, Step 1, Bước 1...)
_EXPLICIT_MARKER_REGEX = re.compile(
    r"(?:^|\s+)(?:Cảnh\s*\d+[\.\:\-]?|E\d+[\.\:\-]?|EV\d+[\.\:\-]?|Event\s*\d+[\.\:\-]?|Bước\s*\d+[\.\:\-]?|Step\s*\d+[\.\:\-]?|Phase\s*\d+[\.\:\-]?|\b\d+[\.\)\:\-]\s*)\s*",
    re.IGNORECASE,
)

# Regex cho các mẫu đánh số phân đoạn
_NUMBERED_PATTERN = re.compile(
    r"(?:^|\s+)(?:(?:(?:\d+|[a-zA-Z])[\.\)\:\-])|(?:\[\d+\])|(?:Bước\s+\d+\:?)|(?:Step\s+\d+\:?)|(?:Event\s+\d+\:?)|(?:E\d+[\.\)\:\-]?))\s+",
    re.IGNORECASE,
)

# Regex cho từ nối thời gian tiếng Việt & tiếng Anh
_TEMPORAL_MARKERS = [
    r"(?<!khoảnh khắc\s)(?<!lần\s)(?<!thời điểm\s)(?<!thời\s)đầu tiên", r"trước tiên", r"trước hết", r"ban đầu",
    r"sau đó", r"tiếp theo", r"tiếp tục", r"kế tiếp", r"liền sau đó", r"ngay sau đó", r"tiếp đó",
    r"cuối cùng", r"sau cùng", r"kết thúc", r"cuối cuộc", r"kết lại",
    r"(?<!the\s)first", r"initially", r"firstly",
    r"then", r"next", r"after that", r"afterwards", r"subsequently",
    r"finally", r"lastly", r"at last", r"in the end",
]
_TEMPORAL_SPLIT_REGEX = re.compile(r"(?:\b(?:" + "|".join(_TEMPORAL_MARKERS) + r")\b)\s*[:,\\-]?\s*", re.IGNORECASE)


def extract_preamble_and_sub_events(query: str, target_num_events: int | None = None) -> tuple[str | None, list[str]]:
    """
    Phân tách query thành preamble (nội dung mô tả tổng quan bối cảnh / đoạn mở đầu video trước E1)
    và danh sách các sub-events độc lập theo thứ tự thời gian.

    Returns:
        tuple[str | None, list[str]]: (preamble_text, sub_events_list)
    """
    if not query or not query.strip():
        return None, []

    text = query.strip()
    preamble = None

    # 1. Khớp theo Explicit Event Markers (E1, E2, E3, Event 1, Bước 1...)
    matches = list(_EXPLICIT_MARKER_REGEX.finditer(text))
    if len(matches) >= 2:
        if matches[0].start() > 0:
            preamble_cand = text[:matches[0].start()].strip()
            if len(preamble_cand) > 5:
                preamble = preamble_cand

        extracted = []
        for i in range(len(matches)):
            start = matches[i].end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            seg = _clean_sub_event(text[start:end])
            if seg:
                extracted.append(seg)
        if len(extracted) >= 2:
            return preamble, _adjust_count(extracted, target_num_events, text)

    # 2. Xuống dòng (Newlines)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) >= 2:
        cleaned = [_clean_sub_event(l) for l in lines if _clean_sub_event(l)]
        if len(cleaned) >= 2:
            return None, _adjust_count(cleaned, target_num_events, text)

    # 3. Đánh số phân đoạn (1. 2. 3.)
    numbered = _NUMBERED_PATTERN.split(text)
    if len(numbered) >= 2:
        cleaned = [_clean_sub_event(p) for p in numbered if _clean_sub_event(p)]
        if target_num_events and len(cleaned) == target_num_events + 1:
            preamble = cleaned[0]
            cleaned = cleaned[1:]
        if len(cleaned) >= 2:
            return preamble, _adjust_count(cleaned, target_num_events, text)

    # 4. Mũi tên / Chấm phẩy (->, =>, ;, |)
    for sep in [r"\s*-->\s*", r"\s*->\s*", r"\s*=>\s*", r"\s*;\s*", r"\s*\|\s*"]:
        parts = re.split(sep, text)
        if len(parts) >= 2:
            cleaned = [_clean_sub_event(p) for p in parts if _clean_sub_event(p)]
            if len(cleaned) >= 2:
                return None, _adjust_count(cleaned, target_num_events, text)

    # 5. Sentence split theo dấu câu (.?!) — Ưu tiên trước khi cắt bằng từ nối giữa câu
    sents = [s.strip() for s in re.split(r"[.?!]\s+", text) if s.strip()]
    if len(sents) >= 2:
        cleaned = [_clean_sub_event(p) for p in sents if _clean_sub_event(p)]
        if target_num_events and len(cleaned) == target_num_events + 1:
            preamble = cleaned[0]
            cleaned = cleaned[1:]
        if len(cleaned) >= 2:
            return preamble, _adjust_count(cleaned, target_num_events, text)

    # 6. Từ nối thời gian (Temporal Connectors)
    temp_parts = _TEMPORAL_SPLIT_REGEX.split(text)
    if len(temp_parts) >= 2:
        cleaned = [_clean_sub_event(p) for p in temp_parts if _clean_sub_event(p)]
        if len(cleaned) >= 2:
            return None, _adjust_count(cleaned, target_num_events, text)

    single = _clean_sub_event(text)
    if target_num_events and target_num_events > 1:
        return None, [single] * target_num_events
    return None, [single]


def parse_sub_events(query: str, target_num_events: int | None = None) -> list[str]:
    """
    Phân tách query thành danh sách các sub-events độc lập theo thứ tự thời gian.
    """
    _, events = extract_preamble_and_sub_events(query, target_num_events)
    return events


def _clean_sub_event(s: str) -> str:
    cleaned = re.sub(r"^[,\-\:\.\s]+", "", s)
    cleaned = re.sub(r"[,\-\:\.\s]+$", "", cleaned)
    # Loại bỏ nhãn E1:, E2:, Bước 1: nếu còn dính ở đầu câu
    cleaned = re.sub(r"^(?:E\d+|EV\d+|Event\s*\d+|Bước\s*\d+|Step\s*\d+|Phase\s*\d+)[\:\.\-\s]+", "", cleaned, flags=re.IGNORECASE)
    # Loại bỏ từ nối thời gian ở đầu câu
    cleaned = re.sub(r"^(?:then|sau đó là|sau đó|tiếp theo là|tiếp theo|cuối cùng là|cuối cùng|next|after that|followed by|và|and|rồi)\s+", "", cleaned, flags=re.IGNORECASE)
    # Bỏ các vụn vặt như "là khoảnh khắc", "là"
    cleaned = re.sub(r"^(?:là\s+khoảnh\s+khắc|là\s+cảnh|là\s+đoạn|là)\s+", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.strip()
    if len(cleaned.split()) < 3:
        return ""
    return cleaned


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


def extract_temporal_cues(sub_events: list[str], raw_query: str = "") -> list[dict[str, any]]:
    """
    Trích xuất temporal cues từ các từ nối giữa các sub-events liên tiếp.
    Do constraint thực tế của dữ liệu AIC2026, khoảng cách giữa 2 sự kiện liên tiếp <= 15s.

    Phân chia thành 3 mức:
      - Fast transition ("ngay sau đó", "liền sau", "vừa ... đã", "lập tức", "immediately", "right after"):
        expected_gap_sec = 2.0s, min_separation_sec = 0.5s
      - Medium transition ("sau đó", "tiếp theo", "tiếp tục", "kế tiếp", "then", "next", "after that"):
        expected_gap_sec = 5.0s, min_separation_sec = 1.0s
      - Long transition ("một lúc sau", "sau một hồi", "lát sau", "cuối cùng", "later", "finally"):
        expected_gap_sec = 10.0s, min_separation_sec = 3.0s

    Returns:
        list[dict]: Danh sách N-1 dict chứa {'expected_gap_sec': float, 'min_separation_sec': float, 'cue_type': str}
    """
    if len(sub_events) < 2:
        return []

    cues: list[dict[str, any]] = []

    fast_patterns = re.compile(
        r"(?:ngay sau đó|liền sau đó|liền sau|vừa\s+\w+\s+đã|ngay lập tức|lập tức|tức thì|ngay khi|right after|immediately|shortly after|as soon as)",
        re.IGNORECASE,
    )
    long_patterns = re.compile(
        r"(?:một lúc sau|sau một hồi|một lát sau|lát sau|hồi lâu sau|cuối cùng|sau cùng|kết thúc|kết lại|much later|a while later|finally|eventually|at last)",
        re.IGNORECASE,
    )
    medium_patterns = re.compile(
        r"(?:sau đó|tiếp theo|tiếp tục|kế tiếp|tiếp đó|đến lượt|theo sau|then|next|after that|afterwards|subsequently|followed by)",
        re.IGNORECASE,
    )

    query_lower = raw_query.lower() if raw_query else ""

    for i in range(len(sub_events) - 1):
        curr_ev = sub_events[i]
        next_ev = sub_events[i + 1]
        context_window = next_ev[:50].lower()

        idx_curr = query_lower.find(curr_ev[:20].lower())
        idx_next = query_lower.find(next_ev[:20].lower())
        between_text = ""
        if 0 <= idx_curr < idx_next:
            between_text = query_lower[idx_curr + len(curr_ev[:20]):idx_next]

        combined_text = f"{between_text} {context_window}"

        if fast_patterns.search(combined_text):
            cues.append({
                "expected_gap_sec": 2.0,
                "min_separation_sec": 0.5,
                "cue_type": "fast",
            })
        elif long_patterns.search(combined_text):
            cues.append({
                "expected_gap_sec": 10.0,
                "min_separation_sec": 3.0,
                "cue_type": "long",
            })
        elif medium_patterns.search(combined_text):
            cues.append({
                "expected_gap_sec": 5.0,
                "min_separation_sec": 1.0,
                "cue_type": "medium",
            })
    return cues


def extract_clean_theme(preamble: str | None) -> str:
    """
    Trích xuất chủ đề cốt lõi (Global Context Theme) từ Preamble,
    lọc bỏ các từ đệm cấu trúc mô tả đề bài / chuyển đoạn.

    Ví dụ:
      - "Video mở đầu bằng flycam bãi biển bình minh. Sau đó là chuỗi 3 sự kiện:"
        -> "flycam bãi biển bình minh"
      - "Video về một khu vườn cây ăn trái ở miền Tây Nam Bộ. Đây là chuỗi liên tiếp các cảnh quay về 4 loại trái cây trong vườn."
        -> "khu vườn cây ăn trái ở miền Tây Nam Bộ"
      - "Cảnh phim lần lượt giới thiệu các nguyên liệu của món ăn qua 3 chuyển cảnh:"
        -> "nguyên liệu của món ăn"
    """
    if not preamble or not preamble.strip():
        return ""

    text = preamble.strip()

    # Bỏ các cụm cấu trúc mở đầu / chuyển đoạn
    patterns_to_remove = [
        r"^(?:video\s+mở\s+đầu\s+bằng|video\s+bắt\s+đầu\s+bằng|đoạn\s+đầu\s+video\s+(?:là|giới\s+thiệu)|mở\s+đầu\s+video\s+(?:là|bằng)|cảnh\s+đầu\s+video\s+(?:là|bằng)|video\s+về|clip\s+về|phim\s+về|cảnh\s+phim\s+(?:về|lần\s+lượt\s+giới\s+thiệu)|hình\s+ảnh\s+về)\s*",
        r"(?:sau\s+đó\s+là\s+chuỗi.*|đây\s+là\s+chuỗi.*|tiếp\s+theo\s+là\s+chuỗi.*|lần\s+lượt\s+là\s+các\s+cảnh.*|qua\s+\d+\s+chuyển\s+cảnh.*|diễn\s+ra\s+qua\s+\d+\s+sự\s+kiện.*)$",
    ]

    for pat in patterns_to_remove:
        text = re.sub(pat, "", text, flags=re.IGNORECASE).strip()

    # Làm sạch dấu câu ở đầu/cuối
    text = re.sub(r"^[\s\:\.\,\;\-]+|[\s\:\.\,\;\-]+$", "", text).strip()
    return text if text else (preamble.strip())

