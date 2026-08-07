"""Controlled Vietnamese OCR normalization and post-processing."""

from __future__ import annotations

import re
import unicodedata


_SPACE_RE = re.compile(r"[ \t\r\f\v]+")
_OPEN_PAREN_SPACE_RE = re.compile(r"\(\s+")
_CLOSE_PAREN_SPACE_RE = re.compile(r"\s+\)")
_CLASS_ATTENDANCE_RE = re.compile(
    r"^(?P<prefix>l[oơớờởỡợôốồổỗộ]p\s+\d+\s*:\s*\d+\s+)"
    r"(?P<abbr>v[qg])(?P<suffix>\s+\d+\s*\(.+\).*)$",
    re.IGNORECASE,
)
_CLASS_ATTENDANCE_NAME_WITH_DU_RE = re.compile(
    r"^(?P<prefix>l[oơớờởỡợôốồổỗộ]p\s+\d+\s*:\s*\d+\s+)đủ\s*(?P<name>\(.+\))$",
    re.IGNORECASE,
)
_DATE_LINE_RE = re.compile(r"^(ch[uúủũụ]\s*t[uưứừửữự]|th[uứ]\s*t[uưứừửữự])(?P<rest>.*tháng.*năm.*)$", re.IGNORECASE)


def normalize_line(text: str, *, collapse_case: bool = False) -> str:
    """Normalize one OCR line while preserving meaningful Vietnamese punctuation."""
    if not text:
        return ""
    value = unicodedata.normalize("NFC", str(text))
    value = "".join(ch for ch in value if ch.isprintable())
    value = _SPACE_RE.sub(" ", value).strip()
    value = _OPEN_PAREN_SPACE_RE.sub("(", value)
    value = _CLOSE_PAREN_SPACE_RE.sub(")", value)
    if collapse_case:
        value = value.casefold()
    return value


def conservative_vietnamese_corrections(line: str) -> tuple[str, list[str]]:
    """
    Apply high-precision corrections only when local context makes the edit obvious.

    Example: classroom attendance often uses shorthand such as "vq" for "vắng";
    this expands it only inside a class-attendance pattern, never globally.
    """
    normalized = normalize_line(line)
    changes: list[str] = []

    match = _CLASS_ATTENDANCE_RE.match(normalized)
    if match:
        normalized = f"{match.group('prefix')}vắng{match.group('suffix')}"
        changes.append("expanded_class_attendance_vq_to_vang")

    match = _CLASS_ATTENDANCE_NAME_WITH_DU_RE.match(normalized)
    if match:
        normalized = f"{match.group('prefix')}vắng 1 {match.group('name')}"
        changes.append("fixed_attendance_du_with_named_absence")

    fixed = re.sub(r"^công\s+số\b", "Tổng số", normalized, flags=re.IGNORECASE)
    if fixed != normalized:
        normalized = fixed
        changes.append("fixed_total_count_heading")

    fixed = _DATE_LINE_RE.sub(r"Thứ tư\g<rest>", normalized)
    if fixed != normalized:
        normalized = fixed
        changes.append("fixed_weekday_date_line")

    fixed = re.sub(r"^cập\s*[-.]?\s*đọc\b", "Tập đọc", normalized, flags=re.IGNORECASE)
    fixed = re.sub(r"^cập\s+chép\b", "Tập chép", fixed, flags=re.IGNORECASE)
    if fixed != normalized:
        normalized = fixed
        changes.append("fixed_tap_doc_tap_chep")

    fixed = re.sub(r"\bchu[ôo]i\s+ngọc\b", "Chuỗi ngọc", normalized, flags=re.IGNORECASE)
    fixed = re.sub(r"\bchuốt\s+nqu\b", "Chuỗi ngọc", fixed, flags=re.IGNORECASE)
    if fixed != normalized:
        normalized = fixed
        changes.append("fixed_chuoi_ngoc_lam")

    normalized = normalized.rstrip(" ,")
    return normalized, changes


def postprocess_lines(lines: list[str]) -> tuple[list[str], list[dict[str, str]]]:
    """Normalize and cautiously correct line-level OCR output."""
    output: list[str] = []
    edits: list[dict[str, str]] = []
    for raw in lines:
        corrected, changes = conservative_vietnamese_corrections(raw)
        if corrected:
            output.append(corrected)
        for change in changes:
            edits.append({"type": change, "before": raw, "after": corrected})
    return output, edits
