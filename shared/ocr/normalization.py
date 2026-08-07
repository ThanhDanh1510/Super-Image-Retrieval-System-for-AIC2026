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


def normalize_text(text: str, *, collapse_case: bool = False) -> str:
    """Normalize multiline OCR text for metric comparison and retrieval indexing."""
    if not text:
        return ""
    lines = [normalize_line(line, collapse_case=collapse_case) for line in str(text).splitlines()]
    return "\n".join(line for line in lines if line)


def conservative_vietnamese_corrections(line: str) -> tuple[str, list[str]]:
    """
    Apply high-precision corrections only when local context makes the edit obvious.

    Example: classroom attendance often uses a teacher shorthand such as "vq" for
    "vắng"; we only expand it inside a class-attendance pattern, never globally.
    """
    normalized = normalize_line(line)
    changes: list[str] = []

    match = _CLASS_ATTENDANCE_RE.match(normalized)
    if match:
        normalized = (
            f"{match.group('prefix')}vắng{match.group('suffix')}"
        )
        changes.append("expanded_class_attendance_vq_to_vang")

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
