"""Line-level OCR quality helpers that can wrap any OCR backend."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Protocol

from shared.ocr.normalization import postprocess_lines


class TextRecognizer(Protocol):
    def __call__(self, image_path: str) -> str:
        ...


@dataclass(frozen=True)
class OCRLine:
    text: str
    confidence: float = 1.0
    bbox: list[int] | None = None
    source: str = "ocr"
    review_reason: str = ""


@dataclass(frozen=True)
class OCRQualityResult:
    text: str
    raw_text: str
    lines: list[OCRLine] = field(default_factory=list)
    edits: list[dict[str, str]] = field(default_factory=list)
    needs_review: bool = False
    review_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "raw_text": self.raw_text,
            "lines": [asdict(line) for line in self.lines],
            "edits": self.edits,
            "needs_review": self.needs_review,
            "review_reasons": self.review_reasons,
        }


def split_ocr_lines(raw_text: str) -> list[str]:
    """Split backend OCR text into lines while supporting comma-joined legacy output."""
    if not raw_text:
        return []
    if "\n" in raw_text:
        return [line.strip() for line in raw_text.splitlines() if line.strip()]
    return [part.strip() for part in raw_text.split(",") if part.strip()]


def assess_review_reasons(lines: list[OCRLine], *, low_confidence_threshold: float = 0.72) -> list[str]:
    reasons: list[str] = []
    if not lines:
        reasons.append("empty_ocr")
    if any(line.confidence < low_confidence_threshold for line in lines):
        reasons.append("low_confidence_line")
    if any(line.review_reason for line in lines):
        reasons.extend(sorted({line.review_reason for line in lines if line.review_reason}))
    return reasons


def build_quality_result(raw_text: str, *, line_confidence: float = 1.0) -> OCRQualityResult:
    """Convert raw backend text into corrected text plus auditable quality metadata."""
    raw_lines = split_ocr_lines(raw_text)
    corrected_lines, edits = postprocess_lines(raw_lines)
    lines = [OCRLine(text=line, confidence=line_confidence) for line in corrected_lines]
    reasons = assess_review_reasons(lines)
    return OCRQualityResult(
        text="\n".join(corrected_lines),
        raw_text=raw_text or "",
        lines=lines,
        edits=edits,
        needs_review=bool(reasons),
        review_reasons=reasons,
    )

