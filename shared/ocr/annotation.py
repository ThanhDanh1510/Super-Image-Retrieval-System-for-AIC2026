"""Annotation schema helpers for OCR ground truth."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from shared.ocr.normalization import normalize_text


@dataclass(frozen=True)
class OCRRegion:
    region_id: str
    text: str
    bbox: list[int] | None = None
    region_type: str = "unknown"
    visibility: str = "clear"
    notes: str = ""


@dataclass(frozen=True)
class OCRAnnotation:
    image_path: str
    image_id: str
    split: str = "regression"
    tags: list[str] = field(default_factory=list)
    regions: list[OCRRegion] = field(default_factory=list)
    raw_text: str = ""
    normalized_text: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OCRAnnotation":
        regions = [OCRRegion(**region) for region in data.get("regions", [])]
        raw_text = data.get("raw_text") or "\n".join(region.text for region in regions)
        normalized_text = data.get("normalized_text") or normalize_text(raw_text)
        return cls(
            image_path=str(data["image_path"]),
            image_id=str(data.get("image_id") or Path(data["image_path"]).stem),
            split=str(data.get("split", "regression")),
            tags=list(data.get("tags", [])),
            regions=regions,
            raw_text=raw_text,
            normalized_text=normalized_text,
        )


def load_annotations(path: str | Path) -> list[OCRAnnotation]:
    """Load OCR annotations from a JSON list file."""
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict) and "annotations" in payload:
        payload = payload["annotations"]
    return [OCRAnnotation.from_dict(item) for item in payload]

