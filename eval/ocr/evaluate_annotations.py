"""Evaluate OCR predictions against annotated ground truth JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from shared.ocr.annotation import load_annotations
from shared.ocr.metrics import score_ocr_text
from shared.ocr.normalization import normalize_text


def _load_predictions(path: str | Path) -> dict[str, str]:
    pred_path = Path(path)
    if pred_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(pred_path)
        id_col = "image_id" if "image_id" in df.columns else "frame_id"
        text_col = "text" if "text" in df.columns else "ocr_text"
        return {str(row[id_col]): str(row[text_col] or "") for _, row in df.iterrows()}
    with open(pred_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict):
        return {str(key): str(value) for key, value in payload.items()}
    raise ValueError(f"Unsupported prediction format: {pred_path}")


def evaluate(annotation_path: str | Path, prediction_path: str | Path) -> dict:
    annotations = load_annotations(annotation_path)
    predictions = _load_predictions(prediction_path)
    rows = []
    for ann in annotations:
        image_key = Path(ann.image_path).stem
        pred = predictions.get(ann.image_id, predictions.get(image_key, ""))
        metrics = score_ocr_text(pred, ann.normalized_text)
        rows.append({
            "image_id": ann.image_id,
            "tags": ann.tags,
            "prediction": normalize_text(pred),
            "reference": ann.normalized_text,
            "cer": metrics.cer,
            "wer": metrics.wer,
            "exact": metrics.exact,
            "char_distance": metrics.char_distance,
            "word_distance": metrics.word_distance,
        })

    total = len(rows)
    return {
        "total": total,
        "exact_match": sum(1 for row in rows if row["exact"]) / max(1, total),
        "mean_cer": sum(row["cer"] for row in rows) / max(1, total),
        "mean_wer": sum(row["wer"] for row in rows) / max(1, total),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate OCR predictions against annotations.")
    parser.add_argument("--annotations", required=True, help="Path to annotation JSON.")
    parser.add_argument("--predictions", required=True, help="Path to OCR JSON/parquet predictions.")
    parser.add_argument("--output", default=None, help="Optional JSON report path.")
    args = parser.parse_args()

    report = evaluate(args.annotations, args.predictions)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

