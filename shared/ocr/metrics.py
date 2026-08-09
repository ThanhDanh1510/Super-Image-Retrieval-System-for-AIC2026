"""OCR text metrics used by regression and slice evaluation."""

from __future__ import annotations

import re
from dataclasses import dataclass

from shared.ocr.normalization import normalize_text


@dataclass(frozen=True)
class OCRMetrics:
    cer: float
    wer: float
    exact: bool
    char_distance: int
    word_distance: int
    ref_chars: int
    ref_words: int


def levenshtein(seq_a: list[str] | str, seq_b: list[str] | str) -> int:
    """Compute Levenshtein edit distance with O(min(n, m)) memory."""
    if len(seq_a) < len(seq_b):
        seq_a, seq_b = seq_b, seq_a
    previous = list(range(len(seq_b) + 1))
    for i, token_a in enumerate(seq_a, start=1):
        current = [i]
        for j, token_b in enumerate(seq_b, start=1):
            cost = 0 if token_a == token_b else 1
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost))
        previous = current
    return previous[-1]


def score_ocr_text(prediction: str, reference: str, *, collapse_case: bool = False) -> OCRMetrics:
    """Return CER/WER/exact after deterministic text normalization."""
    pred = _canonicalize_for_metric(normalize_text(prediction, collapse_case=collapse_case))
    ref = _canonicalize_for_metric(normalize_text(reference, collapse_case=collapse_case))
    char_distance = levenshtein(pred, ref)
    pred_words = pred.split()
    ref_words = ref.split()
    word_distance = levenshtein(pred_words, ref_words)
    ref_chars = len(ref)
    ref_word_count = len(ref_words)
    return OCRMetrics(
        cer=char_distance / max(1, ref_chars),
        wer=word_distance / max(1, ref_word_count),
        exact=pred == ref,
        char_distance=char_distance,
        word_distance=word_distance,
        ref_chars=ref_chars,
        ref_words=ref_word_count,
    )


def _canonicalize_for_metric(text: str) -> str:
    """Ignore harmless OCR spacing differences around punctuation during scoring."""
    text = re.sub(r"\s+([,.:;!?])", r"\1", text)
    text = re.sub(r"([([])\s+", r"\1", text)
    text = re.sub(r"\s+([])])", r"\1", text)
    return text
