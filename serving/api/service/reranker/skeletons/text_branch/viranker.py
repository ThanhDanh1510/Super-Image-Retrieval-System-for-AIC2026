"""
serving/api/service/reranker/skeletons/text_branch/viranker.py
==============================================================
Nhánh rerank văn bản sử dụng ViRanker (BGE-M3 / MMARCO-VI tuned).
"""

from __future__ import annotations

import logging
from typing import Any, Optional
from serving.api.schema.retrieval import CandidateResult

logger = logging.getLogger(__name__)


def score(
    query: str,
    candidates: list[CandidateResult],
    model_bundle: Optional[dict[str, Any]] = None,
) -> list[float]:
    """
    Tính điểm tương đồng ngữ nghĩa giữa query và description + ocr_text + asr_text của từng candidate.

    Args:
        query: Query text tiếng Anh hoặc tiếng Việt đã chuẩn hóa.
        candidates: Danh sách CandidateResult cần rerank.
        model_bundle: Dict chứa model, tokenizer, device (từ ModelRegistry).

    Returns:
        List[float]: Danh sách điểm số (cùng kích thước và thứ tự với candidates).
    """
    if not candidates:
        return []

    # Nếu không truyền model_bundle, nạp standalone
    if model_bundle is None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            model_name = "BAAI/bge-reranker-v2-m3"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map=device,
            )
            model.eval()
            model_bundle = {"model": model, "tokenizer": tokenizer, "device": device}
        except Exception as e:
            logger.error(f"Không thể load model ViRanker: {e}. Fallback score gốc.")
            return [float(c.score) for c in candidates]

    try:
        import torch

        model = model_bundle["model"]
        tokenizer = model_bundle["tokenizer"]
        device = model_bundle.get("device", "cpu")

        # Chuẩn bị query đa dạng hoa/thường cho cross-encoder
        words = query.strip().split()
        if len(words) <= 6:
            query_for_model = f"{query} {query.title()}"
        else:
            query_for_model = query

        # Chuẩn bị cặp (query, doc_text) bao gồm OCR + Description + ASR
        # Ưu tiên OCR ở đầu để không bị cắt xén (truncate) bởi văn bản mô tả dài
        pairs = []
        for c in candidates:
            desc_text = getattr(c, "description", "") or ""
            ocr_text = getattr(c, "ocr_text", "") or ""
            asr_text = getattr(c, "asr_text", "") or ""
            doc = f"Văn bản/Chữ trên hình: {ocr_text} | Mô tả hình ảnh: {desc_text} | Lời thoại âm thanh: {asr_text}".strip()
            if not doc:
                doc = "[EMPTY]"
            if len(doc) > 2000:
                doc = doc[:2000]
            pairs.append([query_for_model, doc])

        inputs = tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(model.device)

        device_type = "cuda" if device == "cuda" else "cpu"
        with torch.no_grad(), torch.amp.autocast(device_type=device_type, enabled=(device == "cuda")):
            outputs = model(**inputs)
            logits = outputs.logits
            if logits.dim() > 1:
                logits = logits.squeeze(-1)
            scores_tensor = torch.sigmoid(logits)
            scores = scores_tensor.cpu().numpy().tolist()

        if isinstance(scores, float):
            scores = [scores]

        q_lower = query.lower().strip()
        final_scores = []
        for i, s in enumerate(scores):
            sc = float(s)
            c = candidates[i]
            ocr = (getattr(c, "ocr_text", "") or "").lower()
            desc = (getattr(c, "description", "") or "").lower()
            if q_lower and (q_lower in ocr or q_lower in desc):
                sc = min(sc + 0.35, 1.0)
            final_scores.append(sc)

        return final_scores
    except Exception as e:
        logger.error(f"Lỗi trong quá trình rerank ViRanker: {e}. Fallback score gốc.")
        return [float(c.score) for c in candidates]
