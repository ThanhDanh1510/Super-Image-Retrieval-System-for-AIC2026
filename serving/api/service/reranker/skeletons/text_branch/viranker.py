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
            from shared.utils.device_utils import resolve_torch_device

            model_name = "BAAI/bge-reranker-v2-m3"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            device = resolve_torch_device()
            dtype = torch.float16 if device == "cuda" else torch.float32
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map=device if device == "cuda" else None,
            )
            if device in ("cuda", "mps"):
                model.to(device)
            model.eval()
            model_bundle = {"model": model, "tokenizer": tokenizer, "device": device}
        except Exception as e:
            logger.error(f"Không thể load model ViRanker: {e}. Fallback score gốc.")
            return [float(c.score) for c in candidates]

    try:
        import torch
        from shared.utils.device_utils import resolve_torch_device

        model = model_bundle["model"]
        tokenizer = model_bundle["tokenizer"]
        device = resolve_torch_device(model_bundle.get("device", "cpu"))

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

        batch_size = 50  # Single batch cho toàn bộ candidates → 1 kernel launch, giảm synchronization overhead
        all_scores: list[float] = []
        device_type = "cuda" if device == "cuda" else ("mps" if device == "mps" else "cpu")

        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i : i + batch_size]
            inputs = tokenizer(
                batch_pairs,
                padding=True,
                truncation=True,
                max_length=384,
                return_tensors="pt",
            ).to(model.device)

            with torch.no_grad(), torch.amp.autocast(device_type=device_type, enabled=(device in ("cuda", "mps"))):
                outputs = model(**inputs)
                logits = outputs.logits
                if logits.dim() > 1:
                    logits = logits.squeeze(-1)
                scores_tensor = torch.sigmoid(logits)
                batch_scores = scores_tensor.cpu().numpy().tolist()

            if isinstance(batch_scores, (float, int)):
                batch_scores = [batch_scores]
            all_scores.extend([float(s) for s in batch_scores])

        return all_scores
    except Exception as e:
        logger.error(f"Lỗi trong quá trình rerank ViRanker: {e}. Fallback score gốc.")
        return [float(c.score) for c in candidates]
