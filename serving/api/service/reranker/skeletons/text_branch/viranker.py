"""
serving/api/service/reranker/skeletons/text_branch/viranker.py
==============================================================
Nhánh rerank văn bản sử dụng ViRanker (BGE-M3 / MMARCO-VI tuned).
"""

from __future__ import annotations

import logging
from typing import Any, Optional
from serving.api.schema.retrieval import CandidateResult
from serving.api.service.reranker.skeletons.text_branch.entity_scorer import compute_entity_coverage_scores

logger = logging.getLogger(__name__)


def _slice_context_around_keywords(text: str, keywords: set[str], window_words: int = 120) -> str:
    """
    Cắt cửa sổ từ ngữ tập trung quanh các từ khóa quan trọng (Keyword-Anchored Context Slicing)
    để Cross-Encoder ViRanker tập trung 100% attention vào thực thể cần so khớp.
    """
    if not text or not text.strip():
        return ""
    words = text.split()
    if len(words) <= window_words or not keywords:
        return " ".join(words[:window_words])

    # Tìm vị trí đầu tiên của từ khóa trong văn bản
    match_idx = -1
    for i, w in enumerate(words):
        w_clean = w.lower().strip(".,?!:;()\"'“”[]{}")
        if w_clean in keywords:
            match_idx = i
            break

    if match_idx == -1:
        return " ".join(words[:window_words])

    half_w = window_words // 2
    start_idx = max(0, match_idx - half_w)
    end_idx = min(len(words), start_idx + window_words)
    return " ".join(words[start_idx:end_idx])


def _build_doc_text(c: CandidateResult, keywords: set[str] | None = None) -> str:
    desc_text = getattr(c, "description", "") or ""
    ocr_text = getattr(c, "ocr_text", "") or ""
    asr_text = getattr(c, "asr_text", "") or ""

    if keywords:
        desc_slice = _slice_context_around_keywords(desc_text, keywords, window_words=120)
        ocr_slice = _slice_context_around_keywords(ocr_text, keywords, window_words=50)
        asr_slice = _slice_context_around_keywords(asr_text, keywords, window_words=50)
        parts = []
        if ocr_slice:
            parts.append(f"OCR: {ocr_slice}")
        if desc_slice:
            parts.append(f"Visual: {desc_slice}")
        if asr_slice:
            parts.append(f"ASR: {asr_slice}")
        doc = " | ".join(parts).strip()
    else:
        doc = f"OCR: {ocr_text[:300]} | Visual: {desc_text[:600]} | ASR: {asr_text[:200]}".strip()

    if not doc:
        doc = "[EMPTY]"
    return doc


def _build_bilingual_query(query: str, enhanced_query: Optional[str] = None) -> str:
    clean_q = query.strip()
    clean_enh = (enhanced_query or "").strip()
    if clean_enh and clean_enh.lower() != clean_q.lower():
        return clean_enh
    elif len(clean_q.split()) <= 6:
        return f"{clean_q} {clean_q.title()}"
    return clean_q


def _extract_keywords(clean_q: str, clean_enh: Optional[str] = None) -> set[str]:
    keywords = set()
    for q_src in [clean_q, clean_enh]:
        if q_src:
            for w in q_src.lower().split():
                if len(w) >= 3 and w not in ("the", "and", "for", "with", "của", "cho", "với", "trong", "một", "các", "những"):
                    keywords.add(w)
    return keywords


def _score_pairs_chunked(
    pairs: list[list[str]],
    model,
    tokenizer,
    device: str = "cpu",
    batch_size: int = 40,
    max_length: int = 384,
) -> list[float]:
    if not pairs:
        return []
    import torch

    device_type = "cuda" if device == "cuda" else "cpu"
    all_scores: list[float] = []

    with torch.no_grad():
        for i in range(0, len(pairs), batch_size):
            chunk = pairs[i : i + batch_size]
            inputs = tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(model.device)

            with torch.amp.autocast(device_type=device_type, enabled=(device == "cuda")):
                outputs = model(**inputs)
                logits = outputs.logits
                if logits.dim() > 1:
                    logits = logits.squeeze(-1)
                scores_tensor = torch.sigmoid(logits)
                sc_list = scores_tensor.cpu().numpy().tolist()
                if isinstance(sc_list, float):
                    all_scores.append(sc_list)
                else:
                    all_scores.extend(sc_list)

    return all_scores


def warmup(model_bundle: Optional[dict[str, Any]] = None) -> None:
    """Chạy 1 lượt inference mẫu để compile CUDA kernels cho ViRanker."""
    if not model_bundle:
        return
    try:
        dummy_pairs = [["khởi động tìm kiếm", "mô tả video khởi động"]]
        _ = _score_pairs_chunked(
            dummy_pairs,
            model_bundle["model"],
            model_bundle["tokenizer"],
            device=model_bundle.get("device", "cpu"),
            batch_size=1,
            max_length=64,
        )
        logger.info("🔥 ViRanker warmup hoàn tất!")
    except Exception as e:
        logger.debug(f"ViRanker warmup: {e}")


def score(
    query: str,
    candidates: list[CandidateResult],
    model_bundle: Optional[dict[str, Any]] = None,
    enhanced_query: Optional[str] = None,
) -> list[float]:
    """
    Tính điểm tương đồng ngữ nghĩa giữa query và description + ocr_text + asr_text của từng candidate.
    Hỗ trợ query song ngữ (Bilingual Query) và cross-lingual keyword boost.
    Tối ưu single-pass batch 40 và max_length=384 để mô hình nhìn thấy trọn vẹn toàn bộ Document.
    """
    if not candidates:
        return []

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
        model = model_bundle["model"]
        tokenizer = model_bundle["tokenizer"]
        device = model_bundle.get("device", "cpu")

        clean_q = query.strip()
        clean_enh = (enhanced_query or "").strip()
        query_for_model = _build_bilingual_query(clean_q, clean_enh)
        keywords = _extract_keywords(clean_q, clean_enh)

        pairs = [[query_for_model, _build_doc_text(c, keywords=keywords)] for c in candidates]
        raw_scores = _score_pairs_chunked(
            pairs, model, tokenizer, device=device, batch_size=40, max_length=384
        )

        entity_coverages = compute_entity_coverage_scores(candidates, query_vi=clean_q, query_en=clean_enh)
        final_scores = []
        for sc, cov in zip(raw_scores, entity_coverages):
            if any(c != 1.0 for c in entity_coverages):
                blended = 0.50 * float(sc) + 0.50 * (float(sc) * float(cov))
            else:
                blended = float(sc)
            final_scores.append(blended)
        return final_scores
    except Exception as e:
        logger.error(f"Lỗi trong quá trình rerank ViRanker: {e}. Fallback score gốc.")
        return [float(c.score) for c in candidates]


def batch_score_events(
    event_queries: list[str],
    event_candidates_list: list[list[CandidateResult]],
    model_bundle: Optional[dict[str, Any]] = None,
    enhanced_queries: Optional[list[str]] = None,
    limit_per_event: int = 30,
    batch_size: int = 32,
    max_length: int = 384,
) -> list[list[CandidateResult]]:
    """
    Rerank thống nhất một lượt (Unified Batch Forward Pass) cho TOÀN BỘ sub-events trong TRAKE.
    Tránh hiện tượng 4 thread cùng ném data vào GPU gây nghẽn CUDA locks và tràn VRAM.
    """
    if not event_candidates_list or not event_queries:
        return event_candidates_list

    if model_bundle is None:
        return event_candidates_list

    try:
        model = model_bundle["model"]
        tokenizer = model_bundle["tokenizer"]
        device = model_bundle.get("device", "cpu")

        # 1. Gom tất cả cặp (query_i, doc_ij) của top limit_per_event ứng viên mỗi event
        all_pairs: list[list[str]] = []
        event_slice_indices: list[tuple[int, int]] = []
        tail_candidates: list[list[CandidateResult]] = []

        start_idx = 0
        for i, q in enumerate(event_queries):
            cands = event_candidates_list[i] if i < len(event_candidates_list) else []
            enh_q = enhanced_queries[i] if enhanced_queries and i < len(enhanced_queries) else None
            q_for_model = _build_bilingual_query(q, enh_q)
            keywords = _extract_keywords(q, enh_q)

            head_cands = cands[:limit_per_event] if len(cands) > limit_per_event else cands
            tail_cands = cands[limit_per_event:] if len(cands) > limit_per_event else []
            tail_candidates.append(tail_cands)

            for c in head_cands:
                all_pairs.append([q_for_model, _build_doc_text(c, keywords=keywords)])

            end_idx = start_idx + len(head_cands)
            event_slice_indices.append((start_idx, end_idx))
            start_idx = end_idx

        if not all_pairs:
            return event_candidates_list

        # 2. Forward pass theo micro-batch 32 trên một CUDA stream duy nhất
        all_raw_scores = _score_pairs_chunked(
            all_pairs, model, tokenizer, device=device, batch_size=batch_size, max_length=max_length
        )

        # 3. Phân rã điểm số ngược lại cho từng sub-event và áp dụng keyword boost
        output_event_candidates: list[list[CandidateResult]] = []
        for i, (s_idx, e_idx) in enumerate(event_slice_indices):
            sub_cands = event_candidates_list[i][:limit_per_event]
            sub_scores = all_raw_scores[s_idx:e_idx]

            q = event_queries[i]
            enh_q = enhanced_queries[i] if enhanced_queries and i < len(enhanced_queries) else None
            keywords = _extract_keywords(q, enh_q)

            reranked_head: list[CandidateResult] = []
            for j, cand in enumerate(sub_cands):
                sc = float(sub_scores[j]) if j < len(sub_scores) else float(cand.score)
                ocr = (getattr(cand, "ocr_text", "") or "").lower()
                desc = (getattr(cand, "description", "") or "").lower()
                asr = (getattr(cand, "asr_text", "") or "").lower()
                combined_doc = f"{ocr} {desc} {asr}"

                if keywords and any(kw in combined_doc for kw in keywords):
                    sc = min(sc + 0.35, 1.0)

                # Clone candidate với score mới
                new_cand = cand.model_copy(update={"score": sc})
                reranked_head.append(new_cand)

            # Sort giảm dần theo điểm số mới
            reranked_head.sort(key=lambda c: c.score, reverse=True)
            output_event_candidates.append(reranked_head + tail_candidates[i])

        return output_event_candidates
    except Exception as e:
        logger.error(f"Lỗi trong batch_score_events ViRanker: {e}. Giữ nguyên candidates gốc.")
        return event_candidates_list
