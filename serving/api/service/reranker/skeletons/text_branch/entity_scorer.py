"""
serving/api/service/reranker/skeletons/text_branch/entity_scorer.py
==================================================================
Generic Multi-Entity Coverage Scorer.
Tính toán độ bao phủ thực thể (Entity Recall / Coverage) một cách tổng quát
cho mọi truy vấn liệt kê nhiều đối tượng / thành phần trong KIS & TRAKE.
"""

from __future__ import annotations

import re
from typing import List, Set, Sequence, Any
from serving.api.schema.retrieval import CandidateResult


# Các stop-phrases & liên từ phổ biến dùng để phân tách thực thể
SEPARATORS = re.compile(
    r"[,;:\n\t\(\)\[\]]|(?:\s+(?:và|hoặc|cùng|kèm|bao gồm|gồm|and|or|with|accompanied by|such as|like)\s+)",
    re.IGNORECASE,
)

# Stop-phrases loại bỏ các câu dẫn chuyện / tiền tố không gian
NARRATIVE_STOP_PATTERNS = [
    r"cảnh(?:\s+quay)?(?:\s+liệt\s+kê)?(?:\s+các)?(?:\s+nguyên\s+liệu)?(?:\s+để\s+nấu)?(?:\s+một)?(?:\s+món\s+ăn)?",
    r"ảnh\s+nền(?:\s+bao\s+gồm)?",
    r"bảng\s+nguyên\s+liệu(?:\s+hiện\s+lên)?",
    r"scene(?:\s+lists)?(?:\s+ingredients)?(?:\s+to\s+cook)?(?:\s+a)?(?:\s+dish)?",
    r"background(?:\s+image)?(?:\s+includes)?",
    r"foreground(?:\s+includes)?",
    r"đặt\s+ở\s+góc(?:\s+bên)?(?:\s+trái|\s+phải|\s+dưới|\s+trên)?(?:\s+phía\s+dưới)?",
    r"đặt\s+phía\s+dưới(?:\s+chén\s+gia\s+vị)?",
    r"placed(?:\s+in|\s+at|\s+on|\s+below|\s+under)?(?:\s+the)?(?:\s+left|\s+right|\s+lower|\s+upper)?(?:\s+corner|\s+side|\s+bottom)?",
    r"below\s+the\s+spice\s+bowl",
    r"hũ\s+thủy\s+tinh(?:\s+nhỏ)?(?:\s+đựng)?",
    r"small\s+glass\s+jar(?:\s+containing)?",
    r"glass\s+jar(?:\s+containing)?",
    r"a\s+small\s+bottle\s+of",
    r"a\s+bunch\s+of",
    r"a\s+plate\s+of",
    r"a\s+packet\s+of",
    r"dĩa|chén|bát|hũ|gói|bó|tép|nhánh|lát|cây",
    r"plate|bowl|jar|packet|bunch|stalk|clove|slice|piece",
]

# Stop-words lọc bỏ các từ ngữ dẫn chuyện / bổ ngữ chung chung
GENERIC_QUERY_STOPWORDS = {
    "cảnh", "quay", "video", "hình", "ảnh", "bối", "không", "gian", "liệt", "kê", "các", "những",
    "nguyên", "liệu", "thành", "phần", "để", "nấu", "món", "ăn", "ở", "trong", "trên", "dưới", "bên", "góc",
    "trái", "phải", "giữa", "phía", "đặt", "được", "có", "này", "đó", "kia", "một", "hai", "ba", "bốn",
    "bảng", "hiện", "lên", "cụ", "thể", "dĩa", "chén", "bát", "hũ", "gói", "bó", "tép", "nhánh", "lát",
    "scene", "video", "image", "photo", "background", "foreground", "lists", "listing", "ingredients",
    "to", "cook", "dish", "in", "on", "at", "under", "beside", "left", "right", "corner", "placed",
    "is", "are", "includes", "including", "specific", "specified", "components", "items", "display",
    "plate", "bowl", "jar", "packet", "bunch", "clove", "stalk", "slice", "table", "counter", "containing",
    "contain", "contains", "small", "large", "fresh", "dried", "dry", "tươi", "khô", "xanh", "đỏ", "trắng"
}

# Bảng đồng nghĩa cơ bản mở rộng (Generic Multi-lingual Entity Synonyms)
SYNONYM_MAP = {
    "nam meo": ["moc nhi", "wood ear", "black fungus", "mushrooms", "mushroom", "fungus"],
    "mộc nhĩ": ["nam meo", "wood ear", "black fungus", "mushrooms", "mushroom"],
    "wood ear": ["mushrooms", "mushroom", "fungus", "nam meo", "moc nhi"],
    "wood ear mushrooms": ["mushrooms", "mushroom", "fungus", "nam meo", "moc nhi"],
    "mushrooms": ["nam meo", "moc nhi", "wood ear", "mushroom"],
    "nuoc cot dua": ["coconut milk", "coconut cream", "milk", "coconut"],
    "nước cốt dừa": ["coconut milk", "coconut cream", "milk", "coconut"],
    "coconut milk": ["nuoc cot dua", "milk", "coconut cream", "coconut"],
    "milk": ["nuoc cot dua", "coconut milk", "coconut cream"],
    "hat nem": ["seasoning", "seasoning packet", "seasoning seeds", "seasoning powder", "aji-ngon", "ajinomoto", "spices"],
    "hạt nêm": ["seasoning", "seasoning packet", "seasoning seeds", "seasoning powder", "aji-ngon", "ajinomoto", "spices"],
    "seasoning": ["hat nem", "seasoning packet", "seasoning seeds", "aji-ngon", "ajinomoto", "spices", "gia vi"],
    "seasoning seeds": ["hat nem", "seasoning", "seasoning packet", "aji-ngon", "ajinomoto", "spices", "gia vi"],
    "bot ca ri": ["curry powder", "ground spices", "spices", "curry", "powder", "seasoning"],
    "bột cà ri": ["curry powder", "ground spices", "spices", "curry", "powder", "seasoning"],
    "curry powder": ["bot ca ri", "ground spices", "spices", "curry", "powder"],
    "ground spices": ["bot ca ri", "curry powder", "spices", "curry", "powder"],
    "sa cay": ["sa", "lemongrass", "lemongrass stalks"],
    "sả cây": ["sả", "lemongrass", "lemongrass stalks"],
    "sa": ["lemongrass", "lemongrass stalks", "sa cay"],
    "sả": ["lemongrass", "lemongrass stalks", "sa cay"],
    "lemongrass": ["sa", "sa cay", "lemongrass stalks"],
    "lemongrass stalks": ["sa", "sa cay", "lemongrass"],
    "ot hiem": ["ot", "ot do", "chili", "red chili", "chili peppers", "peppers"],
    "ớt hiểm": ["ớt", "ớt đỏ", "chili", "red chili", "chili peppers", "peppers"],
    "chili": ["ot", "ot hiem", "red chili", "chili peppers", "peppers"],
    "red chili": ["ot", "ot hiem", "chili", "chili peppers", "peppers"],
    "red chili peppers": ["ot", "ot hiem", "chili", "red chili", "peppers"],
    "thit": ["meat", "pork", "beef", "chicken", "duck", "snails", "raw meat", "cooked meat"],
    "thịt": ["meat", "pork", "beef", "chicken", "duck", "snails", "raw meat", "cooked meat"],
    "meat": ["thit", "pork", "beef", "snails", "raw meat"],
    "la tuoi": ["green leaves", "leaves", "vegetables", "herbs", "fresh leaves"],
    "lá tươi": ["green leaves", "leaves", "vegetables", "herbs", "fresh leaves"],
    "green leaves": ["la tuoi", "leaves", "vegetables", "herbs", "fresh leaves"],
    "fresh green leaves": ["la tuoi", "green leaves", "leaves", "vegetables", "herbs"],
    "leaves": ["la", "la tuoi", "green leaves", "vegetables"],
}


def _clean_str(text: str) -> str:
    text = re.sub(r"[^\w\s]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def extract_entity_clauses(query: str, query_en: str = "") -> List[str]:
    """
    Trích xuất danh sách các cụm thực thể cốt lõi từ câu truy vấn.
    """
    combined_query = f"{query} , {query_en}" if query_en else query
    text = combined_query.lower()

    # Xóa các cụm dẫn chuyện / cấu trúc câu
    for pat in NARRATIVE_STOP_PATTERNS:
        text = re.sub(pat, " , ", text, flags=re.IGNORECASE)

    raw_clauses: List[str] = []
    for part in SEPARATORS.split(text):
        cleaned = _clean_str(part)
        words = [w for w in cleaned.split() if w not in GENERIC_QUERY_STOPWORDS and len(w) >= 2]
        if words:
            phrase = " ".join(words)
            if len(phrase) >= 2 and phrase not in raw_clauses:
                raw_clauses.append(phrase)

    filtered_entities: List[str] = []
    for c in raw_clauses:
        if not any(c != other and c in other for other in raw_clauses):
            filtered_entities.append(c)

    return filtered_entities


def _score_single_entity_list(entities: List[str], clean_doc: str) -> float:
    if not entities:
        return 1.0

    matched_count = 0
    for ent in entities:
        ent_matched = False
        # 1. Khớp nguyên cụm trực tiếp
        if ent in clean_doc:
            ent_matched = True
        else:
            # 2. Khớp từng từ quan trọng trong cụm
            ent_words = ent.split()
            if any(w in clean_doc for w in ent_words if len(w) >= 3):
                ent_matched = True
            else:
                # 3. Khớp qua bảng từ đồng nghĩa (synonyms)
                syns = SYNONYM_MAP.get(ent, [])
                for s in syns:
                    if s in clean_doc or any(sw in clean_doc for sw in s.split() if len(sw) >= 3):
                        ent_matched = True
                        break

        if ent_matched:
            matched_count += 1

    return matched_count / float(len(entities))


def compute_entity_coverage_scores(
    candidates: Sequence[CandidateResult],
    query_vi: str,
    query_en: str = "",
) -> List[float]:
    """
    Tính điểm Entity Coverage cho danh sách candidates:
    - Nếu query có ít hơn 3 thực thể: trả về [1.0] * N (không thiên lệch).
    - Nếu query có >= 3 thực thể: tính tỷ lệ % các thực thể xuất hiện trong Document payload (lấy max giữa VI và EN).
    """
    n = len(candidates)
    if n == 0:
        return []

    entities_vi = extract_entity_clauses(query_vi) if query_vi else []
    entities_en = extract_entity_clauses(query_en) if query_en else []

    max_ent_count = max(len(entities_vi), len(entities_en))
    if max_ent_count < 3:
        return [1.0] * n

    coverage_scores: List[float] = []

    for cand in candidates:
        desc = (getattr(cand, "description", "") or "").lower()
        ocr = (getattr(cand, "ocr_text", "") or "").lower()
        asr = (getattr(cand, "asr_text", "") or "").lower()
        tags = (getattr(cand, "tags", "") or "").lower() if hasattr(cand, "tags") else ""
        doc_text = f"{desc} {ocr} {asr} {tags}"
        clean_doc = _clean_str(doc_text)

        cov_vi = _score_single_entity_list(entities_vi, clean_doc) if len(entities_vi) >= 3 else 0.0
        cov_en = _score_single_entity_list(entities_en, clean_doc) if len(entities_en) >= 3 else 0.0

        best_cov = max(cov_vi, cov_en)
        coverage_scores.append(best_cov)

    return coverage_scores
