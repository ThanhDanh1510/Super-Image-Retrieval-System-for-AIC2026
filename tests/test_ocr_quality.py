import json
import importlib

from shared.ocr.annotation import load_annotations
from shared.ocr.metrics import score_ocr_text
from shared.ocr.normalization import conservative_vietnamese_corrections, normalize_text
from shared.ocr.quality import build_quality_result


def test_contextual_vq_correction_only_for_class_attendance():
    corrected, edits = conservative_vietnamese_corrections("Lớp 3 : 6 vq 1 (Nghĩa)")

    assert corrected == "Lớp 3 : 6 vắng 1 (Nghĩa)"
    assert edits == ["expanded_class_attendance_vq_to_vang"]


def test_vq_not_corrected_without_attendance_context():
    corrected, edits = conservative_vietnamese_corrections("Mã vq đang nằm riêng")

    assert corrected == "Mã vq đang nằm riêng"
    assert edits == []


def test_quality_result_preserves_raw_and_reports_edits():
    quality = build_quality_result("Tổng số : 22\nLớp 3 : 6 vq 1 (Nghĩa)")

    assert quality.raw_text == "Tổng số : 22\nLớp 3 : 6 vq 1 (Nghĩa)"
    assert quality.text == "Tổng số : 22\nLớp 3 : 6 vắng 1 (Nghĩa)"
    assert quality.edits[0]["type"] == "expanded_class_attendance_vq_to_vang"


def test_score_ocr_text_normalizes_spacing_and_punctuation():
    pred = "Thứ tư , 14 tháng 10 năm 2020 .\nTập đọc ."
    ref = "Thứ tư, 14 tháng 10 năm 2020.\nTập đọc."

    metrics = score_ocr_text(pred, ref)

    assert metrics.exact is True
    assert metrics.cer == 0
    assert metrics.wer == 0


def test_load_annotation_fixture():
    annotations = load_annotations("data/ocr_annotations.json")

    assert annotations[0].image_id == "L30_V002_3762"
    assert "chalkboard" in annotations[0].tags
    assert "Chuỗi ngọc lam" in annotations[0].normalized_text
    assert annotations[0].normalized_text == normalize_text(annotations[0].raw_text)


def test_evaluate_annotations_cli_function(tmp_path):
    predictions = {
        "L30_V002_3762": (
            "Tổng số : 22\n"
            "Lớp 1 : 9 đủ\n"
            "Lớp 3 : 6 vắng 1 (Nghĩa)\n"
            "Lớp\n"
            "Thứ tư , 14 tháng 10 năm 2020 .\n"
            "Tập đọc .\n"
            "Chuỗi ngọc lam ( trang 134 ) .\n"
            "Tập chép\n"
            "Chuỗi ngọc lam ."
        )
    }
    pred_path = tmp_path / "predictions.json"
    pred_path.write_text(json.dumps(predictions, ensure_ascii=False), encoding="utf-8")

    evaluator = importlib.import_module("eval.ocr.evaluate_annotations")
    report = evaluator.evaluate("data/ocr_annotations.json", pred_path)

    assert report["total"] == 1
    assert report["exact_match"] == 1.0
    assert report["mean_cer"] == 0

