"""
tests/test_submission_format.py
================================
Kiểm thử tính hợp lệ của định dạng file nộp bài (CSV & ZIP)
theo 100% Quy chế Sơ tuyển AIC 2026 (https://sotuyenaic.oj.io.vn/rules/).
"""

import io
import zipfile
import csv
import pytest


def format_kis_csv(items: list[dict]) -> str:
    """Format KIS: <video_name>, <frame_id> (Tối đa 100 dòng, KHÔNG HEADER)."""
    safe = items[:100]
    lines = [f"{item['video_id']}, {item['frame_id']}" for item in safe]
    return "\r\n".join(lines) + ("\r\n" if lines else "")


def format_qa_csv(items: list[dict]) -> str:
    """Format Q&A: <video_name>, <frame_id>, \"<answer>\" (Answer <= 100 ký tự, KHÔNG HEADER)."""
    safe = items[:100]
    lines = []
    for item in safe:
        ans = (item.get("user_answer") or item.get("answer") or "").strip()
        if not ans:
            ans = "Chưa có đáp án"
        if len(ans) > 100:
            ans = ans[:100]
        escaped = ans.replace('"', '""')
        lines.append(f'{item["video_id"]}, {item["frame_id"]}, "{escaped}"')
    return "\r\n".join(lines) + ("\r\n" if lines else "")


def format_trake_csv(items: list[dict]) -> str:
    """Format TRAKE: <video_name>, <frame_1>, <frame_2>, ..., <frame_N> (KHÔNG HEADER)."""
    safe = items[:100]
    lines = []
    for item in safe:
        frames = item.get("event_frames") or [item["frame_id"]]
        frame_str = ", ".join(str(f) for f in frames)
        lines.append(f"{item['video_id']}, {frame_str}")
    return "\r\n".join(lines) + ("\r\n" if lines else "")


def package_submission_zip(query_files: dict[str, str]) -> bytes:
    """Đóng gói file .zip BẮT BUỘC chứa thư mục 'submission/' bên trong."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for qid, csv_content in query_files.items():
            filename = qid if qid.endswith(".csv") else f"{qid}.csv"
            # Quy chế BTC: BẮT BUỘC đặt trong thư mục submission/
            arcname = f"submission/{filename}"
            zf.writestr(arcname, csv_content.encode("utf-8-sig"))
    return buffer.getvalue()


# =============================================================================
# Test Suites
# =============================================================================

def test_kis_csv_has_no_header_and_valid_columns():
    items = [
        {"video_id": "L00_V000", "frame_id": 1234},
        {"video_id": "L00_V055", "frame_id": 5555},
        {"video_id": "L01_V028", "frame_id": 25300},
    ]
    csv_str = format_kis_csv(items)
    lines = [l.strip() for l in csv_str.strip().splitlines()]
    
    assert len(lines) == 3
    # KHÔNG được có header 'video_id,frame_id'
    assert "video_id" not in lines[0].lower()
    assert lines[0] == "L00_V000, 1234"
    assert lines[1] == "L00_V055, 5555"
    assert lines[2] == "L01_V028, 25300"


def test_kis_csv_limits_to_100_rows():
    items = [{"video_id": f"L01_V{i:03d}", "frame_id": i * 100} for i in range(150)]
    csv_str = format_kis_csv(items)
    lines = [l for l in csv_str.splitlines() if l.strip()]
    assert len(lines) == 100


def test_qa_csv_escapes_quotes_and_truncates():
    long_answer = "A" * 150
    items = [
        {"video_id": "L01_V028", "frame_id": 3450, "answer": "5"},
        {"video_id": "L02_V011", "frame_id": 1200, "answer": "Năm người, gồm nam và nữ"},
        {"video_id": "L03_V005", "frame_id": 2800, "answer": 'Anh ấy nói "Xin chào"'},
        {"video_id": "L04_V012", "frame_id": 4100, "answer": long_answer},
    ]
    csv_str = format_qa_csv(items)
    lines = [l.strip() for l in csv_str.strip().splitlines()]
    
    assert len(lines) == 4
    # No header
    assert "video_id" not in lines[0].lower()
    assert lines[0] == 'L01_V028, 3450, "5"'
    assert lines[1] == 'L02_V011, 1200, "Năm người, gồm nam và nữ"'
    assert lines[2] == 'L03_V005, 2800, "Anh ấy nói ""Xin chào"""'
    # Truncated to 100 chars
    reader = csv.reader(io.StringIO(csv_str), skipinitialspace=True)
    rows = list(reader)
    assert len(rows[3][2]) == 100


def test_trake_csv_formats_multiple_event_frames():
    items = [
        {"video_id": "L10_V001", "event_frames": [1200, 1850, 2100, 2450]},
        {"video_id": "L10_V001", "event_frames": [1180, 1820, 2080, 2420]},
        {"video_id": "L11_V003", "event_frames": [5100, 5700, 6200, 6800]},
    ]
    csv_str = format_trake_csv(items)
    lines = [l.strip() for l in csv_str.strip().splitlines()]
    
    assert len(lines) == 3
    assert lines[0] == "L10_V001, 1200, 1850, 2100, 2450"
    assert lines[1] == "L10_V001, 1180, 1820, 2080, 2420"
    assert lines[2] == "L11_V003, 5100, 5700, 6200, 6800"


def test_submission_zip_contains_submission_folder():
    query_files = {
        "query-1-kis": "L00_V000, 1234\r\nL00_V055, 5555\r\n",
        "query-2-kis": "L01_V028, 25300\r\n",
        "query-3-qa": 'L01_V028, 3450, "5"\r\n',
        "query-4-trake": "L10_V001, 1200, 1850, 2100, 2450\r\n",
    }
    zip_bytes = package_submission_zip(query_files)
    
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
        namelist = zf.namelist()
        assert len(namelist) == 4
        # BẮT BUỘC TẤT CẢ FILE PHẢI NẰM TRONG THƯ MỤC submission/
        for name in namelist:
            assert name.startswith("submission/"), f"File {name} không nằm trong thư mục submission/"
            assert name.endswith(".csv"), f"File {name} không có đuôi .csv"
        
        assert "submission/query-1-kis.csv" in namelist
        assert "submission/query-2-kis.csv" in namelist
        assert "submission/query-3-qa.csv" in namelist
        assert "submission/query-4-trake.csv" in namelist
