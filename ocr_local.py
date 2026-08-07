"""
Vietnamese OCR — PaddleOCR (Detection) + VietOCR (Recognition), Local

Kết hợp:
  - PaddleOCR PP-OCRv6 TextDetection: định vị vùng chứa text (bounding box)
  - VietOCR (vgg_transformer): nhận diện nội dung text tiếng Việt trên từng vùng đã cắt

Lý do kết hợp: PaddleOCR end-to-end (detect+recognize) dùng charset Latin chung,
không chuyên biệt cho dấu thanh tiếng Việt -> hay mất dấu, sai ký tự dù ảnh rõ nét.
VietOCR được huấn luyện chuyên cho tiếng Việt nên nhận diện chính xác hơn nhiều,
nhưng cần một bộ detection riêng vì VietOCR chỉ đọc 1 dòng text đã cắt sẵn.

Cấu trúc input (phẳng, mỗi folder = 1 video, chứa .jpg trực tiếp):
    unified_keyframes/
      ├─ L22_V001/*.jpg
      ├─ L22_V002/*.jpg
      └─ ...

PREFIX dùng để lọc các folder cần xử lý theo tiền tố (vd "L21" -> chỉ xử lý
L21_V001, L21_V002, ...). Kết quả được lưu vào OUTPUT_DIR/<PREFIX>/<folder>.json|.parquet

Cách dùng:
    python ocr_local.py /path/to/unified_keyframes /path/to/output L21

Hoặc sửa DATASET_DIR, OUTPUT_DIR, PREFIX dưới đây rồi chạy:
    python ocr_local.py

Cần cài: pip install paddleocr paddlepaddle pyarrow tqdm opencv-python pillow
         pip install vietocr torch torchvision einops
"""

import os
import sys
import re

# Import PyTorch first to initialize CUDA/DLLs on Windows
try:
    import torch
except ImportError:
    pass

# Đảm bảo stdout/stderr dùng UTF-8 trên Windows (tránh UnicodeEncodeError với tiếng Việt)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
import json
import time
from pathlib import Path
from tqdm import tqdm
import argparse

import numpy as np
if not hasattr(np, 'sctypes'):
    np.sctypes = {
        'int': [np.int8, np.int16, np.int32, np.int64],
        'uint': [np.uint8, np.uint16, np.uint32, np.uint64],
        'float': [np.float16, np.float32, np.float64],
        'complex': [np.complex64, np.complex128],
        'others': [bool, bytes, str, object]
    }
import cv2
from PIL import Image

from shared.ocr.quality import OCRQualityResult, build_quality_result

# Nếu có sẵn bản PaddleOCR / vietocr local trong my-research/, ưu tiên dùng bản đó
_RESEARCH_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_LOCAL_PADDLEOCR_DIR = os.path.join(_RESEARCH_DIR, "PaddleOCR")
_LOCAL_VIETOCR_DIR = os.path.join(_RESEARCH_DIR, "vietocr")
if os.path.isdir(os.path.join(_LOCAL_PADDLEOCR_DIR, "paddleocr")):
    sys.path.insert(0, _LOCAL_PADDLEOCR_DIR)
if os.path.isdir(os.path.join(_LOCAL_VIETOCR_DIR, "vietocr")):
    sys.path.insert(0, _LOCAL_VIETOCR_DIR)

try:
    from paddleocr import TextDetection
except ImportError:
    try:
        from paddleocr import PaddleOCR
        class TextDetection:
            def __init__(self, model_name=None, enable_mkldnn=False, box_thresh=0.55, limit_side_len=1280):
                self.ocr = PaddleOCR(use_angle_cls=False, lang="latin", det=True, rec=False, show_log=False)
            def predict(self, img_path):
                res = self.ocr.ocr(str(img_path), det=True, rec=False)
                if not res or not res[0]:
                    return []
                items = []
                for poly in res[0]:
                    poly_arr = np.array(poly, dtype=np.float32)
                    poly_arr = _order_poly_points(poly_arr)
                    x1, y1 = int(poly_arr[:, 0].min()), int(poly_arr[:, 1].min())
                    x2, y2 = int(poly_arr[:, 0].max()), int(poly_arr[:, 1].max())
                    items.append({"poly": poly_arr, "box": [[x1, y1], [x2, y2]]})
                return items
    except ImportError:
        print("Cần cài paddleocr: pip install paddleocr paddlepaddle")
        sys.exit(1)

try:
    from vietocr.tool.predictor import Predictor
    from vietocr.tool.config import Cfg
except ImportError:
    print("Cần cài vietocr: pip install vietocr torch torchvision einops")
    print(f"(Đã thử dùng bản local tại {_LOCAL_VIETOCR_DIR} nếu tồn tại)")
    sys.exit(1)

# ============================================================
# CẤU HÌNH
# ============================================================

# Thư mục chứa tất cả các folder video (L21_V001, L21_V002, ...)
DEFAULT_DATASET = os.path.abspath(r"data/image/L21/L21")
DATASET_DIR = DEFAULT_DATASET if os.path.exists(DEFAULT_DATASET) else r"data/image"

# Thư mục output gốc
OUTPUT_DIR = os.path.abspath(r"data/ocr_output_local")

# Tiền tố để lọc folder cần xử lý (để trống = xử lý tất cả)
PREFIX = ""

# Chỉ xử lý 1 folder cụ thể (vd "L22_V001"). Để None để xử lý tất cả folder khớp PREFIX.
ONLY_FOLDER = None

# Giới hạn số ảnh xử lý trong mỗi folder (dùng để test nhanh). Để None = xử lý hết.
LIMIT_IMAGES = None

# Detection model (PaddleOCR)
# Đổi từ "medium" sang "server" để tăng recall (2×T4 dư sức chạy bản chính xác hơn).
# Nếu model "server" không tải được (lỗi mạng / chưa có sẵn), code sẽ tự fallback
# về DET_MODEL_NAME_FALLBACK bên dưới.
DET_MODEL_NAME = "PP-OCRv6_server_det"
DET_MODEL_NAME_FALLBACK = "PP-OCRv6_medium_det"
# Ngưỡng score detection box: hạ từ 0.7 → 0.55 để không loại chữ mờ/bị che
# ngay từ bước detect; rác sinh thêm sẽ được lọc lại sau recognize (should_filter_text).
DET_BOX_THRESH = 0.55
# Hệ số "nới" polygon ra khỏi viền chữ khi detect (default PaddleOCR ~1.5).
# Tăng lên để không cắt mất nét chữ khi text bị che khuất một phần.
DET_UNCLIP_RATIO = 1.9
# Giữ nguyên kích thước ảnh 1280×720, không resize xuống 960 (default)
DET_LIMIT_SIDE_LEN = 1280

# Multi-scale detection: chạy detect ở nhiều limit_side_len rồi merge bằng NMS,
# giúp bắt thêm box nhỏ/bị che mà 1 scale duy nhất bỏ sót. Mặc định TẮT vì tốn
# gấp đôi thời gian detect — chỉ bật khi thực sự cần tăng recall tối đa.
ENABLE_MULTISCALE_DET = False
MULTISCALE_SIDE_LENS = [960, 1280]

# VietOCR config
VIETOCR_DEVICE = "cuda"  # 'cuda' nếu có GPU, 'cpu' nếu không
VIETOCR_BEAMSEARCH = False  # Pass 1 (nhanh, greedy): tắt BeamSearch, nhanh hơn 5-10x

# Two-pass recognition: pass 1 greedy cho toàn bộ (nhanh), pass 2 beamsearch chỉ
# cho các box có confidence thấp (khó đọc). Nếu máy có ≥2 GPU, pass 2 chạy trên
# GPU riêng (VIETOCR_DEVICE_PASS2) để chạy song song, không làm chậm pass 1.
ENABLE_TWO_PASS_RECOGNITION = True
REC_CONF_THRESHOLD = 0.7  # confidence dưới ngưỡng này mới đưa qua pass 2 (beamsearch)
VIETOCR_DEVICE_PASS2 = "cuda:1"  # dùng khi có ≥2 GPU; tự fallback nếu không có

# Optional VLM branch for hard real-world images (chalkboard, handwriting,
# posters, low-contrast signs). Kept opt-in because it is slower than local OCR.
DEFAULT_VLM_MODEL = "5CD-AI/Vintern-1B-v3_5"
VLM_INPUT_SIZE = 448

# Tiền xử lý crop trước khi đưa vào VietOCR: upscale ảnh nhỏ + CLAHE + unsharp mask.
# Giúp đọc tốt hơn với chữ mờ/nhỏ. Chi phí CPU rất rẻ so với lợi ích accuracy.
ENABLE_PREPROCESS = True
PREPROCESS_MIN_HEIGHT = 32     # crop thấp hơn ngưỡng này sẽ được upscale
PREPROCESS_TARGET_HEIGHT = 48  # chiều cao sau khi upscale

# Padding thêm quanh polygon khi crop (px). Tăng từ 4 → 8 để không cắt mất nét
# chữ sát viền box (đặc biệt khi chữ bị che khuất một phần).
BOX_PADDING = 8

# ============================================================
# BATCH CONFIG (tối ưu GPU 16GB VRAM)
# ============================================================
# Số ảnh gửi 1 lần vào PaddleOCR để detect
BATCH_DET_SIZE = 16

# Số crop-box gửi 1 lần vào VietOCR predict_batch() (tận dụng GPU cho recognition)
BATCH_REC_SIZE = 128

# ============================================================
# VÙNG LOẠI TRỪ (banner tin tức + watermark)
# ============================================================
# Ảnh có kích thước cố định 1280x720.
# Tất cả tọa độ dùng hệ CHUẨN: gốc (0,0) = góc TRÊN-TRÁI,
# y tăng xuống dưới — đúng với hệ tọa độ của OpenCV / PaddleOCR.
IMG_WIDTH = 1280
IMG_HEIGHT = 720

# --- Hard exclude: mặc định để trống để không bị mất tin tức ở banner dưới cùng ---
EXCLUDE_REGIONS = []

# --- Soft exclude: recognize trước, chỉ lọc nếu text giống channel title ---
# Tùy ảnh, watermark kênh (HTV7 HD, VTV3...) xuất hiện ở 1 trong 2 vị trí.
# Không hard-exclude vì đôi khi vùng này chứa nội dung thật.
CHANNEL_TITLE_REGIONS = [
    # Vị trí 1: góc trên-phải (thường là HTV7 HD + timestamp)
    # Tọa độ chuẩn (gốc trên-trái): (1045, 60) → (1195, 115)
    {
        "name": "channel_top_right",
        "x_min": 1045,
        "x_max": 1195,
        "y_min": 60,
        "y_max": 115,
    },
    # Vị trí 2: góc trên-trái
    # Tọa độ chuẩn (gốc trên-trái): (0, 0) → (130, 85)
    {
        "name": "channel_top_left",
        "x_min": 0,
        "x_max": 130,
        "y_min": 0,
        "y_max": 85,
    },
]

def box_in_exclude_region(box, regions=EXCLUDE_REGIONS, overlap_threshold=0.5):
    """
    Kiểm tra box (list toạ độ [[x,y],...] hoặc [[x1,y1],[x2,y2]]) có nằm (phần lớn)
    trong BẤT KỲ vùng loại trừ nào không. Dùng overlap ratio theo cả 2 trục.
    """
    if not box:
        return False
    ys = [pt[1] for pt in box]
    xs = [pt[0] for pt in box]
    box_y_min, box_y_max = min(ys), max(ys)
    box_x_min, box_x_max = min(xs), max(xs)
    box_height = box_y_max - box_y_min
    box_width = box_x_max - box_x_min
    if box_height <= 0 or box_width <= 0:
        return False

    for region in regions:
        overlap_y = max(0, min(box_y_max, region["y_max"]) - max(box_y_min, region["y_min"]))
        overlap_ratio_y = overlap_y / box_height

        overlap_x = max(0, min(box_x_max, region["x_max"]) - max(box_x_min, region["x_min"]))
        overlap_ratio_x = overlap_x / box_width

        if overlap_ratio_y >= overlap_threshold and overlap_ratio_x >= overlap_threshold:
            return True

    return False

# ============================================================
# SPAM FILTER
# ============================================================

SPAM_KEYWORDS = {
    # Tên kênh truyền hình / Watermark
    "HTV", "HTV7", "HTV9", "HTV7 HD", "HTV9 HD", "Online", "VTV", "VTV1", "VTV2", "VTV3", "VTV4",
    "K+", "K Plus", "FPT", "TV", "VOV", "THVN", "MAX", "VIB", "MTV",
    "TV360", "Cộng Hòa", "Xã Hội", "Chủ Nghĩa",
    # Biến thể lỗi OCR của watermark HTV (quan sát thực tế từ L22_V001, V002)
    "HTV7HP", "HTV7HD", "HTV7H", "HTV9HP", "HTV9HD", "HTV9H",
    "HIMED", "HMT", "HTVV", "HTMIN", "FTVT", "HIMID", "MINTW",
    "HIVID", "HIMVED", "HIMZO", "HIMMY", "HAYW", "HUZZ",
    "SUBSCRIBED",
}

# Pattern cho watermark thời gian dạng HH:MM hoặc HH:MM:SS (vd "18:30:11")
TIME_PATTERN = re.compile(r"^\d{1,2}:\d{2}(:\d{2})?$")

# Pattern mở rộng: timestamp dính liền (vd "1830:18", "1863:01") hoặc số thuần (vd "4900", "160")
TIME_EXTENDED_PATTERN = re.compile(r"^\d{2,}:\d{2,}$")
PURE_NUMBER_PATTERN = re.compile(r"^\d{2,5}$")  # 2-5 chữ số thuần, vd "4900", "160"

# Pattern cho watermark kênh + giờ dính liền (vd "HTV7 HD 18:30:11", "HTV9 20:00")
CHANNEL_TIME_PATTERN = re.compile(
    r"^(HTV\d*\s*(HD)?|VTV\d*|K\+|FPT|VOV|THVN|MTV|TV360)\s*\d{1,2}:\d{2}(:\d{2})?$",
    re.IGNORECASE,
)

# Pattern bắt biến thể hallucinate của watermark HTV:
# Thường là 3-6 ký tự, bắt đầu bằng H/F, toàn uppercase Latin, không phải từ Việt
# VD: HIMED, FTVT, HTMIN, HIMZO, HIVID, ...
HTV_HALLUCINATION_PATTERN = re.compile(
    r"^[HF][A-Z]{2,6}$"
)

# Pattern bắt text Latin ngắn không có dấu (thường là hallucination):
# 3-8 ký tự lowercase hoặc mixed, không chứa nguyên âm có dấu tiếng Việt
# VD: provaled, vix, xxx, murenchion
SHORT_LATIN_GIBBERISH_PATTERN = re.compile(
    r"^[a-zA-Z]{2,5}$"  # từ Latin thuần 2-5 ký tự, rất ngắn = thường rác
)

# Danh sách nguyên âm có dấu tiếng Việt (dùng để phân biệt từ Việt thực vs rác)
_VIET_DIACRITICS = set("àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ"
                       "ÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴĐ")

def should_filter_text(text):
    """Kiểm tra text có phải spam (channel name, watermark, giờ, ký tự ngắn, hallucination)."""
    text_stripped = text.strip()

    # 1. Ký tự đơn Latin hoặc quá ngắn (≤2 ký tự ASCII)
    if len(text_stripped) <= 2 and not any(ord(c) >= 0x0100 for c in text_stripped):
        return True

    # 2. Exact match hoặc tất cả các từ cấu thành thuộc danh sách spam keywords (vd "HTV Online", "HTV 7 HD")
    if text_stripped in SPAM_KEYWORDS or text_stripped.upper() in SPAM_KEYWORDS:
        return True
    words = [w.strip() for w in re.split(r'\s+', text_stripped) if w.strip()]
    if words and all(w in SPAM_KEYWORDS or w.upper() in SPAM_KEYWORDS for w in words):
        return True

    # 3. Timestamp patterns
    if TIME_PATTERN.match(text_stripped):
        return True
    if TIME_EXTENDED_PATTERN.match(text_stripped):
        return True
    if PURE_NUMBER_PATTERN.match(text_stripped):
        return True

    # 4. Channel + time pattern (vd "HTV7 HD 18:30:11")
    if CHANNEL_TIME_PATTERN.match(text_stripped):
        return True

    # 5. HTV hallucination pattern (3-7 uppercase bắt đầu H/F, không phải từ Việt)
    if HTV_HALLUCINATION_PATTERN.match(text_stripped):
        # Cho phép một số từ viết tắt có nghĩa (UBND, HCMC, ...)
        # nhưng hầu hết HIMED, FTVT, HIMZO... đều là rác
        return True

    # 6. Từ Latin thuần 2-5 ký tự, không dấu tiếng Việt = rất có thể là hallucination
    #    Chỉ filter khi đứng RIÊNG LẺ (1 từ duy nhất), không filter nếu nằm trong câu dài
    #    (vì filter này áp dụng cho từng box riêng lẻ, text đã strip)
    if SHORT_LATIN_GIBBERISH_PATTERN.match(text_stripped):
        return True

    # 7. Chuỗi có > 50% là dấu chấm/khoảng trắng/số liên tiếp = rác
    non_content = sum(1 for c in text_stripped if c in '.… 0')
    if len(text_stripped) > 10 and non_content / len(text_stripped) > 0.5:
        return True

    return False

# ============================================================
# DETECTION (PaddleOCR)
# ============================================================

def _order_poly_points(poly):
    """
    Sắp xếp lại 4 điểm polygon theo thứ tự tl, tr, br, bl.
    Cần thiết để warpPerspective duỗi thẳng đúng chiều (không bị lật/xoay sai)
    — PaddleOCR thường trả đúng thứ tự này sẵn, nhưng sắp lại cho chắc chắn.
    """
    pts = np.array(poly, dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def _parse_det_result(result):
    """
    Parse 1 PaddleOCR result object → list item {"poly": 4 điểm góc, "box": [[x1,y1],[x2,y2]]}.
    Giữ nguyên polygon 4 điểm gốc (thay vì chỉ lấy bounding box trục thẳng) để
    có thể duỗi thẳng (perspective warp) dòng chữ nghiêng/cong trước khi recognize.
    "box" (bbox trục thẳng) vẫn được tính kèm để dùng cho các bước lọc vùng
    loại trừ / gom cụm đoạn văn — các bước đó chỉ cần bbox xấp xỉ là đủ.
    """
    if hasattr(result, "json"):
        data = result.json.get("res", result.json)
    elif isinstance(result, dict):
        data = result
    else:
        data = {}

    dt_polys = data.get("dt_polys", data.get("boxes", [])) or []
    items = []
    for poly in dt_polys:
        poly = np.array(poly, dtype=np.float32)
        if poly.shape[0] != 4:
            # Một số backend trả polygon cong (>4 điểm) → quy về 4 điểm qua minAreaRect
            hull = cv2.convexHull(poly.astype(np.float32))
            rect = cv2.minAreaRect(hull)
            poly = cv2.boxPoints(rect)
        poly = _order_poly_points(poly)
        x1 = int(poly[:, 0].min())
        y1 = int(poly[:, 1].min())
        x2 = int(poly[:, 0].max())
        y2 = int(poly[:, 1].max())
        items.append({"poly": poly, "box": [[x1, y1], [x2, y2]]})
    return items


def detect_text_regions(detector, img_path, debug=False):
    """
    Dùng PaddleOCR TextDetection để tìm các vùng chứa text (single image).
    Trả về list item {"poly": [...4 điểm...], "box": [[x1,y1],[x2,y2]]}.
    """
    results = detector.predict(str(img_path))
    if results and isinstance(results[0], dict) and "poly" in results[0]:
        return results
    items = []
    for result in results:
        items.extend(_parse_det_result(result))
    return items


def detect_text_regions_batch(detector, img_paths, debug=False):
    """
    Batch detection: gửi nhiều ảnh cùng lúc vào PaddleOCR để tận dụng GPU.
    Trả về list of list item, mỗi phần tử tương ứng với 1 img_path.
    Nếu PaddleOCR không hỗ trợ batch predict thì fallback sang từng ảnh.
    """
    str_paths = [str(p) for p in img_paths]
    try:
        # PaddlePaddle ≥ 2.6 / PaddleX hỗ trợ predict nhận list
        all_results = detector.predict(str_paths)
        # all_results là flat list gồm N kết quả (1 per image)
        # Mỗi phần tử có thể là 1 result hoặc list[result] → cần gom
        # Kiểm tra nếu trả về ít hơn N → fallback
        if len(all_results) == len(img_paths):
            return [_parse_det_result(r) for r in all_results]
        # Trường hợp trả về đủ N nhưng mỗi r là list
        if len(all_results) > 0 and isinstance(all_results[0], list):
            return [sum((_parse_det_result(r) for r in rs), []) for rs in all_results]
    except Exception:
        pass

    # Fallback: xử lý từng ảnh
    return [detect_text_regions(detector, p) for p in img_paths]


def _iou(box_a, box_b):
    """IoU giữa 2 box dạng [[x1,y1],[x2,y2]]."""
    ax1, ay1 = box_a[0]; ax2, ay2 = box_a[1]
    bx1, by1 = box_b[0]; bx2, by2 = box_b[1]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def merge_items_nms(items, iou_thresh=0.5):
    """Gộp box từ nhiều lần detect (multi-scale), loại box trùng lặp bằng NMS đơn giản
    (giữ box lớn hơn khi 2 box overlap nhiều, giả định box lớn bao trọn text hơn)."""
    items_sorted = sorted(
        items,
        key=lambda it: (it["box"][1][0] - it["box"][0][0]) * (it["box"][1][1] - it["box"][0][1]),
        reverse=True,
    )
    kept = []
    for it in items_sorted:
        if all(_iou(it["box"], k["box"]) < iou_thresh for k in kept):
            kept.append(it)
    return kept


def detect_text_regions_multiscale(detectors, img_path):
    """
    Chạy detect ở nhiều scale (nhiều TextDetection instance, mỗi cái khởi tạo với
    limit_side_len khác nhau) rồi merge bằng NMS — bắt được dòng nhỏ/bị che mà
    1 scale duy nhất bỏ sót. `detectors` là list các TextDetection instance.
    """
    all_items = []
    for det in detectors:
        try:
            all_items.extend(detect_text_regions(det, img_path))
        except Exception:
            continue
    return merge_items_nms(all_items)

# ============================================================
# RECOGNITION (VietOCR)
# ============================================================

def is_channel_title_text(text):
    """
    Kiểm tra text có giống channel title/watermark không.
    Dùng cho các box nằm trong CHANNEL_TITLE_REGIONS (soft exclude).
    Lọc rộng hơn should_filter_text vì đã biết box ở vùng channel.
    """
    t = text.strip()
    if not t:
        return True

    # Nếu text ngắn (≤12 ký tự), kiểm tra kỹ hơn
    if len(t) <= 12:
        # Channel name patterns
        t_upper = t.upper()
        channel_prefixes = ("HTV", "VTV", "FPT", "VOV", "MTV", "THVN", "TV3")
        if any(t_upper.startswith(p) for p in channel_prefixes):
            return True

        # HTV hallucination: 2-7 ký tự, chủ yếu uppercase Latin, không có dấu Việt
        alpha_chars = [c for c in t if c.isalpha()]
        if alpha_chars:
            has_viet = any(c in _VIET_DIACRITICS for c in t)
            all_ascii_alpha = all(c.isascii() for c in alpha_chars)
            if all_ascii_alpha and not has_viet and len(alpha_chars) <= 8:
                return True

    # Timestamp patterns (HH:MM, HH:MM:SS, dính liền...)
    if TIME_PATTERN.match(t) or TIME_EXTENDED_PATTERN.match(t):
        return True
    if PURE_NUMBER_PATTERN.match(t):
        return True

    return False


def cluster_boxes_into_paragraphs(boxes, x_gap_factor=2.5, y_gap_factor=1.8):
    """
    Nhóm các box thành từng cụm (đoạn/khối văn bản) dựa trên khoảng cách không gian,
    không giả định trước số cột/bố cục — tự thích nghi với mọi layout (1 cột, nhiều
    cột, box rải rác...).

    Thuật toán: coi mỗi box là 1 node, nối 2 box bằng 1 cạnh nếu chúng "gần nhau"
    (khoảng cách theo x và y đều nhỏ hơn ngưỡng dựa trên kích thước box trung bình),
    sau đó tìm các connected component (union-find) — mỗi component là 1 đoạn.

    Trả về: list các cụm, mỗi cụm là list các (box, index) đã sort theo thứ tự đọc
    (trên→dưới, trái→phải) trong cụm đó.
    """
    n = len(boxes)
    if n == 0:
        return []
    if n == 1:
        return [[0]]

    # Kích thước box trung bình để tính ngưỡng khoảng cách "gần nhau" theo tỉ lệ
    heights = [b[1][1] - b[0][1] for b in boxes]
    avg_h = sum(heights) / len(heights) if heights else 20
    x_gap_thresh = avg_h * x_gap_factor
    y_gap_thresh = avg_h * y_gap_factor

    def box_distance(a, b):
        """Khoảng cách (dx, dy) giữa 2 box (0 nếu chồng lấn theo trục đó)."""
        ax1, ay1 = a[0]
        ax2, ay2 = a[1]
        bx1, by1 = b[0]
        bx2, by2 = b[1]
        dx = max(0, max(ax1, bx1) - min(ax2, bx2))
        dy = max(0, max(ay1, by1) - min(ay2, by2))
        return dx, dy

    # Union-Find
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(n):
        for j in range(i + 1, n):
            dx, dy = box_distance(boxes[i], boxes[j])
            if dx <= x_gap_thresh and dy <= y_gap_thresh:
                union(i, j)

    clusters = {}
    for i in range(n):
        root = find(i)
        clusters.setdefault(root, []).append(i)

    # Sort các box trong từng cụm theo thứ tự đọc (trên→dưới, trái→phải)
    cluster_list = []
    for indices in clusters.values():
        indices.sort(key=lambda idx: (boxes[idx][0][1], boxes[idx][0][0]))
        cluster_list.append(indices)

    # Sort các cụm theo vị trí trung bình (trên→dưới, trái→phải)
    def cluster_key(indices):
        ys = [boxes[i][0][1] for i in indices]
        xs = [boxes[i][0][0] for i in indices]
        return (min(ys), min(xs))

    cluster_list.sort(key=cluster_key)

    return cluster_list


def expand_poly(poly, padding):
    """
    Nới polygon 4 điểm ra ngoài `padding` px (theo hướng từ tâm ra từng đỉnh).
    Đây là cách nới xấp xỉ (không phải true polygon-offset), nhưng đủ tốt để
    không cắt mất nét chữ sát viền box, tương tự tăng padding cho crop trục thẳng.
    """
    if padding <= 0:
        return poly
    pts = np.array(poly, dtype=np.float32)
    centroid = pts.mean(axis=0)
    expanded = []
    for p in pts:
        direction = p - centroid
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            expanded.append(p)
            continue
        unit = direction / norm
        expanded.append(p + unit * padding)
    return np.array(expanded, dtype=np.float32)


def get_rotate_crop_image(img, points):
    """
    Duỗi thẳng (perspective warp) vùng text theo polygon 4 điểm — kỹ thuật chuẩn
    trong PaddleOCR gốc (get_rotate_crop_image). Thay vì cắt bounding box trục
    thẳng (lấy dư nền, không xử lý được dòng nghiêng), hàm này biến đổi phối cảnh
    để dòng chữ nghiêng/hơi cong trở thành ngang, giúp VietOCR đọc chính xác hơn
    nhiều. Chi phí chỉ là CPU (getPerspectiveTransform + warpPerspective), không
    tốn thêm GPU nên chạy song song rất rẻ.
    """
    points = np.array(points, dtype=np.float32)
    h, w = img.shape[:2]
    points[:, 0] = np.clip(points[:, 0], 0, w - 1)
    points[:, 1] = np.clip(points[:, 1], 0, h - 1)

    img_crop_width = int(max(
        np.linalg.norm(points[0] - points[1]),
        np.linalg.norm(points[2] - points[3]),
    ))
    img_crop_height = int(max(
        np.linalg.norm(points[0] - points[3]),
        np.linalg.norm(points[1] - points[2]),
    ))
    if img_crop_width < 1 or img_crop_height < 1:
        return None

    pts_std = np.float32([
        [0, 0],
        [img_crop_width, 0],
        [img_crop_width, img_crop_height],
        [0, img_crop_height],
    ])
    M = cv2.getPerspectiveTransform(points, pts_std)
    dst_img = cv2.warpPerspective(
        img, M, (img_crop_width, img_crop_height),
        borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC,
    )
    dh, dw = dst_img.shape[:2]
    # Nếu crop ra dạng "cao hơn rộng" (dòng chữ dọc) → xoay lại về ngang cho VietOCR
    if dw > 0 and dh * 1.0 / dw >= 1.5:
        dst_img = np.rot90(dst_img)
    return dst_img


def preprocess_crop(crop_bgr, min_height=PREPROCESS_MIN_HEIGHT, target_height=PREPROCESS_TARGET_HEIGHT):
    """
    Tiền xử lý crop trước khi đưa vào VietOCR:
      - Upscale nếu crop quá nhỏ (chữ nhỏ → VietOCR đọc kém)
      - CLAHE (tăng tương phản cục bộ trên kênh L) → giúp ảnh mờ/thiếu sáng rõ hơn
      - Unsharp mask nhẹ → bù nét cho ảnh blur
    """
    h, w = crop_bgr.shape[:2]
    if h == 0 or w == 0:
        return crop_bgr

    if h < min_height:
        scale = target_height / float(h)
        new_w = max(1, int(round(w * scale)))
        crop_bgr = cv2.resize(crop_bgr, (new_w, target_height), interpolation=cv2.INTER_CUBIC)

    try:
        lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l2 = clahe.apply(l)
        crop_bgr = cv2.cvtColor(cv2.merge((l2, a, b)), cv2.COLOR_LAB2BGR)
    except Exception:
        pass

    try:
        blurred = cv2.GaussianBlur(crop_bgr, (0, 0), sigmaX=1.0)
        crop_bgr = cv2.addWeighted(crop_bgr, 1.5, blurred, -0.5, 0)
    except Exception:
        pass

    return crop_bgr


def recognize_text_vietocr(recognizer, img, items, padding=BOX_PADDING, filter_spam=True,
                           channel_title_regions=None, batch_size=None,
                           recognizer_beam=None, conf_threshold=REC_CONF_THRESHOLD,
                           enable_preprocess=ENABLE_PREPROCESS):
    """
    Dùng VietOCR để nhận diện text trên từng vùng đã crop.
    Khi batch_size > 0, gom toàn bộ crops thành batch rồi gọi predict_batch() 1 lần
    để tận dụng GPU tối đa, thay vì gọi predict() N lần riêng lẻ.

    `items` là list {"poly": 4 điểm, "box": [[x1,y1],[x2,y2]]} từ detect_text_regions
    (đã được lọc bỏ các vùng nằm trong EXCLUDE_REGIONS từ trước).
    Với box trong `channel_title_regions`: recognize trước, chỉ lọc nếu text
    giống channel title (HTV, timestamp, hallucination...).

    Two-pass: pass 1 chạy greedy (nhanh) cho toàn bộ; nếu `recognizer_beam` được
    truyền vào, các box có confidence < conf_threshold sẽ được đọc lại bằng
    beamsearch (chính xác hơn) ở pass 2.

    Text được trả về theo đúng thứ tự đọc tự nhiên: các box được nhóm thành từng
    đoạn/khối theo vị trí không gian (cluster_boxes_into_paragraphs) trước, để
    tránh trộn lẫn nội dung của 2 cột/khối khác nhau nằm cùng độ cao y.

    Trả về list text (đã strip, đã lọc spam nếu filter_spam=True), đã sắp đúng thứ tự.
    """
    if batch_size is None:
        batch_size = BATCH_REC_SIZE

    # Bước 1: Crop tất cả bằng perspective warp (duỗi thẳng polygon) — thay vì
    # cắt bbox trục thẳng thô, giữ đúng hình dạng dòng chữ nghiêng/cong.
    valid = []  # list of (item, pil_img)
    for item in items:
        poly = expand_poly(item["poly"], padding)
        crop = get_rotate_crop_image(img, poly)
        if crop is None or crop.size == 0:
            continue
        if enable_preprocess:
            crop = preprocess_crop(crop)
        pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        valid.append((item, pil_img))

    if not valid:
        return []

    all_items = [v[0] for v in valid]
    all_pils = [v[1] for v in valid]
    all_texts = [""] * len(all_pils)
    all_confs = [1.0] * len(all_pils)  # mặc định "tin cậy" nếu backend không trả prob

    def _predict_single_with_prob(pil):
        try:
            t, p = recognizer.predict(pil, return_prob=True)
            return (t.strip() if t else ""), (float(p) if p is not None else 1.0)
        except Exception:
            try:
                return recognizer.predict(pil).strip(), 1.0
            except Exception:
                return "", 0.0

    # Bước 2: Pass 1 — nhận diện nhanh (greedy), theo batch, kèm confidence nếu có
    if batch_size > 0 and hasattr(recognizer, "predict_batch_padded"):
        for i in range(0, len(all_pils), batch_size):
            batch_pils = all_pils[i:i + batch_size]
            try:
                batch_out = recognizer.predict_batch_padded(batch_pils, return_prob=True)
                for j, out in enumerate(batch_out):
                    text, prob = out if isinstance(out, (tuple, list)) else (out, 1.0)
                    all_texts[i + j] = text.strip() if text else ""
                    all_confs[i + j] = float(prob) if prob is not None else 1.0
            except TypeError:
                # Bản predict_batch_padded không nhận kwarg return_prob
                try:
                    batch_texts = recognizer.predict_batch_padded(batch_pils)
                    for j, t in enumerate(batch_texts):
                        all_texts[i + j] = t.strip() if t else ""
                        # Không có prob → không đưa qua pass 2 (coi như đủ tin cậy)
                        all_confs[i + j] = 1.0
                except Exception:
                    for j, pil in enumerate(batch_pils):
                        all_texts[i + j], all_confs[i + j] = _predict_single_with_prob(pil)
            except Exception:
                for j, pil in enumerate(batch_pils):
                    all_texts[i + j], all_confs[i + j] = _predict_single_with_prob(pil)
    else:
        for i, pil in enumerate(all_pils):
            all_texts[i], all_confs[i] = _predict_single_with_prob(pil)

    # Bước 3: Pass 2 — beamsearch cho các box confidence thấp (GPU riêng nếu có)
    if recognizer_beam is not None:
        low_conf_idx = [i for i, c in enumerate(all_confs) if all_texts[i] and c < conf_threshold]
        for i in low_conf_idx:
            try:
                t2 = recognizer_beam.predict(all_pils[i])
                if t2:
                    all_texts[i] = t2.strip()
            except Exception:
                pass

    # Bước 4: Lọc spam và channel title
    recognized = []  # list of (box, text)
    for item, text in zip(all_items, all_texts):
        if not text:
            continue

        box = item["box"]
        # Soft exclude: box nằm trong vùng channel title → chỉ giữ nếu KHÔNG giống watermark
        if channel_title_regions and box_in_exclude_region(box, channel_title_regions):
            if is_channel_title_text(text):
                continue

        if filter_spam and should_filter_text(text):
            continue

        recognized.append((box, text))

    if not recognized:
        return []

    # Bước 5: Nhóm box (đã lọc) thành từng đoạn, giữ đúng thứ tự đọc
    kept_boxes = [item[0] for item in recognized]
    clusters = cluster_boxes_into_paragraphs(kept_boxes)

    texts = []
    for indices in clusters:
        for idx in indices:
            texts.append(recognized[idx][1])

    return texts

# ============================================================
# OCR PIPELINE (combined)
# ============================================================

def ocr_image(detector, recognizer, img_path, filter_spam=True, filter_exclude_region=False, padding=BOX_PADDING):
    """
    Pipeline đầy đủ: PaddleOCR detect -> lọc box theo vùng loại trừ -> VietOCR recognize -> lọc spam.
    Trả về (text, error).
    """
    quality, error = ocr_image_quality(
        detector,
        recognizer,
        img_path,
        filter_spam=filter_spam,
        filter_exclude_region=filter_exclude_region,
        padding=padding,
    )
    return quality.text, error


def ocr_image_quality(
    detector,
    recognizer,
    img_path,
    filter_spam=True,
    filter_exclude_region=False,
    padding=BOX_PADDING,
) -> tuple[OCRQualityResult, str | None]:
    """
    Pipeline OCR theo kiến trúc quality layer:
      raw line OCR -> conservative Vietnamese postprocess -> auditable metadata.

    Trả về OCRQualityResult để giữ cả text raw, text đã sửa, edits và review flags.
    """
    try:
        boxes = detect_text_regions(detector, img_path)
    except Exception as e:
        return build_quality_result(""), str(e)

    if not boxes:
        return build_quality_result(""), None

    # Lọc bỏ các box nằm trong vùng loại trừ TRƯỚC khi recognize (tiết kiệm thời gian VietOCR)
    if filter_exclude_region and EXCLUDE_REGIONS:
        boxes = [b for b in boxes if not box_in_exclude_region(b, EXCLUDE_REGIONS)]

    if not boxes:
        return build_quality_result(""), None

    img = cv2.imread(str(img_path))
    if img is None:
        return build_quality_result(""), f"Không đọc được ảnh: {img_path}"

    # Không cần sort ở đây nữa — recognize_text_vietocr() tự nhóm box thành đoạn
    # (cluster_boxes_into_paragraphs) và sắp đúng thứ tự đọc trước khi trả về.

    try:
        texts = recognize_text_vietocr(
            recognizer, img, boxes, padding=padding, filter_spam=filter_spam,
            channel_title_regions=CHANNEL_TITLE_REGIONS if filter_exclude_region else None,
        )
    except Exception as e:
        return build_quality_result(""), str(e)

    raw_text = "\n".join(texts)
    return build_quality_result(raw_text), None


def _patch_transformers_for_local_vlm():
    """Patch older local VLM code paths against newer transformers releases."""
    try:
        from transformers import PreTrainedModel

        def get_tied(self):
            return getattr(self, "_all_tied_weights_keys_val", {})

        def set_tied(self, val):
            self._all_tied_weights_keys_val = val

        PreTrainedModel.all_tied_weights_keys = property(get_tied, set_tied)
    except Exception:
        pass


def _build_vlm_transform(input_size=VLM_INPUT_SIZE):
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode

    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std = (0.229, 0.224, 0.225)
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])


def _enhance_for_vlm(image_path):
    """Light global enhancement before VLM tiling, useful for chalk/low contrast."""
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        return Image.open(image_path).convert("RGB")
    try:
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2RGB)
        return Image.fromarray(enhanced)
    except Exception:
        return Image.open(image_path).convert("RGB")


def _load_vlm_tiles(pil_img, input_size=VLM_INPUT_SIZE):
    """Use global + left/right crops; too many tiles can overflow Vintern context."""
    import torch

    w, h = pil_img.size
    mid_w = w // 2
    crops = [
        pil_img,
        pil_img.crop((0, 0, min(w, mid_w + 80), h)),
        pil_img.crop((max(0, mid_w - 80), 0, w, h)),
    ]
    transform = _build_vlm_transform(input_size=input_size)
    tensors = [transform(crop).unsqueeze(0) for crop in crops]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    return torch.cat(tensors, dim=0).to(device=device, dtype=dtype)


_VLM_CACHE = {}

def get_vlm_engine(model_id=DEFAULT_VLM_MODEL):
    """Load VLM model & tokenizer đúng 1 LẦN duy nhất vào VRAM."""
    if model_id in _VLM_CACHE:
        return _VLM_CACHE[model_id]

    import torch
    from transformers import AutoModel, AutoTokenizer

    _patch_transformers_for_local_vlm()
    print(f"📦 Loading Vintern VLM model weights ({model_id}) vào GPU VRAM (LOAD 1 LẦN)...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    device_map = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModel.from_pretrained(
        model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map=device_map,
    ).eval()

    _VLM_CACHE[model_id] = (model, tokenizer)
    print("✅ Vintern VLM model weights ready in VRAM.\n")
    return model, tokenizer

def run_vlm_ocr(image_path, model_id=DEFAULT_VLM_MODEL, model=None, tokenizer=None):
    """
    Run a VLM OCR branch on the whole image plus high-resolution tiles.
    Uses cached global VLM model instance to prevent reloading weights.
    """
    import torch

    if model is None or tokenizer is None:
        model, tokenizer = get_vlm_engine(model_id)

    pil_img = _enhance_for_vlm(image_path)

    pixel_values = _load_vlm_tiles(pil_img)
    question = (
        "<image>\n"
        "Trích xuất OCR tất cả các dòng chữ tiếng Việt xuất hiện trong ảnh (chữ in, chữ viết tay, bảng biểu, tiêu đề, chữ hiển thị trên màn hình). "
        "Xuất mỗi dòng trên một dòng riêng, giữ thứ tự từ trái sang phải rồi từ trên xuống dưới. "
        "Không lặp lại dòng, không mô tả ảnh, không thêm nội dung không nhìn thấy."
    )
    generation_config = dict(max_new_tokens=512, do_sample=False, repetition_penalty=1.15)
    response = model.chat(tokenizer, pixel_values, question, generation_config=generation_config)
    return str(response).strip()


def _candidate_score(text: str) -> tuple[int, int, int]:
    """Rank OCR candidates by useful line count, Vietnamese signal, then length."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    vietnamese_chars = sum(1 for ch in text if ch in _VIET_DIACRITICS)
    return (len(lines), vietnamese_chars, len(text))


def ocr_image_hybrid_quality(
    detector,
    recognizer,
    img_path,
    backend="local",
    vlm_model=DEFAULT_VLM_MODEL,
    filter_spam=True,
    filter_exclude_region=False,
) -> tuple[OCRQualityResult, str | None, dict[str, str]]:
    """Run local/VLM/hybrid OCR and return the selected quality result plus branch raw outputs."""
    candidates: dict[str, str] = {}
    errors: list[str] = []

    if backend in ("local", "hybrid"):
        if detector is None or recognizer is None:
            errors.append("local backend requested but detector/recognizer is not initialized")
        else:
            local_quality, local_error = ocr_image_quality(
                detector,
                recognizer,
                img_path,
                filter_spam=filter_spam,
                filter_exclude_region=filter_exclude_region,
            )
            candidates["local"] = local_quality.raw_text
            if local_error:
                errors.append(f"local: {local_error}")

    if backend in ("vlm", "hybrid"):
        try:
            v_model, v_tok = get_vlm_engine(vlm_model)
            candidates["vlm"] = run_vlm_ocr(img_path, model_id=vlm_model, model=v_model, tokenizer=v_tok)
        except Exception as err:
            errors.append(f"vlm: {err}")

    selected_raw = max(candidates.values(), key=_candidate_score) if candidates else ""
    if filter_spam and selected_raw:
        raw_lines = [line.strip() for line in selected_raw.splitlines() if line.strip()]
        clean_lines = [line for line in raw_lines if not should_filter_text(line)]
        selected_raw = "\n".join(clean_lines)

    quality = build_quality_result(selected_raw)
    return quality, "; ".join(errors) if errors else None, candidates

# ============================================================
# PROCESS FOLDER
# ============================================================

def _quality_record(frame_id, quality: OCRQualityResult):
    """Convert OCRQualityResult into a JSON/parquet-friendly row."""
    payload = quality.to_dict()
    payload["frame_id"] = str(frame_id)
    return payload


def process_folder(folder_name, dataset_dir, output_dir, detector, recognizer,
                    filter_spam=True, filter_exclude_region=False, limit_images=None,
                    write_quality=False):
    """
    OCR toàn bộ ảnh trong 1 folder:
      - Detection: sequential từng ảnh (PaddleOCR chạy CPU, không benefit từ batch list)
      - Recognition: gom TOÀN BỘ crops của 1 ảnh → 1 GPU call (predict_batch_padded)
    Lưu JSON + Parquet với key = tên file (không phần mở rộng).
    """
    folder_path = os.path.join(dataset_dir, folder_name)

    img_files = sorted([
        f for f in os.listdir(folder_path)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))
    ])

    if limit_images is not None:
        img_files = img_files[:limit_images]

    results_json = {}
    quality_json = {}
    results_parquet = []
    quality_parquet = []
    errors = []

    for img_file in tqdm(img_files, desc=f"Processing {folder_name}"):
        frame_id = Path(img_file).stem
        img_path = os.path.join(folder_path, img_file)
        quality, err = ocr_image_quality(
            detector, recognizer, img_path,
            filter_spam=filter_spam,
            filter_exclude_region=filter_exclude_region,
        )
        if err:
            errors.append((img_file, err))
        results_json[frame_id] = quality.text
        results_parquet.append({"frame_id": frame_id, "text": quality.text})
        if write_quality:
            quality_record = _quality_record(frame_id, quality)
            quality_json[frame_id] = quality_record
            quality_parquet.append(quality_record)

    # Save JSON
    json_file = os.path.join(output_dir, f"{folder_name}.json")
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(results_json, f, ensure_ascii=False, indent=2)

    if write_quality:
        quality_file = os.path.join(output_dir, f"{folder_name}.quality.json")
        with open(quality_file, "w", encoding="utf-8") as f:
            json.dump(quality_json, f, ensure_ascii=False, indent=2)
    else:
        quality_file = None

    # Save Parquet
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        table = pa.Table.from_pylist(results_parquet, schema=pa.schema([
            ("frame_id", pa.string()),
            ("text", pa.string()),
        ]))
        parquet_file = os.path.join(output_dir, f"{folder_name}.parquet")
        pq.write_table(table, parquet_file)
        parquet_msg = f"✓ {parquet_file}"
        if write_quality:
            quality_table = pa.Table.from_pylist(quality_parquet)
            quality_parquet_file = os.path.join(output_dir, f"{folder_name}.quality.parquet")
            pq.write_table(quality_table, quality_parquet_file)
            parquet_msg += f"\n     → ✓ {quality_parquet_file}"
    except Exception as e:
        parquet_msg = f"⚠️ Parquet error: {e}"

    print(f"  ✅ {folder_name}: {len(results_json)} frames, {len(errors)} errors")
    print(f"     → {json_file}")
    if quality_file:
        print(f"     → {quality_file}")
    print(f"     → {parquet_msg}")
    if errors:
        print(f"  ⚠️ Errors: {errors[:3]}")

    return results_json, results_parquet, errors


def run_single_image(
    detector,
    recognizer,
    img_path,
    output_dir=None,
    write_quality=True,
    backend="local",
    vlm_model=DEFAULT_VLM_MODEL,
):
    """Run OCR on one image and optionally save a quality JSON artifact."""
    quality, err, candidates = ocr_image_hybrid_quality(
        detector,
        recognizer,
        img_path,
        backend=backend,
        vlm_model=vlm_model,
        filter_spam=True,
        filter_exclude_region=True,
    )
    image_path = Path(img_path)
    image_id = image_path.stem

    print("\n" + "=" * 60)
    print(f"OCR SINGLE IMAGE: {image_path}")
    print(f"BACKEND: {backend}")
    print("=" * 60)
    if err:
        print(f"ERROR: {err}")
    for name, raw in candidates.items():
        print(f"\n[{name.upper()} RAW]")
        print(raw)
    print("\n[RAW OCR]")
    print(quality.raw_text)
    print("\n[POSTPROCESSED OCR]")
    print(quality.text)
    if quality.edits:
        print("\n[EDITS]")
        for edit in quality.edits:
            print(f"- {edit['type']}: {edit['before']} -> {edit['after']}")
    if quality.needs_review:
        print(f"\n[REVIEW] {', '.join(quality.review_reasons)}")

    if output_dir and write_quality:
        os.makedirs(output_dir, exist_ok=True)
        output_path = Path(output_dir) / f"{image_id}.quality.json"
        payload = _quality_record(image_id, quality)
        payload["image_path"] = str(image_path)
        payload["backend"] = backend
        payload["candidates"] = candidates
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\nSaved quality output: {output_path}")

    return quality, err

# ============================================================
# MAIN
# ============================================================

def _init_detector():
    """Khởi tạo PaddleOCR TextDetection trên GPU CUDA."""
    print(f"Khởi tạo PaddleOCR TextDetection ({DET_MODEL_NAME}) trên GPU CUDA...")
    try:
        import paddle
        use_gpu = paddle.device.cuda.device_count() > 0
    except Exception:
        use_gpu = True

    try:
        detector = TextDetection(
            model_name=DET_MODEL_NAME,
            use_gpu=use_gpu,
            device="gpu" if use_gpu else "cpu",
            enable_mkldnn=False,
            box_thresh=DET_BOX_THRESH,
            limit_side_len=DET_LIMIT_SIDE_LEN,
        )
    except TypeError:
        try:
            detector = TextDetection(model_name=DET_MODEL_NAME, use_gpu=use_gpu)
        except TypeError:
            detector = TextDetection(model_name=DET_MODEL_NAME)
    print("✅ TextDetection ready on GPU CUDA.")
    return detector


def _init_recognizer():
    """Khởi tạo VietOCR recognizer theo cấu hình local."""
    print(f"Khởi tạo VietOCR (vgg_transformer, device={VIETOCR_DEVICE})...")
    _vietocr_config_dir = os.path.join(_LOCAL_VIETOCR_DIR, "config")
    _base_cfg_path  = os.path.join(_vietocr_config_dir, "base.yml")
    _vgg_cfg_path   = os.path.join(_vietocr_config_dir, "vgg-transformer.yml")
    if os.path.isfile(_base_cfg_path) and os.path.isfile(_vgg_cfg_path):
        import yaml as _yaml
        with open(_base_cfg_path, encoding="utf-8") as _f:
            _base_cfg = _yaml.safe_load(_f)
        with open(_vgg_cfg_path, encoding="utf-8") as _f:
            _vgg_cfg = _yaml.safe_load(_f)
        _base_cfg.update(_vgg_cfg)
        vietocr_config = Cfg(_base_cfg)
    else:
        vietocr_config = Cfg.load_config_from_name('vgg_transformer')

    vietocr_config['cnn']['pretrained'] = True
    vietocr_config['predictor']['beamsearch'] = VIETOCR_BEAMSEARCH
    vietocr_config['device'] = VIETOCR_DEVICE
    recognizer = Predictor(vietocr_config)
    print("✅ VietOCR ready.\n")
    return recognizer


def main(dataset_dir, output_dir, prefix="", only_folder=None, limit_images=None,
         image_path=None, write_quality=False, backend="local", vlm_model=DEFAULT_VLM_MODEL):
    """Main function."""
    detector = recognizer = None
    if backend in ("local", "hybrid"):
        detector = _init_detector()
        recognizer = _init_recognizer()

    if image_path:
        return run_single_image(
            detector,
            recognizer,
            image_path,
            output_dir,
            write_quality=True,
            backend=backend,
            vlm_model=vlm_model,
        )

    if backend != "local":
        print("⚠️ Batch folder mode hiện chỉ dùng backend local để tránh load VLM cho hàng nghìn frame.")

    if not os.path.isdir(dataset_dir):
        print(f"❌ Dataset dir không tồn tại: {dataset_dir}")
        sys.exit(1)

    all_folders = sorted([
        name for name in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, name))
    ])

    if prefix:
        video_folders = [name for name in all_folders if name.startswith(prefix)]
    else:
        video_folders = all_folders

    if only_folder:
        video_folders = [name for name in video_folders if name == only_folder]

    if not video_folders:
        if only_folder:
            print(f"❌ Không tìm thấy folder '{only_folder}' trong: {dataset_dir}")
        elif prefix:
            print(f"❌ Không tìm thấy folder nào với tiền tố '{prefix}' trong: {dataset_dir}")
        else:
            print(f"❌ Không tìm thấy folder nào trong: {dataset_dir}")
        sys.exit(1)

    final_output_dir = os.path.join(output_dir, prefix) if prefix else output_dir
    os.makedirs(final_output_dir, exist_ok=True)
    output_dir = final_output_dir

    print(f"Dataset dir: {dataset_dir}")
    print(f"Prefix     : {prefix if prefix else '(tất cả)'}")
    print(f"Only folder: {only_folder if only_folder else '(tất cả folder khớp prefix)'}")
    print(f"Limit ảnh  : {limit_images if limit_images is not None else '(không giới hạn)'}")
    print(f"Output dir : {output_dir}")
    print(f"Folders ({len(video_folders)}): {video_folders[:5]}")
    print()

    total_start = time.time()
    total_frames = 0
    all_errors = {}

    for i, folder_name in enumerate(video_folders, 1):
        print(f"\n{'='*60}")
        print(f"🎬 [{i}/{len(video_folders)}] {folder_name}")
        print(f"{'='*60}")

        start = time.time()
        json_res, parquet_res, errors = process_folder(
            folder_name, dataset_dir, output_dir, detector, recognizer,
            filter_spam=True, filter_exclude_region=True, limit_images=limit_images,
            write_quality=write_quality,
        )
        elapsed = time.time() - start

        all_errors[folder_name] = errors
        total_frames += len(json_res)

        per_frame = elapsed / len(json_res) if json_res else 0
        print(f"  ⏱️ {elapsed:.1f}s ({per_frame:.2f}s/frame)")

    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"🏁 XONG!")
    print(f"{'='*60}")
    print(f"Tổng: {total_frames} frames trong {total_elapsed:.1f}s")
    if total_frames:
        print(f"Trung bình: {total_elapsed/total_frames:.2f}s/frame")
    print(f"Output: {output_dir}")
    print()

    # Preview
    print(f"{'='*60}")
    print("📋 PREVIEW (2 folders đầu)")
    print(f"{'='*60}")
    for folder_name in video_folders[:2]:
        json_file = os.path.join(output_dir, f"{folder_name}.json")
        if not os.path.exists(json_file):
            continue
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        keys = list(data.keys())[:3]
        print(f"\n{folder_name}.json ({len(data)} frames):")
        for k in keys:
            text = data[k]
            preview = text[:60] + "..." if len(text) > 60 else text
            print(f"  {k}: {preview}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vietnamese OCR — PaddleOCR (detect) + VietOCR (recognize)")
    parser.add_argument("dataset_dir", nargs="?", default=DATASET_DIR, help="Path to unified_keyframes folder")
    parser.add_argument("output_dir", nargs="?", default=OUTPUT_DIR, help="Path to output folder")
    parser.add_argument("prefix", nargs="?", default=PREFIX, help="Prefix to filter folders, vd 'L21' (để trống = xử lý tất cả)")
    parser.add_argument("--only-folder", default=ONLY_FOLDER, help="Chỉ xử lý 1 folder cụ thể, vd 'L22_V001'")
    parser.add_argument("--limit", type=int, default=LIMIT_IMAGES, help="Giới hạn số ảnh xử lý mỗi folder (dùng để test nhanh)")
    parser.add_argument("--image", default=None, help="Chạy OCR cho một ảnh cụ thể, vd data/image/L30/L30/V002/3762.webp")
    parser.add_argument("--write-quality", action="store_true", help="Lưu thêm *.quality.json/parquet chứa raw/corrected/edits/review.")
    parser.add_argument("--backend", choices=["local", "vlm", "hybrid"], default="local", help="OCR backend cho --image.")
    parser.add_argument("--vlm-model", default=DEFAULT_VLM_MODEL, help="HuggingFace model ID cho backend vlm/hybrid.")
    args = parser.parse_args()

    main(
        args.dataset_dir,
        args.output_dir,
        args.prefix,
        args.only_folder,
        args.limit,
        image_path=args.image,
        write_quality=args.write_quality,
        backend=args.backend,
        vlm_model=args.vlm_model,
    )
