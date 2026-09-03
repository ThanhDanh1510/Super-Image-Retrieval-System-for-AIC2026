# 🔌 Tài Liệu API (API Reference)

Tất cả các endpoint của hệ thống được phục vụ tại tiền tố `/api/v1` và có tài liệu Swagger tương tác tại `http://localhost:8000/docs`.

---

## 1. Danh Sách Endpoint Chính

| Endpoint | Method | Chức Năng |
|---|---|---|
| `/api/v1/kis` | `POST` | Tìm kiếm Known-Item Search (Text, Voice, OCR, Descriptions) |
| `/api/v1/vqa` | `POST` | Visual Question Answering (Truy xuất keyframe bằng chứng & câu trả lời) |
| `/api/v1/trake` | `POST` | Temporal Retrieval (Chuỗi sự kiện theo dòng thời gian DANTE DP) |
| `/api/v1/similar-image` | `POST` | Image-to-Image Search (Tìm ảnh tương đồng từ file ảnh tải lên) |
| `/api/v1/image/{video_id}/{frame_id}` | `GET` | Lấy file ảnh keyframe thực tế phục vụ hiển thị |
| `/api/v1/video/{video_id}` | `GET` | Phát video trực tiếp (HTTP 206 Partial Content Streaming) |
| `/api/v1/video/{video_id}/fps` | `GET` | Lấy FPS chính xác của video |
| `/health` | `GET` | Health check hệ thống, trạng thái GPU và Qdrant |

---

## 2. Chi Tiết Request & Response

### 2.1 `/api/v1/kis` (Known-Item Search)
**Request Body:**
```json
{
  "query": "người phụ nữ mặc áo dài đỏ cầm nón lá đứng bên hồ Gươm",
  "top_k": 100,
  "enable_reranker": true,
  "search_mode": "all",
  "user_choices": {
    "enable_viranker": true,
    "enable_jina_reranker": false,
    "enable_ocr_filter": true
  }
}
```

**Response Body:**
```json
{
  "task_type": "kis",
  "query": "người phụ nữ mặc áo dài đỏ cầm nón lá đứng bên hồ Gươm",
  "enhanced_query": "woman in red ao dai holding conical hat standing by Hoan Kiem Lake",
  "results": [
    {
      "scene_id": "L25_V001_0012",
      "video_id": "L25_V001",
      "frame_id": 340,
      "score": 0.892,
      "timestamp_start": 13.6,
      "timestamp_end": 15.2,
      "fps": 25.0,
      "image_url": "/api/v1/image/L25_V001/340",
      "ocr_text": "HỒ HOÀN KIẾM",
      "asr_text": "chúng ta đang có mặt tại..."
    }
  ],
  "total_candidates": 100,
  "latency_ms": 1850.5
}
```

---

### 2.2 `/api/v1/trake` (Temporal Retrieval)
**Request Body:**
```json
{
  "query": "Đầu tiên là người đàn ông chuẩn bị dùi kẻng. Sau đó ông gõ vào kẻng đồng. Cuối cùng mọi người vỗ tay hưởng ứng.",
  "num_events": 3,
  "top_k": 50
}
```

**Response Body:**
```json
{
  "task_type": "trake",
  "query": "...",
  "sub_events": [
    "người đàn ông chuẩn bị dùi kẻng",
    "ông gõ vào kẻng đồng",
    "mọi người vỗ tay hưởng ứng"
  ],
  "timeline_results": [
    {
      "video_id": "L26_V102",
      "alignment_score": 0.935,
      "keyframes": [
        {"frame_id": 120, "event_idx": 0, "timestamp_sec": 4.8},
        {"frame_id": 245, "event_idx": 1, "timestamp_sec": 9.8},
        {"frame_id": 410, "event_idx": 2, "timestamp_sec": 16.4}
      ]
    }
  ],
  "latency_ms": 2100.2
}
```

---

### 2.3 `/api/v1/similar-image` (Image-to-Image Search)
- **Content-Type:** `multipart/form-data`
- **Params:** `file` (Binary Image JPG/PNG/WebP), `top_k` (default: 100).
