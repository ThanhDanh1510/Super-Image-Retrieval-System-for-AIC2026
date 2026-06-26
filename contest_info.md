# AI Challenge HCMC — Task & Scoring Context (tham chiếu cho pipeline)

## 3 dạng truy vấn (task)

**1. Textual-KIS (Known-Item Search)** — Input: 1 câu/đoạn mô tả tự nhiên. Output: list `(video_id, frame_id)` xếp hạng, top-K ≤ 100. Đúng nếu khớp đúng video VÀ frame nằm trong khoảng `[s, e]` của ground truth.

**2. VQA (Visual Question Answering)** — Input: mô tả + 1 câu hỏi liên quan đến khoảnh khắc đó. Output: `(video_id, frame_id, answer)`. Đúng nếu đúng cả 3: video, frame trong khoảng GT, và câu trả lời khớp GT. Sai 1 trong 3 → 0 điểm.

**3. TRAKE (Temporal Retrieval / multi-event)** — Input: mô tả 1 hành động gồm N khoảnh khắc theo thứ tự (ví dụ các bước của 1 cú nhảy). Output: `(video_id, frame_id_1, ..., frame_id_N)`. Nếu sai video → 0 điểm toàn bộ. Nếu đúng video → điểm là **tỉ lệ phần trăm số khoảnh khắc khớp** (mỗi khoảnh khắc khớp nếu nằm trong sai số ±epsilon quanh frame GT, epsilon do BTC quy định theo từng câu hỏi).

## Công thức chấm điểm

**R-Score** (điểm cho 1 câu trả lời `r_i`):
- KIS: `R-Score = 1[v_i = GT_v ∧ id_i ∈ [s,e]]`
- VQA: `R-Score = 1[v_i = GT_v ∧ id_i ∈ [s,e] ∧ a_i = GT_a]`
- TRAKE: nếu đúng video, `R-Score = (1/N) × Σ 1[id_j ∈ [s_j, e_j]]` — tức trung bình tỉ lệ khớp từng khoảnh khắc; nếu sai video, R-Score = 0.

**Final Score** (điểm cuối cho 1 query, dựa trên toàn bộ list bạn nộp — không chỉ 1 câu trả lời):
1. Tính **R@k** = R-Score cao nhất trong top-k câu trả lời đầu tiên, tại 5 ngưỡng k ∈ {1, 5, 20, 50, 100}.
2. `Final Score = (1/5) × Σ R@k` (trung bình cộng 5 giá trị R@1, R@5, R@20, R@50, R@100).

→ Hệ quả quan trọng cho thiết kế hệ thống: **không chỉ cần tìm ra đáp án đúng, mà phải đẩy nó lên vị trí cao trong list trả về** — vì R@1 chiếm 1/5 trọng số tổng nhưng yêu cầu đáp án đúng phải nằm ở vị trí số 1. Một hệ thống recall tốt nhưng rank kém (đáp án đúng nằm ở vị trí 80) chỉ ăn điểm ở R@100, mất hết điểm ở 4 ngưỡng còn lại — đây chính là lý do reranker quan trọng hơn recall thuần.

