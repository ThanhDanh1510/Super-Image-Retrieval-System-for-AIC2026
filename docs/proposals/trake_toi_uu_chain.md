# 🎯 TỐI ƯU TRAKE: TÌM ĐÚNG CHAIN SỰ KIỆN (Temporal Alignment)

**Mục tiêu:** Tăng R-Score bằng cách tối ưu từng lớp trong pipeline TRAKE để chuỗi N-tuple trả về khớp chính xác với `[s_j, e_j]` ground-truth, trên đúng video.

**Cơ sở:** Tổng hợp từ `trake_evaluation.md` (test suite 24/24 PASSED, nhánh `test/trake-precision-engine`), sắp xếp lại theo **đòn bẩy tác động đến độ chính xác của chain** thay vì theo thứ tự pipeline.

---

## 0. Nguyên tắc chỉ đạo

R-Score chỉ tính điểm khi **đúng video** (`v_i = GT_v`), và trong video đúng, tính theo tỉ lệ sự kiện rơi đúng khoảng `[s_j, e_j]`. Nghĩa là tối ưu chia làm 2 tầng tách biệt, không được đánh đổi lẫn nhau:

1. **Tầng Video Identity** — đưa đúng video lên top (R@1/5/20/50/100)
2. **Tầng Frame Alignment** — trong video đúng, alignment N sự kiện phải rơi đúng khung hẹp

Một pipeline có Tầng 1 tốt nhưng Tầng 2 yếu vẫn mất điểm toàn bộ hàng đó (`R-Score = 0` nếu sai video, hoặc `< 1.0` nếu đúng video nhưng lệch frame). Vì vậy **thứ tự ưu tiên tối ưu nên đi từ Tầng 1 → Tầng 2**, không ngược lại.

---

## 1. TẦNG VIDEO IDENTITY — Đưa đúng video lên top

### 1.1 Siết lại Preamble Intro Fusion (đang ⭐⭐⭐)
Vấn đề hiện tại: trọng số `0.65/0.35` và ngưỡng `35.0s` hardcode, chỉ kích hoạt khi preamble chứa từ khóa "bắt đầu/đầu tiên/mở đầu".

**Cách tối ưu:**
- Đưa `intro_fusion_weight` và `preamble_time_cutoff` vào `fusion_config.yaml`, cho phép A/B test nhiều bộ trọng số trên tập dev.
- Thay ngưỡng cứng bằng ngưỡng **tương đối theo độ dài video**: `cutoff = min(35.0, video_duration * 0.15)` — video ngắn 30s không nên áp cutoff 35s (bằng cả video).
- Mở rộng kích hoạt intro-boost: thay vì chỉ dò từ khóa "bắt đầu", dùng similarity giữa preamble-embedding và frame đầu video như một **soft signal** luôn cộng dồn (trọng số nhỏ), thay vì on/off nhị phân theo từ khóa.

### 1.2 Multi-chain hedging cho video top-1
Hiện tại `max_chains_per_video=1` (hardcode override, trong khi config ghi `5`) — nếu đúng video nhưng alignment sai, không có phương án dự phòng.

**Cách tối ưu:**
- Cho phép **2–3 chains/video** cho video ở top-3 sau bước 3 (Video Candidate Selection), giữ `1 chain/video` cho các video còn lại để không nổ số lượng submission.
- Áp dụng non-max suppression theo **overlap thời gian** giữa các chains cùng video (không phải giữa các video) — 2 chain của cùng 1 video chỉ giữ lại nếu chain thứ 2 lệch khỏi chain thứ nhất tối thiểu vài giây, tránh 2 chain gần như trùng nhau chiếm 2 suất trong 100 submissions.

### 1.3 Fix MatchText → MatchValue (Bug #3, P0)
Full-text tokenizer của Qdrant có thể tách `L01_V001` → `L01` + `V001`, khiến scroll bổ sung timeline nhầm sang video khác — trực tiếp phá Tầng 1. Đây là bug **ưu tiên fix trước tiên** vì nó âm thầm làm sai video mà không báo lỗi.

---

## 2. TẦNG FRAME ALIGNMENT — Rơi đúng khoảng [s_j, e_j]

### 2.1 Dense Similarity Matrix — đã tốt, cần khai thác thêm
$S[i,t]$ loại bỏ hoàn toàn điểm 0.0 giả — nền tảng đúng. Điểm cần khai thác thêm:
- Đảm bảo **L2-normalize** áp dụng nhất quán cho cả $u_i$ (sub-event embedding) và $v_t$ (frame embedding) trước khi tính cosine — sai lệch chuẩn hóa ở 1 phía làm méo toàn bộ ma trận DP ăn theo.
- Với sub-event cực ngắn (hành động diễn ra dưới 1 giây), cosine similarity ở độ phân giải 1.0s/frame có thể "làm mờ" tín hiệu — đây là lý do Dense Frame Extraction (mục 2.3) quan trọng.

### 2.2 DANTE DP Engine — 2 điểm chỉnh để tăng độ chính xác
- **Fix in-place score mutation (Bug #2, P1):** `cand.score = round(event_sims[k], 4)` ghi đè object dùng chung. Clone `CandidateResult` trước khi gán score trong backtracking — nếu không fix, điểm hiển thị UI cho candidate xuất hiện ở nhiều chain sẽ sai, gây khó debug khi tinh chỉnh λ.
- **Tune λ (lambda_penalty) theo mật độ keyframe thực tế của từng video**, thay vì hằng số `0.001` toàn cục. Video có keyframe thưa cần λ nhỏ hơn (ít phạt khoảng cách) để tránh DP bỏ lỡ alignment đúng chỉ vì 2 sự kiện cách nhau xa về thời gian nhưng đúng về nội dung.
- Giữ nguyên **Min-Separation Constraint (1.0s)** và **First-Occurrence Bias** — đây là 2 cơ chế đúng hướng, không cần đổi.

### 2.3 Dense Frame Extraction — lớp quyết định độ chính xác frame-level
Đây là lớp ảnh hưởng trực tiếp nhất đến việc rơi đúng `[s_j, e_j]` (khoảng ground-truth thường dưới 10 frames), nhưng đang bị giới hạn bởi hardcode nhiều nhất:

| Tham số | Hiện tại | Đề xuất tối ưu |
|---|---|---|
| Số video áp dụng dense extraction | Top 2 (hardcode) | Đồng bộ với config `max_dense_videos=3`, ưu tiên video có preamble-score cao nhất |
| Bước nhảy thời gian | 1.0s (hardcode) | Đồng bộ config `0.5s`; với sự kiện cực ngắn nên cho phép 0.25–0.5s |
| Max frames | 150 (hardcode) | Nâng lên ≥300 để cover 5 phút ở bước 1.0s, hoặc tính động: `max_frames = min(600, video_duration / interval_sec)` |
| `keyframe_path` | Giả lập, ảnh không tồn tại | Thêm on-the-fly extract + cache để UI hiển thị đúng thumbnail — không ảnh hưởng R-Score nhưng ảnh hưởng khả năng con người kiểm tra kết quả trước khi submit |

**Ưu tiên đồng bộ config-code trước khi tinh chỉnh thêm** — hiện `max_dense_videos`, `dense_sampling_interval_sec` đọc từ code hardcode thay vì `fusion_config.yaml`, nên mọi thử nghiệm tune config hiện tại **không có tác dụng thực tế**.

### 2.4 Context-Aware Translation — chất lượng đầu vào cho cả 2 tầng
Co-reference resolution ("con lân" → mô tả chi tiết) ảnh hưởng cả việc tìm đúng video lẫn đúng frame, vì embedding query càng cụ thể thì similarity càng sắc nét.

- Domain rules hiện chỉ cover múa lân — cần mở rộng dần theo tần suất xuất hiện chủ đề trong tập thi thực tế (thể thao, giao thông, nấu ăn...), ưu tiên theo dữ liệu thật thay vì làm hết một lượt.
- Với Qwen2.5-1.5B nhỏ, nên thêm bước validate: nếu bản dịch không chứa các danh từ/động từ chính từ câu gốc (kiểm tra bằng keyword overlap đơn giản), fallback ngay sang GoogleTranslator thay vì tin tưởng mù quáng.

---

## 3. FIX P0 BẮT BUỘC TRƯỚC KHI TỐI ƯU THÊM

Theo `trake_evaluation.md`, 2 bug sau **phải fix trước** vì chúng làm hỏng cả 2 tầng, mọi tinh chỉnh khác đều vô nghĩa nếu còn tồn tại:

1. **Voice TRAKE crash** — `base_res.query_text` không tồn tại trên `PipelineResult`, gây `AttributeError` ngay khi dùng voice input. Sửa thành `base_res.transcript or base_res.enhanced_query`.
2. **MatchText → MatchValue** cho filter `video_id` trong Qdrant scroll (`qdrant_repo.py:599,637`) — nguy cơ match nhầm video, phá Tầng Video Identity.

---

## 4. QUY TRÌNH THỬ NGHIỆM ĐỀ XUẤT (A/B trên tập dev)

1. Fix 2 bug P0 → chạy lại 24 test hiện có, đảm bảo không regress.
2. Đồng bộ config-code (`max_dense_videos`, `interval_sec`, `max_chains_per_video`) → đo lại R@1/5/20/50/100 trên tập dev để có baseline "sạch".
3. Bật multi-chain hedging (2–3 chain/video top-3) → so sánh Final Score trước/sau, theo dõi số lượng submission có bị lãng phí vào chain trùng lặp không.
4. Tune λ theo mật độ keyframe (thử 2–3 mức λ theo nhóm video thưa/dày keyframe) → đo ảnh hưởng riêng lên nhóm sub-event cực ngắn (dùi chạm kẻng, chân chạm đất).
5. Đưa intro-fusion weight về dạng config, thử ít nhất 3 bộ trọng số (vd. 0.5/0.5, 0.65/0.35, 0.8/0.2) → chọn bộ tối đa hoá R@1 mà không làm giảm R@5.

---

## 5. TÓM TẮT ƯU TIÊN

| Thứ tự | Việc làm | Tầng ảnh hưởng | Độ khó |
|---|---|---|---|
| 1 | Fix MatchText → MatchValue | Video Identity | Thấp |
| 2 | Fix voice crash | Cả 2 (chặn hoàn toàn voice input) | Thấp |
| 3 | Đồng bộ config-code (dense videos, interval, max_frames) | Frame Alignment | Thấp |
| 4 | Fix in-place score mutation | Frame Alignment (độ tin cậy điểm) | Thấp |
| 5 | Multi-chain hedging top-3 video | Frame Alignment (hedge risk) | Trung bình |
| 6 | Configurable intro-fusion weight + cutoff động theo duration | Video Identity | Trung bình |
| 7 | Tune λ theo mật độ keyframe từng video | Frame Alignment | Trung bình-Cao |
| 8 | Mở rộng domain rules translation | Cả 2 (chất lượng embedding) | Cao (cần dữ liệu thật) |
