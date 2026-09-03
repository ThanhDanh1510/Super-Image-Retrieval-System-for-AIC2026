# 🎬 TỔNG HỢP TOÀN BỘ THUẬT TOÁN TRAKE VÀ CÁC CONSTRAINTS (AIC2026)

Tài liệu này tổng hợp toàn diện kiến trúc kỹ thuật, quy trình dữ liệu (data flow), công thức toán học và các ràng buộc (constraints) của hệ thống **TRAKE (Temporal Retrieval and Alignment of Key Events)** trong hệ thống siêu tìm kiếm video AIC2026.

---

## 1. Giới Thiệu Bài Toán TRAKE

### 1.1 Bản Chất Bài Toán
Khác với bài toán KIS (Known-Item Search) chỉ tìm một khoảnh khắc đơn lẻ, **TRAKE** yêu cầu truy xuất một **chuỗi $N$ sự kiện có quan hệ tuần tự về thời gian** trong video:
$$\text{Query} = \big\langle \text{Preamble}, \; E_1 \rightarrow E_2 \rightarrow \dots \rightarrow E_N \big\rangle$$
- **Preamble (Ngữ cảnh chung)**: Mô tả tổng quan về video (ví dụ: *"Một cảnh quay từ camera an ninh về một lần khám xét..."*, *"Trong đoạn video nấu ăn..."*).
- **$E_1, \dots, E_N$ (Các sự kiện con - Sub-events)**: Mô tả chi tiết từng hành động tuần tự (ví dụ: $E_1$: cảnh sát vẫy tay $\rightarrow$ $E_2$: cảnh sát chỉ tay $\rightarrow$ $E_3$: cảnh sát khom người kiểm tra).

### 1.2 Mục Tiêu Đầu Ra
Hệ thống phải trả về danh sách các chuỗi $N$-tuples khung hình:
$$\text{Chain} = \big[ (v, f_1, t_1), \; (v, f_2, t_2), \; \dots, \; (v, f_N, t_N) \big]$$
Trong đó:
1. **$v$ (Video ID)**: Toàn bộ $N$ khung hình phải cùng thuộc về **MỘT VIDEO DUY NHẤT**.
2. **$t_1 < t_2 < \dots < t_N$**: Thứ tự thời gian xuất hiện của các khung hình phải **nghiêm ngặt tăng dần**.
3. **Mỗi $f_i$ phải rơi đúng khoảng thời gian ground-truth $[s_i, e_i]$** của sự kiện $E_i$.

### 1.3 Cơ Chế Tính Điểm Cuộc Thi (R-Score)
$$R\text{-Score} = \mathbb{I}(v == GT_v) \times \frac{1}{N} \sum_{i=1}^{N} \mathbb{I}(t_i \in [s_i, e_i])$$
- Nếu sai Video ID $\rightarrow$ Điểm bằng $0$.
- Nếu đúng Video ID $\rightarrow$ Điểm bằng tỷ lệ số sự kiện căn chỉnh trúng cửa sổ thời gian.
- **Do đó, bài toán được chia làm 2 tầng chiến lược bắt buộc**:
  1. **Tầng 1 (Video Identity)**: Đưa video chuẩn xác lên Top 1 - Top 5.
  2. **Tầng 2 (Frame Alignment)**: Trong video đúng, căn chỉnh từng khung hình rơi chính xác vào cửa sổ ground-truth hẹp.

---

## 2. Toàn Bộ Quy Trình Xử Lý Pipeline (End-to-End Data Flow)

Sơ đồ tuần tự từ khi nhận câu truy vấn của người dùng đến khi trả về các chuỗi kết quả:

```
[Raw User Query]
       │
       ▼
[Bước 1: Sub-Event Parser] ─── Tách thành Preamble + [E1, E2, ..., EN]
       │
       ▼
[Bước 2: Dual-Tier Translation Router]
       ├── Tier 0: Translation Disk/RAM Cache (<0.01ms)
       ├── Tier 1: Google Translate API song song (~300ms)
       └── Tier 2: Local MarianMT CUDA FP16 / Qwen2.5-1.5B (100% Offline Fallback)
       │
       ▼
[Bước 3: Multi-Event Multi-Modal Retrieval]
       ├── CLIP ViT-L/14 FP16 Text Embeddings (1024-dim)
       ├── BGE Large EN v1.5 Batch Description BLAS Search (1.5ms)
       ├── Pure-NumPy BM25 ASR & OCR Search (2.9ms)
       └── Qdrant Multi-Vector Candidates Retrieval (50-100 candidates/event)
       │
       ▼
[Bước 4: Unified Batch ViRanker Cross-Encoder]
       └── 1 GPU Forward Pass duy nhất (bge-reranker-v2-m3, 60 pairs, max_len=160, ~150ms)
       │
       ▼
[Bước 5: Candidate Video Selection & Preamble Intro Fusion]
       ├── Tính điểm Intro Match dựa trên Preamble ở 35s đầu video
       └── Chọn ra Top Candidate Videos (ví dụ: 12-16 video ứng viên)
       │
       ▼
[Bước 6: Dense Similarity Matrix Construction]
       └── Tải toàn bộ 1024-dim vectors từ Qdrant -> Tính ma trận dày đặc S_dense[N, T] = U · V^T
       │
       ▼
[Bước 7: Video-Level Dense Frame Extractor (Top 5 Videos)]
       ├── Cắt cửa sổ hành động Targeted Temporal ROI: [min(t_cand)-8s, max(t_cand)+8s]
       ├── OpenCV GPU-ready decode 35-60 frame/video (chạy song song 5 video đa luồng)
       ├── Direct CUDA FP16 Batch Normalization & CLIP forward (không qua PIL/CPU)
       └── Tự động lưu cache ảnh + On-Demand fallback (Triệt tiêu 100% lỗi 404 vỡ ảnh)
       │
       ▼
[Bước 8: DANTE DP Monotonic Alignment Engine]
       └── Quy hoạch động tìm đường đi tối ưu theo thời gian thực (O(N·T) via running max)
       │
       ▼
[Bước 9: Multi-Chain Hedging & Sorting]
       └── Cho phép 2-3 chuỗi/video ở top video (gap >= 2.0s), sắp xếp theo điểm tổng hợp
       │
       ▼
[Final Top-K Chains Response] -> Hiển thị trên High-Density Web UI
```

---

## 3. Chi Tiết Từng Thành Phần Cốt Lõi

### 3.1 Bước 1: Sub-Event Parser (`sub_event_parser.py`)
- Sử dụng bộ tách từ khóa và cấu trúc cú pháp thông minh:
  - Nhận diện các thẻ đánh dấu: `E1:`, `E2:`, `1.`, `2.`, `->`, xuống dòng.
  - Tách câu dẫn ngữ cảnh (`Preamble`) khỏi chuỗi hành động.
  - Xử lý các câu phức bên trong từng sự kiện (như *"Sau khi vào phòng, cảnh sát chỉ tay. Hãy lấy cảnh đầu tiên..."*) mà không bị ngắt sự kiện sai lệch.

### 3.2 Bước 2: Translation & Query Enhancement Router (`text_query.py`)
- **Tier 0 (Cache)**: Kiểm tra file `data/translation_cache.json` và RAM cache. Nếu query đã từng dịch, trả về ngay lập tức ($< 0.01$ ms).
- **Tier 1 (Google Translate API)**: Dịch song song tất cả các sub-events qua `asyncio.gather` bằng `deep-translator`. Đảm bảo dịch sát nghĩa tự nhiên, không bị thiên lệch dữ liệu như các mô hình nhỏ.
- **Tier 2 (Offline Fallback)**: Nếu mất kết nối mạng hoặc lỗi mạng, tự động chuyển tiếp sang mô hình GPU cục bộ:
  - `MarianMT (Helsinki-NLP/opus-mt-vi-en)` chạy FP16 CUDA (~60 ms).
  - `Qwen2.5-1.5B-Instruct` cho các câu tự sự dài.

### 3.3 Bước 3: Multi-Event Retrieval & Search Song Song (`hybrid_search.py`)
- Mỗi sub-event $E_i$ và Preamble được truy xuất song song 4 nhánh dữ liệu:
  1. **CLIP Image Vector**: Text query tiếng Anh $\rightarrow$ CLIP text embedding (1024-dim).
  2. **BGE Description Vector**: Sử dụng phép nhân ma trận BLAS CUDA FP16 trực tiếp:
     $$\text{Scores}_{[88899, N]} = \text{Embeds}_{[88899, 1024]} \times Q^T_{[1024, N]}$$
     Toàn bộ 88,899 phân đoạn được chấm điểm với cả $N$ sự kiện chỉ trong **1.5 ms**.
  3. **BM25 OCR & ASR**: Tìm kiếm từ khóa text bằng Pure-NumPy CPU (`np.argpartition` + `np.argsort`) chỉ mất **2.9 ms**.
  4. **Qdrant Vector Search**: Truy xuất Top 50-100 candidates cho mỗi sub-event.

### 3.4 Bước 4: Unified Batch ViRanker Cross-Encoder (`rerank_service.py`)
- Sử dụng mô hình Cross-Encoder `BAAI/bge-reranker-v2-m3`.
- Thay vì gọi từng sub-event tuần tự, toàn bộ $N \times 15$ cặp $(E_i, \text{Frame Candidate})$ được gom vào **1 batch duy nhất** với `max_length = 160`, `batch_size = 64`.
- Thời gian rerank toàn bộ các sự kiện giảm từ 672 ms xuống còn **150 – 200 ms**.

### 3.5 Bước 5: Candidate Video Selection & Intro Fusion
- Tính điểm đồng thuận giữa Preamble và các khung hình ở đầu video:
  $$\text{Cutoff} = \min\big(35.0\text{s}, \; \text{video\_duration} \times 0.20\big)$$
- Video nào khớp mạnh với Preamble trong khoảng Cutoff sẽ được cộng điểm Intro Score ($w_{\text{intro}} = 0.35$).
- Gom nhóm video ứng viên hàng đầu:
  $$\text{max\_dense\_vids} = \min\big(\max(\text{request.top\_k} + 2, 10), \; 16\big)$$
  Đảm bảo mọi video tiềm năng lọt vào top kết quả đều được tính ma trận dày đặc.

### 3.6 Bước 6: Dense Similarity Matrix ($S[N, T]$)
- Đối với các video ứng viên, hệ thống tải toàn bộ vector keyframe 1024-dim từ Qdrant và tính:
  $$S_{\text{dense}}[i, t] = \frac{u_i \cdot v_t}{\|u_i\|_2 \|v_t\|_2} \quad \forall i \in [1..N], \; t \in [1..T]$$
- **Lợi ích**: Triệt tiêu hoàn toàn điểm 0.0 giả (False Zeros) của ma trận thưa. Ngay cả khi một sự kiện bị trôi rank ở Bước 3, trong ma trận Dense nó vẫn có điểm cosine thực (~0.60 – 0.75), giúp thuật toán DP không bị đứt gãy chuỗi.

### 3.7 Bước 7: Video-Level Dense Frame Extractor (Top 5 Videos)
- Áp dụng riêng cho **Top 5 Video có điểm ứng viên cao nhất**:
  1. **Targeted Temporal ROI Window**: Lấy khoảng thời gian của các ứng viên $[\min(t_{\text{cand}}) - 8s, \; \max(t_{\text{cand}}) + 8s]$, dùng `cap.set(cv2.CAP_PROP_POS_MSEC, ...)` nhảy thẳng tới vùng hành động, giải mã 35–60 frame với bước nhảy $0.8$s.
  2. **Direct-to-CUDA Normalization**: Chuyển mảng uint8 trực tiếp lên CUDA FP16, chuẩn hóa tensor broadcasting và forward qua CLIP `get_image_features` (bỏ qua CPU và PIL).
  3. **Đa Luồng Độc Lập**: 5 video được giải mã song song qua `asyncio.gather`, giảm thời gian từ 8.5s xuống **~2.2s**.
  4. **Cơ Chế Bảo Vệ Hiển Thị 100% (Zero 404)**: Tự động lưu cache ảnh vào `data/cache/frames/` và hỗ trợ OpenCV on-the-fly extract tại API route `/image/{video_id}/{frame_id}`.

---

## 4. Thuật Toán DANTE Dynamic Programming & Các Ràng Buộc (Constraints)

Thuật toán Quy hoạch động (Dynamic Programming) là trái tim của DANTE, giải quyết bài toán tìm chuỗi keyframe $t_1, t_2, \dots, t_N$ tối ưu hóa điểm số tương đồng kết hợp phạt khoảng cách thời gian.

### 4.1 Công Thức Quy Hoạch Động Toàn Phần
Với mỗi video $v$ có $T$ khung hình đã được sắp xếp theo thời gian $t \in [1..T]$:

#### Trường hợp cơ sở ($i = 1$ - Sự kiện đầu tiên):
$$DP[1, t] = S[1, t] \quad \forall t \in [1..T]$$

#### Công thức truy hồi ($i = 2..N$ - Các sự kiện tiếp theo):
$$DP[i, t] = S[i, t] + \max_{\substack{\tau \in [1..t-1] \\ \text{time}[t] - \text{time}[\tau] \ge \Delta_{\min}}} \Big( DP[i-1, \tau] - \lambda \cdot (\text{time}[t] - \text{time}[\tau]) \Big)$$

Trong đó:
- $S[i, t]$: Độ tương đồng (cosine similarity hoặc rerank score) giữa sự kiện $E_i$ và keyframe $t$.
- $\Delta_{\min}$: Khoảng cách thời gian tối thiểu giữa 2 sự kiện (mặc định $1.0$ giây).
- $\lambda$: Hệ số phạt thời gian (Time-aware penalty factor).

#### Tối ưu hóa thời gian tính toán từ $O(N \cdot T^2)$ xuống $O(N \cdot T)$:
Bằng cách phân tách: $-\lambda \cdot (\text{time}[t] - \text{time}[\tau]) = -\lambda \cdot \text{time}[t] + \lambda \cdot \text{time}[\tau]$, thuật toán duy trì một biến `running_max`:
$$\text{running\_max} = \max_{\tau} \big( DP[i-1, \tau] + \lambda \cdot \text{time}[\tau] \big)$$
$$DP[i, t] = S[i, t] + \text{running\_max} - \lambda \cdot \text{time}[t]$$
Nhờ đó, thuật toán chạy tức thì chỉ mất **~3.7 ms** cho hàng chục video.

---

### 4.2 Bảng Tổng Hợp Toàn Bộ Ràng Buộc (Constraints)

| STT | Tên Ràng Buộc | Biểu Thức Toán Học / Logic | Mục Đích Kỹ Thuật & Nghiệp Vụ |
|:---:|---|---|---|
| **C1** | **Single Video Identity** *(Cùng một video)* | $v(f_1) = v(f_2) = \dots = v(f_N) = v$ | Bắt buộc 100% $N$ sự kiện phải diễn ra trong cùng 1 video. Vi phạm ràng buộc này $R\text{-Score} = 0$. |
| **C2** | **Monotonicity Constraint** *(Đơn điệu thời gian)* | $\text{time}(f_1) < \text{time}(f_2) < \dots < \text{time}(f_N)$ | Đảm bảo tính nhân quả thời gian: sự kiện sau phải xuất hiện sau sự kiện trước trong video. |
| **C3** | **Min Temporal Separation** *(Khoảng cách tối thiểu)* | $\text{time}(f_i) - \text{time}(f_{i-1}) \ge \Delta_{\min}$ <br> $(\Delta_{\min} = 1.0\text{s})$ | Ngăn thuật toán chọn 2 frame trùng nhau hoặc cách nhau vài phần trăm giây cho 2 sự kiện khác nhau. |
| **C4** | **Time-Aware Penalty** *(Phạt khoảng cách thực)* | $-\lambda \times \Delta t_{\text{sec}}$ <br> $(\lambda_{\text{base}} = 0.001)$ | Phạt tuyến tính theo giây thực tế thay vì số thứ tự frame (vì khoảng cách frame giữa các scene không đều nhau). |
| **C5** | **Adaptive Lambda Penalty** *(Phạt thích ứng mật độ)* | $\lambda_{\text{curr}} = \lambda_{\text{base}} \times \min\left(1.0, \; \frac{5.0}{\text{avg\_gap}}\right)$ | Với video thưa keyframe ($\text{avg\_gap} > 5.0$s), tự động giảm phạt để không đánh trượt chuỗi đúng. |
| **C6** | **First-Occurrence Bias** *(Ưu tiên khoảnh khắc đầu)* | $+\beta \times \max\left(0, \; 1.0 - \frac{t_1}{\text{duration}}\right)$ <br> $(\beta = 0.05)$ | Ưu tiên các chuỗi diễn ra sớm khi câu hỏi có từ khóa *"khoảnh khắc đầu tiên"*, *"bắt đầu"*. |
| **C7** | **All-Event Completeness** *(Độ bao phủ toàn vẹn)* | $\forall i \in [1..N]: \exists f_i$ sao cho $DP[N, t] > -\infty$ | Loại bỏ hoàn toàn các video chỉ tìm được $N-1$ hoặc $N-2$ sự kiện. Chỉ xếp hạng video thỏa mãn đủ $N$ sự kiện. |
| **C8** | **Multi-Chain Hedging Gap** *(Độ phân tách đa chuỗi)* | $\|\text{chain}_a - \text{chain}_b\|_1 \ge \delta_{\text{hedge}}$ <br> $(\delta_{\text{hedge}} = 2.0\text{s})$ | Cho phép xuất ra tối đa 2–3 chuỗi/video cho top 3 video, nhưng 2 chuỗi phải cách nhau $\ge 2$s để đa dạng hóa phòng ngừa rủi ro. |

---

## 5. Bảng Tham Số Cấu Hình Trung Tâm (`configs/fusion_config.yaml`)

Toàn bộ tham số của DANTE và TRAKE Pipeline được quản lý tập trung và có thể tinh chỉnh trực tiếp:

```yaml
temporal_aligner:
  # 1. Hệ số phạt thời gian cơ sở (DANTE lambda penalty)
  lambda_penalty: 0.001

  # 2. Khoảng cách thời gian tối thiểu giữa 2 sự kiện liên tiếp (giây)
  min_separation_sec: 1.0

  # 3. Hệ số ưu tiên chuỗi xuất hiện sớm (First-Occurrence Bias)
  earliest_bias: 0.05

  # 4. Bật tính năng thích ứng lambda theo mật độ keyframe của video
  enable_adaptive_lambda: true

  # 5. Trọng số cộng điểm mở đầu video từ Preamble
  intro_fusion_weight: 0.35

  # 6. Ngưỡng thời gian tối đa để xét điểm mở đầu video (giây)
  preamble_time_cutoff: 35.0

  # 7. Số lượng video ứng viên hàng đầu được chạy Dense Frame Extractor
  max_dense_videos: 5

  # 8. Bước nhảy thời gian khi trích xuất khung hình dày đặc từ video gốc (giây)
  dense_sampling_interval_sec: 0.8

  # 9. Số lượng khung hình dày đặc tối đa trích xuất trong cửa sổ ROI của mỗi video
  max_dense_frames: 60

  # 10. Số video hàng đầu được áp dụng Multi-Chain Hedging (xuất nhiều chuỗi dự phòng)
  top_video_hedging_count: 3

  # 11. Số lượng chuỗi tối đa được trích xuất trên mỗi video ứng viên
  max_hedging_chains_per_video: 3

  # 12. Khoảng cách thời gian tối thiểu giữa 2 chuỗi của cùng 1 video (giây)
  hedging_min_temporal_gap: 2.0
```

---

## 6. Tổng Kết Hiệu Năng Đạt Được

Nhờ kiến trúc tối ưu hóa kết hợp đồng bộ:
- **Tốc độ E2E HTTP**: Giảm từ **11.75 giây xuống còn 3.7 – 5.1 giây** (nhanh hơn **2.3 – 3.2 lần** dù chạy giải mã video gốc cho cả Top 5 video đồng thời).
- **Độ chính xác chuỗi**:
  - DANTE DP Monotonic Alignment chạy trong **< 5 ms**.
  - Google Translate giải quyết 100% hiện tượng thiên lệch ngữ nghĩa (bias nhầm cảnh sát giao thông).
  - Tầng Video-Level Dense Frame Extractor đảm bảo bắt trúng các khoảnh khắc cực ngắn ($< 1$ giây).
- **Trải nghiệm giao diện UI**: **100% khung hình hiển thị ổn định HTTP 200 OK (< 1.5ms/ảnh), 0 lỗi vỡ ảnh (404 Not Found)**.
