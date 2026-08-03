# DANTE — Dynamic Alignment of Narrative Temporal Events

**Nguồn:** Luu et al., *"Integrated Semantic and Temporal Alignment for Interactive Video Retrieval"* (team AIO_Owlgorithms, HCMC AI Challenge 2025), arXiv:2512.13169

**Bối cảnh:** Thuật toán DP giải task **TRAKE** (Temporal Retrieval and Alignment of Key Events) — task đòi hỏi retrieve một **chuỗi N sự kiện có thứ tự thời gian** trong video, thay vì chỉ 1 khoảnh khắc đơn lẻ như KIS thông thường.

---

## 1. Ý tưởng cốt lõi

Thay vì tìm riêng lẻ từng keyframe khớp nhất với từng event query rồi ráp lại (dễ bị scatter, sai thứ tự), DANTE dùng **quy hoạch động (DP)** để đồng thời:
- tối ưu độ tương đồng semantic giữa mỗi query event và keyframe ứng viên,
- ép buộc **thứ tự thời gian đúng** (event sau phải match keyframe đứng sau event trước),
- phạt nhẹ khoảng cách thời gian quá xa giữa 2 event liên tiếp (không loại bỏ hoàn toàn, chỉ giảm điểm).

## 2. Input

- `U = [u1, ..., uN]`: embedding của N event query, encode bằng **BEiT-3**
- `E[1..T]`: embedding của toàn bộ keyframe trong DB (đã precompute, lưu ở **Milvus**)
- JSON metadata: với mỗi video `v`, biết khoảng keyframe `[sv, ev]` của video đó (nhiều video ghép liên tục thành trục `[1, T]`)
- `λ` (penalty factor): hệ số phạt khoảng cách thời gian

## 3. Similarity matrix

```
S[i, t] = cosine_similarity(u_i, E[t])
```
Tính qua Milvus query, cho mọi event `i` và keyframe `t`.

## 4. Công thức DP (theo từng video v, t ∈ [sv, ev])

**Base case (i = 1):**
```
DP[1, t] = S[1, t]
```

**Recurrence (i = 2..N):**
```
DP[i, t] = S[i, t] + max_{τ ∈ [sv, t-1]} ( DP[i-1, τ] - λ(t - τ) )
```
→ Ý nghĩa: điểm match event `i` vào keyframe `t` = độ tương đồng semantic + điểm tốt nhất có thể đạt được khi event `i-1` match vào một keyframe `τ` đứng **trước** `t`, trừ đi penalty tỉ lệ với khoảng cách `(t-τ)`.

**Tối ưu bằng running max** (tránh O(N·T²) → còn **O(N·T)**):

Đặt `running_max` chạy dọc theo `t`, cập nhật:
```
running_max = max(running_max, DP[i-1, t-1] + λ(t-1))
DP[i, t] = S[i, t] + running_max - λt
```
Nhờ tách `λ(t-τ) = λt - λτ`, chỉ cần giữ giá trị tốt nhất của `DP[i-1, τ] + λτ` tính đến thời điểm hiện tại, không cần quét lại toàn bộ τ mỗi lần.

**Độ phức tạp:** O(N·(ev - sv + 1)) cho mỗi video → O(N·T) tổng thể trên toàn bộ keyframe.

## 5. Điểm cuối cho mỗi video

```
DANTE[v] = max_{t ∈ [sv, ev]} DP[N, t]
```

Chuỗi keyframe tối ưu được **backtrack** từ `DP[N, t_max]` (lưu argmax ở mỗi ô DP để truy vết).

## 6. Output

Top-k video có `DANTE[v]` cao nhất, kèm chuỗi N keyframe (mỗi keyframe ứng với 1 event query) từ backtracking.

## 7. Pseudocode (Algorithm 1 trong paper)

```
Require: U = [u1,...,uN], Metadata, E[1..T], λ
1: Compute S[i,t]  ← cosine_similarity(u_i, E[t])   // qua Milvus
2: for each video v in Metadata:
3:     sv, ev = Metadata[v].start, Metadata[v].end
4:     Init DP[1..N, sv..ev]
5:     for t = sv..ev: DP[1,t] = S[1,t]              // base case
6:     for i = 2..N:
7:         running_max = -inf
8:         for t = sv..ev:
9:             if t > sv:
10:                running_max = max(running_max, DP[i-1,t-1] + λ(t-1))
11:            DP[i,t] = S[i,t] + running_max - λt
12:    DANTE[v] = max_t DP[N,t]
13:    backtrack keyframe sequence
14: Return top-k videos theo DANTE[v] + chuỗi keyframe
```

## 8. Tuning λ (kinh nghiệm thực tế của team, vòng chung kết HCMC AIC 2025)

| Khoảng cách index giữa các event ground-truth | λ tối ưu |
|---|---|
| Gap rộng (3–15 keyframe giữa các event) | **λ = 0.001** |
| Gap hẹp, sát nhau (1–3 keyframe) | **λ = 0.01** |

→ λ càng lớn thì penalty theo khoảng cách càng mạnh, ép các event phải nằm sát nhau hơn.

## 9. Vị trí trong hệ thống tổng thể

- Toàn bộ retrieval pipeline dùng chung: TransNetV2 (shot detection) → 4 keyframe/scene → BEiT-3 embedding (Milvus) + Gemini OCR (Elasticsearch).
- DANTE là 1 trong 3 search mode (Semantic / OCR / DANTE) chọn được ở UI, dùng riêng cho truy vấn nhiều sự kiện có thứ tự (TRAKE), nhưng paper cũng cho thấy nó cải thiện cả KIS khi query có nhiều mô tả tuần tự.
- Kết quả: TRAKE đạt mức "Outstanding" tại vòng chung kết HCMC AIC 2025.

## 10. Nhận xét nhanh (liên hệ AIC2026)

- Về bản chất đây là bài toán **sequence alignment / DTW-style DP** áp lên similarity matrix, khá gần với thuật toán alignment kinh điển (Needleman-Wunsch có soft penalty) chứ không phải kỹ thuật hoàn toàn mới — điểm mới là áp dụng cụ thể cho retrieval trên embedding space BEiT-3 + tối ưu O(NT) bằng running max thay vì quét τ.
- Có thể tái sử dụng trực tiếp cho pipeline TRAKE hiện tại: chỉ cần similarity matrix `S[i,t]` từ encoder đang dùng (BEiT3+CLIP/gte-Qwen2), không phụ thuộc Milvus cụ thể.
- Điểm cần cân nhắc khi áp dụng: λ cần tune lại theo mật độ sampling keyframe của hệ thống (paper sample 4 frame/scene TransNetV2, hệ thống khác có density khác nhau → khoảng gap index không tương đương).
