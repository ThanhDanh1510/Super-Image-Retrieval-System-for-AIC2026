# 🎯 ĐỀ XUẤT CẢI TIẾN THUẬT TOÁN TRAKE (AIC2026)

> Ghi chú: các đề xuất bên dưới đã điều chỉnh theo constraint thực tế của dữ liệu — khoảng cách giữa 2 sự kiện liên tiếp **không vượt quá 15 giây**. Điều này loại bỏ nhu cầu xử lý "long-range gap" và cho phép thu hẹp không gian bucket/prior, giảm chi phí tính toán.

---

## 1. Sub-Event Parser — khai thác temporal cue trong câu hỏi

Parser hiện tại chỉ tách $E_1 \to E_2 \to \dots \to E_N$ theo cấu trúc câu, bỏ qua tín hiệu ngôn ngữ về khoảng cách thời gian mà câu hỏi vốn chứa sẵn (ví dụ "ngay sau đó", "một lúc sau").

**Đề xuất**: thêm một **Temporal Cue Extractor** nhỏ chạy song song với Sub-Event Parser, gán cho mỗi cặp $(E_{i-1}, E_i)$ một prior khoảng cách kỳ vọng $\mu_i$, dựa trên cụm từ nối. Do đã biết trần khoảng cách thực tế là 15s, chỉ cần 3 mức:

| Cụm từ nối | $\mu_i$ kỳ vọng |
|---|---|
| "ngay sau đó", "liền sau", "vừa ... đã" | 1–3s |
| "sau đó", "tiếp theo" (không rõ mức độ) | 3–8s |
| "một lúc sau", "cuối cùng" | 8–15s |

Không cần mức "15s+" vì không xảy ra trong dữ liệu — giúp prior gọn và ít nhiễu hơn.

---

## 2. DANTE DP — thay penalty tuyến tính bằng Gaussian prior theo bucket

Công thức gốc:
$$DP[i,t] = S[i,t] + \max_\tau\big(DP[i-1,\tau] - \lambda\cdot(t - \tau)\big)$$

dùng một $\lambda$ chung, giả định "càng gần càng tốt" — không tận dụng được prior ở mục 1.

**Đề xuất công thức**:
$$DP[i,t] = S[i,t] + \max_\tau\Big(DP[i-1,\tau] - \lambda_i\cdot(\Delta t - \mu_i)^2\Big),\quad \Delta t \in [\Delta_{\min},\,15\text{s}]$$

**Cách giữ độ phức tạp $O(N\cdot T)$**: vì trần khoảng cách chỉ 15s, chỉ cần rời rạc hoá $\Delta t$ thành đúng **3 bucket cố định** khớp với bảng ở mục 1 (0–3s / 3–8s / 8–15s), mỗi bucket giữ một biến `running_max` riêng. Với $K=3$ bucket nhỏ và trần 15s, chi phí tăng thêm không đáng kể so với bản gốc ($O(N\cdot T\cdot 3)$), và không cần xử lý case "unbounded gap" như bản đề xuất trước.

---

## 3. C7 (All-Event Completeness) — thêm interpolation fallback thay vì loại cứng

R-Score là trung bình cộng số sự kiện trúng cửa sổ, nghĩa là chuỗi đúng $N-1/N$ vẫn có điểm dương. Nhưng C7 hiện loại bỏ hoàn toàn video nào có một hàng $S[i,\cdot]$ toàn điểm thấp (event "mất tích" do embedding miss).

**Đề xuất**: nếu $S[i,\cdot]$ dưới ngưỡng tin cậy toàn cục, nội suy vị trí kỳ vọng của $f_i$ dựa trên $f_{i-1}$, $f_{i+1}$ đã tìm được. Vì trần gap giữa các sự kiện là 15s, khoảng nội suy luôn bị chặn trong một cửa sổ hẹp $\le 15$s — nội suy tuyến tính đơn giản theo tỷ lệ thời gian là đủ chính xác, không cần mô hình phức tạp hơn. Gán confidence thấp cho $f_i$ nội suy, giữ video trong ranking thay vì loại "oan".

---

## 4. Multi-Chain Hedging (C8) — top-K Viterbi thay vì heuristic gap≥2s

Hiện tại là hedge "thủ công" sau khi đã có 1 best path (2–3 chuỗi/video miễn cách nhau ≥2s).

**Đề xuất**: mở rộng DP thành **top-K DP** (lưu K giá trị tốt nhất tại mỗi state $(i,t)$ thay vì 1, giống k-shortest-path). Vì $T$ đã nhỏ (35–60 frame sau ROI crop ±8s) và không gian bucket $\Delta t$ cũng chỉ còn 3 mức (mục 2), chi phí tăng thêm $O(N\cdot T\cdot K\log K)$ vẫn rất rẻ. Lợi ích: các chuỗi hedge là runner-up có căn cứ xác suất thật, không chỉ "cách xa nhau 2s".

---

## 5. Video Identity (Tầng 1) — Reciprocal Rank Fusion thay vì trọng số cố định

Bước 5 dùng $w_{\text{intro}}=0.35$ cố định để cộng vào điểm ứng viên, trong khi 4 nhánh retrieval (CLIP, BGE, BM25, Qdrant) có phân phối điểm rất khác scale nhau — cộng trực tiếp dễ để nhánh scale lớn áp đảo.

**Đề xuất**: dùng RRF để gộp 4 nhánh trước khi cộng Intro Score:
$$\text{RRF}(v) = \sum_{\text{branch}} \frac{1}{k + \text{rank}_{\text{branch}}(v)} \quad (k = 60)$$
RRF không nhạy scale điểm, chỉ cần thứ hạng — hợp với nhiều nhánh heterogeneous như ở đây.

---

## 6. Adaptive Top-K video candidates thay vì cắt cứng 12–16

$$\text{max\_dense\_vids} = \min(\max(\text{top\_k}+2,\,10),\,16)$$
là hằng số bất kể độ rõ ràng của query.

**Đề xuất**: cắt theo elbow/score-gap detection trên đường cong điểm RRF đã sort — dừng khi $\text{score}[k]/\text{score}[k+1] < 0.9$ liên tiếp 2 lần, sàn min=5, trần max=16.

---

## 7. Preamble Intro Fusion — cutoff 35s cố định không tổng quát

Không phải video nào cũng đặt bối cảnh trong 35s/20% đầu.

**Đề xuất**: thay hard cutoff bằng soft decay $e^{-t/35}$ áp dụng trên toàn video thay vì cắt cứng về 0 sau 35s — vẫn ưu tiên đầu video nhưng không bỏ sót preamble match muộn.

---

## Ưu tiên triển khai (impact / effort)

| Ưu tiên | Đề xuất | Impact | Effort |
|---|---|---|---|
| 1 | C7 → interpolation fallback | Cao | Thấp |
| 2 | Temporal cue extractor + 3-bucket Gaussian penalty | Cao | Trung bình |
| 3 | RRF thay weighted sum ở Tầng 1 | Ổn định | Thấp |
| 4 | Top-K Viterbi cho hedging | Cao | Trung bình |
| 5 | Adaptive top-K candidates (elbow detection) | Trung bình | Thấp |
| 6 | Soft-decay preamble fusion | Trung bình | Thấp |

Hai điểm đáng làm trước nhất là **#1** (chỉ sửa logic DP, không đổi kiến trúc) và **#2** (tận dụng constraint gap≤15s để giữ độ phức tạp thấp mà vẫn thêm tín hiệu ngôn ngữ hữu ích).
