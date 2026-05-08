# Phase 5 — Tổng hợp khoảng cách và xếp hạng

**Module liên quan:**
- `src/distances.py` — kho metric dùng chung
- `src/matcher_hybrid3.py` — tổ hợp 3 nhánh đặc trưng (hybrid3)
- `src/database_hybrid3.py:query` — entry point cho UI/CLI

**Mục tiêu:** Cho 1 ảnh truy vấn, ghép khoảng cách trên **3 nhánh đặc trưng độc lập** (color 576-d, gradient 81-d, compact6 6-d) thành **1 con số duy nhất** để xếp hạng top-k ảnh tương tự.

---

## 1. Câu hỏi quan trọng nhất: có 4 metric (l2, l1, chi2, intersection), có quá phức tạp không?

**Trả lời ngắn:** Không phức tạp về *vận hành*, nhưng nhiều hơn cần thiết về *mặc định*. Cụ thể:

| Vai trò | Có dùng `--distance` chọn không? | Mặc định | Có cần thiết không? |
|---|---|---|---|
| Color histogram (576-d) | ✅ có | `l2` | **Có** — đây là loại đặc trưng *đáng để thử*. Histogram là phân phối xác suất, l2 không tối ưu về mặt lý thuyết. |
| Gradient histogram (81-d) | ❌ không, hard-code l2 | `l2` | Cũng là histogram, *về lý thuyết* nên cho phép chọn — hiện chưa làm vì giữ scope nhỏ. |
| Compact6 (6-d) | ❌ không, hard-code l2 | `l2` | **Đúng** — đây là vector số thực thường, không phải histogram, l2 là mặc định hợp lý. |

→ Phức tạp xuất hiện ở **đúng 1 nhánh duy nhất** (color hist). Nếu thấy không cần thì có thể bỏ flag `--distance`, để mặc định `l2` cho cả 3 nhánh — không mất gì lớn vì:

- `l2` luôn cho ranking *hợp lý* (không tối ưu nhất, nhưng không sai).
- Mean-normalize ở mỗi nhánh (xem mục 4) đã giúp cân bằng amplitude — nên `l2` cũng không bị áp đảo bởi nhánh nào.

**Khuyến nghị giữ lại** `--distance` chỉ vì tính giáo dục/khả năng so sánh, không vì nhu cầu vận hành.

---

## 2. Bốn metric — ý nghĩa toán học và khi nào dùng

Tất cả định nghĩa tại `src/distances.py`. Mỗi hàm nhận `query (D,)` và `db (N, D)`, trả mảng khoảng cách `(N,)`.

### 2.1 Euclidean / L2 — `euclidean_distance_batch` (`distances.py:25-29`)

$$d(p, q) = \sqrt{\sum_i (p_i - q_i)^2}$$

- Generic, hoạt động trên **mọi vector số**.
- Phù hợp khi các chiều **độc lập, cùng đơn vị, có ý nghĩa hình học** (như compact6 sau khi đã chuẩn hóa thang).
- Trên histogram: vẫn cho ranking hợp lý nhưng *không tối ưu*. Hai bin "rất đầy" lệch nhau 0.05 sẽ đóng góp giống hai bin "rất rỗng" lệch 0.05, dù về xác suất bin rỗng quan trọng hơn (hiếm gặp).

### 2.2 Manhattan / L1 — `manhattan_distance_batch` (`distances.py:32-36`)

$$d(p, q) = \sum_i |p_i - q_i|$$

- **Ít nhạy với outlier** hơn L2 (không bình phương).
- Trên histogram đã L1-normalize: gọi là *Total Variation distance × 2*; có ý nghĩa thống kê rõ ràng.

### 2.3 Chi-square — `chi_square_distance_batch` (`distances.py:39-51`)

$$d(p, q) = \frac{1}{2} \sum_i \frac{(p_i - q_i)^2}{p_i + q_i + \varepsilon}$$

- **Chuẩn vàng cho histogram** trong CV cổ điển (Bag-of-Words, color histograms).
- Chia cho `(p_i + q_i)` → bin lớn (xác suất cao) bị **điều chỉnh giảm trọng số**, bin nhỏ (xác suất thấp) tăng trọng số tương đối.
- Hệ quả: hai histogram giống nhau ở bin chính nhưng khác ở bin nhỏ (màu ít gặp) sẽ có chi² lớn hơn l2 — phù hợp với trực giác "màu hiếm phân biệt ảnh tốt hơn màu phổ thông".
- Quan sát thực tế trong test: chi² giảm `color_contribution` ~34% so với L2 ở dataset này.

### 2.4 Histogram intersection — `histogram_intersection_distance_batch` (`distances.py:54-64`)

$$d(p, q) = 1 - \sum_i \min(p_i, q_i)$$

- Yêu cầu cả hai vector đã L1-normalize (ở đây có).
- Khoảng giá trị **chính xác [0, 1]** — 0 trùng tuyệt đối, 1 không giao.
- Cổ điển trong CBIR (Swain & Ballard 1991). Tương đương L1 cho phân phối xác suất, sai khác hằng số.

### Registry

`HIST_METRICS` (`distances.py:67-72`) là `dict` ánh xạ tên → hàm. `get_hist_metric(name)` lookup an toàn (raise nếu sai tên). CLI dùng `HIST_METRICS.keys()` làm `choices`.

---

## 3. Vector hóa: tại sao dùng batch (N,) thay vì lặp từng ảnh

Code không lặp `for i in range(N)` để so query với từng ảnh DB. Mọi metric đều **broadcast** trên `db (N, D)` cùng lúc:

```python
diff = db.astype(float64) - query.astype(float64)   # shape (N, D)
np.sum(diff * diff, axis=1)                          # shape (N,)
```

Với N = 500, D = 576 → toàn bộ phép tính chỉ 1 numpy call ~ 1 ms thay vì 500 ms vòng for Python. Đây là điểm mấu chốt khiến *full-scan vẫn nhanh* và two-stage retrieval *chỉ là tối ưu nice-to-have* trên dataset 500 ảnh.

---

## 4. Mean-normalize per branch — vì sao và làm thế nào

3 nhánh có thang khác nhau:

| Nhánh | Vector | Khoảng giá trị raw distance |
|---|---|---|
| Color (576-d, L1-normalized histogram) | mỗi phần tử ~ 1/576 | distance ~ 0.01–0.1 |
| Gradient (81-d, L1-normalized) | mỗi phần tử ~ 1/81 | distance ~ 0.05–0.3 |
| Compact6 (6-d, ép thang [0,1]) | mỗi phần tử ~ 0–1 | distance ~ 0.1–0.5 |

→ Cộng thẳng `d_color + d_grad + d_compact` thì compact6 sẽ áp đảo, color gần như mất tiếng nói.

### Giải pháp: chia cho mean của chính nhánh đó

`matcher_hybrid3.py:hybrid3_distance_components` tính:

```python
s_hist    = mean(d_hist)      # ★ trung bình distance từ query đến mọi ảnh DB trên nhánh hist
s_grad    = mean(d_grad)
s_compact = mean(d_compact)
```

Sau đó `combine_distances` (`matcher_hybrid3.py:39-50`):

```python
total = w_hist    * (d_hist    / s_hist)
      + w_grad    * (d_grad    / s_grad)
      + w_compact * (d_compact / s_compact)
```

Sau khi chia mean, mỗi nhánh có amplitude trung bình ~ 1.0 → **3 nhánh ngang vai trò** nhau, trọng số `w_*` chỉ điều chỉnh tinh chỉnh.

**Quan trọng:** `s_*` được tính *trên đúng query đó*, không phải hằng số precomputed. Nghĩa là cùng một ảnh DB sẽ có distance khác nhau với 2 query khác nhau — ranking **tương đối** vẫn đúng, giá trị tuyệt đối thì không có ý nghĩa cross-query.

---

## 5. Trọng số fusion — `w_hist`, `w_grad`, `w_compact`

Mặc định: `0.45 / 0.30 / 0.25` (`database_hybrid3.py:209-211`).

Lý do nghiêng về color (0.45):
- Ảnh thiên nhiên phân biệt rõ nhất bằng tone màu (núi tuyết vs bãi biển vs rừng).
- Gradient (0.30) bổ sung khi cùng tone (cùng xanh nhưng cấu trúc khác).
- Compact6 (0.25) như một "soft prior" — ít chiều nên trọng số nhỏ.

CLI có cảnh báo nếu tổng ≠ 1.0 (`query_cli_hybrid3.py:126-131`). **Sai về mặt số nhưng không sai về ranking** — vì mọi nhánh đã mean-normalize, scale tổng không đổi thứ tự `argsort`. Cảnh báo chỉ để tránh hiểu nhầm khi đọc giá trị distance tuyệt đối.

---

## 6. Single-stage và two-stage retrieval

### 6.1 Single-stage — `find_top_k_hybrid3` (`matcher_hybrid3.py:108-149`)

```
1. Tính d_hist, d_grad, d_compact cho TẤT CẢ N ảnh
2. Mean-normalize + cộng có trọng số → d_total (N,)
3. argsort → top-k
```

Chi phí: O(N · (D_hist + D_grad + D_compact)) ≈ O(500 · 663) ≈ 330k phép toán. Trên numpy là ~ 1–3 ms.

### 6.2 Two-stage — `find_top_k_hybrid3_two_stage` (`matcher_hybrid3.py:152-213`)

Lý thuyết CSDL Đa phương tiện: dùng đặc trưng **rẻ** lọc thô, chỉ tính đặc trưng đắt trên ứng viên còn lại.

```
Stage 1: tính d_compact cho TẤT CẢ N ảnh (chỉ 6-d → cực rẻ)
         → giữ M = coarse_top ứng viên gần nhất theo compact6
Stage 2: chạy single-stage hybrid3 ĐẦY ĐỦ trên M ảnh đó
         → top-k cuối
```

Code (`matcher_hybrid3.py:195-213`):

```python
d_compact_all = euclidean_distance_batch(q_compact, db_compact)
cand_idx = np.argsort(d_compact_all)[:coarse_top]
sub_ids = [ids[i] for i in cand_idx]

# Đệ quy gọi single-stage trên subset
return find_top_k_hybrid3(
    db_hist=db_hist[cand_idx],
    db_grad=db_grad[cand_idx],
    db_compact=db_compact[cand_idx],
    ids=sub_ids,
    ...
)
```

### Khi nào lợi?

- **N rất lớn** (10k+ ảnh) → tiết kiệm rõ rệt, vì stage 2 chỉ tính trên M ≪ N.
- **N = 500** (dataset hiện tại) → tiết kiệm ~30% (đo trong test trước: 27.8 ms vs 42.3 ms với coarse_top=50). Không đổi đời nhưng có lợi.

### Rủi ro

Nếu compact6 lọc *bỏ sót* một ảnh thực sự gần (theo metric đầy đủ) thì sẽ không bao giờ vào top-k. Trên dataset hiện tại với coarse_top=50/500 = 10%, top-2 vẫn trùng với single-stage → coarse filter đủ bảo toàn.

→ **Quy tắc thực dụng:** chọn `coarse_top ≥ 5 × k` để giảm rủi ro. Ví dụ `k=5` → `coarse_top ≥ 25`.

---

## 7. Breakdown — phân tích contribution per-branch

`_build_breakdown` (`matcher_hybrid3.py:78-105`) trả list dict cho top-k:

```python
{
    "filename": "01_thien_nhien_0012.jpg",
    "total":               0.6831,
    "color_contribution":  0.156,    # = w_hist * (d_hist / s_hist)
    "grad_contribution":   0.149,    # = w_grad * (d_grad / s_grad)
    "compact_contribution": 0.039,   # = w_compact * (d_compact / s_compact)
    "color_raw":   0.0723,           # raw d_hist trước normalize
    "grad_raw":    0.1856,
    "compact_raw": 0.0124,
}
```

`total = color + grad + compact`. Quan sát từng cột:
- Cột nào đóng góp lớn nhất → nhánh đó *ủng hộ* match.
- Một ảnh top-k có `color_contribution` rất nhỏ và `grad_contribution` cao → nó được xếp cao **chủ yếu nhờ cấu trúc cạnh giống**, dù màu khác.
- Insight này dùng để tinh chỉnh weights `w_*`.

CLI `--breakdown` (`query_cli_hybrid3.py:49-62`) hiển thị bảng này.

---

## 8. Workflow đầy đủ một query

Theo dòng code khi gọi `database_hybrid3.query()`:

```
1. extract_hybrid3_from_path(image_path)        # preprocessing.py + 3 phase trích đặc trưng
   → (q_hist, q_grad, q_compact, _)

2. Nếu coarse_top > 0: gọi find_top_k_hybrid3_two_stage
   Ngược lại:           gọi find_top_k_hybrid3

3. Bên trong: hybrid3_distance_components
   → tính d_hist (theo metric chọn), d_grad (l2), d_compact (l2)
   → tính s_hist, s_grad, s_compact (mean per branch)

4. combine_distances → d_total (N,)

5. argsort → lấy k chỉ số đầu

6. Nếu return_breakdown: build_breakdown thêm chi tiết per-branch

7. Return (q_hist, q_grad, q_compact, top, [breakdown])
```

CLI/UI nhận về và hiển thị.

---

## 9. Đánh giá ngắn

✅ **Mạnh:**
- Mean-normalize giải quyết triệt để vấn đề thang khác nhau giữa các nhánh.
- Vector hóa hoàn toàn — full-scan 500 ảnh chỉ vài ms.
- Two-stage có lý thuyết vững (knowledge.md operational guideline #1) và đã verify trên dữ liệu thực.
- Breakdown cho insight gỡ rối weights.

⚠️ **Hạn chế / điểm cần biết:**
- 4 metric cho color hist là *đáp ứng mục đích giáo dục* hơn là nhu cầu vận hành. Có thể giảm về 1 (`l2`) hoặc 2 (`l2`, `chi2`) cho gọn.
- Mean-normalize tính riêng mỗi query → distance tuyệt đối không so sánh được giữa các query khác nhau (nhưng ranking thì so sánh được).
- Trọng số `w_*` đang là **giả thiết** chưa optimize bằng grid-search trên ground truth — vì dataset chưa có nhãn relevance.
- Two-stage với `coarse_top` quá nhỏ có thể bỏ sót ảnh tốt; chưa có safe-default đề xuất `coarse_top ≥ 5k`.

---

## 10. Map nhanh tới code

| Việc | Vị trí |
|---|---|
| Định nghĩa 4 metric | `src/distances.py:25-64` |
| Registry `HIST_METRICS` | `src/distances.py:67-81` |
| Tính components per-branch | `src/matcher_hybrid3.py:8-36` |
| Mean-normalize + weighted sum | `src/matcher_hybrid3.py:39-50` |
| Single-stage entry | `src/matcher_hybrid3.py:108-149` |
| Two-stage entry | `src/matcher_hybrid3.py:152-213` |
| Breakdown builder | `src/matcher_hybrid3.py:78-105` |
| DB query entry point | `src/database_hybrid3.py:205-252` |
| CLI flags `--distance`, `--coarse-top`, `--breakdown` | `query_cli_hybrid3.py:95-117` |
| Cảnh báo tổng trọng số ≠ 1 | `query_cli_hybrid3.py:126-131` |
