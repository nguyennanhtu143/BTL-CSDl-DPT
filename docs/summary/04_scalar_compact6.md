# Phase 4 — Đặc trưng vô hướng (Compact6)

**Module:** `src/compact6_features.py`
**Output:** vector **6 chiều** float32, mỗi phần tử là một con số tổng hợp toàn ảnh.
**Mục đích:** Mô tả ảnh bằng *6 con số đặc trưng tổng quát* — rất nhẹ, dùng được làm bộ lọc thô (coarse filter) trong two-stage retrieval.

---

## 1. Vector compact6

```python
[mean_rgb, stddev_rgb, skewness_rgb, coarseness, contrast, directionality]
```

| Index | Tên | Loại | Ý nghĩa |
|---|---|---|---|
| 0 | `mean_rgb` | màu | Tông màu nóng/lạnh tổng thể |
| 1 | `stddev_rgb` | màu | Độ phân tán màu (đa dạng hay đơn điệu) |
| 2 | `skewness_rgb` | màu | Phân phối màu lệch về sáng hay tối |
| 3 | `coarseness` | texture | Mức độ thô/mịn của bề mặt |
| 4 | `contrast` | texture | Dải động sáng |
| 5 | `directionality` | texture | Mức độ tập trung của hướng cạnh |

Tất cả đều đã chuẩn hóa về thang gần [0, 1] để có *cùng bậc số*, tránh một thành phần áp đảo các thành phần khác khi tính khoảng cách.

---

## 2. Color moments — `_color_moments_rgb` (`compact6_features.py:34-63`)

3 *moment thống kê* trên 3 kênh RGB:

### 2.1 `mean_rgb` — tông nóng/lạnh

```python
mean_rgb = (means[0] - means[2] + 255) / 510
```

- `means[0]` = trung bình kênh R; `means[2]` = trung bình kênh B.
- Hiệu `R - B` ∈ [-255, 255] → cộng 255 chia 510 → ép về [0, 1].
- **0** = ảnh rất xanh dương (B ≫ R), **1** = ảnh rất đỏ/cam (R ≫ B), **0.5** = trung tính.

Đây là *index* nhanh cho ảnh phong cảnh: trời xanh < hoàng hôn < lá cây ⇒ giá trị `mean_rgb` tăng dần.

### 2.2 `stddev_rgb` — độ đa dạng màu

```python
stds = x.reshape(-1, 3).std(axis=0)         # std từng kênh
stddev_rgb = mean(stds) / 128.0             # trung bình 3 kênh, chia 128 ép về ~[0, 1]
```

- Ảnh sương mù trắng đều → `std` thấp → `stddev_rgb ≈ 0`.
- Ảnh cảnh chợ rực rỡ nhiều màu → `std` cao → `stddev_rgb ≈ 0.6–1.0`.

### 2.3 `skewness_rgb` — lệch sáng/tối

```python
z = (channel - mean) / std                  # chuẩn hóa kênh
skew_per_channel = mean(z³)                 # moment bậc 3
skewness_rgb = tanh(mean(skew_per_channel) / 2.0)
```

- `skew > 0`: phân phối lệch về phía sáng (long tail bên phải) — ảnh chủ yếu tối, có vùng sáng nổi bật.
- `skew < 0`: lệch về phía tối — ảnh chủ yếu sáng, có vùng tối nhỏ.
- `tanh(...)` ép vào [-1, 1] để kiểm soát outliers (skewness lý thuyết không bị chặn).

---

## 3. Texture features

### 3.1 `coarseness` — thô/mịn (`compact6_features.py:73-75`)

```python
gx = np.diff(gray, axis=1, prepend=gray[:,:1])
gy = np.diff(gray, axis=0, prepend=gray[:1,:])
coarseness = mean(sqrt(gx² + gy²)) / 255.0
```

- Đây là **trung bình magnitude gradient** toàn ảnh, dùng `np.diff` (đạo hàm rời rạc 1-pixel) thay vì Sobel 3×3 — đơn giản hơn, nhanh hơn.
- **Cao** → ảnh có nhiều chi tiết nhỏ (lá cây, cát, sỏi). **Thấp** → ảnh phẳng (sương mù, bầu trời).
- ⚠️ Tên hơi *ngược trực giác*: trong văn liệu Tamura, "coarseness" cao = bề mặt **thô** (texel lớn). Ở đây tên giữ nhưng định nghĩa thực chất là **mức năng lượng cạnh** (cao = chi tiết dày). Lưu ý khi đọc kết quả.

### 3.2 `contrast` — dải động robust (`compact6_features.py:78-79`)

```python
p5, p95 = np.percentile(gray, [5, 95])
contrast = (p95 - p5) / 255.0
```

- Dùng **percentile 5 và 95** thay vì max - min → loại outliers (vài pixel cháy sáng/đen tuyền không đại diện cho ảnh).
- Ảnh có cả vùng sáng và vùng tối rõ rệt → dải lớn → `contrast` cao.
- Ảnh phẳng tông → dải hẹp → `contrast` thấp.

### 3.3 `directionality` — mức tập trung hướng (`compact6_features.py:21-31`)

```python
gx = diff(gray, axis=1); gy = diff(gray, axis=0)
mag = sqrt(gx² + gy²) + 1e-8
theta = arctan2(gy, gx)             # hướng từng pixel, [-π, π]

# Dùng 2θ để gộp θ và θ + π (cùng hướng cạnh, ngược chiều gradient)
c = sum(mag * cos(2θ))
s = sum(mag * sin(2θ))
r = sqrt(c² + s²) / sum(mag)
```

Đây là **resultant length của hướng cạnh trên vòng tròn đơn vị**, weighted theo magnitude. Hiểu đơn giản:

- **Mọi cạnh cùng hướng** (sóng biển toàn ngang) → các vector cộng lại không triệt tiêu → `r ≈ 1`.
- **Hướng cạnh hỗn loạn** (lá cây) → các vector triệt tiêu nhau → `r ≈ 0`.

Việc nhân `2θ` là mẹo: cạnh có gradient hướng `+30°` và `−150°` (hai chiều ngược nhau) thực ra cùng *cạnh nghiêng 30°* — sau `2θ` chúng gộp thành `60°` và `60°`, không bị triệt tiêu.

→ `directionality` là *biến thể đơn giản hóa của Tamura directionality*.

---

## 4. Pipeline tổng — `extract_compact6` (`compact6_features.py:66-97`)

```python
mean_rgb, std_n, skew_n = _color_moments_rgb(rgb)
coarseness = compute_from_gray  / 255
contrast   = (p95 - p5) / 255
directionality = _directionality(gray)

vec = [mean_rgb, std_n, skew_n, coarseness, contrast, directionality]
```

Tất cả 6 thành phần ở thang [0, 1] (hoặc [-1, 1] cho skewness, nhưng bị `tanh` chặn).

`_safe_float` (`compact6_features.py:15-18`) chống NaN/Inf đẩy về 0 — phòng ảnh suy biến (ảnh đen hoàn toàn → std = 0 → chia 0 trong skewness).

---

## 5. Vai trò trong các pipeline

| Pipeline | Compact6 đóng vai trò |
|---|---|
| **Compact6 standalone** | Toàn bộ vector đặc trưng (chỉ 6 chiều) |
| **Hybrid** | Bổ sung scalar cho color histogram |
| **Hybrid3** | Bổ sung scalar + dùng làm *coarse filter* trong two-stage |
| **Legacy** | Không dùng (legacy có 4 layout scalars riêng) |

### Two-stage retrieval (xem Phase 5)

Vì compact6 chỉ 6 chiều, tính khoảng cách **rất nhanh**: 500 ảnh × 6 phép toán = vài microsecond. → Dùng compact6 lọc thô top-M ứng viên trước khi rank đầy đủ trên 657 chiều.

---

## 6. Đánh giá ngắn

✅ **Mạnh:**
- Cực gọn (6 chiều) → coarse filter lý tưởng.
- 3 moment màu + 3 texture là tập đặc trưng *cổ điển nhưng vẫn hiệu quả* cho ảnh phong cảnh.
- Mọi thành phần đã ép cùng thang → khoảng cách Euclidean trên 6 chiều có ý nghĩa.

⚠️ **Hạn chế:**
- 6 con số toàn cục → mất hoàn toàn thông tin không gian. Hai ảnh "trên trời dưới đất" và "trên đất dưới trời" có compact6 *gần như giống nhau*.
- `coarseness` không hẳn là Tamura coarseness gốc — chỉ là magnitude gradient trung bình. Tên có thể gây hiểu nhầm.
- Không phù hợp dùng đơn lẻ cho dataset lớn (>10k ảnh) — quá ít chiều, tỉ lệ ảnh "cùng compact6 nhưng nội dung khác" sẽ tăng.

---

## 7. Map nhanh tới code

| Việc | Vị trí |
|---|---|
| Color moments (3 chiều) | `src/compact6_features.py:34-63` |
| Coarseness | `src/compact6_features.py:73-75` |
| Contrast (P95 - P5) | `src/compact6_features.py:78-79` |
| Directionality (2θ trick) | `src/compact6_features.py:21-31` |
| Pipeline tổng | `src/compact6_features.py:66-97` |
| Safe NaN/Inf guard | `src/compact6_features.py:15-18` |
