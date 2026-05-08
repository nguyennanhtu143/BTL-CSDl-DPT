# Phase 3 — Đặc trưng hình dạng (Gradient Histogram / Sobel)

**Module:** `src/gradient_features.py`
**Output:** vector **81 chiều** float32 đã chuẩn hóa L1.
**Mục đích:** Mô tả *cấu trúc cạnh* (edges) — đường chân trời, viền cây, đường mây — không phụ thuộc vào màu sắc.

Đây là biến thể đơn giản của **HOG (Histogram of Oriented Gradients)**.

---

## 1. Tại sao cần đặc trưng hình dạng khi đã có màu?

Hai ảnh có thể *cùng phân phối màu* (đều là cảnh xanh dương + xanh lá) nhưng cấu trúc rất khác:
- Ảnh A: bãi biển ngang (cạnh ngang dài).
- Ảnh B: rừng dọc (cạnh dọc nhiều).

→ Histogram định hướng cạnh phân biệt được; histogram màu thì không.

---

## 2. Sobel — phát hiện cạnh

Sobel là filter tích chập 3×3 ước lượng đạo hàm rời rạc theo 2 hướng:

```
Gx = [-1  0  1]      Gy = [-1 -2 -1]
     [-2  0  2]           [ 0  0  0]
     [-1  0  1]           [ 1  2  1]
```

Định nghĩa tại `gradient_features.py:15-16`. Tương đương đạo hàm trung tâm có *làm trơn theo trục vuông góc*: hàng giữa của `Gx` là `[-2, 0, 2]` thay vì `[-1, 0, 1]` → khử nhiễu tốt hơn.

### Tích chập tự cài — `convolve2d` (`gradient_features.py:19-45`)

Vector hóa: thay vì 4 vòng lặp lồng (output_y, output_x, kernel_i, kernel_j), code chỉ lặp 9 lần (kernel 3×3) — mỗi lần cộng dồn `coef * shifted_image` vào output:

```python
for i in range(kh):
    for j in range(kw):
        coef = float(kernel[i, j])
        if coef == 0.0:
            continue
        output += coef * padded[i:i+h, j:j+w]
```

Padding 0 ở biên (line 35–36) → output cùng kích thước input. Tốc độ ~ tương đương `scipy.ndimage.convolve` cho kernel nhỏ.

---

## 3. Magnitude và angle — `compute_gradient` (`gradient_features.py:48-63`)

```python
dx = convolve2d(gray, SOBEL_GX)   # đạo hàm theo cột
dy = convolve2d(gray, SOBEL_GY)   # đạo hàm theo hàng

magnitude = sqrt(dx² + dy²)        # cường độ cạnh
angle     = degrees(arctan2(dy, dx)) % 180   # hướng cạnh, [0°, 180°)
```

### Vì sao mod 180° (unsigned orientation)?

Cạnh từ đen sang trắng và từ trắng sang đen có cùng "hình dạng" nhưng `arctan2` cho kết quả lệch 180°. Lấy `% 180` gộp chung — phù hợp khi ta quan tâm hình dạng chứ không quan tâm phía nào sáng/tối.

→ Hệ quả: `0°` và `180°` cùng một bin, không có bin "trùng nhau".

---

## 4. Histogram định hướng cho 1 ô — `cell_gradient_histogram` (`gradient_features.py:66-83`)

`GRAD_BINS = 9` (`src/config.py:20`) → chia [0°, 180°) thành **9 bin × 20°/bin**.

```python
bin_idx = (angle / 20.0).astype(int)        # 0..8
return np.bincount(bin_idx.ravel(),
                   weights=mag_cell.ravel(),  # ★ cộng dồn theo magnitude
                   minlength=9)
```

★ **Điểm quan trọng**: dùng `weights=magnitude` — pixel có gradient mạnh đóng góp nhiều, pixel phẳng (magnitude ~ 0) gần như không tính. Kết quả: histogram phản ánh **năng lượng cạnh thực sự**, không bị nhiễu nền phẳng làm loãng.

---

## 5. Lưới 3×3 — dùng lại `split_into_grid` từ Phase 2

```python
mag_cells = split_into_grid(magnitude, grid=3)   # 9 ô
ang_cells = split_into_grid(angle,     grid=3)   # 9 ô
```

Tái sử dụng hàm chia lưới của module color (`gradient_features.py:12`) → cùng lưới với histogram màu. Lợi: ô [i,j] của vector màu và vector gradient *tham chiếu cùng vùng không gian*.

---

## 6. Ghép 9 ô × 9 bin = 81 chiều — `gradient_histogram` (`gradient_features.py:86-99`)

```python
hists = [cell_gradient_histogram(m, a, 9) for m, a in zip(mag_cells, ang_cells)]
return np.concatenate(hists)    # shape (9 * 9,) = (81,)
```

---

## 7. Chuẩn hóa L1 — `extract_gradient_feature` (`gradient_features.py:102-112`)

```python
total = raw.sum()
return raw / total              # tổng = 1
```

Cùng lý do với Phase 2: bỏ ảnh hưởng kích thước/độ sáng tổng thể, ép vector về dạng phân phối xác suất hướng cạnh.

---

## 8. Ví dụ trực quan

Ảnh chứa nhiều **cạnh ngang** (bãi biển):
- `dx ≈ 0`, `dy lớn` → angle ≈ 90° → bin 4 (80°–100°) cao đột biến.

Ảnh chứa nhiều **cạnh chéo 45°** (núi):
- bin 2 (40°–60°) cao.

Ảnh **phẳng đều** (sương mù):
- magnitude rất nhỏ ở mọi pixel → tổng vector rất nhỏ; sau L1-normalize phân phối tương đối *đều* (không có hướng đặc trưng).

---

## 9. Đánh giá ngắn

✅ **Mạnh:**
- Bù được chỗ thiếu của histogram màu — phân biệt cấu trúc dù cùng màu.
- 81 chiều là gọn (so với HOG đầy đủ 3780 chiều), đủ tốt cho dataset 500 ảnh.
- Có thông tin không gian (3×3) như histogram màu.

⚠️ **Hạn chế:**
- Không dùng *block normalization* như HOG gốc (chuẩn hóa cục bộ qua nhiều ô) → nhạy hơn với độ sáng cục bộ.
- Sobel 3×3 chỉ bắt cạnh tần số trung bình–cao, bỏ qua cấu trúc rất lớn (đường chân trời chiếm 1/2 ảnh).
- Không có chống nhiễu trước Sobel (không Gaussian blur). Với JPEG chất lượng cao thì chấp nhận được; với ảnh nhiều noise thì cần thêm.

---

## 10. Map nhanh tới code

| Việc | Vị trí |
|---|---|
| Cấu hình `GRAD_BINS` | `src/config.py:20-21` |
| Sobel kernels | `src/gradient_features.py:15-16` |
| Tích chập vector hóa | `src/gradient_features.py:19-45` |
| Magnitude + unsigned angle | `src/gradient_features.py:48-63` |
| Histogram theo magnitude | `src/gradient_features.py:66-83` |
| Pipeline đầy đủ | `src/gradient_features.py:102-112` |
