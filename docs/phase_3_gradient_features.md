# Phase 3 — Đặc trưng Hình dạng (Gradient Histogram + Sobel)

## Mục tiêu
Bổ sung thông tin kết cấu/cạnh để phân biệt các ảnh có cùng tông màu nhưng khác bản chất (ví dụ: rừng cây vs. đồng cỏ — đều xanh nhưng kết cấu khác). Trích xuất vector hình dạng **81 chiều** dựa trên gradient Sobel + histogram định hướng theo lưới 3×3 (kiểu HOG đơn giản hoá).

## Checklist task
- [x] Định nghĩa kernel Sobel `Gx, Gy` (3×3)
- [x] `convolve2d(image, kernel)`: tự code tích chập 2D, padding 0, vector hoá NumPy
- [x] `compute_gradient(gray)`: trả `(magnitude, angle)` với góc map về `[0°, 180°)`
- [x] `cell_gradient_histogram(mag, ang, bins=9)`: histogram định hướng có trọng số magnitude
- [x] `gradient_histogram(gray)`: 9 bin × 9 ô = **81 chiều** (chưa normalize)
- [x] `extract_gradient_feature(gray)`: pipeline đầy đủ + L1 normalize
- [x] Test: identity convolution, Sobel trên cạnh dọc/ngang tổng hợp, range góc, histogram synthetic, pipeline ảnh thật

## Module
`src/gradient_features.py` (dùng lại `split_into_grid` từ `color_features` để nhất quán cách chia ô).

## Thuật toán chính

### 1. Sobel kernel
```
       [-1  0  1]              [-1 -2 -1]
Gx =   [-2  0  2]      Gy =    [ 0  0  0]
       [-1  0  1]              [ 1  2  1]
```
- `Gx` phát hiện cạnh **dọc** (gradient theo trục x).
- `Gy` phát hiện cạnh **ngang** (gradient theo trục y).

### 2. Tích chập 2D vector hoá
Padding 0 để output cùng shape với input. Thay vì 2 vòng lặp Python qua từng pixel ảnh (chậm O(H·W·9)), ta loop qua **9 ô của kernel** rồi cộng dồn shifted-image:

```python
for i in range(3):
    for j in range(3):
        output += kernel[i, j] * padded[i:i+H, j:j+W]
```

Mỗi phép cộng là một thao tác NumPy vector hoá → tốc độ ~tương đương `scipy.ndimage.convolve` với kernel nhỏ.

> Lưu ý: đây thực chất là **cross-correlation** (không lật kernel). Trong xử lý ảnh, "convolution" và "correlation" thường dùng thay nhau cho Sobel — vì Sobel đối xứng phản theo một trục, kết quả chỉ khác dấu nếu lật, không ảnh hưởng `magnitude`.

### 3. Magnitude và angle
```
magnitude = sqrt(dx² + dy²)
angle     = arctan2(dy, dx) → degrees → mod 180
```

`mod 180` tương đương HOG dùng **unsigned orientation**: cạnh đen→trắng và trắng→đen được gộp chung vào cùng bin (đối xứng 180°). Lý do: chiều của cạnh không quan trọng bằng việc có cạnh và hướng cạnh.

### 4. Histogram định hướng theo ô
Chia `[0°, 180°)` thành **9 bin** đều, mỗi bin rộng **20°**:

| Bin | Khoảng góc |
|-----|-----------|
| 0 | 0°–20° |
| 1 | 20°–40° |
| 2 | 40°–60° |
| ... | ... |
| 8 | 160°–180° |

Mỗi pixel **cộng `magnitude` vào bin theo `angle`**:

```python
bin_idx = (angle / 20).astype(int).clip(0, 8)
hist = np.bincount(bin_idx.ravel(), weights=magnitude.ravel(), minlength=9)
```

Trọng số magnitude (thay vì đếm 1) khiến cạnh mạnh có ảnh hưởng lớn hơn cạnh nhiễu — đây chính là điểm mượn từ HOG.

### 5. Lưới 3×3 + L1 normalize
Áp histogram trên 9 ô (cùng cách chia với Phase 2) → ghép thành vector **81 chiều**. Normalize L1 để tổng = 1, không bị ảnh hưởng bởi độ tương phản tổng thể của ảnh.

## Tham số config
| Hằng số | Giá trị | Ý nghĩa |
|---------|---------|---------|
| `GRAD_BINS` | 9 | 9 bin × 20° = 180° |
| `GRID` | 3 | Lưới 3×3 (dùng chung Phase 2) |
| `GRAD_DIM` | **81** | `9 × 9` |

## Kết quả test
- **Identity convolution**: `output == input` ✓
- **Cạnh tổng hợp** (ảnh nửa đen nửa trắng dọc): `dx[5,4] = 1020 = 4·255` (đúng tổng tuyệt đối Sobel Gx), `dy ≈ 0` ✓. Cạnh ngang đối xứng: `dy ≈ 1020`, `dx ≈ 0` ✓.
- **Range góc**: trên ảnh nhiễu uniform, `angle ∈ [0°, 180°)` ✓.
- **Histogram synthetic**: pixel (10°, 25°, 50°, 175°) với magnitude (2,3,1,4) → bin (0,1,2,8) đúng giá trị ✓.
- **Pipeline ảnh thật** (3 ảnh):

| Ảnh | shape | sum | non_zero / 81 | top-3 bin |
|-----|-------|-----|---------------|-----------|
| `001.jpg` | (81,) | 1.000000 | 81 | 67, 76, 58 |
| `002.jpg` | (81,) | 1.000000 | 81 | 49, 22, 13 |
| `003.jpg` | (81,) | 1.000000 | 81 | 67, 22, 4 |

## Phối hợp với Phase 2
- Cùng dùng `split_into_grid` → 9 ô của Color và Shape **tương ứng cùng vùng không gian**, dễ giải thích trong báo cáo.
- Phase 4 sẽ ghép: `feature = concat(0.7 · v_color_576, 0.3 · v_shape_81)` → vector **657 chiều**.

## Ghi chú
- Không dùng `cv2.Sobel`, `scipy.ndimage`, `skimage.feature.hog`. Toàn bộ Sobel và histogram tự code.
- Padding 0 ở biên có thể tạo cạnh giả ở mép ảnh. Với resize 256×144 và lưới 3×3, hiệu ứng này cục bộ ở viền các ô biên — chấp nhận được; nếu cần khắc phục sau, có thể dùng padding "edge replicate".
