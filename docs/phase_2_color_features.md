# Phase 2 — Đặc trưng Màu sắc (Grid-based RGB Color Histogram)

## Mục tiêu
Trích xuất vector màu **576 chiều** phản ánh đồng thời phân bố màu **và** bố cục không gian: chia ảnh thành lưới 3×3, mỗi ô tính RGB histogram lượng tử hoá 4×4×4, sau đó ghép và normalize L1.

## Checklist task
- [x] `split_into_grid(img, grid=3)`: chia ảnh `144×256` thành 9 ô (xử lý đúng khi 256 không chia hết cho 3)
- [x] `cell_histogram(cell, bins=4)`: lượng tử hoá RGB → vector 64 bin
- [x] `color_histogram(img)`: ghép 9 ô → vector **576 chiều**
- [x] `normalize_l1(vec)`: chuẩn hoá tổng = 1, an toàn khi vec toàn 0
- [x] `extract_color_feature(img)`: pipeline đầy đủ
- [x] Vector hoá bằng `np.bincount` — không loop Python thuần
- [x] Test: synthetic histogram (đen→bin 0, trắng→bin 63, (100,200,50)→bin 28) + ảnh thật

## Module
`src/color_features.py`

## Thuật toán chính

### 1. Chia lưới 3×3
`np.linspace(0, h, 4)` cho ranh giới đều, làm tròn về `int32`:

| Trục | Kích thước | Bound | Width các ô |
|------|-----------|-------|-------------|
| Cao | 144 | `[0, 48, 96, 144]` | `48, 48, 48` |
| Rộng | 256 | `[0, 85, 170, 256]` | `85, 85, 86` |

Tổng pixel của 9 ô = `144 × 256 = 36864` (kiểm tra trong test).

### 2. Lượng tử hoá RGB và bin index
Mỗi kênh chia `[0, 256)` thành 4 khoảng đều rộng 64:

```
r_bin = r // 64    # ∈ {0,1,2,3}
g_bin = g // 64
b_bin = b // 64
bin_idx = r_bin * 16 + g_bin * 4 + b_bin   # ∈ [0, 64)
```

### 3. Đếm bin vector hoá
```python
bin_idx = (img[..., 0] // 64) * 16 + (img[..., 1] // 64) * 4 + (img[..., 2] // 64)
hist = np.bincount(bin_idx.ravel(), minlength=64)
```
Không cần loop pixel.

### 4. Normalize L1
`vec / vec.sum()` để tổng vector = 1 (xác suất). Trả vector zero nếu tổng = 0.

## Tham số config
| Hằng số | Giá trị | Ý nghĩa |
|---------|---------|---------|
| `GRID` | 3 | Lưới 3×3 = 9 ô |
| `COLOR_BINS` | 4 | RGB 4×4×4 = 64 bin/ô |
| `COLOR_DIM` | **576** | `9 × 64` |

## Kết quả test
| Ảnh | shape | sum | non_zero_bins / 576 |
|-----|-------|-----|---------------------|
| `01_thien_nhien_001.jpg` | (576,) | 1.000000 | 113 |
| `01_thien_nhien_002.jpg` | (576,) | 1.000000 | 186 |
| `01_thien_nhien_003.jpg` | (576,) | 1.000000 | 88 |

- Synthetic test: ảnh đen `(0,0,0)` → bin 0, trắng `(255,255,255)` → bin 63, pixel `(100,200,50)` → bin 28 (`= 1*16 + 3*4 + 0`).
- Pipeline: tổng count thô = 36864 = số pixel, sum sau normalize = 1.0 ✓.

## Ghi chú
- Phương án **RGB 4×4×4** chọn vì: đủ phân biệt cho ảnh thiên nhiên (xanh lá / xanh dương / vàng tách rõ trong RGB), không cần chuyển không gian màu phức tạp như HSV.
- Lưới 3×3 cân bằng giữa thông tin không gian và độ ổn định: ô quá nhỏ → nhạy với crop nhẹ; ô quá lớn → mất bố cục.
