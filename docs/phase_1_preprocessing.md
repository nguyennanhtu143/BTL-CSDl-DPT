# Phase 1 — Tiền xử lý ảnh

## Mục tiêu
Đọc và chuẩn hóa toàn bộ 500 ảnh `640×360` về kích thước thống nhất `256×144` (giữ tỉ lệ 16:9), đồng thời cung cấp ảnh xám float32 cho Phase 3 (Sobel).

## Checklist task
- [x] `load_image(path)`: đọc JPG bằng Pillow → `np.uint8` shape `(H, W, 3)` không gian RGB
- [x] `resize_image(img, 256, 144)`: tự code **bilinear interpolation** vector hoá bằng NumPy
- [x] `to_grayscale(img)`: công thức luminance `Y = 0.299R + 0.587G + 0.114B`, trả `float32`
- [x] `preprocess(path)`: pipeline `load → resize → (rgb, gray)`
- [x] Test 5 ảnh: assert shape, dtype, range giá trị
- [x] Fix `UnicodeEncodeError` Windows console: `sys.stdout.reconfigure(encoding="utf-8")`

## Module
`src/preprocessing.py`

## Thuật toán chính

### Bilinear interpolation (vector hoá)
Với mỗi pixel output `(x_out, y_out)`, ánh xạ về toạ độ source:

```
x_src = (x_out + 0.5) * (src_w / target_w) - 0.5
y_src = (y_out + 0.5) * (src_h / target_h) - 0.5
```

Sau đó nội suy 4 neighbor `p00, p01, p10, p11`:

```
out = (1-fx)(1-fy)·p00 + fx(1-fy)·p01 + (1-fx)fy·p10 + fx·fy·p11
```

trong đó `fx = x_src - floor(x_src)`, `fy = y_src - floor(y_src)`.

Toàn bộ phép tính dùng broadcasting NumPy → không có vòng lặp Python.

### Grayscale
Áp dụng trực tiếp công thức luminance. Trả về `float32` (không clip về `uint8`) để Sobel ở Phase 3 không mất độ chính xác.

## Tham số config
| Hằng số | Giá trị | Ý nghĩa |
|---------|---------|---------|
| `IMG_WIDTH` | 256 | Chiều rộng sau resize |
| `IMG_HEIGHT` | 144 | Chiều cao sau resize (giữ tỉ lệ 16:9) |

## Kết quả test
- 5 ảnh đầu tiên: input `(360, 640, 3)` → resize `(144, 256, 3)` → gray `(144, 256)` đều pass.
- So sánh với `PIL.Image.resize(BILINEAR)`: `mean_diff` 2.7–4.5/255, `max_diff` ~71/255 ở vùng cạnh sắc nét. PIL dùng box filter chống alias khi downsample 2.5×; cài đặt 4-neighbor của ta có hiện tượng này nhưng nội dung tổng thể tương đương — chấp nhận được cho histogram-based feature.

## Ghi chú
- File `notebooks/test_preprocessing.py` chạy độc lập: `python notebooks/test_preprocessing.py`.
- Ảnh xám trả về `float32` thay vì `uint8` để giữ tính xác cho phép tích chập Sobel ở Phase 3.
