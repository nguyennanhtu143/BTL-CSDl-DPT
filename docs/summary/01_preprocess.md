# Phase 1 — Tiền xử lý ảnh (Preprocess)

**Module:** `src/preprocessing.py`
**Mục đích:** Đưa mọi ảnh đầu vào về cùng kích thước, cùng định dạng để các phase trích đặc trưng phía sau làm việc trên dữ liệu nhất quán.

---

## 1. Đầu vào / Đầu ra

| Bước | Hàm | Input | Output |
|---|---|---|---|
| Load | `load_image(path)` | đường dẫn `.jpg` | `ndarray (H, W, 3) uint8` (RGB) |
| Resize | `resize_image(img)` | RGB gốc | `ndarray (144, 256, 3) uint8` |
| Grayscale | `to_grayscale(img)` | RGB sau resize | `ndarray (144, 256) float32` |
| Trọn gói | `preprocess(path)` | đường dẫn | tuple `(rgb_resized, gray_resized)` |

Kích thước cố định **256 × 144** lấy từ `src/config.py:9-10` (`IMG_WIDTH=256`, `IMG_HEIGHT=144`) — tỉ lệ 16:9, đủ chi tiết cho ảnh nền thiên nhiên mà vẫn nhẹ.

---

## 2. Load ảnh — `load_image` (`preprocessing.py:10-13`)

```python
img = Image.open(path).convert("RGB")
return np.asarray(img, dtype=np.uint8)
```

- Dùng Pillow đọc JPG, ép về 3 kênh RGB (loại ảnh có alpha hoặc grayscale gốc).
- Trả `uint8` (0–255), giữ nguyên dữ liệu thô — chưa scale, chưa normalize.

---

## 3. Resize bilinear — `resize_image` (`preprocessing.py:16-60`)

Resize **tự cài đặt** bằng NumPy thay vì gọi `PIL.Image.resize`. Vì sao?

- Bài tập yêu cầu hiểu thuật toán nội suy.
- Vector hóa hoàn toàn bằng NumPy → tốc độ xấp xỉ Pillow trên kích thước nhỏ.

### Công thức ánh xạ tọa độ

Với mỗi pixel output `(x_out, y_out)`, tìm tọa độ tương ứng trên ảnh source:

```
x_src = (x_out + 0.5) * (src_w / target_w) - 0.5
y_src = (y_out + 0.5) * (src_h / target_h) - 0.5
```

Đây là **half-pixel center mapping** — chuẩn của OpenCV/Pillow. So với công thức ngây thơ `x_src = x_out * (src_w / target_w)`, cách này tránh lệch về biên trái/trên khi resize.

### Nội suy 4 láng giềng

```python
x0 = floor(x_src);  x1 = x0 + 1
y0 = floor(y_src);  y1 = y0 + 1
fx = x_src - x0;    fy = y_src - y0

out = (1-fx)(1-fy)·p00 + fx(1-fy)·p01 + (1-fx)fy·p10 + fx·fy·p11
```

`p00, p01, p10, p11` là 4 pixel láng giềng. Tất cả ma trận `x0, y0, fx, fy` được build vector hóa nên không có vòng lặp Python.

`np.clip` ở line 37–38 và 42–43 chống tràn biên (khi `x_src > src_w - 1`).

---

## 4. Grayscale luminance — `to_grayscale` (`preprocessing.py:63-71`)

```python
Y = 0.299·R + 0.587·G + 0.114·B
```

- Đây là công thức **luminance ITU-R BT.601** (chuẩn cho video PAL/NTSC, vẫn dùng phổ biến trong xử lý ảnh).
- Trả `float32` (không phải `uint8`) để Phase 3 (Sobel) tích chập được mà không phải cast lại.

Vì sao trọng số bất đối xứng? Mắt người nhạy với G nhất, B kém nhất → công thức cân theo cảm nhận sáng thực tế, không phải trung bình cộng `(R+G+B)/3`.

---

## 5. Hệ quả với các phase sau

| Phase | Dùng output nào |
|---|---|
| Color (Phase 2) | RGB resized `uint8` |
| Gradient (Phase 3) | Gray resized `float32` |
| Compact6 — color moments | RGB resized `uint8` |
| Compact6 — coarseness/contrast/directionality | Gray resized `float32` |

Tức là **chỉ duy nhất Phase 1 đụng tới file ảnh gốc**. Mọi phase trích đặc trưng làm việc trên 2 mảng `rgb` và `gray` đã chuẩn hóa kích thước.

---

## 6. Đánh giá ngắn

✅ **Hợp lý:**
- Kích thước 256×144 đủ giữ chi tiết texture/màu mà đẩy DB build chỉ ~0.5 s/ảnh.
- Bilinear là cân bằng tốt giữa chất lượng và tốc độ (so với nearest = răng cưa, bicubic = chậm hơn ~2×).
- Self-implementation đáp ứng yêu cầu môn học mà không hy sinh hiệu năng.

⚠️ **Hạn chế cần biết:**
- Chỉ resize, **không crop center hay smart-crop**: ảnh tỉ lệ khác 16:9 sẽ bị méo (ảnh dọc 9:16 sẽ bị bóp ngang).
- **Không có chống nhiễu** (Gaussian blur trước Sobel) → Phase 3 có thể nhạy với nhiễu nhỏ; chấp nhận được vì ảnh dataset là JPEG chất lượng cao.

---

## 7. Map nhanh tới code

| Việc | Vị trí |
|---|---|
| Constant kích thước | `src/config.py:9-10` |
| Tính toán bilinear vector hóa | `src/preprocessing.py:32-59` |
| Hệ số luminance | `src/preprocessing.py:71` |
| Hàm tổng `preprocess()` | `src/preprocessing.py:74-78` |
