# Phase 2 — Đặc trưng màu (Color Histogram)

**Module:** `src/color_features.py`
**Output:** vector **576 chiều** float32 đã chuẩn hóa L1 (tổng = 1).
**Mục đích:** Mô tả phân phối màu sắc *theo từng vùng không gian* trong ảnh, không chỉ phân phối toàn cục.

---

## 1. Tại sao chia lưới (grid) thay vì histogram toàn ảnh?

Một histogram toàn ảnh sẽ trả lời câu "ảnh có bao nhiêu màu xanh" nhưng không nói được "xanh ở đâu". Hai ảnh:

- **A**: trời xanh phía trên, đất xanh lá phía dưới
- **B**: đất xanh lá phía trên, trời xanh phía dưới

→ Histogram toàn cục giống hệt nhau, nhưng chia 3×3 sẽ phân biệt rõ.

`GRID = 3` (`src/config.py:12`) → **9 ô**. Mỗi ô tự tính histogram riêng rồi ghép lại.

---

## 2. Lượng tử hóa màu — gom 16.7M màu → 64 nhóm

`COLOR_BINS = 4` (`src/config.py:17`). Mỗi kênh R, G, B chia thành 4 dải:

| Dải | Khoảng giá trị |
|---|---|
| 0 | 0–63 |
| 1 | 64–127 |
| 2 | 128–191 |
| 3 | 192–255 |

Code (`color_features.py:96-97`):

```python
step = 256 // bins        # = 64
quantized = cell // step  # mỗi pixel: (r_q, g_q, b_q), mỗi giá trị 0..3
```

### Mã hóa 3 nhóm thành 1 chỉ số duy nhất

Mỗi pixel có 3 số 0–3, ghép thành chỉ số 0–63 theo hệ cơ số 4:

```python
bin_idx = r_q * 16 + g_q * 4 + b_q     # 4*4 = 16, 4 = bins
```

Ví dụ: pixel `(0, 3, 0)` → `0·16 + 3·4 + 0 = 12` → loại màu số 12.

→ Mỗi ô có vector 64 bin (số pixel trong từng loại màu).

---

## 3. Chia lưới — `split_into_grid` (`color_features.py:18-33`)

Ảnh 256×144, grid=3 → kích thước ô **không chia hết**: `256/3 ≈ 85.33`.

Giải pháp: `np.linspace(0, 256, 4) = [0, 85, 170, 256]` → các ô rộng `[85, 85, 86]`. Phân bố đều nhất có thể, không có ô bị thiếu pixel.

Kết quả: 9 ô, mỗi ô shape `(48, 85)` hoặc `(48, 86)` hoặc `(49, 85)`...

---

## 4. Histogram cho 1 ô — `cell_histogram_binned` (`color_features.py:106-113`)

```python
bin_idx = q[..., 0] * 16 + q[..., 1] * 4 + q[..., 2]
return np.bincount(bin_idx.ravel(), minlength=64)
```

`np.bincount` đếm số lần xuất hiện của mỗi giá trị 0..63 → vector 64 phần tử. Tốc độ rất nhanh nhờ implement bằng C.

---

## 5. Ghép 9 ô → vector 576 chiều — `color_histogram` (`color_features.py:124-144`)

```python
hists = [cell_histogram_binned(cell, bins) for cell in cells]
return np.concatenate(hists)   # shape (9 * 64,) = (576,)
```

Thứ tự ghép: ô [0,0], [0,1], [0,2], [1,0], ... (hàng trước, cột sau). Quan trọng: **mọi ảnh đều theo đúng thứ tự này** → khi so sánh, bin thứ k của query và DB cùng tham chiếu một ô không gian.

---

## 6. Chuẩn hóa L1 — `normalize_l1` (`color_features.py:116-121`)

```python
return vec / vec.sum()        # tổng = 1
```

Sau chuẩn hóa, mỗi giá trị là **xác suất** một pixel (chọn ngẫu nhiên trong toàn ảnh) thuộc loại màu đó tại ô đó. Hai lý do:

1. **Loại bỏ ảnh hưởng kích thước ảnh** — ảnh to/nhỏ đều cùng tỉ lệ.
2. **Cho phép dùng metric trên phân phối xác suất** (chi-square, intersection) — xem Phase 5.

---

## 7. Tùy chọn LAB — `USE_LAB_COLOR_HISTOGRAM`

`src/config.py:15` cờ này mặc định `False` (dùng RGB). Khi bật `True` (`color_features.py:131-135`), chuyển ảnh sang không gian màu CIE-LAB qua `src.lab_color.lab_quantized_indices` rồi vẫn lượng tử 4×4×4.

LAB tách rõ kênh sáng (L*) và 2 kênh sắc (a*, b*) → khoảng cách L2 trong LAB **gần đúng cảm nhận khác biệt màu của mắt người**, tốt hơn RGB cho ảnh có cùng màu nhưng khác độ sáng.

⚠️ Đổi cờ này → **toàn bộ DB phải build lại** (`python build_database*.py --rebuild`).

---

## 8. Ví dụ kích thước và thời gian

- 1 ảnh 256×144 → vector 576 phần tử float32 → **2304 byte** = 2.3 KB.
- 500 ảnh → 1.15 MB cho riêng nhánh color → load toàn bộ vào RAM một lần là ổn.
- Trích đặc trưng 1 ảnh: ~5 ms (chủ yếu là `np.bincount` × 9).

---

## 9. Đánh giá ngắn

✅ **Mạnh:**
- Có thông tin không gian (3×3 grid) — quan trọng với ảnh phong cảnh có cấu trúc trời/đất rõ.
- Chuẩn hóa L1 → dùng được nhiều metric khác nhau (Phase 5).
- Lượng tử 4×4×4 = 64 bin/ô là điểm cân bằng: nhiều bin hơn (8×8×8 = 512) sẽ rỗng nhiều và sensitive với nhiễu màu nhỏ.

⚠️ **Hạn chế:**
- 64 bin/ô trên 4080 pixel/ô → trung bình mỗi bin 64 pixel — đủ thống kê nhưng các bin "ít gặp" (màu hiếm) sẽ noisy.
- Pixel sát biên grid bị chia cứng → ảnh chỉ dịch nhẹ vài pixel có thể đẩy nhiều pixel sang ô khác. Khắc phục được bằng *soft assignment*, hiện chưa làm.

---

## 10. Map nhanh tới code

| Việc | Vị trí |
|---|---|
| Cấu hình `GRID`, `COLOR_BINS` | `src/config.py:12,17` |
| Chia lưới | `src/color_features.py:18-33` |
| Lượng tử + bin index | `src/color_features.py:96-103` |
| Bincount cho 1 ô | `src/color_features.py:106-113` |
| Pipeline đầy đủ | `src/color_features.py:147-155` |
