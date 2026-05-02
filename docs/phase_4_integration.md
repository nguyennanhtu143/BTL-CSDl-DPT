# Phase 4 — Tích hợp đặc trưng & So sánh

## Mục tiêu
Ghép vector màu (576) và vector hình dạng (81) thành **vector đặc trưng cuối cùng 657 chiều** với trọng số `0.7 / 0.3`, đồng thời cài đặt khoảng cách Euclidean và truy vấn top-k.

## Checklist task
- [x] `extract_features(rgb, gray)` ghép `concat(0.7·v_color, 0.3·v_shape)` → vector 657 chiều
- [x] `extract_from_path(path)` pipeline đầu-cuối từ file
- [x] `split_color_shape(feature)` tách lại 2 phần để in console ở Phase 6
- [x] `euclidean_distance(v1, v2)` tự code (`sqrt(sum((v1-v2)²))`), không dùng `np.linalg.norm`
- [x] `euclidean_distance_batch(query, db)` vector hoá để tính 500 ảnh trong 1 broadcast
- [x] `find_top_k(query, db, k=5, ids)` dùng `np.argpartition` cho tốc độ O(N)
- [x] Test: shape, trọng số, Euclidean vs NumPy reference, batch vs single, top-k synthetic, self-match = 0

## Module
- `src/feature_extractor.py`
- `src/matcher.py`

## Thuật toán chính

### 1. Ghép feature có trọng số
```python
v_color = extract_color_feature(rgb)    # 576, sum = 1.0
v_shape = extract_gradient_feature(gray) # 81,  sum = 1.0
feature = concat(W_COLOR · v_color, W_SHAPE · v_shape)   # 657
```

Vì cả 2 đầu vào đã L1-normalize (sum = 1):
- `feature[:576].sum() == W_COLOR == 0.7`
- `feature[576:].sum() == W_SHAPE == 0.3`
- `feature.sum() == 1.0`

**Lý do nhân trọng số _trước_ khi concat (thay vì nhân lúc tính khoảng cách):** mỗi ảnh chỉ trích đặc trưng 1 lần khi build CSDL, sau đó query 500 ảnh chỉ là 1 broadcast trừ + sqrt — tránh nhân trọng số 500 lần mỗi query.

**Tại sao không normalize lại sau concat:** sẽ làm mất tỉ lệ đóng góp 0.7/0.3 — vô hiệu hoá trọng số.

### 2. Euclidean distance
Định nghĩa toán học:

```
d(u, v) = sqrt( Σᵢ (uᵢ - vᵢ)² )
```

Cài đặt vector hoá:
```python
diff = u.astype(float64) - v.astype(float64)
d = sqrt(sum(diff * diff))
```

Dùng `float64` để tránh sai số khi cộng dồn 657 phần tử nhỏ.

### 3. Batch Euclidean (1 query vs N database)
```python
diff = db - query                  # broadcast (N, D)
distances = sqrt(sum(diff² , axis=1))   # (N,)
```

500 ảnh × 657 chiều = ~330k phép trừ — chạy trong vài chục ms.

### 4. Top-k với `np.argpartition`
- `argsort` toàn bộ N phần tử: O(N log N).
- `argpartition` chỉ phân hoạch để tìm k nhỏ nhất: O(N), sau đó `argsort` k phần tử: O(k log k).

Với N=500, k=5: cả hai gần như tức thời, nhưng `argpartition` là practice tốt hơn khi N tăng.

## Tham số config
| Hằng số | Giá trị | Ý nghĩa |
|---------|---------|---------|
| `W_COLOR` | 0.7 | Trọng số phần màu |
| `W_SHAPE` | 0.3 | Trọng số phần hình dạng |
| `TOTAL_DIM` | **657** | `576 + 81` |
| `TOP_K` | 5 | Số ảnh trả về mặc định |

**Lý do 0.7/0.3:** ảnh nền thiên nhiên phân biệt chủ yếu nhờ màu (xanh rừng, vàng sa mạc, xanh biển, trắng tuyết...). Texture đóng vai trò bổ trợ để tách các ảnh cùng tông màu (rừng cây vs đồng cỏ). Đây là gợi ý từ đề bài — sẽ tune lại ở Phase 7 nếu cần.

## Kết quả test

### Đơn vị
- `feature.shape == (657,)`, `dtype == float32` ✓
- `color_sum = 0.7000`, `shape_sum = 0.3000` ✓
- `euclidean_distance` lệch `np.linalg.norm` < 1e-9 ✓
- `euclidean_distance_batch` trùng kết quả single (max diff 0.00e+00) ✓
- `find_top_k` synthetic 4 vector 2D: thứ tự đúng theo lý thuyết ✓

### Self-match (5 ảnh đầu tiên)
| Query | Top-1 | d | Top-2 | d |
|-------|-------|----|-------|----|
| `001.jpg` | `001.jpg` | **0.000000** | `004.jpg` | 0.145402 |
| `002.jpg` | `002.jpg` | **0.000000** | `005.jpg` | 0.143139 |
| `003.jpg` | `003.jpg` | **0.000000** | `004.jpg` | 0.129509 |
| `004.jpg` | `004.jpg` | **0.000000** | `005.jpg` | 0.105082 |
| `005.jpg` | `005.jpg` | **0.000000** | `004.jpg` | 0.105082 |

Quan sát: cặp `(004, 005)` có khoảng cách nhỏ nhất ở cả 2 chiều → ma trận khoảng cách đối xứng đúng kì vọng.

## Sử dụng tiếp ở các phase sau
- **Phase 5 (CSDL)**: gọi `extract_from_path` cho 500 ảnh, lưu vector 657 chiều dưới dạng BLOB.
- **Phase 6 (UI)**: gọi `extract_from_path` cho ảnh upload, dùng `find_top_k`, dùng `split_color_shape` để in vector ra console.
