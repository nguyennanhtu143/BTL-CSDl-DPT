"""Phase 2: Trích xuất đặc trưng màu sắc - Grid-based RGB Color Histogram.

Sơ đồ:
    Ảnh 256x144 -> chia lưới 3x3 -> 9 ô (~85x48)
    Mỗi ô -> histogram 4x4x4 = 64 bin (R/G/B lượng tử 4 mức)
    Ghép 9 histogram -> vector 576 chiều -> normalize L1
"""
from __future__ import annotations

import numpy as np

from src.config import COLOR_BINS, GRID


def split_into_grid(img: np.ndarray, grid: int = GRID) -> list[np.ndarray]:
    """Chia ảnh thành grid x grid ô.

    Khi kích thước không chia hết cho grid (256 / 3 = 85.33), dùng np.linspace
    để phân bố biên đều nhất có thể: 256 -> [0, 85, 170, 256] => ô rộng [85, 85, 86].
    """
    h, w = img.shape[:2]
    h_bounds = np.linspace(0, h, grid + 1, dtype=np.int32)
    w_bounds = np.linspace(0, w, grid + 1, dtype=np.int32)

    cells: list[np.ndarray] = []
    for i in range(grid):
        for j in range(grid):
            cell = img[h_bounds[i]: h_bounds[i + 1], w_bounds[j]: w_bounds[j + 1]]
            cells.append(cell)
    return cells


def cell_histogram_binned(cell_q: np.ndarray, bins: int = COLOR_BINS) -> np.ndarray:
    """Histogram bins^3 cho 1 ô đã lượng tử (mỗi pixel có 3 chỉ số 0..bins-1).

    Cho mỗi pixel với (r_q, g_q, b_q), gán mã duy nhất:
        bin_idx = r_q * bins^2 + g_q * bins + b_q     (= 0..bins^3-1)
    rồi đếm số pixel rơi vào từng mã (np.bincount).
    """
    if cell_q.shape[-1] != 3:
        raise ValueError(f"Yêu cầu ... 3 kênh cuối, nhận {cell_q.shape}")
    q = cell_q.astype(np.int32)
    q = np.clip(q, 0, bins - 1)
    bin_idx = q[..., 0] * (bins * bins) + q[..., 1] * bins + q[..., 2]
    return np.bincount(bin_idx.ravel(), minlength=bins ** 3).astype(np.float32)


def normalize_l1(vec: np.ndarray) -> np.ndarray:
    """Chuẩn hóa L1: chia cho tổng để vec.sum() == 1."""
    total = vec.sum()
    if total == 0:
        return vec.astype(np.float32)
    return (vec / total).astype(np.float32)


def color_histogram(
    img: np.ndarray,
    grid: int = GRID,
    bins: int = COLOR_BINS,
) -> np.ndarray:
    """Ghép histogram của 9 ô thành 1 vector dài grid^2 * bins^3 (= 576).

    1. Lượng tử hoá RGB
    Thay vì dùng 256 mức cho mỗi kênh (quá chi tiết), ta gom thành ít nhóm hơn.

    Ở đây bins = 4 nên mỗi kênh chỉ còn 4 nhóm:

        Nhóm 0: từ 0 đến 63
        Nhóm 1: từ 64 đến 127
        Nhóm 2: từ 128 đến 191
        Nhóm 3: từ 192 đến 255

    Cách làm trong code: chia nguyên cho 64 (vì 256 ÷ 4 = 64).
    Ví dụ:

        10 ÷ 64 = 0 → thuộc nhóm 0
        100 ÷ 64 = 1 → thuộc nhóm 1
        200 ÷ 64 = 3 → thuộc nhóm 3

    Dòng `step = 256 // bins` chính là tính 64 (độ rộng mỗi nhóm).
    Dòng `qimg = (img // step)` nghĩa là: với từng pixel, lấy R, G, B và
    đổi thành 3 nhóm (0–3) tương ứng.

    2. Ghép 3 nhóm thành 1 mã duy nhất (làm trong cell_histogram_binned)
    Sau bước trên, mỗi pixel có 3 số nhỏ (nhóm R, nhóm G, nhóm B), mỗi số từ 0 đến 3.

    Bây giờ ta muốn gán cho pixel một mã số duy nhất từ 0 đến 63, giống như:
        "Màu của pixel này thuộc loại màu số mấy trong bảng 64 loại?"

    Vì có 4 khả năng cho R, 4 cho G, 4 cho B, nên tổng cộng tối đa là:
        4 × 4 × 4 = 64 loại

    Công thức trong code:
        r * (4*4) + g * 4 + b
    chính là cách ghép 3 chữ số (hệ 4) thành một chỉ số giống kiểu ghép số điện thoại
    theo từng phần: (R nhóm nào, G nhóm nào, B nhóm nào) → ra một mã duy nhất.

    Ví dụ: pixel có RGB sau nhóm hoá là (0, 3, 0)
        Mã = 0×16 + 3×4 + 0 = 12
        → pixel đó được xếp vào "loại màu số 12".
    """
    if img.dtype != np.uint8:
        raise ValueError(f"RGB histogram yêu cầu uint8, nhận {img.dtype}")
    step = 256 // bins
    qimg = (img.astype(np.int32) // step).clip(0, bins - 1).astype(np.int32)

    cells = split_into_grid(qimg, grid)
    hists = [cell_histogram_binned(cell, bins) for cell in cells]
    return np.concatenate(hists)


def extract_color_feature(
    img: np.ndarray,
    grid: int = GRID,
    bins: int = COLOR_BINS,
) -> np.ndarray:
    """Pipeline đầy đủ: histogram lưới + chuẩn hoá L1. Trả vector float32."""
    raw = color_histogram(img, grid, bins)
    return normalize_l1(raw)
