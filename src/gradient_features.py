"""Phase 3: Trích xuất đặc trưng hình dạng - Gradient Histogram (Sobel).

Sơ đồ:
    Ảnh xám 256x144 -> Sobel Gx, Gy -> magnitude + angle
    Chia 3x3 ô -> mỗi ô tính histogram 9 bin (0°-180°), trọng số magnitude
    Ghép 9 histogram -> vector 81 chiều -> normalize L1
"""
from __future__ import annotations

import numpy as np

from src.color_features import split_into_grid
from src.config import GRAD_BINS, GRID

SOBEL_GX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
SOBEL_GY = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)


def convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Tích chập 2D (chính xác là cross-correlation - chuẩn dùng trong xử lý ảnh).

    Padding 0 ở biên để output cùng shape với input. Vector hoá bằng NumPy slicing:
    duyệt 9 ô của kernel 3x3, mỗi ô cộng dồn `kernel[i,j] * shifted_image` vào output.
    Tốc độ ~ tương đương scipy.ndimage.convolve cho kernel nhỏ.
    """
    if image.ndim != 2:
        raise ValueError(f"Yêu cầu ảnh 2D, nhận shape {image.shape}")
    if kernel.ndim != 2:
        raise ValueError(f"Yêu cầu kernel 2D, nhận shape {kernel.shape}")

    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    h, w = image.shape

    padded = np.zeros((h + 2 * pad_h, w + 2 * pad_w), dtype=np.float32)
    padded[pad_h:pad_h + h, pad_w:pad_w + w] = image.astype(np.float32)

    output = np.zeros((h, w), dtype=np.float32)
    for i in range(kh):
        for j in range(kw):
            coef = float(kernel[i, j])
            if coef == 0.0:
                continue
            output += coef * padded[i:i + h, j:j + w]
    return output


def compute_gradient(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Tính (magnitude, angle) cho ảnh xám.

    angle được map về [0°, 180°) - HOG dùng "unsigned orientation":
    cạnh đen-trắng và trắng-đen được gộp chung (đối xứng 180°).
    """
    if gray.dtype != np.float32:
        gray = gray.astype(np.float32)

    dx = convolve2d(gray, SOBEL_GX)
    dy = convolve2d(gray, SOBEL_GY)

    magnitude = np.sqrt(dx * dx + dy * dy)
    angle_rad = np.arctan2(dy, dx)
    angle_deg = np.degrees(angle_rad) % 180.0
    return magnitude, angle_deg


def cell_gradient_histogram(
    mag_cell: np.ndarray,
    ang_cell: np.ndarray,
    bins: int = GRAD_BINS,
) -> np.ndarray:
    """Histogram định hướng của 1 ô, cộng dồn theo magnitude.

    Chia [0°, 180°) thành `bins` khoảng đều (mặc định 9 bin × 20°).
    bin_idx = floor(angle / bin_width); với angle = 180° (sau mod sẽ là 0) nên không tràn.
    """
    bin_width = 180.0 / bins
    bin_idx = (ang_cell / bin_width).astype(np.int32)
    bin_idx = np.clip(bin_idx, 0, bins - 1)
    return np.bincount(
        bin_idx.ravel(),
        weights=mag_cell.ravel().astype(np.float64),
        minlength=bins,
    ).astype(np.float32)


def gradient_histogram(
    gray: np.ndarray,
    grid: int = GRID,
    bins: int = GRAD_BINS,
) -> np.ndarray:
    """Vector grid^2 * bins chiều (mặc định 9*9 = 81), chưa normalize."""
    magnitude, angle = compute_gradient(gray)
    mag_cells = split_into_grid(magnitude, grid)
    ang_cells = split_into_grid(angle, grid)

    hists = [
        cell_gradient_histogram(m, a, bins) for m, a in zip(mag_cells, ang_cells)
    ]
    return np.concatenate(hists)


def extract_gradient_feature(
    gray: np.ndarray,
    grid: int = GRID,
    bins: int = GRAD_BINS,
) -> np.ndarray:
    """Pipeline đầy đủ: Sobel -> magnitude/angle -> histogram lưới -> L1 normalize."""
    raw = gradient_histogram(gray, grid, bins)
    total = raw.sum()
    if total == 0:
        return raw
    return (raw / total).astype(np.float32)
