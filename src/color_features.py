"""Phase 2: Trích xuất đặc trưng màu sắc - Grid-based RGB Color Histogram.

Sơ đồ:
    Ảnh 256x144 -> chia lưới 3x3 -> 9 ô (~85x48)
    Mỗi ô -> RGB histogram 4x4x4 = 64 bin
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
            cell = img[h_bounds[i]:h_bounds[i + 1], w_bounds[j]:w_bounds[j + 1]]
            cells.append(cell)
    return cells


def cell_histogram(cell: np.ndarray, bins: int = COLOR_BINS) -> np.ndarray:
    """Tính histogram RGB lượng tử hóa cho 1 ô.

    Lượng tử hoá: chia [0, 256) thành `bins` khoảng đều, mỗi khoảng rộng 256/bins.
    Với bins=4: r_bin = r // 64 (giá trị 0..3).
    Bin index gộp: r_bin * bins^2 + g_bin * bins + b_bin (giá trị 0..bins^3-1).

    Trả về vector dài bins^3 (= 64 với bins=4).
    """
    if cell.dtype != np.uint8:
        raise ValueError(f"Yêu cầu uint8, nhận {cell.dtype}")

    step = 256 // bins
    quantized = (cell.astype(np.int32) // step)
    bin_idx = (
        quantized[..., 0] * (bins * bins)
        + quantized[..., 1] * bins
        + quantized[..., 2]
    )
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
    """Ghép histogram của 9 ô thành 1 vector dài grid^2 * bins^3 (= 576)."""
    cells = split_into_grid(img, grid)
    hists = [cell_histogram(cell, bins) for cell in cells]
    return np.concatenate(hists)


def extract_color_feature(
    img: np.ndarray,
    grid: int = GRID,
    bins: int = COLOR_BINS,
) -> np.ndarray:
    """Pipeline đầy đủ: histogram lưới + chuẩn hoá L1. Trả vector float32."""
    raw = color_histogram(img, grid, bins)
    return normalize_l1(raw)
