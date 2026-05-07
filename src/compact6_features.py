"""Compact6 feature extractor for image retrieval.

Vector 6 chiều:
    [mean_rgb, stddev_rgb, skewness_rgb, coarseness, contrast, directionality]
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from src.preprocessing import load_image, resize_image, to_grayscale


def _safe_float(x: float) -> float:
    if np.isnan(x) or np.isinf(x):
        return 0.0
    return float(x)


def _directionality(gray: np.ndarray) -> float:
    """Mức tập trung hướng cạnh trong [0, 1]."""
    gx = np.diff(gray, axis=1, prepend=gray[:, :1])
    gy = np.diff(gray, axis=0, prepend=gray[:1, :])
    mag = np.sqrt(gx * gx + gy * gy) + 1e-8
    theta = np.arctan2(gy, gx)  # [-pi, pi]
    # Hướng không phân biệt chiều: dùng 2*theta
    c = np.sum(mag * np.cos(2.0 * theta))
    s = np.sum(mag * np.sin(2.0 * theta))
    r = np.sqrt(c * c + s * s) / np.sum(mag)
    return _safe_float(np.clip(r, 0.0, 1.0))


def _color_moments_rgb(rgb: np.ndarray) -> tuple[float, float, float]:
    """3 color moments gọn theo RGB.

    mean_rgb: độ lệch tông nóng/lạnh (R-B) đã chuẩn hóa về [0,1]
    stddev_rgb: độ phân tán màu trung bình 3 kênh
    skewness_rgb: độ lệch phân phối màu trung bình 3 kênh
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Yêu cầu RGB shape (H,W,3), nhận {rgb.shape}")
    x = rgb.astype(np.float64)
    means = x.reshape(-1, 3).mean(axis=0)
    stds = x.reshape(-1, 3).std(axis=0)

    # Skewness từng kênh
    skews = []
    for c in range(3):
        ch = x[..., c].ravel()
        s = stds[c]
        if s < 1e-8:
            skews.append(0.0)
        else:
            z = (ch - means[c]) / s
            skews.append(float(np.mean(z ** 3)))
    skews = np.array(skews, dtype=np.float64)

    # mean_rgb: nhấn khác biệt màu nóng/lạnh
    mean_rgb = (means[0] - means[2] + 255.0) / 510.0
    stddev_rgb = float(np.mean(stds) / 128.0)
    skewness_rgb = float(np.tanh(np.mean(skews) / 2.0))
    return _safe_float(mean_rgb), _safe_float(stddev_rgb), _safe_float(skewness_rgb)


def extract_compact6(rgb: np.ndarray, gray: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    """Trích vector compact6 từ ảnh RGB + gray đã resize."""
    g = gray.astype(np.float32)
    flat = g.ravel().astype(np.float64)
    mean_rgb, std_n, skew_n = _color_moments_rgb(rgb)

    # Coarseness: trung bình gradient magnitude (cao -> thô)
    gx = np.diff(g, axis=1, prepend=g[:, :1])
    gy = np.diff(g, axis=0, prepend=g[:1, :])
    coarseness = float(np.mean(np.sqrt(gx * gx + gy * gy)))

    # Contrast: dải động robust (P95-P5)
    p5, p95 = np.percentile(flat, [5, 95])
    contrast = float(p95 - p5)

    directionality = _directionality(g)

    # Chuẩn hóa thang cơ bản để các thành phần cùng bậc
    coarse_n = _safe_float(coarseness / 255.0)
    contrast_n = _safe_float(contrast / 255.0)
    direc_n = directionality

    vec = np.array([mean_rgb, std_n, skew_n, coarse_n, contrast_n, direc_n], dtype=np.float32)
    meta = {
        "mean": mean_rgb,
        "stddev": std_n,
        "skewness": skew_n,
        "coarseness": coarse_n,
        "contrast": contrast_n,
        "directionality": direc_n,
    }
    return vec, meta


def extract_compact6_from_path(path: str | Path) -> tuple[np.ndarray, dict[str, float]]:
    rgb = resize_image(load_image(path))
    gray = to_grayscale(rgb)
    return extract_compact6(rgb, gray)
