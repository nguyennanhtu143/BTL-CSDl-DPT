"""Chuyển sRGB (uint8) sang CIELAB (D65) — NumPy thuần, không CNN.

Dùng cho histogram màu: tách độ sáng (L*) và sắc độ (a*,b*) tốt hơn RGB,
giảm trường hợp rừng/biển bị coi là gần chỉ vì cùng tông xám-xanh trên RGB.
"""
from __future__ import annotations

import numpy as np


def rgb_uint8_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Ảnh RGB uint8 (H,W,3) -> LAB float32 (H,W,3): L* in [0,100], a*,b* ~ [-128,127]."""
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Yêu cầu RGB uint8 shape (H, W, 3)")

    u = rgb.astype(np.float64) / 255.0
    low = u <= 0.04045
    lin = np.where(low, u / 12.92, ((u + 0.055) / 1.055) ** 2.4)

    r = lin[..., 0]
    g = lin[..., 1]
    b = lin[..., 2]

    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041

    ref = np.array([0.95047, 1.0, 1.08883], dtype=np.float64)
    xyz = np.stack([x, y, z], axis=-1) / ref
    xyz = np.maximum(xyz, 1e-12)

    eps = 216.0 / 24389.0
    kappa = 24389.0 / 27.0

    def f(t: np.ndarray) -> np.ndarray:
        return np.where(t > eps, np.cbrt(t), (kappa * t + 16.0) / 116.0)

    fx = f(xyz[..., 0])
    fy = f(xyz[..., 1])
    fz = f(xyz[..., 2])

    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return np.stack([L, a, b], axis=-1).astype(np.float32)


def _channel_to_bin(v: np.ndarray, lo: float, hi: float, bins: int) -> np.ndarray:
    t = (v.astype(np.float64) - lo) / (hi - lo)
    t = np.clip(t, 0.0, 1.0 - 1e-7)
    return np.minimum((t * bins).astype(np.int32), bins - 1)


def lab_quantized_indices(rgb: np.ndarray, bins: int) -> np.ndarray:
    """RGB uint8 -> chỉ số bin 0..bins-1 cho L*, a*, b* (cùng logic ghép 4^3 bin / ô)."""
    lab = rgb_uint8_to_lab(rgb)
    L = np.clip(lab[..., 0], 0.0, 100.0)
    a = np.clip(lab[..., 1], -128.0, 127.0)
    b = np.clip(lab[..., 2], -128.0, 127.0)
    qL = _channel_to_bin(L, 0.0, 100.0, bins)
    qa = _channel_to_bin(a, -128.0, 127.0, bins)
    qb = _channel_to_bin(b, -128.0, 127.0, bins)
    return np.stack([qL, qa, qb], axis=-1).astype(np.int32)
