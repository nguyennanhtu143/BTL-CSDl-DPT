"""Đặc trưng miền tần số: DCT low-frequency + FFT radial energy."""
from __future__ import annotations

import numpy as np

from src.config import FREQ_DCT_SIDE, FREQ_FFT_BINS


def _dct_1d_type2(x: np.ndarray) -> np.ndarray:
    """DCT-II 1D không cần scipy (dùng FFT trên chuỗi chẵn mở rộng)."""
    n = x.shape[0]
    ext = np.concatenate([x, x[::-1]], axis=0)
    y = np.fft.fft(ext)
    k = np.arange(n)
    w = np.exp(-1j * np.pi * k / (2.0 * n))
    return np.real(y[:n] * w)


def dct2(gray: np.ndarray) -> np.ndarray:
    """DCT-II 2D bằng cách áp DCT theo hàng rồi theo cột."""
    if gray.ndim != 2:
        raise ValueError(f"Yêu cầu ảnh xám 2D, nhận {gray.shape}")
    a = gray.astype(np.float64)
    tmp = np.apply_along_axis(_dct_1d_type2, axis=1, arr=a)
    out = np.apply_along_axis(_dct_1d_type2, axis=0, arr=tmp)
    return out.astype(np.float32)


def dct_low_block_feature(gray: np.ndarray, side: int = FREQ_DCT_SIDE) -> np.ndarray:
    """Lấy block tần số thấp góc trên-trái của DCT (side x side), chuẩn hóa L1."""
    c = dct2(gray)
    block = np.abs(c[:side, :side]).astype(np.float32).ravel()
    s = float(block.sum())
    if s > 0:
        block /= s
    return block


def fft_radial_feature(gray: np.ndarray, bins: int = FREQ_FFT_BINS) -> np.ndarray:
    """Histogram năng lượng FFT theo bán kính (radial bins), chuẩn hóa L1."""
    if gray.ndim != 2:
        raise ValueError(f"Yêu cầu ảnh xám 2D, nhận {gray.shape}")
    g = gray.astype(np.float32)
    f = np.fft.fft2(g)
    mag = np.abs(np.fft.fftshift(f)).astype(np.float32)
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.indices((h, w))
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    rmax = float(rr.max()) + 1e-12
    ridx = np.minimum((rr / rmax * bins).astype(np.int32), bins - 1)
    hist = np.bincount(ridx.ravel(), weights=mag.ravel().astype(np.float64), minlength=bins).astype(
        np.float32
    )
    s = float(hist.sum())
    if s > 0:
        hist /= s
    return hist


def extract_frequency_feature(gray: np.ndarray) -> np.ndarray:
    """Vector tần số đầy đủ: concat(DCT_low, FFT_radial)."""
    dct_feat = dct_low_block_feature(gray, side=FREQ_DCT_SIDE)
    fft_feat = fft_radial_feature(gray, bins=FREQ_FFT_BINS)
    return np.concatenate([dct_feat, fft_feat]).astype(np.float32)
