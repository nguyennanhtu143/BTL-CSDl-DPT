"""Đặc trưng scalar bố cục: eccentricity, contrast, roughness, orderliness."""
from __future__ import annotations

import numpy as np


def _safe01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def extract_layout_scalars(gray: np.ndarray) -> tuple[float, float, float, float]:
    """Trả 4 scalar trong [0,1]:
    - eccentricity: độ lệch tâm phân bố biên/cấu trúc
    - contrast: độ tương phản (std mức xám)
    - roughness: độ thô (năng lượng gradient trung bình)
    - orderliness: mức trật tự (1 - entropy histogram xám chuẩn hóa)
    """
    if gray.ndim != 2:
        raise ValueError(f"Yêu cầu ảnh xám 2D, nhận {gray.shape}")
    g = gray.astype(np.float32)
    h, w = g.shape

    # contrast
    contrast = _safe01(float(np.std(g) / 255.0))

    # roughness: mean gradient magnitude (sai phân bậc 1)
    gx = np.diff(g, axis=1, prepend=g[:, :1])
    gy = np.diff(g, axis=0, prepend=g[:1, :])
    mag = np.sqrt(gx * gx + gy * gy)
    roughness = _safe01(float(np.mean(mag) / 255.0))

    # orderliness: entropy histogram xám (thấp entropy => trật tự hơn)
    hist = np.bincount(np.clip(g.astype(np.int32), 0, 255).ravel(), minlength=256).astype(np.float64)
    p = hist / max(hist.sum(), 1.0)
    nz = p[p > 0]
    entropy = float(-np.sum(nz * np.log2(nz)))
    orderliness = _safe01(1.0 - entropy / 8.0)  # 8 bits max entropy

    # eccentricity: từ ma trận hiệp phương sai có trọng số biên
    yy, xx = np.indices((h, w), dtype=np.float32)
    ww = mag + 1e-6
    sw = float(np.sum(ww))
    mx = float(np.sum(xx * ww) / sw)
    my = float(np.sum(yy * ww) / sw)
    dx = xx - mx
    dy = yy - my
    cxx = float(np.sum(ww * dx * dx) / sw)
    cyy = float(np.sum(ww * dy * dy) / sw)
    cxy = float(np.sum(ww * dx * dy) / sw)
    cov = np.array([[cxx, cxy], [cxy, cyy]], dtype=np.float64)
    vals = np.linalg.eigvalsh(cov)
    l1, l2 = float(vals[1]), float(max(vals[0], 1e-12))
    ecc = _safe01(float(np.sqrt(max(0.0, 1.0 - l2 / max(l1, 1e-12)))))

    return ecc, contrast, roughness, orderliness
