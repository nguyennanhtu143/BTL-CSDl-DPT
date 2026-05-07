"""Các hàm khoảng cách dùng chung cho mọi pipeline CBIR.

Trước đây mỗi matcher_*.py tự định nghĩa lại euclidean_distance_batch.
Module này tập trung các metric phù hợp với từng loại đặc trưng:

    - euclidean (L2): generic, dùng cho gradient và scalar
    - manhattan (L1): ít nhạy outlier, tốt cho histogram
    - chi-square: chuẩn vàng cho histogram phân phối xác suất
    - histogram intersection: cổ điển trong CBIR ảnh màu

Tham khảo: knowledge.md mục 5 — Đo lường độ tương đồng.
"""
from __future__ import annotations

from typing import Callable

import numpy as np


def _validate_shapes(query: np.ndarray, db: np.ndarray) -> None:
    if query.ndim != 1 or db.ndim != 2 or query.shape[0] != db.shape[1]:
        raise ValueError(f"Shape không hợp lệ: query {query.shape}, db {db.shape}")


def euclidean_distance_batch(query: np.ndarray, db: np.ndarray) -> np.ndarray:
    """L2 distance từ query đến từng row của db. Return float64 shape (N,)."""
    _validate_shapes(query, db)
    diff = db.astype(np.float64) - query.astype(np.float64)
    return np.sqrt(np.sum(diff * diff, axis=1))


def manhattan_distance_batch(query: np.ndarray, db: np.ndarray) -> np.ndarray:
    """L1 distance — ít nhạy outlier, hợp histogram."""
    _validate_shapes(query, db)
    diff = db.astype(np.float64) - query.astype(np.float64)
    return np.sum(np.abs(diff), axis=1)


def chi_square_distance_batch(query: np.ndarray, db: np.ndarray) -> np.ndarray:
    """Chi-square distance cho histogram phân phối xác suất.

    chi2(p, q) = 0.5 * sum((p_i - q_i)^2 / (p_i + q_i + eps))
    Khoảng giá trị [0, ~1] với histogram đã L1-normalize.
    """
    _validate_shapes(query, db)
    eps = 1e-12
    q = query.astype(np.float64)[None, :]
    d = db.astype(np.float64)
    num = (d - q) ** 2
    den = d + q + eps
    return 0.5 * np.sum(num / den, axis=1)


def histogram_intersection_distance_batch(query: np.ndarray, db: np.ndarray) -> np.ndarray:
    """Histogram intersection distance = 1 - sum(min(p_i, q_i)).

    Giả định cả hai vector đã L1-normalize (sum = 1).
    Khoảng giá trị [0, 1]: 0 = trùng tuyệt đối, 1 = không giao.
    """
    _validate_shapes(query, db)
    q = query.astype(np.float64)[None, :]
    d = db.astype(np.float64)
    intersection = np.sum(np.minimum(q, d), axis=1)
    return 1.0 - intersection


HIST_METRICS: dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    "l2": euclidean_distance_batch,
    "l1": manhattan_distance_batch,
    "chi2": chi_square_distance_batch,
    "intersection": histogram_intersection_distance_batch,
}


def get_hist_metric(name: str) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Trả hàm distance theo tên. Raise nếu không hỗ trợ."""
    if name not in HIST_METRICS:
        raise ValueError(
            f"Metric không hỗ trợ: {name!r}. Chọn từ {list(HIST_METRICS.keys())}"
        )
    return HIST_METRICS[name]
