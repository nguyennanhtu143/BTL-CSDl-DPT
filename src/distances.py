"""Khoảng cách Euclidean (L2) batch — dùng chung cho mọi nhánh đặc trưng hybrid3."""
from __future__ import annotations

import numpy as np


def euclidean_distance_batch(query: np.ndarray, db: np.ndarray) -> np.ndarray:
    """L2 distance từ query (D,) đến từng row của db (N, D). Return float64 shape (N,)."""
    if query.ndim != 1 or db.ndim != 2 or query.shape[0] != db.shape[1]:
        raise ValueError(f"Shape không hợp lệ: query {query.shape}, db {db.shape}")
    diff = db.astype(np.float64) - query.astype(np.float64)
    return np.sqrt(np.sum(diff * diff, axis=1))
