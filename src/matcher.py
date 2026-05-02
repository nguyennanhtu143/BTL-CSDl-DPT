"""Phase 4: So sánh vector và truy vấn top-k.

Khoảng cách Euclidean tự code (không gọi scipy.spatial.distance).
Hàm find_top_k vector hoá để query 1 ảnh trên CSDL 500 ảnh trong 1 lần broadcast.
"""
from __future__ import annotations

import numpy as np

from src.config import TOP_K


def euclidean_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    """sqrt(sum((v1[i] - v2[i])^2)) — tự code, không dùng np.linalg.norm.

    Hỗ trợ vector 1D cùng shape. Trả float Python.
    """
    if v1.shape != v2.shape:
        raise ValueError(f"Shape không khớp: {v1.shape} vs {v2.shape}")
    diff = v1.astype(np.float64) - v2.astype(np.float64)
    return float(np.sqrt(np.sum(diff * diff)))


def euclidean_distance_batch(query: np.ndarray, db: np.ndarray) -> np.ndarray:
    """Tính khoảng cách Euclidean từ 1 query tới N vector trong DB.

    Tham số:
        query: shape (D,)
        db   : shape (N, D)
    Trả mảng (N,) khoảng cách float64.
    """
    if query.ndim != 1 or db.ndim != 2 or query.shape[0] != db.shape[1]:
        raise ValueError(f"Shape không hợp lệ: query {query.shape}, db {db.shape}")
    diff = db.astype(np.float64) - query.astype(np.float64)
    return np.sqrt(np.sum(diff * diff, axis=1))


def find_top_k(
    query: np.ndarray,
    db_vectors: np.ndarray,
    k: int = TOP_K,
    ids: list | np.ndarray | None = None,
) -> list[tuple[object, float]]:
    """Trả top-k record gần nhất (theo Euclidean) sắp xếp tăng dần khoảng cách.

    Tham số:
        query: vector (D,)
        db_vectors: ma trận (N, D)
        k: số kết quả trả về (default 5)
        ids: danh sách N id ảnh; nếu None dùng index 0..N-1.
    Trả list `[(id, distance), ...]` độ dài min(k, N).
    """
    n = db_vectors.shape[0]
    if n == 0:
        return []
    k = min(k, n)

    distances = euclidean_distance_batch(query, db_vectors)
    top_idx = np.argpartition(distances, k - 1)[:k]
    top_idx = top_idx[np.argsort(distances[top_idx])]

    if ids is None:
        ids_arr = np.arange(n)
    else:
        ids_arr = np.asarray(ids, dtype=object)
        if ids_arr.shape[0] != n:
            raise ValueError(f"len(ids)={ids_arr.shape[0]} khác N={n}")

    return [(ids_arr[i], float(distances[i])) for i in top_idx]
