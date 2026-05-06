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

    if ids is None:
        ids_arr = np.arange(n)
    else:
        ids_arr = np.asarray(ids, dtype=object)
        if ids_arr.shape[0] != n:
            raise ValueError(f"len(ids)={ids_arr.shape[0]} khác N={n}")

    # Sắp xếp toàn bộ chỉ số theo khoảng cách, bỏ qua id đã gặp (phòng DB/ids lặp hoặc ghép dữ liệu lỗi).
    order = np.argsort(distances)
    out: list[tuple[object, float]] = []
    seen: set[object] = set()
    for i in order:
        key = ids_arr[i]
        if key in seen:
            continue
        seen.add(key)
        out.append((key, float(distances[i])))
        if len(out) >= k:
            break
    return out


def weighted_distance_batch(
    q_color: np.ndarray,
    q_shape: np.ndarray,
    q_freq: np.ndarray | None,
    db_color: np.ndarray,
    db_shape: np.ndarray,
    db_freq: np.ndarray | None,
    q_layout: np.ndarray | None,
    db_layout: np.ndarray | None,
    w_color: float,
    w_shape: float,
    w_freq: float = 0.0,
    w_layout: float = 0.0,
) -> np.ndarray:
    """Khoảng cách hợp thành: w_color*d_color_norm + w_shape*d_shape_norm.

    d_color_norm, d_shape_norm được scale theo trung bình khoảng cách trên toàn DB
    để 2 nhánh có biên độ gần nhau trước khi cộng trọng số.
    """
    d_color = euclidean_distance_batch(q_color, db_color)
    d_shape = euclidean_distance_batch(q_shape, db_shape)
    eps = 1e-12
    s_color = max(float(np.mean(d_color)), eps)
    s_shape = max(float(np.mean(d_shape)), eps)
    out = w_color * (d_color / s_color) + w_shape * (d_shape / s_shape)
    if q_freq is not None and db_freq is not None and w_freq > 0.0:
        d_freq = euclidean_distance_batch(q_freq, db_freq)
        s_freq = max(float(np.mean(d_freq)), eps)
        out = out + w_freq * (d_freq / s_freq)
    if q_layout is not None and db_layout is not None and w_layout > 0.0:
        d_layout = euclidean_distance_batch(q_layout, db_layout)
        s_layout = max(float(np.mean(d_layout)), eps)
        out = out + w_layout * (d_layout / s_layout)
    return out


def find_top_k_weighted(
    q_color: np.ndarray,
    q_shape: np.ndarray,
    q_freq: np.ndarray | None,
    db_color: np.ndarray,
    db_shape: np.ndarray,
    db_freq: np.ndarray | None,
    q_layout: np.ndarray | None,
    db_layout: np.ndarray | None,
    k: int = TOP_K,
    ids: list | np.ndarray | None = None,
    w_color: float = 0.7,
    w_shape: float = 0.3,
    w_freq: float = 0.0,
    w_layout: float = 0.0,
) -> list[tuple[object, float]]:
    """Top-k theo khoảng cách tổng hợp từ 2 vector thành phần."""
    n = db_color.shape[0]
    if n == 0:
        return []
    if db_shape.shape[0] != n:
        raise ValueError("db_color và db_shape phải cùng số dòng")
    k = min(k, n)

    distances = weighted_distance_batch(
        q_color=q_color,
        q_shape=q_shape,
        q_freq=q_freq,
        db_color=db_color,
        db_shape=db_shape,
        db_freq=db_freq,
        q_layout=q_layout,
        db_layout=db_layout,
        w_color=w_color,
        w_shape=w_shape,
        w_freq=w_freq,
        w_layout=w_layout,
    )

    if ids is None:
        ids_arr = np.arange(n)
    else:
        ids_arr = np.asarray(ids, dtype=object)
        if ids_arr.shape[0] != n:
            raise ValueError(f"len(ids)={ids_arr.shape[0]} khác N={n}")

    order = np.argsort(distances)
    out: list[tuple[object, float]] = []
    seen: set[object] = set()
    for i in order:
        key = ids_arr[i]
        if key in seen:
            continue
        seen.add(key)
        out.append((key, float(distances[i])))
        if len(out) >= k:
            break
    return out
