from __future__ import annotations

import numpy as np


def euclidean_distance_batch(query: np.ndarray, db: np.ndarray) -> np.ndarray:
    if query.ndim != 1 or db.ndim != 2 or query.shape[0] != db.shape[1]:
        raise ValueError(f"Shape không hợp lệ: query {query.shape}, db {db.shape}")
    diff = db.astype(np.float64) - query.astype(np.float64)
    return np.sqrt(np.sum(diff * diff, axis=1))


def hybrid_distance_batch(
    q_hist: np.ndarray,
    q_compact: np.ndarray,
    db_hist: np.ndarray,
    db_compact: np.ndarray,
    w_hist: float = 0.65,
    w_compact: float = 0.35,
) -> np.ndarray:
    d_hist = euclidean_distance_batch(q_hist, db_hist)
    d_comp = euclidean_distance_batch(q_compact, db_compact)
    eps = 1e-12
    s_hist = max(float(np.mean(d_hist)), eps)
    s_comp = max(float(np.mean(d_comp)), eps)
    return w_hist * (d_hist / s_hist) + w_compact * (d_comp / s_comp)


def find_top_k_hybrid(
    q_hist: np.ndarray,
    q_compact: np.ndarray,
    db_hist: np.ndarray,
    db_compact: np.ndarray,
    ids: list[str],
    k: int,
    w_hist: float = 0.65,
    w_compact: float = 0.35,
) -> list[tuple[str, float]]:
    n = db_hist.shape[0]
    if n == 0:
        return []
    if db_compact.shape[0] != n:
        raise ValueError("db_hist và db_compact phải cùng số dòng")
    if len(ids) != n:
        raise ValueError(f"len(ids)={len(ids)} khác N={n}")
    k = min(k, n)
    d = hybrid_distance_batch(
        q_hist=q_hist,
        q_compact=q_compact,
        db_hist=db_hist,
        db_compact=db_compact,
        w_hist=w_hist,
        w_compact=w_compact,
    )
    order = np.argsort(d)[:k]
    return [(ids[i], float(d[i])) for i in order]
