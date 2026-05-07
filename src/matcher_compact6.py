from __future__ import annotations

import numpy as np


def euclidean_distance_batch(query: np.ndarray, db: np.ndarray) -> np.ndarray:
    if query.ndim != 1 or db.ndim != 2 or query.shape[0] != db.shape[1]:
        raise ValueError(f"Shape không hợp lệ: query {query.shape}, db {db.shape}")
    diff = db.astype(np.float64) - query.astype(np.float64)
    return np.sqrt(np.sum(diff * diff, axis=1))


def find_top_k_compact6(
    query: np.ndarray,
    db_vectors: np.ndarray,
    k: int,
    ids: list[str],
) -> list[tuple[str, float]]:
    n = db_vectors.shape[0]
    if n == 0:
        return []
    k = min(k, n)
    distances = euclidean_distance_batch(query, db_vectors)
    order = np.argsort(distances)[:k]
    return [(ids[i], float(distances[i])) for i in order]
