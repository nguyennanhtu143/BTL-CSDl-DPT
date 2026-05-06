from __future__ import annotations

from pathlib import Path

import numpy as np


def _kmeans(x: np.ndarray, k: int, iters: int = 20, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    n, d = x.shape
    k = max(1, min(k, n))
    rng = np.random.default_rng(seed)
    init_idx = rng.choice(n, size=k, replace=False)
    centroids = x[init_idx].astype(np.float32, copy=True)
    assign = np.zeros(n, dtype=np.int32)
    for _ in range(iters):
        diff = x[:, None, :] - centroids[None, :, :]
        dist2 = np.sum(diff * diff, axis=2)
        assign = np.argmin(dist2, axis=1).astype(np.int32)
        for c in range(k):
            mask = assign == c
            if np.any(mask):
                centroids[c] = x[mask].mean(axis=0).astype(np.float32)
            else:
                centroids[c] = x[rng.integers(0, n)]
    return centroids.astype(np.float32), assign


def build_ivf(x: np.ndarray, nlist: int, iters: int = 20, seed: int = 42) -> dict[str, np.ndarray]:
    """Build IVF đơn giản: kmeans centroid + inverted lists nén (offset/order)."""
    if x.ndim != 2:
        raise ValueError(f"Yêu cầu ma trận 2D, nhận {x.shape}")
    n = x.shape[0]
    centroids, assign = _kmeans(x.astype(np.float32), nlist, iters=iters, seed=seed)
    order = np.argsort(assign, kind="stable").astype(np.int32)
    sorted_assign = assign[order]
    k = centroids.shape[0]
    counts = np.bincount(sorted_assign, minlength=k).astype(np.int32)
    offsets = np.zeros(k + 1, dtype=np.int32)
    offsets[1:] = np.cumsum(counts)
    return {"centroids": centroids, "order": order, "offsets": offsets}


def save_ivf(path: str | Path, ivf: dict[str, np.ndarray]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        centroids=ivf["centroids"].astype(np.float32),
        order=ivf["order"].astype(np.int32),
        offsets=ivf["offsets"].astype(np.int32),
    )


def load_ivf(path: str | Path) -> dict[str, np.ndarray]:
    data = np.load(Path(path))
    return {
        "centroids": data["centroids"].astype(np.float32),
        "order": data["order"].astype(np.int32),
        "offsets": data["offsets"].astype(np.int32),
    }


def ivf_candidates(q: np.ndarray, ivf: dict[str, np.ndarray], nprobe: int) -> np.ndarray:
    """Lấy candidate ids từ top-nprobe centroid gần nhất."""
    centroids = ivf["centroids"]
    order = ivf["order"]
    offsets = ivf["offsets"]
    nprobe = max(1, min(nprobe, centroids.shape[0]))
    d2 = np.sum((centroids - q[None, :]) ** 2, axis=1)
    probe_ids = np.argsort(d2)[:nprobe]
    chunks: list[np.ndarray] = []
    for cid in probe_ids:
        a, b = int(offsets[cid]), int(offsets[cid + 1])
        if b > a:
            chunks.append(order[a:b])
    if not chunks:
        return np.zeros(0, dtype=np.int32)
    return np.concatenate(chunks).astype(np.int32)
