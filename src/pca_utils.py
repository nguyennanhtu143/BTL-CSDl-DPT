from __future__ import annotations

from pathlib import Path

import numpy as np


def fit_pca(x: np.ndarray, out_dim: int) -> tuple[np.ndarray, np.ndarray]:
    """Fit PCA bằng SVD: trả (mean, components[out_dim, D])."""
    if x.ndim != 2:
        raise ValueError(f"Yêu cầu ma trận 2D, nhận {x.shape}")
    n, d = x.shape
    if n == 0:
        raise ValueError("Không thể fit PCA với ma trận rỗng")
    k = max(1, min(out_dim, d, n))
    mean = x.mean(axis=0, dtype=np.float64)
    xc = x.astype(np.float64) - mean
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    comps = vt[:k].astype(np.float32)
    return mean.astype(np.float32), comps


def transform_pca(x: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    """Chiếu dữ liệu theo PCA đã fit."""
    if x.ndim == 1:
        return ((x.astype(np.float32) - mean) @ components.T).astype(np.float32)
    return ((x.astype(np.float32) - mean[None, :]) @ components.T).astype(np.float32)


def save_pca_bundle(
    path: str | Path,
    *,
    color_mean: np.ndarray,
    color_components: np.ndarray,
    shape_mean: np.ndarray,
    shape_components: np.ndarray,
    freq_mean: np.ndarray,
    freq_components: np.ndarray,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        color_mean=color_mean.astype(np.float32),
        color_components=color_components.astype(np.float32),
        shape_mean=shape_mean.astype(np.float32),
        shape_components=shape_components.astype(np.float32),
        freq_mean=freq_mean.astype(np.float32),
        freq_components=freq_components.astype(np.float32),
    )


def load_pca_bundle(path: str | Path) -> dict[str, np.ndarray]:
    data = np.load(Path(path))
    return {
        "color_mean": data["color_mean"].astype(np.float32),
        "color_components": data["color_components"].astype(np.float32),
        "shape_mean": data["shape_mean"].astype(np.float32),
        "shape_components": data["shape_components"].astype(np.float32),
        "freq_mean": data["freq_mean"].astype(np.float32),
        "freq_components": data["freq_components"].astype(np.float32),
    }
