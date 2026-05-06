from __future__ import annotations

from pathlib import Path

import numpy as np

from src.config import (
    IVF_INDEX_PATH,
    IVF_NPROBE,
    PCA_MODEL_PATH,
    W_COLOR,
    W_FREQ,
    W_LAYOUT,
    W_SHAPE,
)
from src.database import DatabaseSnapshot
from src.ivf_index import ivf_candidates, load_ivf
from src.matcher import find_top_k_weighted
from src.pca_utils import load_pca_bundle, transform_pca


class AccelRetriever:
    """Truy vấn tăng tốc bằng PCA + IVF trên vector giảm chiều."""

    def __init__(
        self,
        db: DatabaseSnapshot,
        pca_path: str | Path = PCA_MODEL_PATH,
        index_path: str | Path = IVF_INDEX_PATH,
        nprobe: int = IVF_NPROBE,
    ) -> None:
        self.db = db
        self.nprobe = nprobe
        self.pca = load_pca_bundle(pca_path)
        self.ivf = load_ivf(index_path)

        self.db_color_pca = transform_pca(
            db.color_vectors, self.pca["color_mean"], self.pca["color_components"]
        )
        self.db_shape_pca = transform_pca(
            db.shape_vectors, self.pca["shape_mean"], self.pca["shape_components"]
        )
        self.db_freq_pca = transform_pca(
            db.freq_vectors, self.pca["freq_mean"], self.pca["freq_components"]
        )
        self.db_joined_pca = np.concatenate(
            [W_COLOR * self.db_color_pca, W_SHAPE * self.db_shape_pca, W_FREQ * self.db_freq_pca], axis=1
        ).astype(np.float32)

    def query(
        self,
        q_color: np.ndarray,
        q_shape: np.ndarray,
        q_freq: np.ndarray,
        q_layout: np.ndarray,
        k: int,
    ) -> list[tuple[str, float]]:
        q_color_pca = transform_pca(
            q_color, self.pca["color_mean"], self.pca["color_components"]
        )
        q_shape_pca = transform_pca(
            q_shape, self.pca["shape_mean"], self.pca["shape_components"]
        )
        q_freq_pca = transform_pca(
            q_freq, self.pca["freq_mean"], self.pca["freq_components"]
        )
        q_joined = np.concatenate([W_COLOR * q_color_pca, W_SHAPE * q_shape_pca, W_FREQ * q_freq_pca]).astype(
            np.float32
        )

        cand = ivf_candidates(q_joined, self.ivf, nprobe=self.nprobe)
        if cand.size == 0:
            cand = np.arange(len(self.db.filenames), dtype=np.int32)

        top = find_top_k_weighted(
            q_color=q_color_pca,
            q_shape=q_shape_pca,
            q_freq=q_freq_pca,
            db_color=self.db_color_pca[cand],
            db_shape=self.db_shape_pca[cand],
            db_freq=self.db_freq_pca[cand],
            q_layout=q_layout,
            db_layout=self.db.layout_scalars[cand],
            k=k,
            ids=np.asarray(self.db.filenames, dtype=object)[cand],
            w_color=W_COLOR,
            w_shape=W_SHAPE,
            w_freq=W_FREQ,
            w_layout=W_LAYOUT,
        )
        return [(str(name), float(dist)) for name, dist in top]
