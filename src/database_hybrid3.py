from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from src.color_features import extract_color_feature
from src.compact6_features import extract_compact6
from src.config import GRAD_DIM
from src.gradient_features import extract_gradient_feature
from src.matcher_hybrid3 import find_top_k_hybrid3, find_top_k_hybrid3_two_stage
from src.preprocessing import load_image, resize_image, to_grayscale

HIST_DIM = 576
COMPACT6_DIM = 6
VECTOR_DTYPE = np.float32

SCHEMA = """
CREATE TABLE IF NOT EXISTS images_hybrid3 (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT UNIQUE NOT NULL,
    width INTEGER,
    height INTEGER,
    file_size INTEGER,
    mean REAL,
    stddev REAL,
    skewness REAL,
    coarseness REAL,
    contrast REAL,
    directionality REAL,
    color_hist_vector BLOB NOT NULL,
    grad_vector BLOB NOT NULL,
    compact6_vector BLOB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_images_hybrid3_filename ON images_hybrid3(filename);
"""


@dataclass
class Hybrid3Snapshot:
    filenames: list[str]
    hist_vectors: np.ndarray
    grad_vectors: np.ndarray
    compact_vectors: np.ndarray

    def __len__(self) -> int:
        return len(self.filenames)


def connect(db_path: str | Path) -> sqlite3.Connection:
    p = Path(db_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(p)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(db_path: str | Path) -> None:
    with connect(db_path) as conn:
        conn.executescript(SCHEMA)
        conn.commit()


def reset_db(db_path: str | Path) -> None:
    with connect(db_path) as conn:
        conn.execute("DROP TABLE IF EXISTS images_hybrid3")
        conn.executescript(SCHEMA)
        conn.commit()


def serialize_vec(vec: np.ndarray, dim: int) -> bytes:
    if vec.shape != (dim,):
        raise ValueError(f"Vector phải shape ({dim},), nhận {vec.shape}")
    return vec.astype(VECTOR_DTYPE).tobytes()


def deserialize_vec(blob: bytes, dim: int) -> np.ndarray:
    arr = np.frombuffer(blob, dtype=VECTOR_DTYPE)
    if arr.shape != (dim,):
        raise ValueError(f"BLOB sai shape: {arr.shape}, mong đợi ({dim},)")
    return arr.copy()


def extract_hybrid3_from_path(
    path: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    rgb = resize_image(load_image(path))
    gray = to_grayscale(rgb)
    hist = extract_color_feature(rgb)
    grad = extract_gradient_feature(gray)
    compact, scalars = extract_compact6(rgb, gray)
    return (
        hist.astype(np.float32),
        grad.astype(np.float32),
        compact.astype(np.float32),
        scalars,
    )


def read_image_meta(path: Path) -> tuple[int, int, int]:
    with Image.open(path) as im:
        width, height = im.size
    return width, height, path.stat().st_size


def upsert_image(
    conn: sqlite3.Connection,
    filename: str,
    width: int,
    height: int,
    file_size: int,
    hist: np.ndarray,
    grad: np.ndarray,
    compact: np.ndarray,
    scalars: dict[str, float],
) -> None:
    conn.execute(
        """
        INSERT INTO images_hybrid3
        (filename, width, height, file_size, mean, stddev, skewness, coarseness, contrast, directionality, color_hist_vector, grad_vector, compact6_vector)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(filename) DO UPDATE SET
            width=excluded.width,
            height=excluded.height,
            file_size=excluded.file_size,
            mean=excluded.mean,
            stddev=excluded.stddev,
            skewness=excluded.skewness,
            coarseness=excluded.coarseness,
            contrast=excluded.contrast,
            directionality=excluded.directionality,
            color_hist_vector=excluded.color_hist_vector,
            grad_vector=excluded.grad_vector,
            compact6_vector=excluded.compact6_vector,
            created_at=CURRENT_TIMESTAMP
        """,
        (
            filename,
            width,
            height,
            file_size,
            float(scalars["mean"]),
            float(scalars["stddev"]),
            float(scalars["skewness"]),
            float(scalars["coarseness"]),
            float(scalars["contrast"]),
            float(scalars["directionality"]),
            serialize_vec(hist, HIST_DIM),
            serialize_vec(grad, GRAD_DIM),
            serialize_vec(compact, COMPACT6_DIM),
        ),
    )


def existing_filenames(db_path: str | Path) -> set[str]:
    if not Path(db_path).exists():
        return set()
    with connect(db_path) as conn:
        try:
            rows = conn.execute("SELECT filename FROM images_hybrid3").fetchall()
            return {r[0] for r in rows}
        except sqlite3.OperationalError:
            return set()


def count_images(db_path: str | Path) -> int:
    if not Path(db_path).exists():
        return 0
    with connect(db_path) as conn:
        try:
            (n,) = conn.execute("SELECT COUNT(*) FROM images_hybrid3").fetchone()
            return int(n)
        except sqlite3.OperationalError:
            return 0


def load_database(db_path: str | Path) -> Hybrid3Snapshot:
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT filename, color_hist_vector, grad_vector, compact6_vector FROM images_hybrid3 ORDER BY id"
        ).fetchall()
    if not rows:
        return Hybrid3Snapshot(
            filenames=[],
            hist_vectors=np.zeros((0, HIST_DIM), dtype=VECTOR_DTYPE),
            grad_vectors=np.zeros((0, GRAD_DIM), dtype=VECTOR_DTYPE),
            compact_vectors=np.zeros((0, COMPACT6_DIM), dtype=VECTOR_DTYPE),
        )
    filenames = [r[0] for r in rows]
    hist_vectors = np.stack([deserialize_vec(r[1], HIST_DIM) for r in rows])
    grad_vectors = np.stack([deserialize_vec(r[2], GRAD_DIM) for r in rows])
    compact_vectors = np.stack([deserialize_vec(r[3], COMPACT6_DIM) for r in rows])
    return Hybrid3Snapshot(
        filenames=filenames,
        hist_vectors=hist_vectors,
        grad_vectors=grad_vectors,
        compact_vectors=compact_vectors,
    )


def query(
    image_path: str | Path,
    db: Hybrid3Snapshot,
    k: int,
    w_hist: float = 0.45,
    w_grad: float = 0.30,
    w_compact: float = 0.25,
    coarse_top: int | None = None,
    return_breakdown: bool = False,
):
    """Query hybrid3 với khoảng cách Euclidean (L2) trên cả 3 nhánh.

    Tham số nâng cao:
        coarse_top: nếu > 0, dùng two-stage retrieval (compact6 lọc thô).
        return_breakdown: nếu True, trả thêm list dict per-branch contribution.
    """
    q_hist, q_grad, q_compact, _ = extract_hybrid3_from_path(image_path)

    common_kwargs = dict(
        q_hist=q_hist,
        q_grad=q_grad,
        q_compact=q_compact,
        db_hist=db.hist_vectors,
        db_grad=db.grad_vectors,
        db_compact=db.compact_vectors,
        ids=db.filenames,
        k=k,
        w_hist=w_hist,
        w_grad=w_grad,
        w_compact=w_compact,
        return_breakdown=return_breakdown,
    )

    if coarse_top is not None and coarse_top > 0:
        result = find_top_k_hybrid3_two_stage(coarse_top=coarse_top, **common_kwargs)
    else:
        result = find_top_k_hybrid3(**common_kwargs)

    if return_breakdown:
        top, breakdown = result
        return q_hist, q_grad, q_compact, top, breakdown
    return q_hist, q_grad, q_compact, result
