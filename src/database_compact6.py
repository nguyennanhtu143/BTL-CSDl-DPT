from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from src.compact6_features import extract_compact6_from_path
from src.matcher_compact6 import find_top_k_compact6

FEATURE6_DIM = 6
VECTOR_DTYPE = np.float32


@dataclass
class Compact6Snapshot:
    filenames: list[str]
    vectors: np.ndarray  # (N, 6)

    def __len__(self) -> int:
        return len(self.filenames)


SCHEMA = """
CREATE TABLE IF NOT EXISTS images_compact6 (
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
    feature6_vector BLOB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_images_compact6_filename ON images_compact6(filename);
"""


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
        conn.execute("DROP TABLE IF EXISTS images_compact6")
        conn.executescript(SCHEMA)
        conn.commit()


def serialize_vector6(vec: np.ndarray) -> bytes:
    if vec.shape != (FEATURE6_DIM,):
        raise ValueError(f"Vector phải shape (6,), nhận {vec.shape}")
    return vec.astype(VECTOR_DTYPE).tobytes()


def deserialize_vector6(blob: bytes) -> np.ndarray:
    arr = np.frombuffer(blob, dtype=VECTOR_DTYPE)
    if arr.shape != (FEATURE6_DIM,):
        raise ValueError(f"BLOB sai shape: {arr.shape}, mong đợi (6,)")
    return arr.copy()


def read_image_meta(path: Path) -> tuple[int, int, int]:
    with Image.open(path) as im:
        width, height = im.size
    return width, height, path.stat().st_size


def existing_filenames(db_path: str | Path) -> set[str]:
    if not Path(db_path).exists():
        return set()
    with connect(db_path) as conn:
        try:
            rows = conn.execute("SELECT filename FROM images_compact6").fetchall()
            return {r[0] for r in rows}
        except sqlite3.OperationalError:
            return set()


def count_images(db_path: str | Path) -> int:
    if not Path(db_path).exists():
        return 0
    with connect(db_path) as conn:
        try:
            (n,) = conn.execute("SELECT COUNT(*) FROM images_compact6").fetchone()
            return int(n)
        except sqlite3.OperationalError:
            return 0


def upsert_image(
    conn: sqlite3.Connection,
    filename: str,
    width: int,
    height: int,
    file_size: int,
    vec6: np.ndarray,
    scalars: dict[str, float],
) -> None:
    conn.execute(
        """
        INSERT INTO images_compact6
        (filename, width, height, file_size, mean, stddev, skewness, coarseness, contrast, directionality, feature6_vector)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            feature6_vector=excluded.feature6_vector,
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
            serialize_vector6(vec6),
        ),
    )


def load_database(db_path: str | Path) -> Compact6Snapshot:
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT filename, feature6_vector FROM images_compact6 ORDER BY id"
        ).fetchall()
    if not rows:
        return Compact6Snapshot(filenames=[], vectors=np.zeros((0, FEATURE6_DIM), dtype=VECTOR_DTYPE))
    filenames = [r[0] for r in rows]
    vectors = np.stack([deserialize_vector6(r[1]) for r in rows])
    return Compact6Snapshot(filenames=filenames, vectors=vectors)


def query(image_path: str | Path, db: Compact6Snapshot, k: int) -> tuple[np.ndarray, list[tuple[str, float]]]:
    vec, _ = extract_compact6_from_path(image_path)
    top = find_top_k_compact6(vec, db.vectors, k=k, ids=db.filenames)
    return vec, top
