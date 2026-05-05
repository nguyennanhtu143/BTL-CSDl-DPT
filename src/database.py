"""Phase 5: CSDL đặc trưng SQLite."""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from src.config import (
    COLOR_DIM,
    DB_PATH,
    FREQ_DIM,
    GRAD_DIM,
    TOP_K,
    TOTAL_DIM,
    W_COLOR,
    W_FREQ,
    W_SHAPE,
)
from src.feature_extractor import extract_components_from_path
from src.matcher import find_top_k_weighted

VECTOR_DTYPE = np.float32

SCHEMA = """
CREATE TABLE IF NOT EXISTS images (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT UNIQUE NOT NULL,
    width INTEGER,
    height INTEGER,
    file_size INTEGER,
    color_vector BLOB,
    shape_vector BLOB,
    freq_vector BLOB,
    feature_vector BLOB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_images_filename ON images(filename);
"""


@dataclass
class DatabaseSnapshot:
    ids: np.ndarray
    filenames: list[str]
    vectors: np.ndarray
    color_vectors: np.ndarray
    shape_vectors: np.ndarray
    freq_vectors: np.ndarray

    def __len__(self) -> int:
        return len(self.filenames)


def connect(db_path: str | Path = DB_PATH) -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(db_path: str | Path = DB_PATH) -> None:
    with connect(db_path) as conn:
        conn.executescript(SCHEMA)
        _ensure_split_columns(conn)
        conn.commit()


def reset_db(db_path: str | Path = DB_PATH) -> None:
    with connect(db_path) as conn:
        conn.execute("DROP TABLE IF EXISTS images")
        conn.executescript(SCHEMA)
        _ensure_split_columns(conn)
        conn.commit()


def _ensure_split_columns(conn: sqlite3.Connection) -> None:
    cols = {r[1] for r in conn.execute("PRAGMA table_info(images)").fetchall()}
    if "color_vector" not in cols:
        conn.execute("ALTER TABLE images ADD COLUMN color_vector BLOB")
    if "shape_vector" not in cols:
        conn.execute("ALTER TABLE images ADD COLUMN shape_vector BLOB")
    if "freq_vector" not in cols:
        conn.execute("ALTER TABLE images ADD COLUMN freq_vector BLOB")


def serialize_vector_fixed(vec: np.ndarray, dim: int) -> bytes:
    if vec.shape != (dim,):
        raise ValueError(f"Vector phải shape ({dim},), nhận {vec.shape}")
    return vec.astype(VECTOR_DTYPE).tobytes()


def serialize_vector(vec: np.ndarray) -> bytes:
    return serialize_vector_fixed(vec, TOTAL_DIM)


def deserialize_vector_fixed(blob: bytes, dim: int) -> np.ndarray:
    arr = np.frombuffer(blob, dtype=VECTOR_DTYPE)
    if arr.shape != (dim,):
        raise ValueError(f"BLOB sai shape: {arr.shape}, mong đợi ({dim},)")
    return arr.copy()


def deserialize_vector(blob: bytes) -> np.ndarray:
    return deserialize_vector_fixed(blob, TOTAL_DIM)


def insert_image(
    conn: sqlite3.Connection,
    filename: str,
    width: int,
    height: int,
    file_size: int,
    vector: np.ndarray,
    color_vector: np.ndarray,
    shape_vector: np.ndarray,
    freq_vector: np.ndarray,
) -> int:
    cur = conn.execute(
        "INSERT INTO images (filename, width, height, file_size, color_vector, shape_vector, freq_vector, feature_vector) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            filename,
            width,
            height,
            file_size,
            serialize_vector_fixed(color_vector, COLOR_DIM),
            serialize_vector_fixed(shape_vector, GRAD_DIM),
            serialize_vector_fixed(freq_vector, FREQ_DIM),
            serialize_vector(vector),
        ),
    )
    return cur.lastrowid


def upsert_image(
    conn: sqlite3.Connection,
    filename: str,
    width: int,
    height: int,
    file_size: int,
    vector: np.ndarray,
    color_vector: np.ndarray,
    shape_vector: np.ndarray,
    freq_vector: np.ndarray,
) -> int:
    cur = conn.execute(
        """
        INSERT INTO images (filename, width, height, file_size, color_vector, shape_vector, freq_vector, feature_vector)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(filename) DO UPDATE SET
            width = excluded.width,
            height = excluded.height,
            file_size = excluded.file_size,
            color_vector = excluded.color_vector,
            shape_vector = excluded.shape_vector,
            freq_vector = excluded.freq_vector,
            feature_vector = excluded.feature_vector,
            created_at = CURRENT_TIMESTAMP
        """,
        (
            filename,
            width,
            height,
            file_size,
            serialize_vector_fixed(color_vector, COLOR_DIM),
            serialize_vector_fixed(shape_vector, GRAD_DIM),
            serialize_vector_fixed(freq_vector, FREQ_DIM),
            serialize_vector(vector),
        ),
    )
    return cur.lastrowid


def count_images(db_path: str | Path = DB_PATH) -> int:
    db_path = Path(db_path)
    if not db_path.exists():
        return 0
    with connect(db_path) as conn:
        try:
            (n,) = conn.execute("SELECT COUNT(*) FROM images").fetchone()
            return int(n)
        except sqlite3.OperationalError:
            return 0


def existing_filenames(db_path: str | Path = DB_PATH) -> set[str]:
    if not Path(db_path).exists():
        return set()
    with connect(db_path) as conn:
        try:
            rows = conn.execute("SELECT filename FROM images").fetchall()
            return {r[0] for r in rows}
        except sqlite3.OperationalError:
            return set()


def load_database(db_path: str | Path = DB_PATH) -> DatabaseSnapshot:
    with connect(db_path) as conn:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(images)").fetchall()}
        has_split2 = {"color_vector", "shape_vector"}.issubset(cols)
        has_freq = "freq_vector" in cols
        if has_split2 and has_freq:
            rows = conn.execute(
                "SELECT id, filename, color_vector, shape_vector, freq_vector, feature_vector FROM images ORDER BY id"
            ).fetchall()
        elif has_split2:
            rows = conn.execute(
                "SELECT id, filename, color_vector, shape_vector, feature_vector FROM images ORDER BY id"
            ).fetchall()
        else:
            rows = conn.execute("SELECT id, filename, feature_vector FROM images ORDER BY id").fetchall()

    if not rows:
        return DatabaseSnapshot(
            ids=np.zeros(0, dtype=np.int64),
            filenames=[],
            vectors=np.zeros((0, TOTAL_DIM), dtype=VECTOR_DTYPE),
            color_vectors=np.zeros((0, COLOR_DIM), dtype=VECTOR_DTYPE),
            shape_vectors=np.zeros((0, GRAD_DIM), dtype=VECTOR_DTYPE),
            freq_vectors=np.zeros((0, FREQ_DIM), dtype=VECTOR_DTYPE),
        )

    ids = np.array([r[0] for r in rows], dtype=np.int64)
    filenames = [r[1] for r in rows]
    if has_split2 and has_freq:
        color_vectors = np.stack([deserialize_vector_fixed(r[2], COLOR_DIM) for r in rows])
        shape_vectors = np.stack([deserialize_vector_fixed(r[3], GRAD_DIM) for r in rows])
        freq_vectors = np.stack([deserialize_vector_fixed(r[4], FREQ_DIM) for r in rows])
        vectors = np.stack([deserialize_vector(r[5]) for r in rows])
    elif has_split2:
        color_vectors = np.stack([deserialize_vector_fixed(r[2], COLOR_DIM) for r in rows])
        shape_vectors = np.stack([deserialize_vector_fixed(r[3], GRAD_DIM) for r in rows])
        vectors = np.stack([deserialize_vector(r[4]) for r in rows])
        if vectors.shape[1] >= COLOR_DIM + GRAD_DIM + FREQ_DIM:
            freq_vectors = vectors[:, COLOR_DIM + GRAD_DIM:COLOR_DIM + GRAD_DIM + FREQ_DIM].astype(
                VECTOR_DTYPE, copy=True
            )
        else:
            freq_vectors = np.zeros((vectors.shape[0], FREQ_DIM), dtype=VECTOR_DTYPE)
    else:
        vectors = np.stack([deserialize_vector(r[2]) for r in rows])
        color_vectors = vectors[:, :COLOR_DIM].astype(VECTOR_DTYPE, copy=True)
        shape_vectors = vectors[:, COLOR_DIM:COLOR_DIM + GRAD_DIM].astype(VECTOR_DTYPE, copy=True)
        if vectors.shape[1] >= COLOR_DIM + GRAD_DIM + FREQ_DIM:
            freq_vectors = vectors[:, COLOR_DIM + GRAD_DIM:COLOR_DIM + GRAD_DIM + FREQ_DIM].astype(
                VECTOR_DTYPE, copy=True
            )
        else:
            freq_vectors = np.zeros((vectors.shape[0], FREQ_DIM), dtype=VECTOR_DTYPE)
    return DatabaseSnapshot(
        ids=ids,
        filenames=filenames,
        vectors=vectors,
        color_vectors=color_vectors,
        shape_vectors=shape_vectors,
        freq_vectors=freq_vectors,
    )


def query(
    image_path: str | Path,
    db: DatabaseSnapshot | None = None,
    k: int = TOP_K,
    db_path: str | Path = DB_PATH,
) -> tuple[np.ndarray, list[tuple[str, float]]]:
    if db is None:
        db = load_database(db_path)
    if len(db) == 0:
        raise RuntimeError(f"CSDL rỗng tại {db_path}. Hãy chạy build_database.py.")

    q_color, q_shape, q_freq = extract_components_from_path(image_path)
    q = np.concatenate([W_COLOR * q_color, W_SHAPE * q_shape, W_FREQ * q_freq]).astype(np.float32)
    top = find_top_k_weighted(
        q_color=q_color,
        q_shape=q_shape,
        q_freq=q_freq,
        db_color=db.color_vectors,
        db_shape=db.shape_vectors,
        db_freq=db.freq_vectors,
        k=k,
        ids=db.filenames,
        w_color=W_COLOR,
        w_shape=W_SHAPE,
        w_freq=W_FREQ,
    )
    return q, [(str(name), dist) for name, dist in top]


def get_image_meta(filename: str, db_path: str | Path = DB_PATH) -> dict | None:
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT id, filename, width, height, file_size, created_at FROM images WHERE filename = ?",
            (filename,),
        ).fetchone()
    if not row:
        return None
    keys = ("id", "filename", "width", "height", "file_size", "created_at")
    return dict(zip(keys, row))


def read_image_meta(path: Path) -> tuple[int, int, int]:
    with Image.open(path) as im:
        width, height = im.size
    file_size = path.stat().st_size
    return width, height, file_size
