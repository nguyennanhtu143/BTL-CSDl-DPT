"""Phase 5: CSDL đặc trưng SQLite.

Schema:
    images (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT UNIQUE NOT NULL,
        width INTEGER,
        height INTEGER,
        file_size INTEGER,
        feature_vector BLOB NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )

Vector lưu dạng BLOB = `np.ndarray(657,) float32 .tobytes()` -> 657*4 = 2628 byte/ảnh.
500 ảnh ~ 1.3 MB BLOB + metadata ~ vài chục KB.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from src.config import DB_PATH, TOP_K, TOTAL_DIM
from src.feature_extractor import extract_from_path
from src.matcher import find_top_k

VECTOR_DTYPE = np.float32

SCHEMA = """
CREATE TABLE IF NOT EXISTS images (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT UNIQUE NOT NULL,
    width INTEGER,
    height INTEGER,
    file_size INTEGER,
    feature_vector BLOB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_images_filename ON images(filename);
"""


@dataclass
class DatabaseSnapshot:
    """Toàn bộ CSDL nạp về RAM cho query nhanh."""

    ids: np.ndarray         # shape (N,) int64
    filenames: list[str]    # len N
    vectors: np.ndarray     # shape (N, TOTAL_DIM) float32

    def __len__(self) -> int:
        return len(self.filenames)


def connect(db_path: str | Path = DB_PATH) -> sqlite3.Connection:
    """Mở kết nối SQLite, đảm bảo thư mục cha tồn tại."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(db_path: str | Path = DB_PATH) -> None:
    """Tạo bảng nếu chưa có."""
    with connect(db_path) as conn:
        conn.executescript(SCHEMA)
        conn.commit()


def reset_db(db_path: str | Path = DB_PATH) -> None:
    """Xóa toàn bộ dữ liệu (drop + recreate). Dùng cho --rebuild."""
    with connect(db_path) as conn:
        conn.execute("DROP TABLE IF EXISTS images")
        conn.executescript(SCHEMA)
        conn.commit()


def serialize_vector(vec: np.ndarray) -> bytes:
    """Chuẩn hoá về float32 -> bytes BLOB."""
    if vec.shape != (TOTAL_DIM,):
        raise ValueError(f"Vector phải shape ({TOTAL_DIM},), nhận {vec.shape}")
    return vec.astype(VECTOR_DTYPE).tobytes()


def deserialize_vector(blob: bytes) -> np.ndarray:
    """BLOB -> ndarray (TOTAL_DIM,) float32. np.frombuffer trả về view read-only;
    copy để tránh lỗi khi caller muốn modify in-place."""
    arr = np.frombuffer(blob, dtype=VECTOR_DTYPE)
    if arr.shape != (TOTAL_DIM,):
        raise ValueError(f"BLOB sai shape: {arr.shape}, mong đợi ({TOTAL_DIM},)")
    return arr.copy()


def insert_image(
    conn: sqlite3.Connection,
    filename: str,
    width: int,
    height: int,
    file_size: int,
    vector: np.ndarray,
) -> int:
    """Insert 1 record. Trả lastrowid. UNIQUE(filename) sẽ lỗi nếu đã có."""
    cur = conn.execute(
        "INSERT INTO images (filename, width, height, file_size, feature_vector) "
        "VALUES (?, ?, ?, ?, ?)",
        (filename, width, height, file_size, serialize_vector(vector)),
    )
    return cur.lastrowid


def upsert_image(
    conn: sqlite3.Connection,
    filename: str,
    width: int,
    height: int,
    file_size: int,
    vector: np.ndarray,
) -> int:
    """Insert hoặc update theo filename. Hữu ích khi build lại từng phần."""
    cur = conn.execute(
        """
        INSERT INTO images (filename, width, height, file_size, feature_vector)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(filename) DO UPDATE SET
            width = excluded.width,
            height = excluded.height,
            file_size = excluded.file_size,
            feature_vector = excluded.feature_vector,
            created_at = CURRENT_TIMESTAMP
        """,
        (filename, width, height, file_size, serialize_vector(vector)),
    )
    return cur.lastrowid


def count_images(db_path: str | Path = DB_PATH) -> int:
    """Số record hiện có. 0 nếu bảng chưa tạo."""
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
    """Danh sách filename đã có để skip khi build incremental."""
    if not Path(db_path).exists():
        return set()
    with connect(db_path) as conn:
        try:
            rows = conn.execute("SELECT filename FROM images").fetchall()
            return {r[0] for r in rows}
        except sqlite3.OperationalError:
            return set()


def load_database(db_path: str | Path = DB_PATH) -> DatabaseSnapshot:
    """Đọc toàn bộ CSDL về RAM thành ma trận numpy (N, TOTAL_DIM)."""
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT id, filename, feature_vector FROM images ORDER BY id"
        ).fetchall()

    if not rows:
        return DatabaseSnapshot(
            ids=np.zeros(0, dtype=np.int64),
            filenames=[],
            vectors=np.zeros((0, TOTAL_DIM), dtype=VECTOR_DTYPE),
        )

    ids = np.array([r[0] for r in rows], dtype=np.int64)
    filenames = [r[1] for r in rows]
    vectors = np.stack([deserialize_vector(r[2]) for r in rows])
    return DatabaseSnapshot(ids=ids, filenames=filenames, vectors=vectors)


def query(
    image_path: str | Path,
    db: DatabaseSnapshot | None = None,
    k: int = TOP_K,
    db_path: str | Path = DB_PATH,
) -> tuple[np.ndarray, list[tuple[str, float]]]:
    """Trích xuất feature ảnh truy vấn rồi tìm top-k trong CSDL.

    Tham số:
        image_path: ảnh đầu vào.
        db: snapshot đã load sẵn; nếu None sẽ load từ db_path (chậm hơn).
        k: số kết quả.
        db_path: đường dẫn SQLite (chỉ dùng khi `db is None`).

    Trả `(query_vector, [(filename, distance), ...])`.
    """
    if db is None:
        db = load_database(db_path)
    if len(db) == 0:
        raise RuntimeError(f"CSDL rỗng tại {db_path}. Hãy chạy build_database.py.")

    q = extract_from_path(image_path)
    top = find_top_k(q, db.vectors, k=k, ids=db.filenames)
    return q, [(str(name), dist) for name, dist in top]


def get_image_meta(filename: str, db_path: str | Path = DB_PATH) -> dict | None:
    """Lấy metadata 1 ảnh theo filename. Trả None nếu không có."""
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT id, filename, width, height, file_size, created_at "
            "FROM images WHERE filename = ?",
            (filename,),
        ).fetchone()
    if not row:
        return None
    keys = ("id", "filename", "width", "height", "file_size", "created_at")
    return dict(zip(keys, row))


def read_image_meta(path: Path) -> tuple[int, int, int]:
    """Đọc width, height, file_size mà không cần load toàn bộ pixel."""
    with Image.open(path) as im:
        width, height = im.size
    file_size = path.stat().st_size
    return width, height, file_size
