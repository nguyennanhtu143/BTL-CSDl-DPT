"""Cấu hình toàn cục cho hệ thống CBIR ảnh nền thiên nhiên."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "Images_Dataset" / "01_thien_nhien"
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "features.db"

IMG_WIDTH = 256
IMG_HEIGHT = 144

GRID = 3

COLOR_BINS = 4
COLOR_DIM = (COLOR_BINS ** 3) * (GRID ** 2)

GRAD_BINS = 9
GRAD_DIM = GRAD_BINS * (GRID ** 2)

TOTAL_DIM = COLOR_DIM + GRAD_DIM

W_COLOR = 0.7
W_SHAPE = 0.3

TOP_K = 5
