"""Cấu hình toàn cục cho hệ thống CBIR ảnh nền thiên nhiên (pipeline hybrid3)."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "Images_Dataset" / "01_thien_nhien"
DATA_DIR = PROJECT_ROOT / "data"

IMG_WIDTH = 256
IMG_HEIGHT = 144

GRID = 3

COLOR_BINS = 4

GRAD_BINS = 9
GRAD_DIM = GRAD_BINS * (GRID ** 2)

TOP_K = 5
