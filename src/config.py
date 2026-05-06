"""Cấu hình toàn cục cho hệ thống CBIR ảnh nền thiên nhiên."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "Images_Dataset" / "01_thien_nhien"
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "features.db"

IMG_WIDTH = 256
IMG_HEIGHT = 144

GRID = 3

# True: histogram màu trên CIELAB (tách sáng/sắc tốt hơn RGB). Đổi cờ -> cần build lại CSDL.
USE_LAB_COLOR_HISTOGRAM = False

COLOR_BINS = 4
COLOR_DIM = (COLOR_BINS ** 3) * (GRID ** 2)

GRAD_BINS = 9
GRAD_DIM = GRAD_BINS * (GRID ** 2)

FREQ_DCT_SIDE = 8
FREQ_DCT_DIM = FREQ_DCT_SIDE * FREQ_DCT_SIDE
FREQ_FFT_BINS = 32
FREQ_DIM = FREQ_DCT_DIM + FREQ_FFT_BINS

TOTAL_DIM = COLOR_DIM + GRAD_DIM + FREQ_DIM

W_COLOR = 0.55
W_SHAPE = 0.20
W_FREQ = 0.25
W_LAYOUT = 0.10

TOP_K = 5

# PCA + IVF index artifacts (không dùng CNN)
PCA_COLOR_DIM = 96
PCA_SHAPE_DIM = 24
PCA_FREQ_DIM = 24
PCA_MODEL_PATH = DATA_DIR / "pca_model.npz"
IVF_INDEX_PATH = DATA_DIR / "ivf_index.npz"
IVF_NLIST = 64
IVF_NPROBE = 6
