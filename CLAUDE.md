# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Content-Based Image Retrieval (CBIR) system for nature background images. Searches for visually similar images from a 500-image dataset using feature vectors stored in SQLite.

## Setup

```bash
pip install -r requirements.txt
```

Uses a `.venv` local virtual environment. Activate with `.venv\Scripts\activate` (Windows).

## Common Commands

**Build feature databases (run once per pipeline):**
```bash
python build_database.py                      # legacy (657-dim), incremental
python build_database.py --rebuild            # force rebuild from scratch
python build_database.py --limit 50           # quick test with 50 images
python build_database_compact6.py             # compact6 (6-dim) pipeline
python build_database_hybrid.py               # hybrid (color hist + compact6)
python build_database_hybrid3.py              # hybrid3 (color + gradient + compact6)
```

**Run the web UI:**
```bash
streamlit run ui/app.py
```

**Benchmark all 4 pipelines:**
```bash
python benchmark_pipelines.py --query-limit 10 --k 5
```

**Inspect a database:**
```bash
python inspect_db.py
python inspect_db.py --filename 01_thien_nhien_001.jpg --full
python inspect_db.py --schema
```

**Run tests (no test framework, plain Python scripts):**
```bash
python notebooks/test_database.py
python notebooks/test_preprocessing.py
python notebooks/test_color_features.py
python notebooks/test_gradient_features.py
python notebooks/test_integration.py
```

## Architecture

### 4 Search Pipelines

Each pipeline has its own database module, snapshot dataclass, matcher, and SQLite file.

| Pipeline | DB file | Feature | Matcher |
|---|---|---|---|
| **Legacy** | `data/features.db` | 657-dim (color+shape+freq+layout) | `matcher.py` |
| **Compact6** | `data/features_compact6.db` | 6-dim scalar | `matcher_compact6.py` |
| **Hybrid** | `data/features_hybrid.db` | 576+6-dim | `matcher_hybrid.py` |
| **Hybrid3** | `data/features_hybrid3.db` | 576+81+6-dim | `matcher_hybrid3.py` |

### Legacy Pipeline Feature Vector (657-dim)

Built in `src/feature_extractor.py`, composed of three branches that are L1-normalized before concatenation with weights:

- **Color** (576-dim): Grid-based RGB histogram — 3×3 spatial grid × 4³ bins per cell — extracted in `src/color_features.py`
- **Shape** (81-dim): Sobel gradient histogram — 3×3 grid × 9 orientation bins — extracted in `src/gradient_features.py`
- **Frequency** (96-dim): DCT (8×8) + FFT radial bins — extracted in `src/frequency_features.py`
- **Layout scalars** (4 values, separate from the blob): eccentricity, contrast, roughness, orderliness — extracted in `src/layout_features.py`

Weights are configured in `src/config.py`: `W_COLOR=0.55`, `W_SHAPE=0.20`, `W_FREQ=0.25`, `W_LAYOUT=0.10`.

### Compact6 Feature Vector (6-dim)

Built in `src/compact6_features.py`: `[mean_rgb, stddev_rgb, skewness_rgb, coarseness, contrast, directionality]`. Used standalone in the compact6 pipeline, and as a component in hybrid/hybrid3.

### Distance Computation

All matchers compute Euclidean distance per component branch, normalize each branch by its mean distance across the DB (so all branches have equal amplitude), then combine with weights. This is in `matcher.py:weighted_distance_batch`.

### Data Flow

1. Image loaded via `src/preprocessing.py` (load → resize to 256×144 → grayscale)
2. Feature extraction per branch
3. Features stored as float32 BLOBs in SQLite via the respective `database_*.py` module
4. At query time, the full DB is loaded into a `*Snapshot` dataclass (all vectors in RAM as numpy arrays)
5. Vectorized batch distance computed against all N=500 vectors in one numpy broadcast
6. Top-k returned sorted by ascending distance

### Key Config (src/config.py)

- `DATASET_DIR`: `Images_Dataset/01_thien_nhien/`
- `DATA_DIR`: `data/`
- `IMG_WIDTH=256`, `IMG_HEIGHT=144` (resize target)
- `GRID=3` (spatial grid cells per side)
- `USE_LAB_COLOR_HISTOGRAM=False` — if changed, all DBs must be rebuilt with `--rebuild`

### PCA + IVF Index

`build_database.py` optionally fits PCA on each feature branch and builds an IVF (k-means inverted file) index. Artifacts saved to `data/pca_model.npz` and `data/ivf_index.npz`. Used by `src/retrieval_accel.py` for accelerated search. Skip with `--no-pca` / `--no-index`.

### Streamlit UI (ui/app.py)

Loads the legacy database once into `@st.cache_resource`. On search, extracts features from uploaded image bytes (in-memory, no temp file), runs `find_top_k_weighted`, displays results as a grid. Logs query details to terminal.
