"""
Konfigurasi terpusat untuk Image Preprocessing Playground.
"""

from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# Dataset configurations
DATASET_SLUG = "frmgtx/facial-emotion-recognition-images-raw-dataset"
DATASET_URL = f"https://www.kaggle.com/datasets/{DATASET_SLUG}"
SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")

# Default random seed for reproducibility
DEFAULT_SEED = 42

# Default visualization settings
DEFAULT_FIGSIZE = (12, 8)
DEFAULT_CMAP_GRAY = "gray"
