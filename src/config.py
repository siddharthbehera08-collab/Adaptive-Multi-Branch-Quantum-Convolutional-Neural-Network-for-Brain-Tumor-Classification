from pathlib import Path

# =============================================================================
# Project Paths
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_DATASET_PATH = PROJECT_ROOT / "Data" / "Training"
TEST_DATASET_PATH = PROJECT_ROOT / "Data" / "Testing"

# Original Kaggle Dataset:
# https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

# =============================================================================
# Image Size
# =============================================================================

IMG_SIZE = (128, 128)