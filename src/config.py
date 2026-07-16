from pathlib import Path

# =============================================================================
# Project Paths
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "Training"
TEST_DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "Testing"

# Original Kaggle Dataset:
# https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset