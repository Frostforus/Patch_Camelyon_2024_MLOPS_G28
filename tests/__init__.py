import torch
from patch_camelyon_2024_mlops_g28.models.model import SimpleCNN


torch.manual_seed(42)
NETWORK = SimpleCNN()
DATA_PDW = "data/processed/test_dataset.pkl"

IMAGES_PWD = "data/raw/camelyonpatch_level_2_split_valid_x.h5.gz"
LABELS_PWD = "data/raw/camelyonpatch_level_2_split_valid_y.h5.gz"
