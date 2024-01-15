import os.path
import torch
import pytest

from Patch_Camelyon_2024_MLOPS_G28.data.make_dataset import h5gz_to_tensor

IMAGES_PWD = "tests\.filesForTesting\camelyonpatch_level_2_split_valid_x.h5.gz"
LABELS_PWD = "tests\.filesForTesting\camelyonpatch_level_2_split_valid_y.h5.gz"


@pytest.mark.skipif(not os.path.exists(IMAGES_PWD), reason="Images file not found")
def test_data_processing_images():
    data = h5gz_to_tensor(src=IMAGES_PWD, images=True)
    assert data.shape == (2**15, 3, 96, 96)  # The shape of the loaded images tensor is inadecuate


@pytest.mark.skipif(not os.path.exists(LABELS_PWD), reason="Labels file not found")
def test_data_processing_labels():
    targets = h5gz_to_tensor(src=LABELS_PWD, images=False)
    assert targets.shape == torch.Size([2**15])  # The shape of the loaded targets tensor is inadecuate
    assert targets.dtype == torch.uint8  # The type of the loaded targets is incorrect
