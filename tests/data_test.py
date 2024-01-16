import os.path
import torch
import pytest

from tests import IMAGES_PWD, LABELS_PWD
from Patch_Camelyon_2024_MLOPS_G28.data.make_dataset import h5gz_to_tensor

if os.path.exists(IMAGES_PWD):
    IMAGES = h5gz_to_tensor(src=IMAGES_PWD, images=True)

if os.path.exists(LABELS_PWD):
    LABELS = h5gz_to_tensor(src=LABELS_PWD, images=False)


@pytest.mark.skipif(not os.path.exists(IMAGES_PWD), reason="Images file not found")
def test_data_processing_images():
    assert IMAGES.shape == (2**15, 3, 96, 96)  # The shape of the loaded images tensor is inadecuate


@pytest.mark.skipif(not os.path.exists(LABELS_PWD), reason="Labels file not found")
def test_data_processing_labels_shape():
    assert LABELS.shape == torch.Size([2**15])  # The shape of the loaded targets tensor is inadecuate


@pytest.mark.skipif(not os.path.exists(LABELS_PWD), reason="Labels file not found")
def test_data_processing_labels_type():
    assert LABELS.dtype == torch.uint8  # The type of the loaded targets is incorrect
