import os.path
import torch
import pytest

from tests import IMAGES_PWD, LABELS_PWD
from patch_camelyon_2024_mlops_g28.data.make_dataset import h5gz_to_tensor, main

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


@pytest.mark.skipif(
    not os.path.exists(IMAGES_PWD) or not os.path.exists(LABELS_PWD), reason="Images or labels file not found"
)
@pytest.fixture
def get_main():
    return main(IMAGES, LABELS)


@pytest.mark.skipif(
    not os.path.exists(IMAGES_PWD) or not os.path.exists(LABELS_PWD), reason="Images or labels file not found"
)
def test_data_main_test_ds_data(get_main):
    assert get_main[1][:][0].shape == (int(len(IMAGES) * 0.1), 3, 96, 96)  # The shape of the test split is inadecuate


@pytest.mark.skipif(
    not os.path.exists(IMAGES_PWD) or not os.path.exists(LABELS_PWD), reason="Images or labels file not found"
)
def test_data_main_test_ds_targets(get_main):
    assert get_main[1][:][1].shape[0] == (int(len(IMAGES) * 0.1))  # The shape of the test split is inadecuate


@pytest.mark.skipif(
    not os.path.exists(IMAGES_PWD) or not os.path.exists(LABELS_PWD), reason="Images or labels file not found"
)
def test_data_main_val_ds_data(get_main):
    assert get_main[2][:][0].shape == (
        len(IMAGES) - int(len(IMAGES) * 0.8) - int(len(IMAGES) * 0.1),
        3,
        96,
        96,
    )  # The shape of the validation split is inadecuate


@pytest.mark.skipif(
    not os.path.exists(IMAGES_PWD) or not os.path.exists(LABELS_PWD), reason="Images or labels file not found"
)
def test_data_main_val_ds_targets(get_main):
    assert get_main[2][:][1].shape[0] == (
        len(IMAGES) - int(len(IMAGES) * 0.8) - int(len(IMAGES) * 0.1)
    )  # The shape of the validation split is inadecuate


@pytest.mark.skipif(
    not os.path.exists(IMAGES_PWD) or not os.path.exists(LABELS_PWD), reason="Images or labels file not found"
)
def test_data_main_train_ds_data(get_main):
    assert get_main[0][:][0].shape == (int(len(IMAGES) * 0.8), 3, 96, 96)  # The shape of the train split is inadecuate


@pytest.mark.skipif(
    not os.path.exists(IMAGES_PWD) or not os.path.exists(LABELS_PWD), reason="Images or labels file not found"
)
def test_data_main_train_ds_targets(get_main):
    assert get_main[0][:][1].shape[0] == (int(len(IMAGES) * 0.8))  # The shape of the train split is inadecuate


@pytest.mark.skipif(
    not os.path.exists(IMAGES_PWD) or not os.path.exists(LABELS_PWD), reason="Images or labels file not found"
)
def test_data_main_test_images(get_main):
    assert get_main[3][:].shape == (
        int(len(IMAGES) * 0.1),
        3,
        96,
        96,
    )  # The shape of the test only images split is inadecuate
