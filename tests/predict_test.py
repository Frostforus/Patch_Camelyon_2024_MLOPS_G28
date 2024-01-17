import torch
import os
import pytest

from tests import DATA_PDW, NETWORK, MODEL_PDW, PKL_PDW, PT_PDW, NPY_PDW, OTHER_PDW

from torch.utils.data import DataLoader
from patch_camelyon_2024_mlops_g28.predict_model import predict, PredictionDataset, main

if os.path.exists(DATA_PDW):
    DATA = torch.load(DATA_PDW)
    torch.save(DATA[0:10], PT_PDW)
    torch.save(DATA[0:10], NPY_PDW)
    ONE_IMAGE_DATASET = PredictionDataset(DATA[0][0])
    DATASET = PredictionDataset(DATA[:][0])
    DATALOADER = DataLoader(dataset=DATASET, batch_size=5)
    ONE_IMAGE_DATALOADER = DataLoader(dataset=ONE_IMAGE_DATASET, batch_size=1)


@pytest.mark.skipif(not os.path.exists(DATA_PDW), reason="Data file not found")
def test_predictionDataset_len():
    assert len(DATASET) == len(
        DATA
    )  # The length function of the PredictionDataset class does not return the correct value


@pytest.mark.skipif(not os.path.exists(DATA_PDW), reason="Data file not found")
def test_predictionDataset_get_item():
    for i in range(0, len(DATA), int(len(DATA) / 5)):
        assert torch.all(
            DATASET.__getitem__(i).eq(DATA[i][0])
        )  # The __getitem__ function of the PredictionDataset class does not return the correct value


@pytest.mark.skipif(not os.path.exists(DATA_PDW), reason="Data file not found")
def test_predictionDataset_one_image_len():
    assert len(ONE_IMAGE_DATASET) == len(
        ONE_IMAGE_DATASET
    )  # The length function of the PredictionDataset class does not return the correct value


@pytest.mark.skipif(not os.path.exists(DATA_PDW), reason="Data file not found")
def test_predictionDataset_one_image_get_item():
    assert torch.all(
        ONE_IMAGE_DATASET.__getitem__(0).eq(DATA[0][0])
    )  # The __getitem__ function of the PredictionDataset class does not return the correct value


@pytest.mark.skipif(not os.path.exists(DATA_PDW), reason="Data file not found")
def test_prediction_function():
    assert predict(model=NETWORK, dataloader=DATALOADER).shape == torch.Size(
        [len(DATASET)]
    )  # The output shape of the prediction function is incorrect


@pytest.mark.skipif(not os.path.exists(PKL_PDW) or not os.path.exists(MODEL_PDW), reason="Data file not found")
def test_main_function_PKL():
    assert main(MODEL_PDW, PKL_PDW).shape == torch.Size([len(DATA)])


@pytest.mark.skipif(not os.path.exists(PT_PDW) or not os.path.exists(MODEL_PDW), reason="Data file not found")
def test_main_function_PT():
    assert main(MODEL_PDW, PT_PDW).shape == torch.Size([10])


@pytest.mark.skipif(not os.path.exists(NPY_PDW) or not os.path.exists(MODEL_PDW), reason="Data file not found")
def test_main_function_NPY():
    assert main(MODEL_PDW, NPY_PDW).shape == torch.Size([10])


@pytest.mark.skipif(not os.path.exists(OTHER_PDW) or not os.path.exists(MODEL_PDW), reason="Data file not found")
def test_main_function_other():
    with pytest.raises(Exception):
        main(MODEL_PDW, OTHER_PDW)
