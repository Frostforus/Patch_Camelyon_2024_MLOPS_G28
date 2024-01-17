import torch
import os
import pytest

from tests import DATA_PDW, NETWORK

from torch.utils.data import DataLoader
from patch_camelyon_2024_mlops_g28.predict_model import predict

if os.path.exists(DATA_PDW):
    DATASET = torch.load(DATA_PDW)[0:2][0]
    DATALOADER = DataLoader(dataset=DATASET, batch_size=1)


@pytest.mark.skipif(not os.path.exists(DATA_PDW), reason="Data file not found")
def test_prediction_function():
    assert predict(model=NETWORK, dataloader=DATALOADER).shape == (len(DATASET), 2)
