import torch
from Patch_Camelyon_2024_MLOPS_G28.models.model import SimpleCNN

torch.manual_seed(42)
NETWORK = SimpleCNN()
DATA = torch.ones((100, 3, 96, 96))
LABELS = torch.ones(10)
PREDS_01 = torch.cat((torch.zeros((10, 1)), torch.ones((10, 1))), dim=1)
PREDS_10 = torch.cat((torch.ones((10, 1)), torch.zeros((10, 1))), dim=1)


def test_model_forward():
    assert NETWORK.forward(DATA).shape == (
        len(DATA),
        2,
    )  # The forward pass of the NN does not return a tensor of a valid shape


def test_model_accuracy_100():
    assert NETWORK.accuracy(PREDS_01, LABELS) == 1  # Accuracy does not identify that both tensors are equal


def test_model_accuracy_0():
    assert NETWORK.accuracy(PREDS_10, LABELS) == 0  # Accuracy does not identify that both tensoes are oposite
