import torch
from Patch_Camelyon_2024_MLOPS_G28.models.model import SimpleCNN

torch.manual_seed(42)
NETWORK = SimpleCNN()
DATA = torch.ones((100, 3, 94, 94))
LABELS = torch.ones(10)
PREDS_01 = torch.cat((torch.zeros(10), torch.ones(10)), dim=0)
PREDS_10 = torch.cat((torch.ones(10), torch.zeros(10)), dim=0)


def test_model_forward():
    assert NETWORK.forward(DATA).shape == (
        len(DATA),
        1024,
        2,
    )  # The forward pass of the network does not return a tensor of a valid shape


def test_model_accuracy():
    assert NETWORK.accuracy(LABELS, PREDS_01) == 1  # Accuracy does not identify that both tensors are equal
    assert NETWORK.accuracy(LABELS, PREDS_10) == 0  # Accuracy does not identify that both tensoes are oposite
