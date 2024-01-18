import sys
import torch
from torch.utils.data import Dataset, DataLoader
import os


def predict(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader) -> None:
    """Run prediction for a given model and dataloader.

    Args:
        model: model to use for prediction
        dataloader: dataloader with batches

    Returns
        Tensor of shape [N, d] where N is the number of samples and d is the output dimension of the model

    """
    return torch.cat([model(batch).argmax(dim=-1) for batch in dataloader], 0)


class PredictionDataset(Dataset):
    """Torch's Dataset class specialization for inderence."""

    def __init__(self, data) -> None:
        """Asign the input Data to the attribute data unsqueezing
        if the data is unbathed (single images)

        Args:
            data: tensor of (x,3,96,96) or (3,96,96) images"""
        self.data = data if data.ndim > 3 else data.unsqueeze(dim=0)

    def __len__(self) -> int:
        """Overwrites the length method of the class so it returns the
        length value of its data atribute"""
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Overwrites the getitem method of the class so it returns the
        element of the input index (idx) of the classes data attribute.

        Args:
            idx: index of the element to retrieve

        Returns
            Tensor of shape (3,96,96)
        """
        sample = self.data[idx]
        return sample


def main(modelpath: str, datapath: str) -> torch.Tensor:
    """
    1. Loads the model specified by the input modelpath.
    2. Loads the data specified by the input datapath and
    performs any modifications required to its structure
    based on the files extension.
    3. If the data file format is not supported it raises an exception.
    4. Gathers the predictions and returns them.

    Args:
        modelpath: path to the model to load
        datapath: path to the data to load

    Returns
        Tensor of shape (N,1) where N is the size of the data to predict
    """
    # checks if the number of arguments passed to the script is three
    model = torch.load(modelpath)
    datafile = datapath  # data file path
    if ".pt" in datafile:
        data = torch.load(datafile)
    elif ".pkl" in datafile:
        # Dataset file, only take images from it
        data = torch.load(datafile)
        data = torch.stack([image for image in data[0]])  # get only the umages from the file
    else:
        print(f'File format not suported: {datafile.split(".")[-1]}')
        raise Exception(f'File format not suported: {datafile.split(".")[-1]}')
    preds = predict(model, DataLoader(PredictionDataset(data), batch_size=64))
    print(f"Predictions: {preds.tolist()}")
    return preds  # Dimensions: length of dataset X 1


if __name__ == "__main__":
    if len(sys.argv) == 3:
        preds = main(*sys.argv[1:])
        torch.save(preds, f"./data/predictions/predictions_{os.path.basename(sys.argv[1]).split('.')[0]}.pt")
    else:
        print("Incorrect usage of script arguments")
