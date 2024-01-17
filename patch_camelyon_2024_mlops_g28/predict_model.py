import sys
import numpy as np
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
    def __init__(self, data):
        # For unbatched data (single images)
        self.data = data if data.ndim > 3 else data.unsqueeze(dim=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return sample


def main(modelpath, datapath):
    # checks if the number of arguments passed to the script is three
    model = torch.load(modelpath)
    datafile = datapath  # data file path
    if ".npy" in datafile:
        data = np.moveaxis(np.load(datafile), -1, -3)  # traspose data
    elif ".pt" in datafile:
        data = torch.load(datafile)
    elif ".pkl" in datafile:
        # Dataset file, only take images from it
        data = torch.load(datafile)
        data = torch.stack([image for image, _ in data])  # get only the umages from the file
    else:
        print(f'File format not suported: {datafile.split(".")[-1]}')
        raise Exception(f'File format not suported: {datafile.split(".")[-1]}')
    preds = predict(model, DataLoader(PredictionDataset(data), batch_size=64))
    torch.save(preds, f"./data/predictions/predictions_{os.path.basename(datafile).split('.')[0]}.pt")
    print(f"Predictions: {preds.tolist()}")
    return preds


if __name__ == "__main__":
    if len(sys.argv) == 3:
        main(*sys.argv[1:])
    else:
        print("Incorrect usage of script arguments")
