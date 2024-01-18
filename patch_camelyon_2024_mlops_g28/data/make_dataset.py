import os
import gzip

import h5py
import numpy as np
import torch
from torch.utils.data import TensorDataset, random_split
import typing


def h5gz_to_tensor(src: str, dest: str = None, images: bool = False) -> torch.Tensor:
    """
    Convers the data on the source path to a tensor and stores it on the destination path.

    Args:
        src: path to the source data file
        dest: path to the destination data file
        images: True for loading images and False for targets

    Returns
        A tensor of size (N,3,96,96) for Images == True or (N,1) else with the corresponding
        images or targets in the source data file. WhereN >= 1.
    """
    if dest is not None and os.path.exists(dest):
        ans = input(f"{src} is already processed, do you want to do it again? [y/n]: ")
        if ans.lower() not in ["y", "yes"]:
            return None
    with gzip.open(src, "r") as gzf:
        with h5py.File(gzf, "r") as h5f:
            ds_name = list(h5f.keys())[0]
            ds = torch.tensor(np.array(h5f.get(ds_name)))
            if images:
                # torch works with BxCxHxW and float
                ds = ds.float().permute(0, 3, 1, 2)
            else:
                # The dataset stores labels unsqeezed (Bx1x1x1) for some reason.
                ds = ds.squeeze()
            if dest is not None:
                torch.save(ds, dest)
            else:
                return ds


def main(
    data: torch.Tensor, targets: torch.Tensor
) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generates the data splits for the data.

    Args:
        data: tensor of shape (N, 3, 96, 96) where N >= 1 and N == X
        targets: tensor of shape (X,1) where X >= 1 and X == N

    Returns
        A tuple of four tendors with the different data splits.
        The first 3 tensors are of shape ([N,3,96,96],[N,1]) images and targets
        and the last one is of shape (N,3,96,96) only images with no tagets.
    """

    # Ensure that the random split is always the same (but still 'random')
    torch.manual_seed(42)

    ds = TensorDataset(data, targets)
    train_size = int(len(ds) * 0.8)
    test_size = int(len(ds) * 0.1)
    val_size = len(ds) - train_size - test_size

    train_ds, test_ds, val_ds = random_split(ds, [train_size, test_size, val_size])

    # Save test images separately for easy prediction
    test_images = torch.stack([image for image, _ in test_ds])

    return train_ds, test_ds, val_ds, test_images


if __name__ == "__main__":
    """
    Gets the data splits and saves them locally.
    """
    data = h5gz_to_tensor(src="./data/raw/camelyonpatch_level_2_split_valid_x.h5.gz", images=True)
    targets = h5gz_to_tensor(src="./data/raw/camelyonpatch_level_2_split_valid_y.h5.gz", images=False)
    train_ds, test_ds, val_ds, test_images = main(data, targets)

    torch.save(test_images, "./data/processed/test_images.pt")
    torch.save(test_images[:20], "./data/processed/test_prediction_images.pt")

    torch.save(train_ds, "./data/processed/train_dataset.pkl")
    torch.save(test_ds, "./data/processed/test_dataset.pkl")
    torch.save(val_ds, "./data/processed/validation_dataset.pkl")
