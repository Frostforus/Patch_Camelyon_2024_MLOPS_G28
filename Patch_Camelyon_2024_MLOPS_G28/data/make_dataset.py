import os
import gzip

import h5py
import numpy as np
import torch
from torch.utils.data import TensorDataset, random_split


def h5gz_to_tensor(src, dest=None, images=False):
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


if __name__ == "__main__":
    # Ensure that the random split is always the same (but still 'random'
    torch.manual_seed(42)

    data = h5gz_to_tensor(src="./data/raw/camelyonpatch_level_2_split_valid_x.h5.gz", images=True)
    targets = h5gz_to_tensor(src="./data/raw/camelyonpatch_level_2_split_valid_y.h5.gz", images=False)
    ds = TensorDataset(data, targets)
    train_size = int(len(ds) * 0.8)
    test_size = int(len(ds) * 0.1)
    val_size = len(ds) - train_size - test_size

    train_ds, test_ds, val_ds = random_split(ds, [train_size, test_size, val_size])
    torch.save(train_ds, "./data/processed/train_dataset.pkl")
    torch.save(test_ds, "./data/processed/test_dataset.pkl")
    torch.save(val_ds, "./data/processed/validation_dataset.pkl")
