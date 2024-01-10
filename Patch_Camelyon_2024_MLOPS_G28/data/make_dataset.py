import os
import gzip

import h5py
import numpy as np
import torch

def h5gz_to_tensor(src, dest, images=False):
    if os.path.exists(dest):
        ans = input(f'{src} is already processed, do you want to do it again? [y/n]: ')
        if ans.lower() not in ['y', 'yes']:
            return None
    with gzip.open(src, 'r') as gzf:
        with h5py.File(gzf, 'r') as h5f:
            ds_name = list(h5f.keys())[0]
            ds = torch.tensor(np.array(h5f.get(ds_name)))
            if images:
                # torch works with BxCxHxW and float
                ds = ds.float().permute(0,3,1,2)
            else:
                # The dataset stores labels unsqeezed (Bx1x1x1) for some reason.
                ds = ds.squeeze()
            torch.save(ds, dest)

if __name__ == '__main__':
    print('Extracting tensors from test data...')
    h5gz_to_tensor('./data/raw/camelyonpatch_level_2_split_test_x.h5.gz', './data/processed/test_images.pt', True)
    h5gz_to_tensor('./data/raw/camelyonpatch_level_2_split_test_y.h5.gz', './data/processed/test_target.pt')
    print('Extracting tensors from validation data...')
    h5gz_to_tensor('./data/raw/camelyonpatch_level_2_split_valid_x.h5.gz', './data/processed/val_images.pt', True)
    h5gz_to_tensor('./data/raw/camelyonpatch_level_2_split_valid_y.h5.gz', './data/processed/val_target.pt')
    print('Extracting tensors from training data (may take a couple minutes)...')
    h5gz_to_tensor('./data/raw/camelyonpatch_level_2_split_train_x.h5.gz', './data/processed/train_images.pt', True)
    h5gz_to_tensor('./data/raw/camelyonpatch_level_2_split_train_y.h5.gz', './data/processed/train_target.pt')