import os
import torch
import pandas as pd
import numpy as np
import h5py
from torch import nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

__all__ = ["CmapDataset", "cmap_dset", "dataloader"]

__author__ = "Joseph D. Romano"
__email__ = "jdr2160@cumc.columbia.edu"


class CmapDataset(Dataset):
    """GCTX files are just HDF5, so we read them as such"""
    def __init__(self, gctx_file, root_dir, verbose=True):
        self.h5_file = h5py.File(root_dir + gctx_file, 'r')
        self.cmap_data = self.h5_file['0/DATA/0/matrix']

    def __len__(self):
        return self.cmap_data.shape[0]

    def __getitem__(self, idx):
        return self.cmap_data[idx]


cmap_dset = CmapDataset(
    gctx_file = "GSE70138_Broad_LINCS_Level5_COMPZ_n118050x12328_2017-03-06.gctx",
    root_dir = "../../Data/l1000/"
)

dataloader = DataLoader(
    cmap_dset,
    batch_size=16,
    shuffle=True,
)
