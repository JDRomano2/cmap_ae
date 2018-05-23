import os
import torch
import pandas as pd
import numpy as np
import h5py
from torch import nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

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

INPUT = cmap_dset.cmap_data.shape[1]
HIDDEN_1 = 5000
HIDDEN_2 = 1000
HIDDEN_3 = 500
LEARNING_RATE_START = 1e-4
NUM_EPOCHS = 100


def checkpoint(state, fname_stub="ae", fname_suffix=""):
    fname = "./checkpoints/{}{}.pt".format(fname_stub,fname_suffix)
    torch.save(state, fname)


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(INPUT, HIDDEN_1),
            nn.Tanh(),
            #nn.Linear(HIDDEN_1, HIDDEN_2),
            #nn.Tanh(),
            #nn.Linear(HIDDEN_2, HIDDEN_3)
        )
        self.decoder = nn.Sequential(
            #nn.Linear(HIDDEN_3, HIDDEN_2),
            #nn.Tanh(),
            #nn.Linear(HIDDEN_2, HIDDEN_1),
            #nn.Tanh(),
            nn.Linear(HIDDEN_1, INPUT),
            nn.Tanh()
        )
        self.lr = LEARNING_RATE_START
        self.init_weights()

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def init_weights(self):
        init_std = 0.02
        for layer in self.encoder:
            if hasattr(layer, 'weight'):
                layer.weight.data.normal_(0, init_std)
                layer.bias.data.fill_(0)

    def reduce_lr(self, mult_factor=0.1):
        self.lr *= mult_factor

    def compute_loss(self, output):
        # todo
        return


print("Instantiating model and optimizer.")
model = Autoencoder().cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr = model.lr(),
    weight_decay = 1e-5
)


print("Beginning to train model.")
for epoch in range(NUM_EPOCHS):
    # note: checkpoint at the BEGINNING of each epoch
    checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict()
    })
    for i, data in enumerate(dataloader):
        sig = data
        sig = Variable(sig).cuda()
        output = model(sig)
        loss = criterion(output, sig)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 10 == 0:
            print('[{}/{}] batch: {}, loss: {}'.format(
                epoch + 1,
                NUM_EPOCHS,
                i,
                loss.data
            ))
    print(
        'epoch [{}/{}], loss:{:.4f}'.format(
            epoch + 1,
            NUM_EPOCHS,
            loss.data
        ))
