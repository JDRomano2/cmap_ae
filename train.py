import argparse
import os
import torch
import pandas as pd
import numpy as np
import h5py
from torch import nn
from torch.autograd import Variable

from models import *
from 

__author__ = "Joseph D. Romano"
__email__ = "jdr2160@cumc.columbia.edu"


class ArgParser(object):
    def __init__(self):
        parser = argparse.ArgumentParser(description='Autoencoder Analysis of l1000 Arrays')
        parser.add_argument(
            '--batch-size',
            type=int,
            default=8,
            metavar='N',
            help='Minibatch size for training (default: 8)'
        )
        parser.add_argument(
            '--epochs',
            type=int,
            default=50,
            metavar='N',
            help='Number of epochs to run (i.e., passes over the training data; default: 50)'
        )
        parser.add_argument(
            '--no-cuda',
            action='store_true',
            default=False,
            help='Flag to force training on CPU instead of GPU'
        )
        parser.add_argument(
            '--model',
            type=str,
            choices=[
                'ae',
                'dae',
                'vae'
            ],
            default='ae',
            help='Choose autoencoder model to train (default: \'vanilla\' autoencoder)'
        )

        self.parser = parser

        self.args = self.parser.parse_args()
        self.args.cuda = not self.args.no_cuda and torch.cuda.is_available()
        if not self.args.no_cuda and not torch.cuda.is_available():
            print('Warning: No cuda device available; continuing using CPU')



if __name__ == "__main__":

    ap = ArgParser()

    models = {
        'ae': Autoencoder,
        'dae': DenoisingAE,
        'vae': VariationalAE
    }

    model = models[ap.args.model]
    print(model)

