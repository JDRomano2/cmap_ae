import argparse
import os
import torch
import numpy as np
import h5py
import math
import time
import ipdb
from tqdm import tqdm
from torch import nn
from torch.autograd import Variable, Function
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR

# from torchviz import make_dot
from tensorboard_logger import configure, log_value

from data import CmapDataset

__all__ = ["Autoencoder", "DenoisingAE", "VariationalAE"]

__author__ = "Joseph D. Romano"
__email__ = "jdr2160@cumc.columbia.edu"


VAE = False
AE = False


parser = argparse.ArgumentParser(description='Generate a new dataset from CMap data.')
parser.add_argument(
    '-m',
    '--model',
    help='Autoencoder variation to use',
    choices=[
        'ae',
        'vae',
        'da',
        'sda',
        'none'
    ],
    required=True
)
parser.add_argument(
    '-p',
    '--pert_names',
    nargs='+',
    help='Perturbagen names to subset from the dataset',
    required=False
)
parser.add_argument(
    '-t',
    '--pert_types',
    nargs='?',
    help='Perturbagen types to select',
    choices=[
        'trt_cp',           # Compound
        'trt_lig',          # Peptides and biological agents (e.g., cytokines)
        'trt_oe',           # cDNA for overexpression of wt gene
        'trt_oe.mut',       # cDNA for overexpression of mutated gene
        'trt_sh',           # shRNA for LoF of gene
        'trt_sh.cgs',       # Consensus sig for shRNAs targeting the same gene
        'trt_sh.css',       # Controls - consensus sig from shRNAs that share common seed sequence
        'ctl_vehicle',      # Controls - vehicle for compound treatment (e.g., DMSO)
        'ctl_vehicle.cns',  # Controls - consensus signature of vehicles
        'ctl_vector',       # Controls - vector for genetic perturbation (e.g., empty vector, GFP)
        'ctl_vector.cns',   # Controls - consensus signature of vectors
        'ctl_untrt',        # Controls - untreated cells
        'ctl_untrt.cns'     # Controls - consensus signature of many untreated wells
    ],
    required=False
)


class L1Penalty(Function):

    @staticmethod
    def forward(ctx, inpt, l1weight):
        ctx.save_for_backward(input)
        ctx.l1weight = l1weight
        return inpt

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_variables
        grad_input = input.clone().sign().mul(self.l1weight)
        grad_input += grad_output
        return grad_input


class Autoencoder(nn.Module):
    def __init__(self, config_dict):
        super(Autoencoder, self).__init__()

        self.arch = config_dict

        self.encoder = nn.Sequential(
            nn.Linear(self.arch['input_dim'], self.arch['hidden_1']),
            nn.Tanh(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.arch['hidden_1'], self.arch['input_dim']),
            nn.Tanh()
        )
        self.lr = self.arch['lr_start']
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


# class RegularizedAE(Autoencoder):
#     def __init__(self):
#         super(RegularizedAE, self).__init__()


class DenoisingAE(Autoencoder):
    def __init__(self):
        super(DenoisingAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(INPUT, HIDDEN_1),
            nn.Tanh(),
            #nn.Linear(HIDDEN_1, HIDDEN_2),
            #nn.Tanh(),
            #nn.Linear(HIDDEN_2, HIDDEN_3)
        )


class VariationalAE(nn.Module):
    """Variational Autoencoder---an explicit generative model that somewhat
    resembles 'true' (vanilla) autoencoders, but with entirely different
    methematics. VAEs are designed to cope with (1) the frequent intractability
    of computing the marginal likelihood p(x) and/or the true posterior density
    p(z|x), and (2) large datasets where batch gradient descent and Monte Carlo
    EM are practically impossible.

    ## Mathematical Preliminaries

    z                --- A vector of latent variables.
    x                --- A vector of observed variables.
    X                --- A dataset of i.i.d. samples from `x`.
    \theta*          --- The true parameters of the generative process.
    \theta           --- Estimated parameters for the generative model.
                         Also referred to as the *generative parameters*.
    \phi             --- Parameters for the recognition model (encoder) that
                         produces `z` given `x`. Also referred to as the
                         *variational parameters*.

    p_{\theta*}(z)   --- The (true) prior distribution (specifying `z`).
    p_{\theta*}(x|z) --- The (true) conditional distribution specifying the
                         likelihood of `x`.

    p_{\theta}(z|x)  --- The (intractable) true posterior over the latent
                         variables.
    q_{\phi}(z|x)    --- A recognition model that approximates `p_{\theta}(z|x)`.

    X is observed.
    \theta* and z are unknown.

    We want to learn `\theta` and `\phi`. `z` can then be sampled from
    `q_{\phi}(z|x)` directly for a given `x`.


    ### Relationship to Vanilla AEs

    - The recognition model `q_{\phi}(z|x)` is essentially an encoder.

    - The latent variables `z` are analogous to encoded data.

    - The conditional distribution `p_{\theta}(x|z)` is essentially a decoder.


    ## Computational Pragmatics

    Fortunately, we don't need to worry about all of the details of the AEVB
    algorithm in order to train a VAE with most modern deep learning libraries,
    thanks to automated differentiation.

    However, we do need to apply the *reparametrization trick*, which makes
    optimization of the latent loss possible by substituting the mean and
    standard deviation vectors of encoded minibatches for the actual encoded
    outputs. The generation loss can be evaluated simultaneously by sampling
    the multivariate Gaussian implied by the mean and standard devation
    vectors, and computing the mean square error.

    """

    def __init__(self, config_dict):
        super(VariationalAE, self).__init__()

        self.arch = config_dict

        self.n = self.arch['input_dim']

        self.len_z = self.arch['len_z']
        # self.len_z = self.n

        self.recognition = nn.Sequential(
            nn.Linear(self.arch['input_dim'], self.arch['recog_1']),
            nn.Tanh(),
            nn.Linear(self.arch['recog_1'], self.arch['recog_2']),
            nn.Tanh(),
        )
        # self.recognition = nn.Sequential(
        #     nn.Linear(self.n, self.n),
        #     nn.Tanh(),
        # )

        self._enc_mean = nn.Linear(self.arch['recog_2'], self.len_z)
        self._enc_log_sigma = nn.Linear(self.arch['recog_2'], self.len_z)
        # self._enc_mean = nn.Linear(self.n, self.n)
        # self._enc_log_sigma = nn.Linear(self.n, self.n)

        self.generator = nn.Sequential(
            nn.Linear(self.len_z, self.arch['recog_2']),
            nn.Tanh(),
            nn.Linear(self.arch['recog_2'], self.arch['recog_1']),
            nn.Tanh(),
            nn.Linear(self.arch['recog_1'], self.arch['input_dim']),
            nn.Tanh(),
        )
        # self.generator = nn.Sequential(
        #     nn.Linear(self.n, self.n),
        #     nn.Tanh(),
        # )

        self._gen_loss_func = nn.MSELoss()

        self.init_weights()

    def validate_architecture(self):
        """TODO"""
        return True

    def init_weights(self):
        """See http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
        for information about Xavier initialization.

        Biases are initialized to zero.
        """

        # recognition network
        for layer in self.recognition:
            if hasattr(layer, 'weight'):
                nn.init.xavier_normal_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)

        # reparametrization layers
        nn.init.xavier_normal_(self._enc_mean.weight)
        nn.init.constant_(self._enc_mean.bias, 0.0)
        nn.init.xavier_normal_(self._enc_log_sigma.weight)
        nn.init.constant_(self._enc_log_sigma.bias, 0.0)

        # generator network
        for layer in self.generator:
            if hasattr(layer, 'weight'):
                nn.init.xavier_normal_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)

    def encode(self, x):
        """The recognition model `q_{\phi}(z|x)`, an MLP that maps inputs
        to a multivariate Gaussian that serves as a prior to the generator.
        """
        recog = self.recognition(x)
        mu = self._enc_mean(recog)
        log_sigma = self._enc_log_sigma(recog)
        return mu, log_sigma

    def reparametrize(self, mu, log_sigma):
        """Output is `z`; the 'encoded' data"""
        std = torch.exp(0.5*log_sigma)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        """The desired generative model, p_{\theta}()"""
        return self.generator(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z)

    def gen_loss(self, x):
        out = self.forward(x)
        return self._gen_loss_func(out, x)

    def latent_loss(self, x):
        enc_mu, enc_std = self.encode(x)
        mu_sq = enc_mu * enc_mu
        sigma_sq = enc_std * enc_std
        kl_div = 0.5 * torch.mean(mu_sq + sigma_sq - torch.log(sigma_sq) - 1)
        return kl_div


if __name__=="__main__":
    # testing zone

    fname_token = time.time()

    args = parser.parse_args()

    args

    cmap = CmapDataset(
        gctx_file = "annotated_GSE92742_Broad_LINCS_Level5_COMPZ_n473647x12328.gctx",
        #root_dir = "../../Data/l1000/",
        #gctx_file = "testdata_n1000x978.gctx",
        root_dir = "../../Data/cmap/",
    )

    #cmap.filter_pert_type(['trt_cp'])
    cmap.filter_pert_iname(['imatinib'])
    cmap.filter_cell_type()

    perts = cmap.extract_annotation('pert_iname')
    cells = cmap.extract_annotation('cell_id')

    cmap.toggle_filter_lmark()

    if args.model == 'vae':
        configure("runs/vae/{0}".format(fname_token), flush_secs=5)

        net_config = dict(
            input_dim = cmap.cmap_data.shape[1],
            # input_dim = 1000,
            recog_1 = 8000,
            recog_2 = 4000,
            len_z = 1000,
            gen_1 = 4000,
            gen_2 = 8000,
            lr_start = 0.0001,
            batch_size = 32,
            n_epochs = 50,
        )

        dataloader = DataLoader(
            cmap,
            batch_size=net_config['batch_size'],
            shuffle=True,
        )

        vae = VariationalAE(net_config).cuda()

        optimizer = torch.optim.Adam(vae.parameters(), lr=net_config['lr_start'])

        # make_dot(vae, params=dict(vae.parameters()))

        # x_test = next(iter(dataloader))
        # x_test = Variable(x_test).cuda()
        # output_test = vae(x_test)

        for epoch in range(net_config['n_epochs']):
            for i, data in enumerate(dataloader):
                step = (epoch*len(dataloader)) + i
                inputs = data
                inputs = Variable(inputs).cuda()
                optimizer.zero_grad()
                dec = vae(inputs)
                ll = vae.latent_loss(inputs)
                gl = vae.gen_loss(inputs)
                loss = ll + gl
                loss.backward()
                optimizer.step()
                l = loss.data[0]
                log_value('latent_loss', ll, step)
                log_value('generative_loss', gl, step)
                log_value('loss', l, step)
                if math.isnan(l):
                    exit

                if i % 10 == 0:
                    print("  {0}/{1}     (loss: {2})".format(i, len(dataloader), l))
                    print("    latent loss:     {0}".format(ll))
                    print("    generative loss: {0}".format(gl))
            print("Epoch: ", epoch, ". Loss: ", l)

    if args.model == 'ae':
        configure("runs/ae/{0}".format(fname_token), flush_secs=5)
        state_path = "runs/ae/state-{0}".format(fname_token)
        print("input dim: ", len(cmap[0]))

        # net_config = dict(
        #     input_dim = cmap.data.shape[1],
        #     hidden_1 = 8000,
        #     lr_start = 0.0001,
        #     batch_size = 32,
        #     n_epochs = 50
        # )
        net_config = dict(
            input_dim = len(cmap[0]),
            hidden_1 = len(cmap[0]),
            lr_start = 0.0001,
            batch_size = 32,
            n_epochs = 50
        )

        ipdb.set_trace()

        dataloader = DataLoader(
            cmap,
            batch_size=net_config['batch_size'],
            shuffle=True,
        )

        ae = Autoencoder(net_config).cuda()
        optimizer = torch.optim.Adam(ae.parameters(), lr=net_config['lr_start'])
        scheduler = StepLR(optimizer, step_size=3, gamma=0.5)
        loss_func = nn.MSELoss()

        for epoch in range(net_config['n_epochs']):
            scheduler.step()
            for i, data in enumerate(dataloader):
                step = (epoch*len(dataloader)) + i
                inputs = data
                inputs = Variable(inputs).cuda()
                # FORWARD
                outputs = ae(inputs)
                loss = loss_func(outputs, inputs)
                # BACKWARD
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                log_value('loss', loss, step)

                if i % 10 == 0:
                    print("  batch:{}/{}, loss:{:.4f}".format(i, len(dataloader), loss.data.item()))
            print("epoch:{}/{}".format(epoch+1, net_config['n_epochs']))
            print("Current learning rate: {0}".format(scheduler.get_lr()))
            print("Saving model parameter checkpoint...")
            torch.save(ae.state_dict(), state_path)
            print("...done.")
