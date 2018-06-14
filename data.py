import os
import torch
import pandas as pd
import numpy as np
import h5py
import argparse
import warnings
import ipdb
from torch import nn
from torch.autograd import Variable
from torch.utils.data import Dataset

__all__ = ["CmapDataset"]

__author__ = "Joseph D. Romano"
__email__ = "jdr2160@cumc.columbia.edu"


class CmapDataset(Dataset):
    """GCTX files are just HDF5, so we read them as such"""
    def __init__(self, gctx_file, root_dir, verbose=True):
        self.h5_file = h5py.File(root_dir + gctx_file, 'r')
        self.data = self.h5_file['0/DATA/0/matrix']
        self.row_meta = self.h5_file['0/META/ROW']
        self.col_meta = self.h5_file['0/META/COL']
        self.view_idxs = None
        self.verbose = verbose
        self.lmark_idxs = None
        self.lmark_range = None
        self.filter_lmark = False

        self.test_contiguous_lmark()

    def set_view(self, criterion=None, keep_existing=True):
        """Build a 'view' into the database that filters by some
        criterion specified by the user.

        Args:
          criterion: tuple of length 2
            (SIG/GENE, META_KEY, BOOL_EXPR)
        """
        if criterion == None:
            warnings.warn(
                "No criterion provided for filtering the data--view unchanged",
                UserWarning
            )
            return

        if not keep_existing:
            self.reset_view()

        rc, meta_key, bool_expr = criterion
        metasub = ('self.row_meta' if (rc == 'GENE') else 'self.col_meta')

        eval_key = "{0}['{1}'].value{2}".format(metasub, meta_key, bool_expr)
        view_mask = eval(eval_key)
        self.view_idxs = np.where(view_mask)[0]

    def filter_pert_iname(self, pert_inames=[]):
        if pert_inames == []:
            warnings.warn(
                "No pert inames provided; skipping filter",
                UserWarning
            )
            return

        mask_matrix = np.zeros(
            shape=(len(pert_inames), self.data.shape[0]),
            dtype="bool"
        )

        for i, pt in enumerate(pert_inames):
            enc_pt = pt.encode()
            mask_matrix[i,:] = (self.col_meta['pert_iname'].value == enc_pt)

        mask_cmp = np.any(mask_matrix, axis=0)  # performs logical or along axis 0
        view_newidxs = np.where(mask_cmp)[0]    # converts bool array to array of indices

        if self.view_idxs is None:
            self.view_idxs = view_newidxs
        else:
            self.view_idxs = np.intersect1d(self.view_idxs, view_newidxs)

    def filter_pert_type(self, pert_types=[]):
        if pert_types == []:
            warnings.warn(
                "No pert types provided; skipping filter",
                UserWarning
            )
            return

        mask_matrix = np.zeros(
            shape=(len(pert_types), self.data.shape[0]),
            dtype="bool"
        )

        for i, pt in enumerate(pert_types):
            enc_pt = pt.encode()
            mask_matrix[i,:] = (self.col_meta['pert_type'].value == enc_pt)

        mask_cmp = np.any(mask_matrix, axis=0)  # performs logical or along axis 0
        view_newidxs = np.where(mask_cmp)[0]    # converts bool array to array of indices

        if self.view_idxs is None:
            self.view_idxs = view_newidxs
        else:
            self.view_idxs = np.intersect1d(self.view_idxs, view_newidxs)

    def filter_cell_type(self, cell_lines=[
        'A375',
        'A549',
        'HA1E',
        'HCC515',
        'HEPG2',
        'HT29',
        'MCF7',
        'PC3',
        'VCAP'
    ]):
        """Note: The default arg for cell_lines is the 9 'canonical' cell
        lines in the CMap dataset.
        """
        if cell_lines == []:
            warnings.warn(
                "No pert types provided; skipping filter",
                UserWarning
            )
            return

        mask_matrix = np.zeros(
            shape=(len(cell_lines), self.data.shape[0]),
            dtype="bool"
        )

        for i, pt in enumerate(cell_lines):
            enc_pt = pt.encode()
            mask_matrix[i,:] = (self.col_meta['cell_id'].value == enc_pt)

        mask_cmp = np.any(mask_matrix, axis=0)  # performs logical or along axis 0
        view_newidxs = np.where(mask_cmp)[0]    # converts bool array to array of indices

        if self.view_idxs is None:
            self.view_idxs = view_newidxs
        else:
            self.view_idxs = np.intersect1d(self.view_idxs, view_newidxs)

    def get_lmark_idxs(self):
        if self.lmark_idxs is None:
            # make the index
            lmk = [int(x) for x in self.row_meta['pr_is_lm'].value]
            self.lmark_idxs = np.where([bool(x) for x in lmk])[0]
        return self.lmark_idxs

    def reset_view(self):
        """Reset view into the database to the default (all data)."""
        if self.verbose:
            print("Resetting view (filters)...")
        self.view_idxs = None

    def extract_annotation(self, meta_field="pert_iname"):
        meta = self.col_meta[meta_field].value[self.view_idxs]
        return np.array([m.decode() for m in meta], dtype=str)

    def test_contiguous_lmark(self):
        lmk = self.get_lmark_idxs()
        if (lmk[-1] - lmk[0] + 1) == len(lmk):
            self.lmark_range = (lmk[0], lmk[-1])
            return True
        return False

    def toggle_filter_lmark(self, state=None):
        if state is None:
            self.filter_lmark = not self.filter_lmark
        else:
            self.filter_lmark = state
        if self.filter_lmark is False:
            self.lmark_range = None
        else:
            self.test_contiguous_lmark()

    def cond_lmark(self, idx):
        if self.lmark_range is None:
            return self.data[idx]
        else:
            i1, i2 = self.lmark_range
            return self.data[idx,i1:i2]

    def __len__(self, unmask=False):
        if unmask == False and self.view_idxs is not None:
            return self.view_idxs.shape[0]
        return self.data.shape[0]

    def __getitem__(self, idx):
        """Get a signature by index from the currently stored view"""
        if self.view_idxs is None:
            #return self.data[idx]
            return self.cond_lmark(idx)
        else:
            view_sub = self.view_idxs[idx]
            #return self.data[view_sub]
            return self.cond_lmark(view_sub)



# GCTX cheatsheet:
"""
Column (SIG) metadata keys:
cell_id             # Broad ID for a cell line
distil_id           # List (pipe-delimited) of Broad IDs for individual replicates used to compute level-5 profile
id                  # Broad ID for a signature
pert_dose           # Amount of compound used to treat cells
pert_dose_unit      # Unit (string; usually micromolar) of dose
pert_id             # Broad ID for a perturbagen
pert_idose          # Concatenation of pert_dose and pert_dose_unit (string)
pert_iname          # CMap-designated name of a perturbagen (string). When genetic, HUGO symbol is used
pert_itime          # Concatenation of pert_time and pert_time_unit (string)
pert_time           # Length of time perturbagen was exposed to the cells
pert_time_unit      # Unit of pert_time (string)
pert_type           # Abbreviated designation of perturbagen type (string) - see argparse code for choices

Row (GENE) metadata keys:
id                  # NCBI gene ID
pr_gene_symbol      # Gene symbol (alphanumeric string)
pr_gene_title       # Gene name (string with spaces)
pr_is_bing          # Best Inferred Gene (boolean; 0/1) (lm plus well-inferred genes)
pr_is_lm            # Landmark gene (boolean; 0/1) (978 directly measured genes)
"""
