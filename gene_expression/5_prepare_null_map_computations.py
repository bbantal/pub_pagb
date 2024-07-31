"""
Created on Jun 4 2023

@author: botond

This script contains various code to produce inputs for
the computation of surrogate (null) maps using neuromaps
that takes place in a later step.

"""

# %%

import os
import sys
import pandas as pd
import numpy as np
import nibabel as nib
from nilearn import image
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm
from neuromaps import nulls
from scipy import spatial


# %%
# =============================================================================
# Setup
# =============================================================================

# Define filepaths
HOMEDIR = os.path.expanduser("~/")
SRCDIR = "/shared/projects/gene_expression_brain_maps/data/"
OUTDIR = "/shared/projects/gene_expression_brain_maps/results/alff/"


# %%
# =============================================================================
# Prepare inputs
# =============================================================================

# %%
# Null models: build mask
# -----------------------------------------------------------------------------

# Load data
gene_exp_data = pd.read_csv(f"/shared/projects/gene_expression_brain_maps/" \
    "data/parcellated_gene_expression/parcellated_gene_expression_seitzman.csv", index_col=0)

# Mask
# ----

# Load info
atlas_info = pd.read_csv(HOMEDIR + "utils/Seitzman/seitzman_300_info.csv")

# Structure to include
STRUCT = "cortex"

# Construct mask based on structure
mask_struct = atlas_info.eval(f"structure == '{STRUCT}'")

# Construct mask based on missing values
mask_nan = ~np.isnan(gene_exp_data.reset_index(drop=True)).any(axis=1)

# Merge the two masks -> 1D mask
mask_gene_exp = mask_nan & mask_struct

# %%
# Null models: mask atlas for null models
# ---------------------------------------------------------------------

# 1D mask to use
mask_1D = mask_gene_exp

# Load parcellation image
img_parc = image.index_img(nib.load(HOMEDIR + "utils/Seitzman/ROIs_300inVol_MNI.nii"), 0)

# Extract data from parcellation image
img_parc_data = img_parc.get_fdata()

# Save original number of nonzero voxels
N_nonzero_voxels_og = img_parc_data.astype(bool).sum()

# Get ROIs to keep based on previously made mask
rois_to_keep = np.argwhere(mask_1D.values) + 1

# 3D (voxel space) mask of regions to keep
mask_keep = np.isin(img_parc_data, rois_to_keep)

# Zero out excluded regions
img_parc_data[~mask_keep] = 0

# Create new nifti image
img_parc_modified = nib.Nifti1Image(img_parc_data, img_parc.affine)

# Save modified nifti image
nib.save(img_parc_modified, HOMEDIR + "utils/Seitzman/seitzman_atlas_cortical_only_v1.nii")

# Print information
print(
    f"ROIs kept: {rois_to_keep.shape[0]}/{mask_1D.shape[0]}\n",
    f"Voxels kept: {100*img_parc_data.astype(bool).sum()/N_nonzero_voxels_og:.1f}%")
