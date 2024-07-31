"""
Created on Nov 1 2023

@author: botond

This script computes spatial null maps for parcellated gene expression data.
"""

import sys
import os
import numpy as np
import pandas as pd
import nibabel as nib
from neuromaps import nulls
import multiprocessing

sys.path.append(os.path.expanduser("~/utils/"))

import helpers

# %%
# Setup
# ------------------------------------------------------------------------------

# Define filepaths
HOMEDIR = os.path.expanduser("~/")
SRCDIR = "/shared/projects/gene_expression_brain_maps/data/"
OUTDIR = "/shared/projects/gene_expression_brain_maps/data/null_maps/contrasts/seitzman_alff/"

# Save script
helpers.save_script(HOMEDIR)

# Settings
STRUCT = "cortex"  # Structure to include
IMG_PARC_FP = HOMEDIR + "utils/Seitzman/seitzman_atlas_cortical_only_v1.nii"
# Make sure it only containts cortical+retained ROIs! May need prior preprocessing
N_PERM = 1000  # Number of permutations
N_THREADS = 25

# %%
# Construct mask of cortex/retained ROIs to be applied to gene expression data
# ------------------------------------------------------------------------------

# Load gene expression data
gene_exp_data = pd.read_csv(SRCDIR + "parcellated_gene_expression/" \
                "parcellated_gene_expression_seitzman.csv", index_col=0)

# Load info
atlas_info = pd.read_csv(HOMEDIR + "utils/Seitzman/seitzman_300_info.csv")

# Construct mask based on structure
mask_struct = atlas_info.eval(f"structure == '{STRUCT}'")

# Construct mask based on missing values
mask_nan = ~np.isnan(gene_exp_data.reset_index(drop=True)).any(axis=1)

# Merge the two masks
mask_gene_exp = mask_nan & mask_struct

# %%
# Compute null models
# ------------------------------------------------------------------------------

# List of genes to compute nullmaps for
genes = list(gene_exp_data.columns)

# Function computing null maps
def compute_null_map(gene):

    try:

        # Status
        print(f"Computing null map for {gene}...")

        # Extract data for current gene
        data = gene_exp_data[gene].reset_index(drop=True).to_numpy()[mask_gene_exp]

        # Compute null maps
        null_map = nulls.burt2020(data, atlas="mni152", density="1mm",
                parcellation=IMG_PARC_FP, n_perm=N_PERM)

        # Save results
        np.save(OUTDIR + f"null_maps_{gene}.npy", null_map)

    except Exception as e:
        print(e)

# Multiprocessing
if __name__ == '__main__':

    # Create a pool of worker processes
    pool = multiprocessing.Pool(processes=N_THREADS)

    # Map the function to the arguments
    pool.map(compute_null_map, genes)

    # Close the pool of worker processes
    pool.close()
    pool.join()
