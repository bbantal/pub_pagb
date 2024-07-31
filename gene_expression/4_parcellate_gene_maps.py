"""
Created on Oct 22 2023

@author: botond

This script parcellates brain maps of gene expression into a provided parcellation scheme.

"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import abagen
from scipy import stats

# %%
# =============================================================================
# Setup
# =============================================================================

# Define filepaths
HOMEDIR = os.path.expanduser("~/")
SRCDIR = "/shared/projects/gene_expression_brain_maps/data/"
OUTDIR = "/shared/projects/gene_expression_brain_maps/data/parcellated_gene_expression/"

# %%
# =============================================================================
# Parcellate gene maps
# =============================================================================

# Pacellation label
parc_label = "ukb_gm"

# Load atlas
atlas = {
    "image": HOMEDIR + "utils/Seitzman/ROIs_300inVol_MNI.nii",
    "info": HOMEDIR + "utils/Seitzman/seitzman_300_info.csv"
    }

# Check to make sure the specified image is in the correct format
atlas_checked = abagen.images.check_atlas(atlas["image"], atlas["info"])

# Perform parcellation
exp_parcelled = abagen.get_expression_data(
    atlas["image"], atlas["info"], lr_mirror="bidirectional",
    ibf_threshold=0.2)

# Region labels
region_labels = pd.read_csv(atlas["info"])["label"]

# Set them to index
exp_parcelled = exp_parcelled.set_index(region_labels.values)

# Inspect
exp_parcelled
exp_parcelled.shape
"SLC2A4" in exp_parcelled.columns

# Save parcellation
exp_parcelled.to_csv(OUTDIR + f"parcellated_gene_expression_{parc_label}.csv")

