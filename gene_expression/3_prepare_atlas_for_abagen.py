"""
Created on Nov 1 2023

@author: botond

This script prepares an info file for any parcellation, needed as an input for abagen.

"""

import os
import sys
import pandas as pd
import numpy as np
import nibabel as nib
from scipy import stats
import itertools
from nilearn import image, masking, plotting, maskers
from tqdm import tqdm

# %%
# Setup
# -----------------------------------------------------------------------------

# Project name
project_name = "keck_bolus"

# Filepaths
HOMEDIR = os.path.expanduser("~/Documents/School/Stony_Brook_University/LCNeuro/Work/")
SRCDIR = HOMEDIR + f"data/{project_name}/"
OUTDIR = HOMEDIR + f"data/{project_name}/mechanistic_map_comparisons/"

# Path and filename for output
outpath = "Seitzman/seitzman_300_info.csv"

# %%
# Load atlases
# -----------------------------------------------------------------------------

# Load parcellation
img_parc = image.index_img(
    nib.load(HOMEDIR + "tools/misc/anatomical_templates/Seitzman/ROIs_300inVol_MNI.nii"), 0)

# Open labels
info_parc = pd.read_csv(
    HOMEDIR + "tools/misc/anatomical_templates/Seitzman/subnetworks_seitzman.csv")


# -----
# Number of ROIs
N = int(np.unique(img_parc.get_fdata()).shape[0] - 1)

# Open AAL labels
aal_labels = pd.read_csv(
    HOMEDIR + "tools/misc/anatomical_templates/" \
    "AAL (mni)/roi_MNI_V4.txt",
    sep="\t",
    header=None,
    names=["label", "region", "id"])

# Open AAL atlas
aal_atlas = nib.load(
    HOMEDIR + "tools/misc/anatomical_templates/" \
    "AAL (mni)/AAL.nii")

# %%
# Map seitzman ROIs to AAL space for the removal of ROIs in the cerebellum
# -----------------------------------------------------------------------------

# Resample AAL atlas to seitzman
aal_atlas_resampled = image.resample_to_img(
    source_img=aal_atlas,
    target_img=img_parc,
    interpolation="nearest")

# Regions
roi_anat_inds = np.zeros(N)

# Substitute zeros with nans in AAL atlas
aal_atlas_resampled_data = aal_atlas_resampled.get_fdata()
aal_atlas_resampled_data_nan = np.where(
    aal_atlas_resampled_data!=0, aal_atlas_resampled_data, np.nan)

# Iterate over seitzman ROIs
for i, roi in tqdm(enumerate(np.unique(img_parc.get_fdata())[1:]), total=N):

    # Find most common value in AAL atlas
    most_common = stats.mode(
        aal_atlas_resampled_data_nan[img_parc.get_fdata() == roi],
        nan_policy="omit")[0][0]
    
    # Save
    roi_anat_inds[i] = most_common

# Coarsen anatomical regions into larger subgroups based on first digit
roi_anat_inds_coarse = roi_anat_inds//1000

# Mask for hemisphere
mask_left = (roi_anat_inds%2).astype(bool)

# Build 1D cortical mask
mask_cortical = \
    (roi_anat_inds_coarse != 9) & \
    (roi_anat_inds_coarse != 7) & \
    (roi_anat_inds != 4101) & \
    (roi_anat_inds != 4102) & \
    (roi_anat_inds != 4201) & \
    (roi_anat_inds != 4202)

# Build 1D subcortical mask
mask_subcortical = \
    (roi_anat_inds_coarse == 7) | \
    (roi_anat_inds == 4101) | \
    (roi_anat_inds == 4102) | \
    (roi_anat_inds == 4201) | \
    (roi_anat_inds == 4202)


# Build 1D cerebellum mask
mask_cerebellum = (roi_anat_inds_coarse == 9)

# %%
# Build info file
# -----------------------------------------------------------------------------

# Initiate dataframe
df = pd.DataFrame([], columns=["id", "label", "hemisphere", "structure",])

# Fill columns
df["id"] = np.arange(1, N+1)  # ID column
df["label"] = df["id"].astype(str) + "_" + info_parc["network"]  # Label column
df["hemisphere"] = list(map(lambda x: "L" if x else "R", mask_left))  # L/R column

# Structure column
df.loc[mask_cortical, "structure"] = "cortex"  # Cortex column
df.loc[mask_subcortical, "structure"] = "subcortex/brainstem"  # Cortex column
df.loc[mask_cerebellum, "structure"] = "cerebellum"  # Cortex column

# Save
df.to_csv(HOMEDIR + f"tools/misc/anatomical_templates/{outpath}", index=False)
