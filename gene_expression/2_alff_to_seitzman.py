#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 00:52:41 2021

@author: botond

Notes: This script parcellates the ALFF maps into 300 ROIs using the Seitzman atlas.


"""

import os
import numpy as np
import pandas as pd
import nibabel as nib
import itertools
from nilearn import image, input_data

# Directories
HOMEDIR = os.path.abspath(os.path.join(__file__, "../../../..")) + "/"
SRCDIR = "/shared/datasets/public/hcp-a/derivatives/alff/normalized/"
OUTDIR = "/shared/datasets/public/hcp-a/derivatives/alff/parcelled/"

# Region image
# ---------

# Import region image
roi_img = nib.load("/shared/home/botond/utils/Seitzman/ROIs_300inVol_MNI.nii")

# Gray matter mask
# -------

# Threshold for gm
GM_THR = 0.5

# Load MNI gm mask
gm_mask_raw = nib.load(HOMEDIR + "utils/mni_icbm152_gm_tal_nlin_asym_09c.nii")

# Binarize
gm_mask = image.math_img(f'img > {GM_THR}', img=gm_mask_raw)

# Masking
# -----

# Parceller object
parceller = input_data.NiftiLabelsMasker(
    roi_img, mask_img=gm_mask, smoothing_fwhm=None, standardize=False
    )

# Files read in from directory instead
files = [file for file in os.listdir(SRCDIR) if ".nii.gz" in file]

# Init collection
df_rows = []

# It over files
for i, file in enumerate(files):

    try:
        # Load alff img
        alff_img = nib.load(SRCDIR + file)

        # Status
        print(f"Loaded {file}. ({i+1}/{len(files)})")

        # Apply parceller
        roi_ts = parceller.fit_transform(alff_img)

        # Create row from dataframe
        # pd.DataFrame(roi_ts).to_csv(OUTDIR + file[:-4] + ".csv", header=False, index=False)
        row = pd.concat([
            pd.DataFrame([{item.split("-")[0]:item.split("-")[1] \
                  for item in file.replace(".nii.gz", "").split("_")}]),
                pd.DataFrame(roi_ts)], axis=1)

        # Add to collection
        df_rows.append(row)

        # Check for zero/close-to-zero ROIs
        roi_means =  np.mean(np.abs(roi_ts), axis=0)
        zero_rois = np.argwhere(roi_means < 1e-10)
        if len(zero_rois) > 0:
            print(f"[!] Zero ROI(s) found in {file}: {zero_rois}!")


    except Exception as E:
        print(E)
        continue

# Create df
df = pd.concat(df_rows, axis=0)

# Save df
df.to_csv(OUTDIR + "alff_parcelled_300.csv", header=True, index=False)
