#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 00:52:41 2021

@author: botond

Notes:
This script performs parcellation
Seitzman (300 ROIs) atlas
Smoothing is done (5mm)

"""

import os
import numpy as np
import pandas as pd
import nibabel as nib
import itertools
from nilearn import image, input_data


# Directories
HOMEDIR = os.path.abspath(os.path.join(__file__, "../../../..")) + "/"
SRCDIR = "/shared/datasets/private/keck_bolus/derivatives/clean_voxel_bpf/004hz/"
OUTDIR = "/shared/datasets/private/keck_bolus/derivatives/parcelled/300/004hz/"

# Spatial smoothing
FWHM = 5

# Region image
# ---------

# Import parcellation atlas
roi_img = nib.load("/shared/home/botond/utils/Seitzman/ROIs_300inVol_MNI.nii")

# Gray matter mask
# -------

# Threshold for gm
GM_THR = 0

# Load MNI gm mask
gm_mask_raw = nib.load(HOMEDIR + "utils/mni_icbm152_gm_tal_nlin_asym_09c.nii")

# Binarize
gm_mask = image.math_img(f'img > {GM_THR}', img=gm_mask_raw)

# Masking
# -----

# Parceller object
parceller = input_data.NiftiLabelsMasker(
    roi_img, mask_img=gm_mask, smoothing_fwhm=FWHM, standardize=True
    )

# Get user inputs regarding files to analyze
subjects = [int(x) for x in input("Subject ID (XXX format): ").split(" ")]
sessions = [x for x in input("Session (bhb and/or glc): ").split(" ")]

# Given info
tasks = ["rest"]
runs = ["1", "2"]

# Items to analyze
items = [item for item in list(itertools.product(subjects, sessions, tasks, runs))]

# Convert items to filenames
files = ["sub-{0:0>3}_ses-{1}_task-{2}_run-{3}.nii".format(*item) for item in items]

# It over files
for i, file in enumerate(files):

    # Load image once and for good
    func_img = nib.load(SRCDIR + file)

    # Status
    print(f"Loaded {file}. ({i+1}/{len(files)})")

    # Apply parceller
    roi_ts = parceller.fit_transform(func_img)

    # Save
    pd.DataFrame(roi_ts).to_csv(OUTDIR + file[:-4] + ".csv", header=False, index=False)

    # Check for zero/close-to-zero ROIs
    roi_means =  np.mean(np.abs(roi_ts), axis=0)
    zero_rois = np.argwhere(roi_means < 1e-10)
    if len(zero_rois) > 0:
        print(f"[!] Zero ROI(s) found in {file}: {zero_rois}!")
