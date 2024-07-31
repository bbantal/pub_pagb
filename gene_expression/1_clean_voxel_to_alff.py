#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on May 17 2023

@author: botond

This script computes ALFF in voxel space for the HCP-A dataset

"""

import os
import sys
import traceback
import pandas as pd
import numpy as np
from scipy import stats
from nilearn import image, masking
import nibabel as nib
from multiprocessing import Pool
from contextlib import closing
import time
import multiprocessing_logging
import logging

# =============================================================================
# Setup
# =============================================================================

# Filepaths
HOMEDIR = os.path.expanduser("~/")
SRCDIR = "/shared/datasets/public/hcp-a/derivatives/clean_bpf/"
OUTDIR = "/shared/datasets/public/hcp-a/derivatives/alff/"

dirs = [HOMEDIR, SRCDIR, OUTDIR]

# Preprocessing pars
FWHM = 5  # Spatial smoothing kernel fwhm in mm

# Computation pars
N_THREADS = 1  # Number of threads in multiprocessing

print(f"Number of parallel threads to be used: {N_THREADS}\n")

# Set up logger
logger = logging.getLogger("main")
logger.setLevel(logging.INFO)
formatter = logging.Formatter(
    fmt='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')

f_hl = logging.FileHandler("../logs/log_alff.log")
f_hl.setFormatter(formatter)
logger.addHandler(f_hl)
s_hl = logging.StreamHandler()
s_hl.setFormatter(formatter)
logger.addHandler(s_hl)

# Catch errors
def log_except_hook(*exc_info):
    text = "".join(traceback.format_exception(*exc_info))
    logging.error("Unhandled exception: %s", text)

sys.excepthook = log_except_hook

# Establish multithread logging
multiprocessing_logging.install_mp_handler()

# Status
logger.info("Started analysis.")

#raise

# Select files to analyze
files = sorted([file for file in os.listdir(SRCDIR) if ".nii" in file])

# Status
logger.info(f"Number of files to analyze: {len(files)}")

# Open up one of the images and extract the zeroeth slice to become the template image
# for shape and affine
template_img = image.index_img(image.load_img(SRCDIR + files[0]), 0)

# Status
logger.info(f"Template image shape: {template_img.shape}")

# Brain mask
# -----

# Load MNI brain mask
brain_mask_raw = image.load_img(HOMEDIR + "utils/mni_icbm152_t1_tal_nlin_asym_09c_mask.nii")

# Resample
brain_mask = image.resample_img(
        brain_mask_raw, target_affine=template_img.affine,
        target_shape=template_img.shape, interpolation="nearest",
        fill_value=0)

# Convert to numpy
brain_mask_np = brain_mask.get_fdata()

# Gray matter mask
# -------

# Threshold for gm
GM_THR = 0.5

# Load MNI gm mask
gm_mask_raw = image.load_img(HOMEDIR + "utils/mni_icbm152_gm_tal_nlin_asym_09c.nii")

# Binarize
gm_mask_preresampling = image.math_img(f'img > {GM_THR}', img=gm_mask_raw)

# Resample
gm_mask = image.resample_img(
        gm_mask_preresampling, target_affine=template_img.affine,
        target_shape=template_img.shape, interpolation="nearest",
        fill_value=0)

# =============================================================================
# Functions for computing ALFF
# =============================================================================
def comp_alff(bold_fp):
    """
    Function for computing ALFF.
    Input time-series must already be cleaned (confs removed)!
    """

    # Import bold image
    bold_img = image.load_img(bold_fp)

    # Smooth bold image
    bold_img = image.smooth_img(bold_img, FWHM)

    # Compute ALFF
    alff = np.std(bold_img.get_fdata(), axis=3)

    # Zero out voxels outside the brain mask
    alff = np.where(brain_mask_np>0, alff, np.nan)

    # Compute mean signal while bold is still loaded
    mean_signal = masking.apply_mask([bold_img], mask_img=gm_mask).mean()

    # Clean up
    del bold_img

    # Convert back to nifti
    alff_nifti = image.new_img_like(template_img, alff, affine=template_img.affine)

    # Compute mean normalized alff (unmasked!)
    # ---------

    # Compute mALFF
    alff_norm = alff/np.nanmean(alff)

    # Convert back to nifti
    alff_norm_nifti = image.new_img_like(template_img, alff_norm, affine=template_img.affine)

    # Compute average alff within gray matter mask
    # ---------

    # Extract signal from gray matter voxels
    alff_nifti_masked = masking.apply_mask(alff_nifti, mask_img=gm_mask)

    # Compute mean signal
    alff_mean = alff_nifti_masked.mean()

    # Return
    return [alff_nifti, alff_norm_nifti, alff_mean, mean_signal]


def alff_wrapper(item):
    """ Wrapper function for computing alff. dirs variable is global """

    # Unpack directories
    [homedir, workdir, outdir] = dirs

    ## Status
    text = f"Current item: {item}"
    logger.info(text)

    # Start timer
    time_start = time.time()

    # Define bold function filepath
    bold_fp = SRCDIR + item

    try:
        # Compute alff
        [alff_nifti, alff_norm_nifti, alff_mean, mean_signal] = comp_alff(bold_fp)

        # Finish timer
        time_end = time.time()

        # Elapsed time
        time_elapsed = f"{time_end - time_start:0.2f}"

        # Save as nifti
        nib.save(alff_nifti, OUTDIR + f"raw/{item}")
        nib.save(alff_norm_nifti, OUTDIR + f"normalized/{item}")

        # Status
        text = f"Finished item: {item}, duration: {time_elapsed}s"
        logger.info(text)

        # Return mean alff
        return [alff_mean, mean_signal]

    except Exception as exc:
        # print(f"Exception in {item}: ", exc)
        raise Exception(f"Failed item: {item}") from exc
    
    finally:
        return [np.nan, np.nan]

# =============================================================================
# Perform computations
# =============================================================================

# Run computation through multiprocessing
with closing(Pool(processes=N_THREADS)) as pool:
   mean_coll = pool.map(alff_wrapper, files)
   pool.terminate()

# =============================================================================
# Save average alff
# =============================================================================

# Meta coll
meta_coll = []

# It over
for i, file in enumerate(files):

    # Store meta
    meta_coll.append(
            {item.split("-")[0]:item.split("-")[1] \
              for item in file.replace(".nii", "").split("_")}
            )

# Create df
df = pd.concat(
    (pd.DataFrame(meta_coll), pd.DataFrame(mean_coll, columns=["mean_alff", "mean_signal"])),
    axis=1
        )

# Save df
df.to_csv(OUTDIR + "mean/alff_mean.csv")

# Status
text = "\nFinished analysis."
logger.info(text)








