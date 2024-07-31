# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 20:40:35 2019

@author: botond

This script cleans up preprocessed voxel space time-series data. It:
    -detrends
    -standardizes
    -bandpass filters
    -regresses out confounds
    -saves the cleaned image (voxel-space)

"""

import os
import sys
import pandas as pd
import numpy as np
import json
import nibabel as nib
from nilearn import image
from multiprocessing import Pool
import logging
import multiprocessing_logging
import matplotlib.pyplot as plt
import itertools
import datetime

# %%
# =============================================================================
# Setup
# =============================================================================


# Filepaths
# -----
HOMEDIR = os.path.abspath(os.path.join(__file__, "../../../..")) + "/"
SRCDIR = "/shared/datasets/private/keck_bolus/derivatives/fmriprep/"
OUTDIR = f"/shared/datasets/private/keck_bolus/derivatives/clean_voxel_bpf/004hz/"

# Inputs
# ------

# Computation
N_THREADS = 2  # Number of threads in multiprocessing

# Handling
CUTOFF = 20  # Trimming timeseries up to this frame

# QC - motion
FD_threshold = 0.5
FD_criteria = 0.05

# Status
print(f"Number of cores to be used: {N_THREADS}")

# Items to be preprocessed
# ------
subjects = [int(x) for x in input("Subject ID (XXX format): ").split(" ")]
sessions = [x for x in input("Session (bhb and/or glc): ").split(" ")]
tasks = ["rest"]
runs = ["1", "2"]

items = [item for item in list(itertools.product(subjects, sessions, tasks, runs))]

# Set up dependencies
# ------

# Set up logger
logger = logging.getLogger("main")
logger.setLevel(logging.INFO)
case = "single" if len(subjects) == 1 else "multi"
curr_datetime = datetime.datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
hl = logging.FileHandler("../logs/" + f"log_comp_ts_{subjects[0]}_{case}_" \
           f"{curr_datetime}.log")
hl.setFormatter(logging.Formatter(
    fmt='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
    ))
logger.addHandler(hl)

# Establish multithread logging
multiprocessing_logging.install_mp_handler()

# Status
text = f"Case: {subjects[0]}_{case}\nTime: {curr_datetime}"
logger.info(text)

# %%
# =============================================================================
# Analysis
# =============================================================================

def comp_timeseries(item):
    """
    Function to process voxel space time-series
    """

    # Status
    logger.info(f"Current item: {item}")

    # Get filepaths
    bold_fp = SRCDIR + (
        "sub-{0:0>3}/ses-{1}/func/" \
        "sub-{0:0>3}_ses-{1}_task-{2}_run-{3}_space-MNI152NL"
        "in2009cAsym_desc-preproc_bold.nii.gz"
        ).format(*item)

    conf_fp = SRCDIR + (
        "sub-{0:0>3}/ses-{1}/func/" \
        "sub-{0:0>3}_ses-{1}_task-{2}_run-{3}_desc-confounds_"
        "timeseries.tsv").format(*item)

    conf_info_fp = SRCDIR + (
        "sub-{0:0>3}/ses-{1}/func/" \
        "sub-{0:0>3}_ses-{1}_task-{2}_run-{3}_desc-confounds_"
        "timeseries.json").format(*item)

    # Load the image and get rid of first n frames for which confounds were not
    # computed for
    func_img = image.index_img(image.load_img(bold_fp), slice(CUTOFF, None))

    # Status
    logging.info("Loaded functional image")

    # Select confound regressors to be used
    # ---------

    # Open info of confound regressors
    conf_file = open(conf_info_fp, "r")
    conf_info = json.load(conf_file)
    conf_file.close()

    # Collect all confound regressors to be used
    conf_labels = [
                   "white_matter",
                   "csf",
                   "trans_x",
                   "trans_y",
                   "trans_z",
                   "rot_x",
                   "rot_y",
                   "rot_z",
                   ]

    # -----------
    # Load confounds
    confounds = pd.read_csv(conf_fp, sep='\t') \
                    .loc[CUTOFF:, conf_labels]

    # Make sure confounds do not contain zeros
    try:
        assert 0.0 not in confounds.iloc[0, :6].values, "Zero element " \
        f"found in confound component: {item}!"
    except Exception as E:
        print(E)

    # Clean img
    img_clean = image.clean_img(
            func_img, detrend=True, confounds=confounds.values,
            standardize=False, t_r=0.802, low_pass=0.1, high_pass=0.04,
            sample_mask=None
                )

    # Save cleaned img
    nib.save(img_clean, OUTDIR + "sub-{0:0>3}_ses-{1}_task-{2}_run-" \
            "{3}.nii".format(*item))

    # Quality control
    # ----------------------------------


    # Check for framewise displacement
    fdisp  = pd.read_csv(conf_fp, sep='\t') \
                        .loc[CUTOFF:, ["framewise_displacement"]] \
                        .to_numpy()
    outlier_ratio = fdisp[fdisp>FD_threshold].shape[0]/fdisp.shape[0]
    if outlier_ratio > FD_criteria:
        # print(f"[!] Excessive amount of motion in {item}: " \
        #       f"{100*outlier_ratio:0.2f}%!")
        logger.info(f"[!] Excessive amount of motion in {item}: " \
                    f"{100*outlier_ratio:0.2f}%!")
    #    bins = [0.01, 0.02, 0.03, 0.04, 0.05]

    text = "sub-{0:0>3}_ses-{1}_task-{2}_run-{3}: Framewise Displacement" \
            .format(item[0], item[1].lower(), item[2], item[3])

    plt.figure(figsize=(11, 8))
    plt.hist(fdisp)  #, bins=bins)
    plt.annotate(text, xy=(0.1, 0.5), xycoords='figure fraction', fontsize=16)
    plt.savefig(OUTDIR + "../quality_control/motion/framewise_disp_sub"
                "-{0:0>3}_ses-{1}_task-{2}_run-{3}.png" \
                .format(item[0], item[1].lower(), item[2], item[3]))
    plt.close("all")


###############################################################################

if __name__ == "__main__":

    # Testimg
    # map(comp_timeseries, items)

    # Run computation through multiprocessing
    pool = Pool(processes=N_THREADS)
    pool.map(comp_timeseries, items)
    pool.close()
    pool.join()

    # Status
    logging.info("Finished execution.")

    # Shut down logging
    logging.shutdown()

    # Collect warnings into a separate file
    with open("../logs/" + f"log_comp_ts_{subjects[0]}_{case}_" \
               f"{curr_datetime}.log", "a+") as log:
        log.seek(0)
        lines_all = log.readlines()
        motion_warnings = sorted([line[0:] for line in lines_all \
                                  if "Excessive amount" in line])
        zroi_warnings = sorted([line[0:] for line in lines_all \
                                if "Zero ROI(s) found in" in line])

    with open("../logs/" + \
              f"warnings_comp_ts_{subjects[0]}_{case}_{curr_datetime}.txt", "a+") as warnings:
        warnings.write(
            f'###############\n' \
            f'Case: {subjects[0]}_{case}\n' \
            f'Time: {curr_datetime}\n' \
            f'###############\n'
            )
        warnings.write("\nMotion:\n")
        for mot_warn in motion_warnings:
            warnings.write(f"{mot_warn}")

        warnings.write("\nZero Roi:\n")
        for zroi_warn in zroi_warnings:
            warnings.write(f"{zroi_warn}")
