#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 17:37:06 2020

@author: botond

This script is called by a bash script.
1. It takes in raw anatomical images
2. Corrects for biasfield using SPM
3. Outputs biasfield corrected image
"""


import os
import sys
import itertools
import nibabel as nb
from nipype.interfaces import spm

# =============================================================================
# Setup
# =============================================================================

# Filepaths
SRCDIR = "/shared/datasets/private/keck_bolus/sourcedata/anat_for_bf_corr/"
OUTDIR = "/shared/datasets/private/keck_bolus/"

# Set path for matlab and SPM
spm.SPMCommand.set_mlab_paths(
        paths="/shared/home/botond/apps/SPM12/spm12/",
        matlab_cmd="matlab -nodesktop -nosplash")

# Settings
# --------------------

# Parameters
BIAS_FWHM = 60   # Bias field FWHM (use higher for smoother biasfields) [30-130]
BIAS_REG = 1e-5   # Bias field regularization factor (use higher for weaker
# biasfields) [0/0.1 - 10.0]

# Inputs
SUBJECTS = [int(subid) for subid in sys.argv[1].split(" ")]
SESSIONS = [ses for ses in sys.argv[2].split(" ")]
RUNS = [1, 2]  # Runs

# =============================================================================
# Function to perform biasfield correction for an input file
# =============================================================================

def correct_bias_field(SRCDIR=None, outdir=None, file=None, bias_fwhm=60,
                       bias_reg=1e-5):
    """ Performs biasfield correction within segmentation workflow for input
    image """

    # Convert nii.gz to .nii
    # ---------

#    # Read in .nii.gz image
#    img_gz = nb.loadsave.load(SRCDIR + file)
#
#    # Get rid of .gz extension from filename
#    file = file.replace(".gz", "")
#
#    # Write image into .nii
#    nb.loadsave.save(img_gz, SRCDIR + file)

    # Perform bias field correction within segment within interface
    # ---------

    # Initiate segment object
    segment = spm.Segment()

    # Define parameters of segment for segment object
    segment.inputs.data = SRCDIR + file
    segment.inputs.bias_fwhm = BIAS_FWHM
    segment.inputs.bias_regularization = BIAS_REG
    segment.inputs.save_bias_corrected = True
    segment.inputs.gm_output_type = [False, False, True]
    segment.inputs.wm_output_type = [False, False, True]
    segment.inputs.csf_output_type = [False, False, True]

    # Run segmentation for the purpose of bias field correction
    segment_output = segment.run()

    # Rename and copy final output from SPM
    # ---------

    # Read in bias field corrected SPM output
    img_bfc = nb.loadsave.load(segment_output.outputs.bias_corrected_image)

    # Ranema and copy into output directory
    nb.loadsave.save(img_bfc, outdir + file)

    return

# =============================================================================
# Run biasfield correction
# =============================================================================

# Take combinations of all subjects, sessions and runs
items = list(itertools.product(SUBJECTS, SESSIONS, RUNS))

# Iterate over all items
for item in items:

    # Extact sub, session and run identifiers
    sub, ses, run = item

    # Construct sourcepath
    srcpath = SRCDIR + f"sub-{sub:0>3}/ses-{ses}/anat/"

    # Construct outpath
    outpath = OUTDIR + f"sub-{sub:0>3}/ses-{ses}/anat/"

    # Construct file name
    file = f"sub-{sub:0>3}_ses-{ses}_run-{run}_T1w.nii"

    # Print status
    print(f"Starting {file}")

    # Call function to perform biasfield segmentation
    correct_bias_field(
            SRCDIR=srcpath,
            outdir=outpath,
            file=file,
            bias_fwhm=BIAS_FWHM,
            bias_reg=BIAS_REG)

    # Print status
    print(f"Finished {file}")

