#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jun 4 2023

@author: botond

This script computes spatial correlation between gene expression maps
and statistical maps of age-related effects in ALFF from UKB and HCP-A datasets.

"""

# %%

import os
import pandas as pd
import numpy as np
import itertools
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm


# %%
# =============================================================================
# Setup
# =============================================================================

# Define filepaths
HOMEDIR = os.path.expanduser("~/")
SRCDIR = "/shared/projects/gene_expression_brain_maps/data/"
OUTDIR = "/shared/projects/gene_expression_brain_maps/results/alff/"

# Number of ROIs
N = 300

# Initialize collection for maps
maps = {}
    
# Initialize collection for masks
masks = {}

# Function that yields masks for outliers based on IQR
def iqr_removal(arr, thr=1.5):

    q1, q3 = np.percentile(arr, [25, 75])
    iqr = q3 - q1
    lb = q1 - thr*iqr
    ub = q3 + thr*iqr
    mask = (arr > lb) & (arr < ub)

    return mask

# %%
# =============================================================================
# Prepare maps
# =============================================================================

# %%
# UKB age
# -----------------------------------------------------------------------------

# Load data
data = pd \
    .read_csv("/shared/datasets/public/ukb/derivatives/alff/" \
        "parcelled/alff_parcelled_300.csv") \
    .rename({"sub": "subject"}, axis=1)

# Load matched demographics
demo = pd \
    .read_csv("/shared/datasets/public/ukb/derivatives/" \
        "design_matrices/matched_regressors_age_alff_v1.csv",
         index_col=0) \
    .rename({"eid": "subject"}, axis=1) \
    .loc[:, ["subject", "age"]]

# Merge data with age
df = data.merge(demo, on="subject")

# Collection for stats
map_ = pd.Series(index=range(N), dtype=float)

# Iterate over all features
for i in tqdm(range(N)):

    try:
        # Mask outliers
        mask = iqr_removal(df[f"{i}"])

        # Print percentage of kept elements
        # print(f"ROI {i}: {np.sum(mask)/len(mask)*100:.2f}%")

        # Normalizing factor (normalize beta coefficient by mean of feature)
        norm_fact = np.mean(df[f"{i}"][mask])

        # Use statsmodels to fit linear model
        model = sm.OLS(df[f"{i}"][mask], sm.add_constant(df["age"][mask])).fit()

        # Save to collection
        map_[i] = model.params["age"]/norm_fact # model.tvalues["age"]/norm_fact

    except:
        map_[i] = np.nan

# Assign map to collection
maps["age-ukb"] = map_.to_numpy()
    
# Build a mask for those ROIs that do not have any data
mask_map = ~np.isnan(map_)

# If all values were excluded, do not mask out anything
if np.isnan(map_).sum() == len(map_):
    mask_map = np.ones(len(map_), dtype=bool)

# Assign mask to collection
masks["age-ukb"] = mask_map

# Print sample size
print(f"UKB sample size: {df.shape[0]}")

# %%
# HCP-A age
# ---------------------------------------------------------------------------

# Load data
data = pd \
    .read_csv("/shared/datasets/public/hcp-a/derivatives/alff/" \
        "parcelled/alff_parcelled_300.csv") \
    .rename({"sub": "subject"}, axis=1)

# Load age info
demo = pd \
    .read_csv("/shared/datasets/public/hcp-a/rsfMRI/fmriresults01.txt",
        delimiter="\t") \
    .loc[1:, ["src_subject_id", "interview_age"]] \
    .rename({"src_subject_id": "subject",
                "interview_age": "age"}, axis=1) \
    .pipe(lambda df: df.assign(age=df.age.astype(float)/12)) \
    .query('age < 100')

# Merge data with age
df = data.merge(demo, on="subject")

# Collection for stats
map_ = pd.Series(index=range(N), dtype=float)

# Iterate over all features
for i in tqdm(range(N)):

    try:
        # Mask outliers
        mask = iqr_removal(df[f"{i}"])

        # Print percentage of kept elements
        # print(f"ROI {i}: {np.sum(mask)/len(mask)*100:.2f}%")

        # Normalizing factor (normalize beta coefficient by mean of feature)
        norm_fact = np.mean(df[f"{i}"][mask])

        # Use statsmodels to fit linear model
        model = sm.OLS(df[f"{i}"][mask], sm.add_constant(df["age"][mask])).fit()

        # Save to collection
        map_[i] = model.params["age"]/norm_fact # model.tvalues["age"]/norm_fact

    except:
        map_[i] = np.nan

# Assign map to collection
maps["age-hcp"] = map_.to_numpy()

# Build a mask for those ROIs that do not have any data
mask_map = ~np.isnan(map_)

# Assign mask to collection
masks["age-hcp"] = mask_map

# Print sample size
print(f"HCP-A sample size: {df.shape[0]}")

# %%
# Gene expression maps
# ---------------------------------------------------------------------

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

# Merge the two masks
mask_gene_exp = mask_nan & mask_struct

# Load genes
# ----

def load_gene(gene_name):
    maps[gene_name] = gene_exp_data[gene_name].reset_index(drop=True).to_numpy()
    masks[gene_name] = mask_gene_exp

# Metabolic genes
load_gene("SLC2A1")  # GLUT1
load_gene("SLC2A3")  # GLUT3
load_gene("SLC2A4")  # GLUT4
load_gene("SLC16A1")  # MCT1
load_gene("SLC16A7")  # MCT2
load_gene("APOE")  # APOE

# Vascular health related genes
load_gene("NOS1")  # NOS1
load_gene("ACE")  # NOS2
load_gene("EDN1")  # NOS3
load_gene("VEGFA")  # VEGFA
load_gene("VEGFB")  # VEGFB
load_gene("FLT1") # VEGFR1

# Inflammation related genes
load_gene("TNF")  # TNF
load_gene("TNFRSF1A")  # TNF receptor
load_gene("IL1B") # IL1B
load_gene("IL6R")  # IL6
load_gene("IL23A")  # IL-23A
load_gene("P2RX7")  # P2X7

# Unrelated negative control genes
load_gene("ACTB")  # ACTB
load_gene("NEFL")  # NEFL
load_gene("GAPDH")  # GAPDH
load_gene("PGK1")  # PGK1
load_gene("EEF1A1")  # EEF1A1
load_gene("RPL13A")  # RPL13A

# %%
# =============================================================================
# Compute similarity
# =============================================================================

# Function to compute p value
def compute_similarity_with_perm(map_a_input, map_b_input, null_map_path=None):
    """
    Function to compute p value from permutation test with surrogate maps
    """

    # Outlier removal for empirical test
    iqr_mask1 = iqr_removal(map_a_input)
    iqr_mask2 = iqr_removal(map_b_input)
    iqr_mask = iqr_mask1 & iqr_mask2
    map_a = map_a_input[iqr_mask]
    map_b = map_b_input[iqr_mask]
    
    # Compute correlation
    r_empirical, _ = stats.spearmanr(map_a, map_b)

    # Load surrogate maps for gene
    null_maps = np.load(null_map_path)

    # Initialize collection for correlations
    r_null_coll = np.zeros(null_maps.shape[1])

    # Iterate over surrogate maps
    for i, null_map in enumerate(null_maps.T):

        # Remove outliers
        iqr_mask2 = iqr_removal(null_map)
        iqr_mask = iqr_mask1 & iqr_mask2
        map_a = map_a_input[iqr_mask]
        null_map = null_map[iqr_mask]

        # Compute correlation and save to collection
        r_null_coll[i], _ = stats.spearmanr(map_a, null_map)

    # Fit normal distribution
    surr_mean, surr_std = r_null_coll.mean(), r_null_coll.std()

    # # Plot gaussian with surr_mean and surr_std
    # x = np.linspace(-.5, .5, 100)
    # plt.hist(r_null_coll, color="dodgerblue", edgecolor="black", linewidth=2, alpha=0.7, density=True)
    # plt.plot(x, stats.norm.pdf(x, surr_mean, surr_std), color="orangered", lw=3)
    # plt.axvline(r_empirical, color="limegreen", lw=2)

    # Compute z score
    z_score = (r_empirical - surr_mean)/surr_std

    # Compute p value
    p_val_onetailed = stats.norm.cdf(abs(z_score))

    # Perform two-tailed adjustment
    p_val = 2 * min(p_val_onetailed, 1 - p_val_onetailed)

    # Return p value
    return r_empirical, p_val

# Define null map paths
# --------

# Initiate collection
null_map_paths = {}

# Iterate over all keys
for key in list(maps.keys()):
    
    # If it is an age related contrast
    if "age-" in key:
        # Do not assign null maps
        null_map_paths[key] = None
    
    # If it is a special contrast
    elif any([val in key for val in ["-ukb", "-camcan", "-hcp", "-mgh"]]):
        # Assign path unique to contrasts
        null_map_paths[key] = \
                "/shared/projects/gene_expression_brain_maps/data/null_maps/contrasts/" \
                f"seitzman_alff/null_maps_{key}.npy"
    
    # Else, it is assumed to be a gene
    else:
        # Assign path unique to gene maps
        null_map_paths[key] = \
            "/shared/projects/gene_expression_brain_maps/data/" \
            f"null_maps/seitzman_alff/null_map_{key}.npy"

# Iterate through all combinations
# --------

# Combos
combos = list(itertools.combinations(maps.keys(), 2))

# Collections for stats
r_coll = []
pval_coll = []

# Corresponding upper triangular indices
indices = np.triu_indices(len(maps.keys()), k=1)

# Iterate over combos
for i, combo in tqdm(enumerate(combos), total=len(combos)):
    
    # Print status
    # print(f"Comparing {combo[0]} vs {combo[1]}")

    # Get maps
    map1 = maps[combo[0]]
    map2 = maps[combo[1]]

    # Get masks
    mask1 = masks[combo[0]]
    mask2 = masks[combo[1]]
    
    # Combine masks
    mask = mask1 & mask2

    # Apply mask
    map1 = map1[mask]
    map2 = map2[mask]

    # Compute p value
    # ----
    # Take null map from the second item of the comparions (ensures hierarchy)
    null_map_path = null_map_paths[combo[1]]

    # If path is none, null map does note exist
    if null_map_path is None:

        # Outlier removal
        iqr_mask1 = iqr_removal(map1)
        iqr_mask2 = iqr_removal(map2)
        iqr_mask = iqr_mask1 & iqr_mask2
        map1 = map1[iqr_mask]
        map2 = map2[iqr_mask]
        
        # Compute similarity
        r, _ = stats.spearmanr(map1, map2)

        # Assign 1 to p value
        pval = 1

    # Else, run permutation tests with existing null_map
    else:
        r, pval = compute_similarity_with_perm(map1, map2, null_map_path=null_map_path)
    
    # Save r value to collection
    r_coll.append(r)

    # Save computed p value to collection
    pval_coll.append(pval)

# %%
# Visualize
# -------------------------------------------------------------------------------------

# Bonferroni correction
correction_factor = 24 # Equals to the number of relevant tests run in parallel
alpha = 0.05 / correction_factor

# Zero out p values for tiles where significance is too low (but not where it wasn't evaluated)
r_coll_filtered = [r if pval_coll[i] < alpha else r if pval_coll[i] == 1 \
            else 0 for i, r in enumerate(r_coll)]

# Convert to matrix
sim_mat = np.zeros(2*[len(maps)])
sim_mat[indices] = r_coll_filtered
# sim_mat = sim_mat.T  # To display only the bottom part
sim_mat = sim_mat + sim_mat.T
sim_mat = sim_mat + np.eye(len(maps.keys()))

# Figure
plt.figure(figsize=(12, 10))

# Plot
ax = sns.heatmap(
    sim_mat, annot=True, cmap="seismic", center=0, vmin=-1, vmax=1, mask=sim_mat==0,
    xticklabels=maps.keys(), yticklabels=maps.keys(), linewidth=0.5, linecolor="black")

# Spines
for sp in ['bottom', 'top', 'left', 'right']:
    plt.gca().spines[sp].set_visible(True)
    plt.gca().spines[sp].set_linewidth(1.5)
    plt.gca().spines[sp].set_color("black")

# Format
plt.xticks(rotation=45)
plt.yticks(rotation=45)

# # Save
# np.save("/shared/projects/gene_expression_brain_maps/data/similarity/alff/" \
#         f"sim_mat_{LT}_{UT}.npy", sim_mat)
# np.save("/shared/projects/gene_expression_brain_maps/data/similarity/alff/" \
#          "sim_mat_keys.npy", np.array(list(maps.keys())))


# Extract p values
p_mat = np.zeros(2*[len(maps)])
p_mat[indices] = pval_coll
p_mat = p_mat + p_mat.T
p_df = pd.DataFrame(p_mat, columns=maps.keys(), index=maps.keys())
p_df.loc["age-hcp", "SLC2A4"]
p_df.loc["age-hcp", "SLC16A7"]
p_df.loc["age-hcp", "APOE"]

# %%
# =============================================================================
# Figures
# =============================================================================

# %%
# Plot pairwise scatterplots (Figure 2C)
# ---------------------

# rc formatting
plt.rcdefaults()
plt.rcParams["font.size"] = 9
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['font.family'] = "dejavu sans"

# Combos
combos = [
    ["age-ukb", "SLC2A4"],
    ["age-ukb", "SLC16A7"],
    ["age-ukb", "APOE"],
]

# Start figure
plt.subplots(3, 1, sharex=True, figsize=(3.25, 6))

# # Current combo
# combo = ["age-ukb", "SLC16A7"]

# Itereate over combos
for j, combo in enumerate(combos):

    # Specific settings
    colors = {
        "SLC2A4": "salmon",
        "SLC16A7": "tab:blue",
        "APOE": "indianred",
    }

    names = {
        "SLC2A4": "GLUT4",
        "SLC16A7": "MCT2",
        "APOE": "APOE",
    }

    # Get maps
    map1 = maps[combo[0]]*100
    map2 = maps[combo[1]]

    # Get masks
    mask1 = masks[combo[0]]
    mask2 = masks[combo[1]]

    # Combine masks
    mask = mask1 & mask2

    # Apply mask
    map1 = map1[mask]
    map2 = map2[mask]

    # Compute p value
    # ----
    # Take null map from the second item of the comparions (ensures hierarchy)
    null_map_path = null_map_paths[combo[1]]

    # If path is none, null map does note exist
    if null_map_path is None:

        # Outlier removal
        iqr_mask1 = iqr_removal(map1)
        iqr_mask2 = iqr_removal(map2)
        iqr_mask = iqr_mask1 & iqr_mask2
        map1 = map1[iqr_mask]
        map2 = map2[iqr_mask]
        
        # Compute similarity
        r, _ = stats.spearmanr(map1, map2)

        # Assign 1 to p value
        pval = 1

    # Else, run permutation tests with existing null_map
    else:
        r, pval = compute_similarity_with_perm(map1, map2, null_map_path=null_map_path)

    # Visualize
    # -----


    # np.random.random(3)}
        
    # Select current axis
    plt.subplot(3, 1, j+1)
    ax = plt.gca()

    # Plot
    sns.regplot(
        x=map1, y=map2, ax=ax,
        scatter_kws={"s": 20, "linewidth": 0.4,
        "edgecolor": "k", "facecolor": colors[combo[1]], "zorder": 2, "alpha": 1}, #colors[i]},
        line_kws={"color": "black", "linewidth": 1.5})

    # Annotate stats
    plt.annotate(
        f"r={np.round(r, 2)}\np={pval:.0e}",
        xy=(0.72, 0.79),
        xycoords="axes fraction",
        fontsize=8) #, fontweight="bold")


    # Format
    plt.title(names[combo[1]])
    # plt.title("Brain GLUT4 expression spatially correlates\nwith aging patterns from fMRI")
    # plt.grid(zorder=1)
    plt.ylim([-0.05, 1.03])
    # plt.ylabel(combo[1])
    plt.xticks(rotation=0)
    plt.ylabel(f"Gene Expression")
    if j == 2:
        plt.xlabel("Age-Related Effect in ALFF\n(% change per year)")



    # Spines
    for sp in ['bottom', 'top', 'left', 'right']:
        plt.gca().spines[sp].set_linewidth(0.75)
        plt.gca().spines[sp].set_color("black")

# Save
plt.tight_layout() #rect=[ 0, 0.02, 1, 1])
# plt.savefig("/shared/home/botond/results/pagb/" + "fig_geneexp_scatter_column.pdf",
#             transparent=True, dpi=300)
# plt.savefig(OUTDIR + f"scatter_{combo[0]}_{combo[1]}.pdf", transparent=True, dpi=300)
# plt.savefig(f"/Users/benett/Desktop/{combo}.png", dpi=300)

# %%
# Plot cropped similarity matrix (Figure 2B)
# ---------------------

# Names
names = {
    "age-ukb": "Age\n(UKB)",
    "age-hcp": "Age\n(HCP-A)",
    "SLC2A1": "GLUT1",
    "SLC2A3": "GLUT3",
    "SLC2A4": "GLUT4",
    "SLC16A1": "MCT1",
    "SLC16A7": "MCT2",
    "APOE": "APOE",
    "TNF": "TNF",
    "TNFRSF1A": "TNFRSF1A",
    "IL1B": "IL-1B",
    "IL6R": "IL-6R",
    "IL23A": "IL-23A",
    "P2RX7": "P2RX7",
    "NOS1": "NOS1",
    "ACE": "ACE",
    "EDN1": "ET-1",
    "VEGFA": "VEGFA",
    "VEGFB": "VEGFB",
    "FLT1": "VEGFR1",
    "ACTB": "β-actin",
    "NEFL": "NF-L",
    "GAPDH": "GAPDH",
    "PGK1": "PGK1",
    "EEF1A1": "EEF1A1",
    "RPL13A": "RPL13A",
    
}

# Ticklabels
xticklabels = [names[item] for item in list(maps.keys())[:2]]
yticklabels = [names[item] for item in list(maps.keys())[2:]]

# Crop sim matrix
mat_gf_full = sim_mat[:2, 2:].T

# Figure
plt.figure(figsize=(4, 7.5))

# Get subset of genes for current axis
inds = np.arange(24)
mat_gf = mat_gf_full[inds, :]

# Plot heatmap for current axis
sns.heatmap(
    mat_gf, annot=True, cmap="RdBu_r", center=0, vmin=-0.5, vmax=0.5,
    mask=mat_gf==0,
    linewidth=0.5, linecolor="black", annot_kws={"fontsize": 8, "rotation": 0},
    xticklabels=xticklabels, yticklabels=np.array(yticklabels)[inds],
    square=True, cbar=False)

# Spines
for sp in ['bottom', 'top', 'left', 'right']:
    plt.gca().spines[sp].set_visible(True)
    plt.gca().spines[sp].set_linewidth(1)
    plt.gca().spines[sp].set_color("black")

# Format
plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)
plt.tick_params(axis='x', which='major', pad=20)
plt.xticks(rotation=90, fontsize=11, ha="center", va="center")
plt.yticks(rotation=0, fontsize=11)

# Save
plt.tight_layout()
# plt.savefig("/shared/home/botond/results/pagb/" + f"fig_geneexp_simmat_vertical.pdf",
#             transparent=True, dpi=300)
