"""
Created on Oct 3 2024

@author: bbantal

This script is used to compute similarity between aging patterns
and all available gene maps.

Commented out sections are for the HCP-A dataet.

"""

# %%

import os
import pandas as pd
import numpy as np
import statsmodels as sm
import statsmodels.api as sma
from scipy import stats
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt

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
# Age contrast in ALFF
# =============================================================================

# %%
# UKB
# ----------------------

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
        model = sma.OLS(df[f"{i}"][mask], sma.add_constant(df["age"][mask])).fit()

        # Save to collection
        map_[i] = model.params["age"]/norm_fact # model.tvalues["age"]/norm_fact

    except:
        map_[i] = np.nan

# Assign map to collection
map_alff_raw = map_.to_numpy()
    
# Build a mask for those ROIs that do not have any data
mask_alff = ~np.isnan(map_)

# If all values were excluded, do not mask out anything
if np.isnan(map_).sum() == len(map_):
    mask_alff = np.ones(len(map_), dtype=bool)

# Print sample size
print(f"UKB sample size: {df.shape[0]}")

# %%
# HCP-A
# ----------------------

# # Load data
# data = pd \
#     .read_csv("/shared/datasets/public/hcp-a/derivatives/alff/" \
#         "parcelled/alff_parcelled_300.csv") \
#     .rename({"sub": "subject"}, axis=1)

# # Load age info
# demo = pd \
#     .read_csv("/shared/datasets/public/hcp-a/rsfMRI/fmriresults01.txt",
#         delimiter="\t") \
#     .loc[1:, ["src_subject_id", "interview_age"]] \
#     .rename({"src_subject_id": "subject",
#                 "interview_age": "age"}, axis=1) \
#     .pipe(lambda df: df.assign(age=df.age.astype(float)/12)) \
#     .query('age < 100')

# # Merge data with age
# df = data.merge(demo, on="subject")

# # Collection for stats
# map_ = pd.Series(index=range(N), dtype=float)

# # Iterate over all features
# for i in tqdm(range(N)):

#     try:
#         # Mask outliers
#         mask = iqr_removal(df[f"{i}"])

#         # Print percentage of kept elements
#         # print(f"ROI {i}: {np.sum(mask)/len(mask)*100:.2f}%")

#         # Normalizing factor (normalize beta coefficient by mean of feature)
#         norm_fact = np.mean(df[f"{i}"][mask])

#         # Use statsmodels to fit linear model
#         model = sma.OLS(df[f"{i}"][mask], sma.add_constant(df["age"][mask])).fit()

#         # Save to collection
#         map_[i] = model.params["age"]/norm_fact # model.tvalues["age"]/norm_fact

#     except:
#         map_[i] = np.nan

# # Assign map to collection
# map_alff_raw = map_.to_numpy()

# # Build a mask for those ROIs that do not have any data
# mask_alff = ~np.isnan(map_)

# # Print sample size
# print(f"HCP-A sample size: {df.shape[0]}")

# %%
# =============================================================================
# Load gene maps
# =============================================================================

# Load data
gene_exp_data = pd.read_csv(f"/shared/projects/gene_expression_brain_maps/" \
    "data/parcellated_gene_expression/parcellated_gene_expression_seitzman.csv",
    index_col=0)

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
mask_gene = mask_nan & mask_struct

# Null map path
null_map_path = "/shared/projects/gene_expression_brain_maps/data/null_maps/seitzman_alff/"

# Full list of genes
genes = list(gene_exp_data.columns)

# %%
# =============================================================================
# Compute similarity
# =============================================================================

"""
This section can take hours!
"""

# Collections for stats
r_coll = []
pval_coll = []

# Combine masks
mask = mask_alff & mask_gene

# Apply mask to alff
map_alff_input = map_alff_raw[mask]

# Compute IQR mask for alff map
iqr_mask1 = iqr_removal(map_alff_input)

# -----------
# Iterate over all genes

for gene in tqdm(genes, total=len(genes)):

    # Load gene
    map_gene_raw = gene_exp_data[gene].reset_index(drop=True).to_numpy()

    # Apply mask
    map_gene_input = map_gene_raw[mask]

    # Outlier removal for empirical test
    iqr_mask2 = iqr_removal(map_gene_input)
    iqr_mask = iqr_mask1 & iqr_mask2
    map_alff = map_alff_input[iqr_mask]
    map_gene = map_gene_input[iqr_mask]

    # Compute correlation
    r_empirical, _ = stats.spearmanr(map_alff, map_gene)

    # Load surrogate maps for gene
    null_maps = np.load(null_map_path + f"null_map_{gene}.npy")

    # Initialize collection for correlations
    r_null_coll = np.zeros(null_maps.shape[1])

    # Iterate over surrogate maps
    for i, null_map in enumerate(null_maps.T):

        # Remove outliers
        iqr_mask2 = iqr_removal(null_map)
        iqr_mask = iqr_mask1 & iqr_mask2
        map_alff = map_alff_input[iqr_mask]
        null_map = null_map[iqr_mask]

        # Compute correlation and save to collection
        r_null_coll[i], _ = stats.spearmanr(map_alff, null_map)

    # Fit normal distribution
    surr_mean, surr_std = r_null_coll.mean(), r_null_coll.std()

    # Compute z score
    z_score = (r_empirical - surr_mean)/surr_std

    # Compute p value
    p_val_onetailed = stats.norm.cdf(abs(z_score))

    # Perform two-tailed adjustment
    p_val = 2 * min(p_val_onetailed, 1 - p_val_onetailed)

    # Save r value to collection
    r_coll.append(r_empirical)

    # Save computed p value to collection
    pval_coll.append(p_val)

try:

    # Create dataframe with results
    df_results = pd.DataFrame({
        "Gene": genes,
        "r": r_coll,
        "p": pval_coll
    })

    # Save to csv
    df_results.to_csv(
        OUTDIR + "gene_map_similarity_alff_ukb_age_fullset_pca.csv", index=False)

except:

    # Pickle it
    with open(OUTDIR + "gene_map_similarity_alff_ukb_age_fullset_pca.pkl", "wb") as f:
        pickle.dump((genes, r_coll, pval_coll), f)


# %%
# =============================================================================
# Take common subset of significantly associated genes across the two datasets
# =============================================================================

"""
Run this only after the above sections have been run 
"""

# rc formatting
plt.rcdefaults()
plt.rcParams["font.size"] = 9
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['font.family'] = "dejavu sans"

# Load statistics for genes
df_ukb = pd.read_csv(OUTDIR + "gene_map_similarity_alff_ukb_age_fullset.csv")
df_hcp = pd.read_csv(OUTDIR + "gene_map_similarity_alff_hcp_age_fullset.csv")

# Merge
df = df_ukb.merge(df_hcp, on="Gene", suffixes=("_ukb", "_hcp"))

# Load curated list (it's a txt)
with open(OUTDIR + "BrainGMTv2_HumanOrthologs.gmt.txt", "r") as f:
    genes_curated = f.read()

# Clean up the list
curated_list = [item for item in genes_curated.split("\t") \
                if ("_" not in item) and ("/" not in item) and (item != "")]

# Remove duplicates
curated_set = list(set(curated_list))

# Make series of curated genes
curated_series = pd.Series(curated_set, name="curated")

# Merge
df_curated = df.merge(curated_series, left_on="Gene", right_on="curated", how="inner")

# Do multiple comparison correction
# method = "fdr_tsbky"
# df_curated["p_ukb_corr"] = sm.stats.multitest.multipletests(df_curated["p_ukb"], alpha=0.05, method=method)[1]
# df_curated["p_hcp_corr"] = sm.stats.multitest.multipletests(df_curated["p_hcp"], alpha=0.05, method=method)[1]
df_curated["p_ukb_corr"] = df_curated["p_ukb"]
df_curated["p_hcp_corr"] = df_curated["p_hcp"]

# # # Filter for significant genes - uncorrected
# df.query("p_ukb < 0.05")
# df.query("p_hcp < 0.05")
# df_subset = df.query("p_ukb < 0.05 and p_hcp < 0.05")

# Filter for significant genes - corrected
df_curated.query("p_ukb_corr < 0.05")
df_curated.query("p_hcp_corr < 0.05")
df_filtered = df_curated.query("p_ukb_corr < 0.05 and p_hcp_corr < 0.05")

# %%
# Volcano plot
# ----------------

"""
Turn off FDR for this section! 
"""

# Threshold
threshold = 0.05/24
dataset = "ukb"

# x, y, coordinates for volcano plot
x_cut = df_curated.query(f"p_{dataset}_corr >= {threshold}")[f"r_{dataset}"]
y_cut = -np.log10(df_curated.query(f"p_{dataset}_corr >= {threshold}")[f"p_{dataset}_corr"])
x_kept_pos = df_curated.query(f"(p_{dataset}_corr < {threshold}) & (r_{dataset} > 0)")[f"r_{dataset}"]
y_kept_pos = -np.log10(df_curated.query(f"(p_{dataset}_corr < {threshold}) & (r_{dataset} > 0)")[f"p_{dataset}_corr"])
x_kept_neg = df_curated.query(f"(p_{dataset}_corr < {threshold}) & (r_{dataset} < 0)")[f"r_{dataset}"]
y_kept_neg = -np.log10(df_curated.query(f"(p_{dataset}_corr < {threshold}) & (r_{dataset} < 0)")[f"p_{dataset}_corr"])

# Figure
plt.figure(figsize=(3.625, 3))

# Visualize volcano plot
plt.scatter(x_cut, y_cut, s=0.5, color="dimgray")
plt.scatter(x_kept_pos, y_kept_pos, s=0.5, color="indianred")
plt.scatter(x_kept_neg, y_kept_neg, s=0.5, color="dodgerblue")

# Format
plt.xlabel("Spearman Correlation")
plt.ylabel("-log10(p)")
plt.xlim(-0.55, 0.55)
plt.ylim(-0.4, 6)
# plt.axhline(-np.log10(threshold), color="black", lw=1, ls="--")
plt.title("UK Biobank Dataset" if dataset == "ukb" else "HCP-A Dataset")

# List of selected genes
genes = { #UKB
    "SLC2A1": ["GLUT1", -30, -2],
    "SLC2A3": ["GLUT3", -50, -4],
    "SLC2A4": ["GLUT4", -40, -2],
    "SLC16A1": ["MCT1", -30, 2],
    "SLC16A7": ["MCT2", 40, -2],
    "APOE": ["APOE", 30, -2],
    "TNF": ["TNF", -30, -2],
    "TNFRSF1A": ["TNF rec1", 50, 5],
    "IL1B": ["IL-1B", 35, -2],
    "IL6R": ["IL-6", -30, -2],
    "IL23A": ["IL-23A", -40, -6],
    "P2RX7": ["P2RX7", 40, -10],
    "NOS1": ["NOS1", 40, -4],
    "ACE": ["ACE", 20, 5],
    "EDN1": ["ET-1", 40, 7],
    "VEGFA": ["VEGFA", -40, 3],
    "VEGFB": ["VEGFB", 50, -2],
    "FLT1": ["VEGFR1", 40, -2],
    "ACTB": ["Beta-actin", 45, 4],
    "GAPDH": ["GAPDH", 40, -2],
    "NEFL": ["NF-L", 45, -2],
    "EEF1A1": ["EEF1A1", 10, -10],
    "RPL13A": ["RPL13A", -30, -8],
    "PGK1": ["PGK1", 40, -2]
}

# genes = {  # HCP-A
#     "SLC2A1": ["GLUT1", -40, -6], #
#     "SLC2A3": ["GLUT3", -80, -8],
#     "SLC2A4": ["GLUT4", -40, -2], #
#     "SLC16A1": ["MCT1", -35, 2],
#     "SLC16A7": ["MCT2", 40, -2],
#     "APOE": ["APOE", 30, -2], #
#     "TNF": ["TNF", 30, -2], #
#     "TNFRSF1A": ["TNF rec1", 50, -7], #
#     "IL1B": ["IL-1B", -30, -2], #
#     "IL6R": ["IL-6", 25, -6], #
#     "IL23A": ["IL-23A", 25, -9], #
#     "P2RX7": ["P2RX7", 40, 0], #
#     "NOS1": ["NOS1", 40, 7], #
#     "ACE": ["ACE", 15, 10],
#     "EDN1": ["ET-1", 40, -2],
#     "VEGFA": ["VEGFA", -30, -7], # 
#     "VEGFB": ["VEGFB", 30, -2], # 
#     "FLT1": ["VEGFR1", 30, 12],  #
#     "ACTB": ["Beta-actin", 45, 6], # 
#     "GAPDH": ["GAPDH", -40, -2], #
#     "NEFL": ["NF-L", -35, -3], #
#     "EEF1A1": ["EEF1A1", -30, -8],
#     "RPL13A": ["RPL13A", -45, -3],
#     "PGK1": ["PGK1", -30, 2] # 
# }


# Iterate through all selected genes
for gene, [protein, x_offset, y_offset] in genes.items():
    
    # x, y coordinates for current gene
    x_gene = df_curated.query(f'Gene=="{gene}"')[f"r_{dataset}"]
    y_gene = -np.log10(df_curated.query(f'Gene=="{gene}"')[f"p_{dataset}_corr"])

    # Annotate a specific point
    plt.annotate(
        protein,
        (x_gene, y_gene),
        textcoords="offset points",
        xytext=(x_offset, y_offset),
        ha='center',
        arrowprops=dict(arrowstyle='->', lw=0.75),
        fontsize=7
    )

# Save
plt.tight_layout()
# plt.savefig("/shared/home/botond/results/pagb/" + f"volcano_plot_{dataset}.pdf",
#             dpi=300, transparent=True)

# %%
# Save subset for GSEA analysis
# -------------------

# Suffix (all, pos, or neg)
extra = "_all"

# Take only genes with positive correlation
statement = "r_ukb > 0 and r_hcp > 0" if extra == "_pos" else "r_ukb < 0 and r_hcp < 0" if extra == "_neg" else "r_ukb * r_hcp > 0"
df_subset = df_filtered.query(statement)

# Check if there are any opposite signs (currently redundant)
assert sum(sum([df_subset["r_ukb"] * df_subset["r_hcp"] > 0])) == df_subset.shape[0]

# Take the top N genes
df_subset = df_subset.assign(abs_mean=abs((df_subset["r_ukb"] + df_subset["r_hcp"])/2))

# %%
# Visualize GSEA analysis results
# -------------------

# Disable pandas SettingWithCopyWarning
pd.set_option('mode.chained_assignment', None)

# Version
VER = "all"

# Colors
colors = {
    "pos": "orangered",
    "neg": "dodgerblue",
    "all": "teal"
}

# Load data
with open(OUTDIR + f"overlap_{VER}.tsv", 'r') as file:
    file_content = file.read()

# Parse into lines
lines = file_content.splitlines()

# Parse meta results into dataframe
gsea_meta = pd.DataFrame([line.split('\t') for line in lines[3:8]], columns=["description", "value"])

# Extract number of observed and background genes
N_observed_genes = int(gsea_meta.query("description=='# genes in comparison (n):'")["value"].values[0])
N_background_genes = int(gsea_meta.query("description=='# genes in universe (N):'")["value"].values[0])

# Find range of lines with relevant results
start_ind = 9
stop_ind = [i for i in range(len(lines)) if "Gene/Gene Set Overlap Matrix" in lines[i]][0]-2

# Parse relevant results into dataframe
gsea_results = pd.DataFrame([line.split('\t') for line in lines[start_ind:stop_ind]])
gsea_results.columns = gsea_results.iloc[0]
gsea_results = gsea_results[1:].reset_index(drop=True)
gsea_results = gsea_results.apply(pd.to_numeric, errors='ignore')

# Assign values to display
gsea_results = gsea_results.assign(**{
    "display_name": gsea_results["Gene Set Name"] \
        .apply(lambda x: ' '.join(x.lower().split("_")[1:]).capitalize()),
    "logp-": -np.log10(gsea_results["p-value"].astype(float)),
    "fold_enrichment": (gsea_results["# Genes in Overlap (k)"]/N_observed_genes) / \
        (gsea_results["# Genes in Gene Set (K)"]/N_background_genes)
        })


for i, item in enumerate(gsea_results["display_name"]):
    
    # If the item has more than 4 words, insert a new line at every 2nd word    
    if len(item.split(" ")) > 3:
        words = item.split(" ")
        new_item = " ".join(words[:2]) + "\n" + " ".join(words[2:])
        gsea_results["display_name"][i] = new_item


# Figure
plt.figure(figsize=(3.625, 3))

# Plot results
plt.scatter(gsea_results["fold_enrichment"], gsea_results["logp-"],
        s=gsea_results["# Genes in Gene Set (K)"], color=colors[VER], alpha=0.3)

# Annotation offsets
x_offsets = {
    "all": [0, 0, 0, 25, -25, -30, -10, 30, 0, 50],
    "pos": [0, 0, -10, 30, 30, -5, 0, -60, 40, -20],
    "neg": [0, 0, 0, 10, 5, -5, -10, 0, -10, -20]
}

y_offsets = {
    "all": [0, 0, 0, 25, 25, -10, -25, 20, 0, -30],
    "pos": [0, 0, 20, 20, -20, 20, 0, 10, -20, -30],
    "neg": [0, 0, 0, 10, 5, -5, -10, 0, -10, -20]
}

# Annotate label and connect it to the center
for i, txt in enumerate(gsea_results["display_name"]):
    plt.annotate(txt, (gsea_results["fold_enrichment"][i], gsea_results["logp-"][i]),
        xytext=(x_offsets[VER][i], y_offsets[VER][i]), textcoords='offset points', ha='center', va="center",
        arrowprops=dict(arrowstyle='-', lw=1, color="black"), fontsize=7)

# Format
plt.xlim([0, 10])
plt.ylim([15, 35])
plt.xlabel("Fold Enrichment")
plt.ylabel("-log10(p)")

# Save
plt.tight_layout()
# plt.savefig("/shared/home/botond/results/pagb/" + f"fig_gsea_results_{VER}.pdf",
#             dpi=300, transparent=True)
