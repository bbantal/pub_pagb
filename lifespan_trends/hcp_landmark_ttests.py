"""
Created on Nov 15 2023

@author: botond

This script evaluates changes in peripheral biomarkers around landmark points derived from the lifespan trends.
These analyses were all done in the HCP-A dataset.

"""

# %%

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# %%
# =============================================================================
# Setup
# =============================================================================

# Define filepaths
HOMEDIR = os.path.expanduser("~/")

# Load demographics info
demo = pd \
    .read_csv("/shared/datasets/public/hcp-a/rsfMRI/fmriresults01.txt",
        delimiter="\t") \
    .loc[1:, ["src_subject_id", "interview_age", "sex"]] \
    .rename({"src_subject_id": "subject",
                "interview_age": "age",}, axis=1) \
    .pipe(lambda df: df.assign(age=df.age.astype(float)/12)) \
    .query('age < 100')

# Helper functions
# ---------

# Function that yields masks for outliers based on IQR
def iqr_removal(arr, thr=1.5):

    q1, q3 = np.nanpercentile(arr, [25, 75])
    iqr = q3 - q1
    lb = q1 - thr*iqr
    ub = q3 + thr*iqr
    mask = (arr > lb) & (arr < ub)

    return mask

# %%
# =============================================================================
# Load data
# =============================================================================

# %%
# Load physical measures
# ---------------------------------------------------------------------

# Blood results
# -----

# Load blood results
bsc = pd.read_csv("/shared/datasets/public/hcp-a/summary_tables/bsc01.txt", sep='\t')

# Define relevant columns to extract
rel_cols = {
    "src_subject_id": "subject", # Subject ID
    "fasting": "fasting",  # Fasting yes/no
    "laba8": "crp",  # High sensitivity C-reactive protein
    "a1crs": "hba1c", # HbA1c
    "glucose": "glucose",  # Fasting glucose
    "insomm": "insulin",  # Fasting insulin
}

# Extract relevant columns
bsc_selected = bsc \
    [rel_cols.keys()] \
    .drop(0) \
    .rename(rel_cols, axis=1) \
    .pipe(lambda df: df.assign(**{"insulin": df["insulin"].str.replace(" uU/mL", "")})) \
    .apply(pd.to_numeric, errors="ignore")

# Vitals
# ----
vitals = pd.read_csv("/shared/datasets/public/hcp-a/summary_tables/vitals01.txt", sep='\t')

# Define relevant columns to extract
rel_cols = {
    "src_subject_id": "subject", # Subject ID
    "bp": "bp",  # Blood pressure
}

# Extract relevant columns
vitals_selected = vitals \
    [rel_cols.keys()] \
    .drop(0) \
    .rename(rel_cols, axis=1) \
    .pipe(lambda df: pd.concat([df, df['bp'].str.split('/', expand=True)], axis=1)) \
    .rename({0: "systolic", 1: "diastolic"}, axis=1) \
    .drop("bp", axis=1) \
    .apply(pd.to_numeric, errors="ignore")

# Merge vitals and blood results and clean data
# ----

# Merge
data_phys = pd.merge(bsc_selected, vitals_selected, on="subject", how="inner")

# Select fasting only
data_phys = data_phys.query('fasting == 1')

# Remove compromised values
data_phys = data_phys.query('hba1c < 15')

# # Compute HOMA-IR
# data_phys["HOMA-IR"] = data_phys["insulin"] * data_phys["glucose"] / 405

# Plot histograms for phys data
# ----------------------------

# Grid size for plots
I = 2
J = 3

# Ranges
ranges = {
    "crp": (0, 10),
    "hba1c": (3.5, 8),
    "glucose": (50, 200),
    "insulin": (0, 30),
    "systolic": (50, 200),
    "diastolic": (30, 120),
}

# Thresholds
thresh = {
    "crp": 1,
    "hba1c": 5.7,
    "glucose": 100,
    "insulin": 10,
    "systolic": 120,
    "diastolic": 80,
}

# Columns
cols = data_phys.columns[2:]

# Redefine thresholds - at 50%
thresh = {key: data_phys[key].median() for key in cols}

# Start figure and axes
fig, axes = plt.subplots(I, J, figsize=(10, 8), sharex=False, sharey=True)

# Iterate over columns
for c, col in enumerate(cols):

    i, j = np.unravel_index(c, (I, J))

    # Plot histogram
    axes[i, j].hist(data_phys[col], bins=np.linspace(*ranges[col], 20),
    color="dodgerblue", edgecolor="black", linewidth=1, alpha=0.8)

    # Show line at threshold
    axes[i, j].axvline(x=thresh[col], color="red", linestyle="--")

    # Show counts
    count_total = data_phys.query(f"{col} == {col}").shape[0]
    count_high = data_phys.query(f"{col} > {thresh[col]}").shape[0]
    axes[i, j].text(0.5, 0.9, f"{count_total - count_high}/{count_high}",
        transform=axes[i, j].transAxes)

    # X label
    axes[i, j].set_xlabel("value")
    # plt.ylabel("count")

    # Set x lim
    # axes[i, j].set_xlim(lims[col])

    # Set title
    axes[i, j].set_title(col)

# Super formatting
plt.gcf().supylabel("count")
plt.tight_layout()

# Plot Age/sex trends before outlier removal
# -----------

# Temporary dataframe for plotting
tdf = data_phys.merge(demo, on="subject")

# Start figure and axes
fig, axes = plt.subplots(I, J, figsize=(10, 8), sharex=False, sharey=False)

# Iterate over columns
for c, col in enumerate(cols):

    # Unravel
    i, j = np.unravel_index(c, (I, J))

    # Plot
    sns.scatterplot(data=tdf, x="age", y=col, hue="sex", palette=["blue", "red"], ax=axes[i, j])    

plt.suptitle("Before outlier removal")
plt.tight_layout()

# Remove outliers
# -----------------

# Iterate over columns
for col in cols:

    # Get mask
    mask = iqr_removal(data_phys[col])

    # Assign nan to outlier in column
    data_phys.loc[~mask, col] = np.nan

# Plot Age/sex trends again following outlier removal
# -----------
# Temporary dataframe for plotting
tdf = data_phys.merge(demo, on="subject")

# Start figure and axes
fig, axes = plt.subplots(I, J, figsize=(10, 8), sharex=False, sharey=False)

# Iterate over columns
for c, col in enumerate(cols):

    # Unravel
    i, j = np.unravel_index(c, (I, J))

    # Plot
    sns.scatterplot(data=tdf, x="age", y=col, hue="sex", palette=["blue", "red"], ax=axes[i, j])    
    sns.regplot(data=tdf, x="age", y=col, ax=axes[i, j], scatter=False, color="black")

plt.suptitle("After outlier removal")
plt.tight_layout()

# Correlations
plt.figure()
sns.heatmap(data_phys[cols].corr(), cmap="seismic", center=0, annot=True)


# %%
# T-tests around landmark points (Figure 1B)
# ----------------------------------------------------------------------------------------------

# Some rc formatting
plt.rcdefaults()
plt.rcParams["font.size"] = 9
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['font.family'] = "dejavu sans"

# Merge demo and phys data
sdf = pd.merge(demo, data_phys, on="subject", how="inner")

# Columns to analyze
# cols_rel = list(sdf.columns[4:])
cols_rel = ["hba1c", "systolic", "diastolic", "crp"]

# Column labels
col_labels = {
    "hba1c": "HbA1c",
    # "glucose": "Fasting\nBlood\nGlucose",
    "systolic": "SysBP",
    "diastolic": "DiaBP",
    "crp": "CRP",
}

# colors = ["#0E67A6", "#F5A94D"]
colors = ["#0E67A6", "orange"]

# Get reference changes
# ----------

# Initiate dict for reference amplitudes
norm_fact = {}
norm_fact_cis = {}

# Normalization factor: std deviation
# ----

# Iterate through all columns
for col in cols_rel:

    # Compute std and add to dict
    norm_fact[col] = sdf[col].std()

    # Uncertainty
    norm_fact_cis[col] = None

# Compute and plot mean differences
# ------

# Start figure
plt.subplots(1, 2, sharey=True, figsize=(3.625, 2.9))

# Irerate over thresholds
for j, threshold_str in enumerate(["alpha", "I"]):

    # # Specific threshold to use
    # threshold_str = "alpha"

    # Breakpoints
    alpha = 43.7
    I = 66.7
    beta1 = 89.7

    # Split into groups around landmarks
    groups = [sdf.query(f"(age > {l1}) & age <= {l2}") \
        for l1, l2 in zip([0, alpha, I, beta1], [alpha, I, beta1, 100])]

    # Assign pairs of groups (before vs after) to landmarks
    groups = {
        "alpha": groups[0:2],
        "I": groups[1:3],
        "beta1": groups[2:4],
        }

    # Define threshold (from string)
    threshold = eval(threshold_str)

    # Extract groups
    sdf_pre, sdf_post = groups[threshold_str]

    # Get sample sizes
    ss = [sdf_pre.shape[0], sdf_post.shape[0]]

    # Initiate collections
    diffs_t = {}
    diffs_p = {}
    diffs_means = {}
    diffs_errors = {}
    sample_sizes = {}

    # Iterate over columns
    for col in cols_rel:

        # Extract corresponding data
        d1, d2 = sdf_pre[col].dropna(), sdf_post[col].dropna()

        # Run ttest
        t, p = stats.ttest_ind(d1, d2, equal_var=False, nan_policy="omit")

        # Degrees of freedom
        deg_f = len(d1) + len(d2) - 2

        # Two-tailed critical value for the t-distribution
        critical_value = stats.t.ppf((1 + 0.95) / 2, deg_f)

        # Standard errors
        std_error = np.sqrt(np.var(d1)/len(d1) + np.var(d2)/len(d2))

        # Margin of error
        margin_of_error = critical_value * std_error

        # Mean difference
        mean_difference = np.mean(d2) - np.mean(d1)

        # Confidence interval
        confidence_interval = (mean_difference - margin_of_error, mean_difference + margin_of_error)

        # Half width of confidence interval
        half_width = (confidence_interval[1] - confidence_interval[0]) / 2

        # Add to collections
        diffs_t[col] = t
        diffs_p[col] = p
        diffs_means[col] = mean_difference
        diffs_errors[col] = half_width
        sample_sizes[col] = [d1.shape, d2.shape]
    
    # Age ranges
    age_range = sdf_post["age"].mean() - sdf_pre["age"].mean()

    # Apply scaling to mean differences (multiply by 100 for %)
    diffs_normalized = {key: diffs_means[key] / (norm_fact[key]*age_range) * 100 \
                        for key in diffs_means.keys()}

    # Apply scaling to errors (multiply by 100 for %)
    diffs_errors_normalized = {key: np.abs(diffs_errors[key] / (norm_fact[key]*age_range)) * 100 \
                                for key in diffs_means.keys()}

    # Print sample sizes, T scores and p values
    # --------
    print(f"\n{threshold_str} ({ss}):\n")
    for key in diffs_p.keys():
        print(f"{key}: T={diffs_t[key]:.2e}; p={diffs_p[key]:.1e}")


    # Visualize
    # ------

    # Current subplot
    plt.subplot(1, 2, j+1)

    # Plot
    plt.errorbar(np.arange(0, len(cols_rel)), diffs_normalized.values(),
                 yerr=list(diffs_errors_normalized.values()),
                color=colors[j], fmt="o", capsize=5, elinewidth=1.5, capthick=1.5, zorder=2)
    
    
    
    # Formatting
    plt.xlim([plt.xlim()[0] - 0.3, plt.xlim()[1] + 0.3])
    plt.ylim([plt.ylim()[0], plt.ylim()[1]*1.1])  # Increase y lim by 10%
    plt.ylim([-2.3, 5.6])
    plt.yticks(np.arange(-2, 6, 1))

    title_symbol = r"$\mathbf{\alpha=}$" if threshold_str == "alpha" \
            else r"$\mathbf{I=}$"
    plt.title(
        title_symbol + f"{eval(threshold_str):.1f}y", fontweight="bold", pad=5)
    plt.axhline(y=0, color="black", lw=1)
    plt.xticks(np.arange(0, len(cols_rel)), col_labels.values(), fontsize=9, rotation=45)
    if j==0:
        plt.ylabel(" "*0 + "Normalized Mean Change [%]", labelpad=0)

    # Annotate p values
    for i, col in enumerate(cols_rel):
        text = ("n.s." if diffs_p[col] > 0.05 else "*" if diffs_p[col] > 0.01 \
                else " **" if diffs_p[col] > 0.001 else " ***" if diffs_p[col] > 1e-5 else "  ****")
        x = i
        y = 0.98
        plt.annotate(text, xy=[x, y], va="top", ha="center", fontsize=11, xycoords=("data", "axes fraction"))

# # Save
plt.tight_layout(w_pad=0, rect=(0, 0, 1, 1.))
# plt.savefig("/shared/home/botond/results/pagb/fig_hcp_landmark_ttests_errorbars_2panel.pdf",
#             transparent=True, dpi=300)
