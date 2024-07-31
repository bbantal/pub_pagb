"""
Created on Nov 30 2023

@author: botond

"""

# %%

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
from scipy import stats
import pingouin as pg
import statsmodels.formula.api as smf

# %%
# =============================================================================
# Setup
# =============================================================================

# Settings
pd.options.display.float_format = '{:.3f}'.format

# Some rc formatting
plt.rcdefaults()
plt.rcParams["font.size"] = 9
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['font.family'] = "dejavu sans"

# Filepaths
HOMEDIR = os.path.expanduser("~/")
SRCDIR = "/shared/datasets/private/keck_bolus/derivatives/network_instability/"
OUTDIR = "/shared/home/botond/results/pagb/"

# Load data
# --------

# Load meta
meta = pd.read_csv(SRCDIR + "../../participants.csv")

# Load instability data
data_raw = pd \
    .read_csv(SRCDIR + "brain_network_stability_20231210_004_30.csv") \
    .groupby(["sub", "ses", "task", "run", "tau", "subnetwork"]) \
    .mean() \
    ["stability"] \
    .reset_index() \
    .rename({"sub": "subject",
            "ses": "session",
            "subnetwork": "network",
            "stability": "instability"},
            axis=1)

            
# Merge data with demo
df = pd.merge(meta[["subject", "age"]], data_raw, on="subject", how="inner")

# %%
# Helper functions
# -----------------------------------------------------------------------------

# Binning function
def bin_df(df, factor="age", binw=5, start=0, end=100):
    return df \
        .assign(
            **{f"{factor}_bin": pd.cut(df[factor], np.arange(start, end, binw), right=False).astype(str)
        }) \
        .pipe(lambda df: df.assign(**{
                f"{factor}_group": df[f"{factor}_bin"] \
                    .apply(lambda x: df.groupby([f"{factor}_bin"])[factor].mean()[x])
            })) \
        .sort_values(by=factor)

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
# Analyze
# =============================================================================

# Settings
TAU = 1
TASK = "rest"
LB = 0
UB = 100

# Subjects to exclude
subjects_to_exclude = [113, 117, 147]  # FD<0.5, rest

# Subnetwork
selected = ["whole_brain"]
# selected = ["Auditory", "Visual", "CinguloOpercular"]

# Color palettes
palette = sns.color_palette(["red", "gray"])

# Transform and select data
wdf = df \
    .query(f"subject not in {subjects_to_exclude}") \
    .query(f'tau == {TAU} & task == "{TASK}"') \
    .sort_values(by="network") \
    .query(f'network in {selected}') \
    .groupby(["subject",  "session", "run"]) \
    .mean(numeric_only=True) \
    .reset_index() \
    .query(f'(age >= {LB}) & (age < {UB})') \
    .pipe(lambda df: bin_df(df, factor="age", binw=20, start=20)) \
    .set_index(["session", "subject", "age", "age_bin", "age_group"]) \
    .pivot(columns="run") \
    .swaplevel(i=0, j=1, axis=1) \
    .pipe(lambda df: df[2] - df[1]) \
    .reset_index() \
    .dropna() \
    .drop(["tau"], axis=1) \
    .sort_values(by="age")

# Sample sizes
full_sample_size = int(wdf.shape[0]/2)
print(wdf.groupby("age_bin")["subject"].nunique())

# Age distribution
print(wdf.age.describe())

# Get sex distribution
print(wdf.merge(meta[["subject", "sex"]], on="subject")["sex"].value_counts()/2)

# Apply outlier removal based on IQR
mask = wdf.groupby(["age_bin", "session"])["instability"].transform(lambda x: ~iqr_removal(x.values))
print(f"Excluded images: {mask.sum()}, subjects: {wdf[mask]['subject'].values}")
wdf = wdf.query(f'subject not in {list(wdf[mask]["subject"].values)}')


# %%
# Statistics
# -----------------------------------------------------------------------------

# Collections
stats_bl = {}
stats_ses = {}
group_age_vals = {}
group_sample_sizes = {}

# Iterate through age groups, run stats
for i, (group, sdf) in enumerate(wdf.groupby("age_bin")):

    # Compute statistics for within session comparisons (post vs pre bolus)
    stats_bl[group + " - bhb"] = pg.ttest(x=sdf.query("session == 'bhb'")["instability"], y=0)
    stats_bl[group + " - glc"] = pg.ttest(x=sdf.query("session == 'glc'")["instability"], y=0)

    # Compute statistics for between session comparisons (glc vs bhb)
    stats_ses[group] = pg.pairwise_tests(data=sdf, within="session", dv="instability",
                                         subject="subject")

    # Store average age of age group
    group_age_vals[group] = np.mean([float(val) for val in group[1:-1].split(", ")])
    
    # Store sample size
    group_sample_sizes[group] = sdf.shape[0]/2

# %%
# Visualize
# -----------------------------------------------------------------------------

# Figure
plt.figure(figsize=(3.625, 2.8))

# Plot
sns.lineplot(data=wdf, hue="session", y="instability", x="age_bin", ci=68, marker="o",
             hue_order=["glc", "bhb"], lw=1.5, ms=5, palette=["black", "red"], err_style="bars", markeredgewidth=0,
             err_kws={"capsize": 0, "capthick": 2.5, "linewidth": 1.5, "zorder":-1})
plt.axhline(0, linestyle="--", color="black", zorder=4)

# Legend
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles=handles[::1], labels=['GLC', 'D-βHB'], fontsize=8.2, frameon=False)

# legend_elements = [Patch(facecolor='crimson', edgecolor='black', label='D-βHB'),
#                     Patch(facecolor='black', edgecolor='black', label='Glucose')]
# plt.legend(handles=legend_elements[::-1], loc=2, fontsize=8)

# Annotate statistics in three lines
# -----

# Preparations
assign_stars = lambda pval: "n.s." if pval > 0.05 else "*" if pval > 0.01 else "**" \
        if pval > 0.001 else "***" if pval > 0.0001 else "****"
x_val_bin_index = dict(zip(sorted(wdf["age_bin"].unique()), np.arange(1e4)))

# Iterate over groups
for i, group_key in enumerate(stats_ses):

    # Get text for GLC
    pval = stats_bl[group_key + ' - glc']['p-val'].values[0]
    text1 = f"GLC: {assign_stars(pval)}"
    
    # f"GLC: p={pval:.1g}{assign_stars(pval)}"

    # Get text for BHB
    pval = stats_bl[group_key + ' - bhb']['p-val'].values[0]
    text2 = f"D-βHB: {assign_stars(pval)}"

    # Get text for between sessions
    pval = stats_ses[group_key]['p-unc'].values[0]
    text3 = f"△: {assign_stars(pval)}"

    # Annotate
    plt.annotate(
        "\n".join([text1, text2, text3]),
        xy=(x_val_bin_index[group_key]-0.25*i, [0.05, 0.335, 0.05][i]),
        xycoords=("data", "axes fraction"), ha="left", fontsize=8, linespacing=1.2)

# Format
# plt.title(f"Metabolic Intervention Dataset (N={int(wdf.shape[0]/2)})")
plt.title(f"Metabolic Intervention Dataset (N={full_sample_size})") 
plt.xlabel("Age in Years")
plt.ylabel(f"$\Delta$ Brain Network Instability") # $\\tau={TAU}$")

# Spines
for sp in ['bottom', 'top', 'left', 'right']:
    plt.gca().spines[sp].set_linewidth(.75)
    plt.gca().spines[sp].set_color("black")

# Layout
plt.tight_layout()

# # Save
plt.savefig(OUTDIR + f"fig_bolus_ns.pdf", transparent=True, dpi=300)

# %%
# Print statistics for every group
# -------------------
for group_key in stats_ses:
    print(f"Age group: {group_key}")
    print("GLC:", [f"{val:.2g}" for val in stats_bl[group_key + ' - glc'][['T', 'p-val']].values[0]])
    print("BHB:", [f"{val:.2g}" for val in stats_bl[group_key + ' - bhb'][['T', 'p-val']].values[0]])
    print("△:", [f"{val:.2g}" for val in stats_ses[group_key][['T', 'p-unc']].values[0]])
    print("\n")

# %%
# =============================================================================
# Checkig Motion as a potential confounder
# =============================================================================

from matplotlib.ticker import MultipleLocator

# Settings
THR = 0.5  # mm
SES, color, color2, title= "bhb", "tomato", "red", "D-βHB"
# SES, color, color2, title= "glc", "gray", "black", "GLC"

# Load motion
motion = pd \
    .read_csv(f"/shared/home/botond/results/keck_bolus/reports/motion_fdthr_{THR}.csv", index_col=0)

# Transform motion
motion = motion \
    .set_index(
        pd.MultiIndex.from_tuples(
            motion["subject"] \
                .apply(lambda item: [x.split("-")[1] for x in item.split("_")]),
                names=["sub", "session", "task", "run"]
                )) \
    .drop("subject", axis=1) \
    .reset_index() \
    .pipe(lambda df: df.assign(**{"subject": df["sub"].astype(int), "run": df["run"].astype(int)})) \
    .drop("sub", axis=1) \
    .query(f'task == "{TASK}" & session == "{SES}"')


# Baseline motion
sdf_blc = motion \
    .set_index(["subject"]) \
    .pivot(columns="run") \
    [["FD_mean"]] \
    .swaplevel(i=0, j=1, axis=1) \
    .pipe(lambda df: df[2] - df[1]) \
    .reset_index() \
    .merge(wdf.query(f"session=='{SES}'"), on="subject", how="inner") 

# Whole sample
# --------------

# Remove outliers
sdf_blc = sdf_blc[iqr_removal(sdf_blc["FD_mean"])]

# Stats
r, p = stats.pearsonr(sdf_blc["FD_mean"], sdf_blc["instability"])

# Visualize
plt.figure(figsize=(3.6, 3.625))
sns.regplot(data=sdf_blc, x="FD_mean", y="instability",
            scatter_kws={"s": 25, "edgecolor": "black", "linewidth": 1., "color": color},
            line_kws={"color": color2, "lw": 2})
plt.annotate(f"r={r:.2f}\np={p:.2g}", xy=[0.7, 0.05], xycoords="axes fraction")
plt.gca().xaxis.set_major_locator(MultipleLocator(0.05))
plt.title(f"Bolus: {title}")
plt.xlabel("Δ Framewise Displacement")
plt.ylabel("Δ Brain Network Instability")

# Save
plt.tight_layout()
plt.savefig(f"/shared/home/botond/results/pagb/fig_motion_vs_instability_{SES}.pdf",
            transparent=True, dpi=300)


# %%
# =============================================================================
# Test-retest reliability
# =============================================================================

# %%
# Test-retest correlation
# -------------

# Settings
TAU = 1
TASK = "rest"
# TASK = "task"
LB = 00
UB = 100

# Subjects to exclude
subjects_to_exclude = [113, 117, 147]  # FD<0.5, rest, BHB

# Subnetwork
selected = ["whole_brain"]
# selected = ["Auditory", "Visual", "CinguloOpercular"]

# Color palettes
palette = sns.color_palette(["red", "gray"])

# Transform and select data
wdf = df \
    .query(f"subject not in {subjects_to_exclude}") \
    .query(f'tau == {TAU} & task == "{TASK}"') \
    .sort_values(by="network") \
    .query(f'network in {selected}') \
    .groupby(["subject",  "session", "run"]) \
    .mean(numeric_only=True) \
    .reset_index() \
    .query(f'(age >= {LB}) & (age < {UB})') \
    .pipe(lambda df: bin_df(df, factor="age", binw=20, start=20)) \
    .query('run == 1') \
    .sort_values(by="subject")

# Extract data to compare
x = wdf.query("session == 'bhb'")["instability"]
y = wdf.query("session == 'glc'")["instability"]

# Remove outliers
maskx = iqr_removal(x.values)
masky = iqr_removal(y.values)
# x = x[maskx&masky]
# y = y[maskx&masky]


# T-test
x.mean()
y.mean()
ttest = stats.ttest_rel(x, y)
print(ttest)
# plt.boxplot([x,y])

# Test-retest correlation
r, p = stats.pearsonr(x, y)

# Make pretty scatterplot
plt.figure(figsize=(3.625, 3.625))
plt.scatter(x, y, color="skyblue", linewidth=1, edgecolor="black", s=30)
sns.regplot(x=x, y=y, color="navy", scatter=False, ci=None, line_kws={"linewidth": 1})
plt.xlabel("Pre D-βHB Bolus (Day A)")
plt.ylabel("Pre GLC Bolus (Day B)")
plt.xlim([0.505, 0.585])
plt.ylim([0.505, 0.585])
plt.xticks(np.arange(0.51, 0.59, 0.01))
plt.yticks(np.arange(0.51, 0.59, 0.01))
plt.title("Test-retest Correlation of\nBrain Network Instability")
plt.annotate(f"r={r:.2f}\np={p:.2g}", xy=(0.75, 0.05), xycoords="axes fraction", fontsize=10)
plt.tight_layout()
plt.savefig("/shared/home/botond/results/pagb/fig_test_retest.pdf", transparent=True, dpi=300)

# %%
# SI: Comparisons between age groups
# ---------------------------

SES = "bhb"

# Compute ANOVA
anova = pg.anova(data=wdf.query(f"session == '{SES}'"), dv="instability", between="age_bin", detailed=True)

# Post-hoc
posthoc = pg.pairwise_tests(data=wdf.query(f"session == '{SES}'"), dv="instability", between="age_bin", padjust="bonf")

# Print
print(anova)
print(posthoc)

# Visualize
# -------

# Open figure
plt.figure(figsize=(3.625, 3.625))

# Plot
sns.swarmplot(data=wdf.query(f"session == '{SES}'"),
              x="age_bin", y="instability", palette=["lightblue", "mediumseagreen", "darkorange"],
              s=5, edgecolor='black', linewidth=1)

# Add line at 0
plt.axhline(0, linestyle="--", color="black", zorder=0)

# Add signifiance bars
yshift = 0.0  # Shift for glc
plt.text(0.5, 0.032+yshift, f"n.s.", ha="center", va="center", fontsize=12)
plt.annotate("", xy=(0, 0.028+yshift), xytext=(1, 0.028+yshift), arrowprops=dict(arrowstyle="-", lw=1.5))
plt.text(1.5, 0.030+yshift, f"*", ha="center", va="center", fontsize=12)
plt.annotate("", xy=(1, 0.028+yshift), xytext=(2, 0.028+yshift), arrowprops=dict(arrowstyle="-", lw=1.5))
plt.text(1, 0.040+yshift, f"n.s.", ha="center", va="center", fontsize=12)
plt.annotate("", xy=(0, 0.036+yshift), xytext=(2, 0.036+yshift), arrowprops=dict(arrowstyle="-", lw=1.5))

# Format
plt.title(f"D-βHB Bolus")
# plt.title(f"GLC Bolus")
plt.xlabel("Age group")
plt.ylabel(f"$\Delta$ Brain Network Instability")
plt.ylim([-0.062, 0.045+yshift])
plt.tight_layout()

# Save
plt.savefig(f"/shared/home/botond/results/pagb/fig_age_comparisons_{SES}.pdf", transparent=True, dpi=300)
