"""

Created on Feb 8 2024

author: botond

This script graphs network stability as a function of age for the HCP-A dataset.

"""

# %%
import os
import numpy as np
import pandas as pd
import functools
import itertools
import glob
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import ticker
import seaborn as sns
from scipy import stats, optimize
import pingouin as pg
from sklearn import linear_model
from sklearn import preprocessing
import statsmodels.formula.api as smf

# %%
# =============================================================================
# # Setup
# =============================================================================

# Some rc formatting
plt.rcdefaults()
plt.rcParams["font.size"] = 9
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['font.family'] = "dejavu sans"

pd.options.display.float_format = '{:.3f}'.format

# Fix seed
np.random.seed(42)

# Filepaths
HOMEDIR = os.path.expanduser("~/")
SRCDIR = "/shared/datasets/public/hcp-a/"
OUTDIR = "/shared/home/botond/results/pagb/"

# %%
# Helper functions
# ---------------------------------------------------------------------------------

# Binning function
def bin_df(df, factor="age", binw=5, start=0, end=100):
    return df \
        .assign(
              **{f"{factor}_bin": pd.cut(df[factor], np.arange(start, end, binw)).astype(str)
          }) \
        .pipe(lambda df: df.assign(**{
                f"{factor}_group": df[f"{factor}_bin"] \
                    .apply(lambda x: df.groupby([f"{factor}_bin"])[factor].mean()[x])
              })) \
        .sort_values(by=factor)

# Sigmoid function
def sigmoid(x, L ,x0, k, b):
    y = L/(1 + np.exp(-k*(x - x0))) + b
    return (y)
    
# Sigmoid fitting function
def fit_sigmoid(wdf, factor1="age", factor2="instability", initial_guess=None):

    xdata = wdf[factor1]
    ydata = wdf[factor2]

    # initial_guess = [max(ydata), np.median(xdata), .5, min(ydata)]

    popt, pcov = optimize.curve_fit(sigmoid, xdata, ydata, p0=initial_guess, maxfev=int(1e4))

    # Results
    xfit = np.arange(xdata.min()-5, xdata.max()+5, step=0.01)
    yfit = sigmoid(xfit, *popt)

    infl_point_val = [popt[1], sigmoid(popt[1], *popt)]
    infl_point_err = np.sqrt(np.diag(pcov))[1] #/np.sqrt(wdf.shape[0])

    # Compute error measure (sum of squared residuals)
    ssr = np.sum((ydata - sigmoid(xdata, *popt))**2)

    return ((xfit, yfit), (infl_point_val, infl_point_err), popt, pcov, ssr)

# %%
# =============================================================================
# Analysis
# =============================================================================

# %%
# Load and transform demographics
# ------

info = pd \
    .read_csv(SRCDIR + "rsfMRI/fmriresults01.txt", delimiter="\t") \
    .iloc[1:, :] \
    [["src_subject_id", "interview_age", "sex"]] \
    .set_axis(["subject", "age", "sex"], axis=1) \
    .pipe(lambda df: df.assign(**{
        "age": df["age"].apply(lambda x: round(int(x)/12))}))


# Load and transform network instability data
# ------

# Tau
TAU = 1

# Load data
data = pd \
    .read_csv(SRCDIR + f"derivatives/network_instability/" \
              "brain_network_stability_20240415_205910.csv") \
    .rename({
            "sub": "subject",
            "subnetwork": "network",
            "stability": "instability"
                }, axis=1) \
    .query(f"tau == {TAU}") \
    .groupby(["subject", "network"]) \
    .mean() \
    .reset_index()

# Merge with info
df = pd.merge(info, data, on="subject")

# Binarize sex
df["sex"] = df["sex"].apply(lambda x: 0 if x == "F" else 1)

# Dropnas
df = df.dropna()

# Networks
networks = [label for  label in df.network.unique() if label not in ["unlabeled", "whole"]]
selected = ["whole_brain"]
# selected = ["Auditory", "Visual", "CinguloOpercular"]
# selected = networks

# Transform and select data
wdf = df \
    .query(f'network in {selected} & age < 100 & age >= 36') \
    .groupby("subject") \
    .mean(numeric_only=True) \
    .pipe(lambda df: bin_df(df, factor="age", binw=5, start=1, end=120))

# Describe characteristics
print(wdf.age.describe())
print(wdf.sex.value_counts())

# %%
# [x] Fit and plot sigmoid
# -----------------------------------------------------------------------------

# Perform fitting from multiple initializations for robust results
# ------

# Number of initizalizations
N = 20

# Collection
fitting_results = []

# Iterate through initializations
for j in range(N):
    
    # Draw a new random initialization
    initial_guess = [
        np.random.uniform(min(wdf["instability"]), max(wdf["instability"])),
        np.random.uniform(min(wdf["age"]), max(wdf["age"])),
        np.random.uniform(0, 1),
        np.random.uniform(min(wdf["instability"]), max(wdf["instability"]))]
    
    # Fit sigmoid to group
    try:
        out = fit_sigmoid(wdf, factor1="age", factor2="instability", initial_guess=initial_guess)
    
        # Append to collection
        fitting_results.append(out)

    except:
        pass
        
# Select best fit
best_fit_index = np.argmin(np.array([item[-1] for item in fitting_results]))
(xfit, yfit), (infl_point_val, infl_point_err), popt, pcov, srr = fitting_results[best_fit_index]

# Get fitted parameters
L, x0, k, b = popt

# Compute locations for landmark points
Ix = x0
ax = x0 - np.log(0.95/0.05)/k
b1x = x0 + np.log(0.95/0.05)/k

# Print
print(f"alpha: {ax:.2f}, I: {Ix:.2f}, beta1: {b1x:.2f}")

# Visualize
# --------

# Colors
# colors = ["#0E67A6", "#F5A94D", "red"]
colors = ["#0E67A6", "orange", "red"]

# Figure
plt.figure(figsize=(3.625, 2.85))
# plt.figure(figsize=(7.5, 5)
sns.lineplot(data=wdf, x="age_group", y="instability", ci=68,
             linestyle="", lw=.5, ms=3, err_style="bars", marker="o",
             err_kws={"capsize": 0, "capthick": 1.5, "linewidth": 1.5},
             markeredgecolor ="k",
             color="k", zorder=10)
plt.plot(xfit, yfit, color="k", lw=1, zorder=2)
# plt.axvline(x=infl_point_val[0], color="navy", lw=1)
# plt.errorbar(infl_point_val[0], yfit.mean()*0.99,
#              xerr=infl_point_err, color="navy", capsize=3, capthick=1.5, linewidth=1.5)

# Pretty up
# plt.title(f"HCP-A Dataset, Resting-State, Whole-Brain\nN={wdf.shape[0]}, " \
#           "length=2x13 minutes, TR=0.8s, winlen=24s")

plt.title(f"HCP-A Dataset (N={wdf.shape[0]})")
# plt.title(f"HPF={HPF}Hz, W={W}s")
plt.xlabel("Age in Years", fontsize=10)
plt.ylabel(f"Brain Network Instability") # \n(Whole-Brain)") # $\\tau={TAU}$")
# plt.xticks(np.arange(30, 100, 10))
# plt.ylabel("")
# plt.grid()

# Formatting
# plt.ylim([0.569, 0.586])
# plt.ylim([0.565, 0.595])
plt.xlim([34, 93])
# ytick_pos = [0.57, .575, 0.58, 0.585]
# plt.gca().yaxis.set_major_locator(ticker.FixedLocator(ytick_pos))
for sp in ['bottom', 'top', 'left', 'right']:
    plt.gca().spines[sp].set_linewidth(0.75)
    plt.gca().spines[sp].set_color("black")
plt.tight_layout() #rect=[ 0, 0.02, 1, 1])

# Show landmark points
# ------

# Compute the y values for landmark points
Iy = sigmoid(Ix, *popt)
ay = sigmoid(ax, *popt)
b1y = sigmoid(b1x, *popt)

# Plot
plt.scatter(ax, ay, color=colors[0], marker="o", s=30, zorder=10)
plt.scatter(Ix, Iy, color=colors[1], marker="o", s=30, zorder=10)
plt.scatter(b1x, b1y, color=colors[2], marker="o", s=30, zorder=10)

# Annotate
plt.annotate(r"$\alpha$", xy=(ax-1.5, ay-1.9e-3), color=colors[0], fontsize=14)
plt.annotate(r"$I$", xy=(Ix-1, Iy+6e-4), color=colors[1], fontsize=14)
plt.annotate(r"$\beta_{1}$", xy=(b1x-1.2, b1y+8e-4), color=colors[2], fontsize=14)

# Annotate values
plt.annotate(r"$\alpha=$"+f"{ax:.1f}y", xy=(0.1, 0.9),
xycoords="axes fraction", color="k", fontsize=10)
plt.annotate(r"$I=$"+f"{Ix:.1f}y", xy=(0.1, 0.8),
xycoords="axes fraction", color="k", fontsize=10)
plt.annotate(r"$\beta_{1}=$"+f"{b1x:.1f}y", xy=(0.1, 0.7),
xycoords="axes fraction", color="k", fontsize=10)

# Save
plt.tight_layout(rect=(0, 0, 1.02, 1))
plt.savefig(OUTDIR + "fig_hcp_sigmoid.pdf", transparent=True, dpi=300)

# %%
# F-test
# -----------------------------------------------------------------------------

# Function to get residuals for sigmoid model
def get_sigmoid_residuals(wdf, factor1="age", factor2="instability", popt=popt, pcov=pcov):

    # Unpack data
    xdata = wdf[factor1]
    ydata = wdf[factor2]

    # Results
    xfit = np.arange(xdata.min(), xdata.max(), step=0.01)
    yfit = sigmoid(xfit, *popt)

    infl_point_val = [popt[1], sigmoid(popt[1], *popt)]
    infl_point_err = np.sqrt(np.diag(pcov))[1] #/np.sqrt(wdf.shape[0])

    resid = ydata - sigmoid(xdata, *popt)
    ss_res = np.sum(resid**2)
    ss_tot = np.sum((ydata - ydata.mean())**2)
    r2 = 1 - (ss_res/ss_tot)

    return resid

# Residuals
res_model = get_sigmoid_residuals(wdf, factor1="age", factor2="instability").values
res_null = wdf["instability"].values - wdf["instability"].mean()
# (Linear model)
m, b = np.polyfit(wdf["age"].values, wdf["instability"].values, 1)
res_lin = wdf["instability"].values - (m*wdf["age"].values + b)

# Sum of squares of residuals
rss_model = np.sum(res_model**2)
rss_null = np.sum(res_null**2)
rss_lin = np.sum(res_lin**2)

# Degrees of freedom
df_model = wdf.shape[0] - 4
df_null = wdf.shape[0] - 1
df_lin = wdf.shape[0] - 2

# F-statistic
F_sig_null = ((rss_null - rss_model) / (df_null - df_model)) / (rss_model / df_model)
F_sig_lin = ((rss_lin - rss_model) / (df_lin - df_model)) / (rss_model / df_model)
F_lin_null = ((rss_null - rss_lin) / (df_null - df_lin)) / (rss_lin / df_lin)

# P values
p_value_sig_lin = 1 - stats.f.cdf(F_sig_lin, df_lin, df_model)
print("F-test (sig vs lin):", F_sig_lin, p_value_sig_lin)

p_value_lin_null = 1 - stats.f.cdf(F_lin_null, df_null, df_lin)
print("F-test (lin vs null):", F_lin_null, p_value_lin_null)

# Fit a linear model using stats models
model = smf.ols(formula="instability ~ age", data=wdf)
results = model.fit()
print(results.summary())
results.pvalues["age"]
