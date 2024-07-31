"""
Created on Dec 12 2023

@author: benett

This script investigates age trends in network instability for the Mayo dataset.

"""

# %%

import sys
import os
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, optimize
import statsmodels.formula.api as smf
import pickle

# %%
# =============================================================================
# Setup
# =============================================================================

# Filepaths
HOMEDIR = os.path.expanduser("~/")
SRCDIR = "/shared/datasets/public/mayo/derivatives/network_instability/"
OUTDIR = "/shared/home/botond/results/pagb/"

# Some rc formatting
plt.rcdefaults()
plt.rcParams["font.size"] = 9
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['font.family'] = "dejavu sans"

# Helper functions
# ------

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
    xfit = np.arange(xdata.min()-5, xdata.max(), step=0.01)
    yfit = sigmoid(xfit, *popt)

    infl_point_val = [popt[1], sigmoid(popt[1], *popt)]
    infl_point_err = np.sqrt(np.diag(pcov))[1] #/np.sqrt(wdf.shape[0])

    # Compute error measure (sum of squared residuals)
    ssr = np.sum((ydata - sigmoid(xdata, *popt))**2)

    return ((xfit, yfit), (infl_point_val, infl_point_err), popt, pcov, ssr)

# %%
# Load data
# ---------------------------------------------------------------------

# Settings
TAU = 1

# Age
demo = pd \
    .read_excel(SRCDIR + "../../sourcedata/Strey_clinical.xlsx", index_col=0) \
    .reset_index() \
    [["ID", "AgeVis", "Male"]] \
    .rename({"ID": "subject", "AgeVis": "age", "Male": "sex"}, axis=1)

# Network instability
data = pd.read_csv(SRCDIR + "brain_network_stability_20240423_202558.csv") \
    .rename({"sub": "subject"}, axis=1)

# Network
selected = ["whole_brain"]
selected = ["Auditory", "Visual", "CinguloOpercular"]

# Filter data
data_filtered = data \
    .query(f"subnetwork == {selected}") \
    .drop(["subnetwork"], axis=1) \
    .groupby(["subject", "tau"]) \
    .mean() \
    .drop(["time"], axis=1) \
    .reset_index() \
    .query("tau == @TAU") \
    .drop(["tau"], axis=1)

# Merge with demo
df = pd.merge(demo, data_filtered, on="subject", how="inner")

# Bin age
df = df.pipe(lambda df: bin_df(df, factor="age", binw=5, start=0, end=120))

# Age filter
# df = df.query("age >= 50")

# Describe characteristics
print(df.age.describe())
print(df.sex.value_counts())

# %%
# =============================================================================
# [x] Analysis
# =============================================================================

# Perform fitting from multiple initializations for robust results
# ------

# Number of initizalizations
N = 1

# Collection
fitting_results = []

# Iterate through initializations
for j in range(N):
    
    # Draw a new random initialization
    initial_guess = [
        np.random.uniform(min(df["stability"]), max(df["stability"])),
        np.random.uniform(min(df["age"]), max(df["age"])),
        np.random.uniform(0, 1),
        np.random.uniform(min(df["stability"]), max(df["stability"]))]
    
    # Fit sigmoid to group
    try:
        out = fit_sigmoid(df, factor1="age", factor2="stability", initial_guess=initial_guess)
    
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
colors = ["#0E67A6", "#F5A94D", "red"]

# Figure
plt.figure(figsize=(3.625, 2.25))

# Plot
# plt.figure(figsize=(7.5, 5)
sns.lineplot(data=df, x="age_group", y="stability", ci=68,
             linestyle="", lw=.5, ms=3, err_style="bars", marker="o",
            err_kws={"capsize": 0, "capthick": 1.5, "linewidth": 1.5},
             markeredgecolor ="k",
             color="k", zorder=10)
plt.plot(xfit, yfit, color="red", lw=1, zorder=2)
# plt.axvline(x=infl_point_val[0], color="navy", lw=1)
# plt.errorbar(infl_point_val[0], yfit.mean()*0.995,
#              xerr=infl_point_err, color="navy", capsize=3, capthick=1.5, linewidth=1.5)

# Format
plt.title(f"Mayo Dataset (N={df.shape[0]:,})")
# plt.title(f"HPF={HPF}Hz, W={W}s")
plt.xlabel("Age in Years")
plt.ylabel(f"Brain Network Instability") # $\\tau={TAU}$")
# plt.grid()

# Formatting
plt.xticks(np.arange(20, 100, 10))
plt.xlim([15, 92])
# plt.xticks(np.arange(30, 100, 10))
# plt.yticks(np.arange(0.375, 0.395, 0.005))
plt.ylim([0.3718, 0.3952])
for sp in ['bottom', 'top', 'left', 'right']:
    plt.gca().spines[sp].set_linewidth(.75)
    plt.gca().spines[sp].set_color("black")
plt.tight_layout() #rect=[ 0, 0.02, 1, 1])

# Show landmark points
# ------

# Compute the y values for landmark points
Iy = sigmoid(Ix, *popt)
ay = sigmoid(ax, *popt)
b1y = sigmoid(b1x, *popt)

# # Plot
# plt.scatter(ax, ay, color=colors[0], marker="o", s=30, zorder=10)
# plt.scatter(Ix, Iy, color=colors[1], marker="o", s=30, zorder=10)
# plt.scatter(b1x, b1y, color=colors[2], marker="o", s=30, zorder=10)

# # Annotate
# plt.annotate(r"$\alpha$", xy=(ax-0.8, ay+3e-4), color=colors[0], fontsize=14)
# plt.annotate(r"$I$", xy=(Ix-0.8, Iy+2e-4), color=colors[1], fontsize=14)
# plt.annotate(r"$\beta$", xy=(b1x-0.8, b1y+4e-4), color=colors[2], fontsize=14)

# # Annotate values
# plt.annotate(r"$\alpha=$"+f"{ax:.1f}y", xy=(0.1, 0.9), xycoords="axes fraction", color="k", fontsize=10)
# plt.annotate(r"$I=$"+f"{Ix:.1f}y", xy=(0.1, 0.8), xycoords="axes fraction", color="k", fontsize=10)
# plt.annotate(r"$\beta=$"+f"{b1x:.1f}y", xy=(0.1, 0.7), xycoords="axes fraction", color="k", fontsize=10)

# Save
plt.tight_layout(rect=(0, 0, 1.02, 1))
plt.savefig(OUTDIR + "fig_mayo_sigmoid_subnetwork.pdf", transparent=True, dpi=300)

# Pickle results
res = {}
res["df"] = df
res["xfit"] = xfit
res["yfit"] = yfit
with open(OUTDIR + "data_mayo_subnetwork.pkl", "wb") as f:
    pickle.dump(res, f)


# %%
# F-test
# -----------------------------------------------------------------------------

# Sigmoid fitting function
def get_sigmoid_residuals(wdf, factor1="age", factor2="instability"):

    def sigmoid(x, L ,x0, k, b):
        y = L / (1 + np.exp(-k*(x-x0)))+b
        return (y)

    xdata = wdf[factor1]
    ydata = wdf[factor2]

    initial_guess = [max(ydata), np.median(xdata), 1, min(ydata)]

    popt, pcov = optimize.curve_fit(sigmoid, xdata, ydata, p0=initial_guess, maxfev=int(1e4))

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
res_model = get_sigmoid_residuals(df, factor1="age", factor2="stability").values
res_null = df["stability"].values - df["stability"].mean()
# (Linear model)
m, b = np.polyfit(df["age"].values, df["stability"].values, 1)
res_lin = df["stability"].values - (m*df["age"].values + b)

# Sum of squares of residuals
rss_model = np.sum(res_model**2)
rss_null = np.sum(res_null**2)
rss_lin = np.sum(res_lin**2)

# Degrees of freedom
df_model = df.shape[0] - 4
df_null = df.shape[0] - 1
df_lin = df.shape[0] - 2

# F-statistic
F_sig_null = ((rss_null - rss_model) / (df_null - df_model)) / (rss_model / df_model)
F_sig_lin = ((rss_lin - rss_model) / (df_lin - df_model)) / (rss_model / df_model)
F_lin_null = ((rss_null - rss_lin) / (df_null - df_lin)) / (rss_lin / df_lin)

# P values
p_value_sig_lin = 1 - stats.f.cdf(F_sig_lin, df_lin, df_model)
print("F-test (sig vs lin):", F_sig_lin, p_value_sig_lin)

p_value_lin_null = 1 - stats.f.cdf(F_lin_null, df_null, df_lin)
print("F-test (lin vs null):", F_lin_null, p_value_lin_null)

p_value_sig_null = 1 - stats.f.cdf(F_sig_null, df_null, df_model)
print("F-test (sig vs null):", F_sig_null, p_value_sig_null)

# Fit a linear model using stats models
model = smf.ols(formula="stability ~ age", data=df)
results = model.fit()
print(results.summary())
results.pvalues["age"]


# %%
# =============================================================================
# DEV: T2DM
# =============================================================================

# T2DM info
demo = pd \
    .read_excel(SRCDIR + "../../sourcedata/Strey_clinical.xlsx", index_col=0) \
    .reset_index() \
    [["ID", "AgeVis", "Any_E4", "HbA1c", "DMtypeGS"]] \
    .rename({"ID": "subject", "AgeVis": "age", "Any_E4": "apoe4",
             "HbA1c": "hba1c", "DMtypeGS": "diab"}, axis=1)

# Drop T1DM
demo = demo.query("diab != '1=TYPE 1'")

# Relabel columns
demo["diab"] = demo["diab"].apply(lambda x: "T2DM+" if x == "2=TYPE 2" else "Control")
demo["apoe4"] = demo["apoe4"].apply(lambda x: "ε4+" if x == "1=Yes" else "Control")

# Merge with data
df = pd.merge(demo, data_filtered, on="subject", how="inner")
df.value_counts("apoe4")
# Bin age
df = df.pipe(lambda df: bin_df(df, factor="age", binw=5, start=1, end=120))

# Age filter
df = df.query("age >= 52")

# Figure
plt.figure(figsize=(3.625, 2.85))

# Plot
sns.lineplot(data=df, x="age_group", y="stability", hue="apoe4", ci=68, marker="o",
             lw=1.5, ms=5, palette=["black", "red"], err_style="bars", markeredgewidth=0,
             err_kws={"capsize": 0, "capthick": 2.5, "linewidth": 1.5, "zorder":-1})

# Format
plt.xlabel("Age in Years")
plt.ylabel(f"Brain Network Instability") # $\\tau={TAU}$")e
plt.title(f"Mayo Dataset (N={df.shape[0]:,})")
plt.legend(title=None)

# Spines
for sp in ['bottom', 'top', 'left', 'right']:
    plt.gca().spines[sp].set_linewidth(.75)
    plt.gca().spines[sp].set_color("black")

# Layout
plt.tight_layout()
