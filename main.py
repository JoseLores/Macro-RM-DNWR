# Author: Jose Lores Diz

import matplotlib.pyplot as plt
from matplotlib import rc
import econpizza as ep
import os
import copy
import numpy as np
import pandas as pd
from utilities import _create_directory

# Control #

# Models
baseline = "./models/baseline_model.yaml"
dnwr = "./models/baseline_model_dnwr.yaml"

# Shock
shk = ("e_beta", 0.02)
shockname = "positive_beta_shock"  # folder for the standard irf
Simulation_dir = "Simulation"  # folder for the simulation
new_shockname = shockname + "_tight"  # folder for the irfs with stronger CB responses
ZLB = "stronger_shock_hit_ZLB"

# Variables to plot IRFs
variables = "y", "pi", "R", "w", "n", "k", "i", "c", "mc", "Rk", "RR", "hhdf"
var_names = (
    "Output",
    "Inflation",
    "Nominal Interest Rate",
    "Real Wages",
    "Labor",
    "Capital",
    "Investment",
    "Consumption",
    "Marginal Costs",
    "Rk",
    "Real Interest Rate",
    "Household Discount Factor",
)

# Style for plots
plt.rc("font", family="serif")
plt.rcParams["font.size"] = "14"

# Solve #
# Load baseline
mod_b = ep.load(baseline)
_ = mod_b.solve_stst()
xSS = mod_b["stst"].copy()

# Load new model
mod_dnwr = ep.load(dnwr)
_ = mod_dnwr.solve_stst()
ySS = mod_dnwr["stst"].copy()

# Change Steady State inflation (CB's target)
dnwr_dict0 = ep.parse(dnwr)
dnwr_dict1 = copy.deepcopy(dnwr_dict0)
dnwr_dict1["steady_state"]["fixed_values"]["pi"] = 1
mod_dnwr2 = ep.load(dnwr_dict1)
_ = mod_dnwr2.solve_stst()
zSS = mod_dnwr2["stst"].copy()

# Find IRFs
x, flag_x = mod_b.find_path(shock=shk)
y, flag_y = mod_dnwr.find_path(shock=shk)
z, flag_z = mod_dnwr2.find_path(shock=shk)

# Find variables index
inds_b = [mod_b["variables"].index(v) for v in variables]
inds_dnwr = [mod_dnwr["variables"].index(v) for v in variables]
inds_dnwr2 = [mod_dnwr2["variables"].index(v) for v in variables]

# Directories for saving the plots
_create_directory(shockname)

# Plot #

# Produce the IRFs

for i in range(len(variables)):
    # We dont want interest rate as a percentage deviation, only as deviation
    if variables[i] == "R" or variables[i] == "hhdf" or variables[i] == "beta":
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(
            (x[0:30, inds_b[i]] - xSS[variables[i]]) * 100,
            marker="o",
            linestyle="-",
            label="baseline",
            color="rebeccapurple",
            alpha=0.9,
        )
        ax.plot(
            (y[0:30, inds_dnwr[i]] - ySS[variables[i]]) * 100,
            marker="d",
            linestyle="-",
            label="dwnr 2% inflation target",
            color="darkred",
            alpha=0.9,
        )
        ax.plot(
            (z[0:30, inds_dnwr2[i]] - zSS[variables[i]]) * 100,
            marker="^",
            linestyle="-",
            label="dwnr 0% inflation target",
            color="darkslategrey",
            alpha=0.9,
        )
        ax.set_title(var_names[i], size="18")
        ax.set_xlabel("Quarters")
        ax.set_ylabel("Absolute Deviation")
        ax.legend()
        plt.savefig(os.path.join("bld", new_shockname, var_names[i] + ".pdf"))
        plt.close()
    else:
        fig, ax = plt.subplots(figsize=(8, 5))
        # plot as % deviation from Steady State
        ax.plot(
            (x[0:30, inds_b[i]] - xSS[variables[i]]) / xSS[variables[i]] * 100,
            marker="o",
            linestyle="-",
            label="baseline",
            color="rebeccapurple",
            alpha=0.9,
        )
        ax.plot(
            (y[0:30, inds_dnwr[i]] - ySS[variables[i]]) / ySS[variables[i]] * 100,
            marker="d",
            linestyle="-",
            label="dwnr 2% inflation target",
            color="darkred",
            alpha=0.9,
        )
        ax.plot(
            (z[0:30, inds_dnwr2[i]] - zSS[variables[i]]) / zSS[variables[i]] * 100,
            marker="^",
            linestyle="-",
            label="dwnr 0% inflation target",
            color="darkslategrey",
            alpha=0.9,
        )
        ax.set_title(var_names[i], size="18")
        ax.set_xlabel("Quarters")
        ax.set_ylabel("Percent")
        ax.legend()
        plt.savefig(os.path.join("bld", shockname, var_names[i] + ".pdf"))
        plt.close()

#########################################################################################
# Simulate Phillips
_create_directory(Simulation_dir)
# Seed
np.random.seed(1)

# Get index for Phillips
y_ind = []
pi_ind = []
for i in [mod_b, mod_dnwr, mod_dnwr2]:
    y_ind.append(i["variables"].index("y"))
    pi_ind.append(i["variables"].index("pi"))

# Shock
shocks = np.random.normal(0, 0.995 * 0.02, size=1000)

# Get Data
A = np.empty(6)

for s in shocks:
    shk = ("e_beta", s)
    x, flag_x = mod_b.find_path(shock=shk)
    y, flag_y = mod_dnwr.find_path(shock=shk)
    z, flag_z = mod_dnwr2.find_path(shock=shk)

    rows = np.column_stack(
        [
            x[1:2, y_ind[0]],
            y[1:2, y_ind[1]],
            z[1:2, y_ind[2]],
            x[1:2, pi_ind[0]],
            y[1:2, pi_ind[1]],
            z[1:2, pi_ind[2]],
        ]
    )

    A = np.vstack([A, rows])

df = pd.DataFrame(
    A, columns=["y_baseline", "y_dnwr", "y_dnwr2", "pi_baseline", "pi_dnwr", "pi_dnwr2"]
)
df = df.iloc[1:]

df.to_csv("./bld/Simulation/simulation_data.csv")

# Get deviations from Steady State
df["y_baseline"] = (df["y_baseline"] - xSS["y"]) / xSS["y"] * 100
df["y_dnwr"] = (df["y_dnwr"] - ySS["y"]) / ySS["y"] * 100
df["y_dnwr2"] = (df["y_dnwr2"] - zSS["y"]) / zSS["y"] * 100

# Sort them for the plot
df_baseline = df[["y_baseline", "pi_baseline"]]
df_baseline = df_baseline.sort_values(["y_baseline"])

df_dnwr = df[["y_dnwr", "pi_dnwr"]]
df_dnwr = df_dnwr.sort_values(["y_dnwr"])

df_dnwr2 = df[["y_dnwr2", "pi_dnwr2"]]
df_dnwr2 = df_dnwr2.sort_values(["y_dnwr2"])


# Plot #

fig, ax = plt.subplots(figsize=(8, 5))

ax.scatter(
    df_baseline["y_baseline"],
    df_baseline["pi_baseline"],
    label="baseline",
    color="rebeccapurple",
    marker=".",
)

ax.scatter(
    df_dnwr["y_dnwr"],
    df_dnwr["pi_dnwr"],
    label="dwnr 2% inflation target",
    color="darkred",
    marker=".",
)

ax.scatter(
    df_dnwr2["y_dnwr2"],
    df_dnwr2["pi_dnwr2"],
    label="dwnr 0% inflation target",
    color="darkslategrey",
    marker=".",
)

ax.set_ylim(bottom=0.99, top=1.05)
ax.set_xlim(left=-2.5, right=2)
ax.set_xlabel("Output Gap in %")
ax.set_ylabel("Quarterly Inflation")
ax.legend()
plt.savefig(os.path.join("bld", Simulation_dir, "Phillips.pdf"))
plt.close()

###################################################################
# Create plots for stronger monetary policy responses
_create_directory(new_shockname)
# Recall to avoid problems with the simulation shocks
shk = ("e_beta", 0.02)

# Change Steady State inflation (CB's target)
dnwr_dict2 = ep.parse(dnwr)
dnwr_dict_baseline = ep.parse(baseline)

# 0% target and dnwr
dnwr_dict1["steady_state"]["fixed_values"]["phi_pi"] = 5

# 2% target and dnwr
dnwr_dict2["steady_state"]["fixed_values"]["phi_pi"] = 5

# baseline
dnwr_dict_baseline["steady_state"]["fixed_values"]["phi_pi"] = 5

# Load baseline
mod_b = ep.load(dnwr_dict_baseline)
_ = mod_b.solve_stst()
xSS = mod_b["stst"].copy()

# Load new model
mod_dnwr = ep.load(dnwr_dict2)
_ = mod_dnwr.solve_stst()
ySS = mod_dnwr["stst"].copy()

# Load 0% target model
mod_dnwr2 = ep.load(dnwr_dict1)
_ = mod_dnwr2.solve_stst()
zSS = mod_dnwr2["stst"].copy()

# Find IRFs
x, flag_x = mod_b.find_path(shock=shk)
y, flag_y = mod_dnwr.find_path(shock=shk)
z, flag_z = mod_dnwr2.find_path(shock=shk)

# Find variables index
inds_b = [mod_b["variables"].index(v) for v in variables]
inds_dnwr = [mod_dnwr["variables"].index(v) for v in variables]
inds_dnwr2 = [mod_dnwr2["variables"].index(v) for v in variables]

# Plot #

# Produce the IRFs

for i in range(len(variables)):
    # We dont want interest rate as a percentage deviation, only as deviation
    if variables[i] == "R" or variables[i] == "hhdf" or variables[i] == "beta":
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(
            (x[0:30, inds_b[i]] - xSS[variables[i]]) * 100,
            marker="o",
            linestyle="-",
            label="baseline",
            color="rebeccapurple",
            alpha=0.9,
        )
        ax.plot(
            (y[0:30, inds_dnwr[i]] - ySS[variables[i]]) * 100,
            marker="d",
            linestyle="-",
            label="dwnr 2% inflation target",
            color="darkred",
            alpha=0.9,
        )
        ax.plot(
            (z[0:30, inds_dnwr2[i]] - zSS[variables[i]]) * 100,
            marker="^",
            linestyle="-",
            label="dwnr 0% inflation target",
            color="darkslategrey",
            alpha=0.9,
        )
        ax.set_title(var_names[i], size="18")
        ax.set_xlabel("Quarters")
        ax.set_ylabel("Absolute Deviation")
        ax.legend()
        plt.savefig(os.path.join("bld", new_shockname, var_names[i] + ".pdf"))
        plt.close()
    else:
        fig, ax = plt.subplots(figsize=(8, 5))
        # plot as % deviation from Steady State
        ax.plot(
            (x[0:30, inds_b[i]] - xSS[variables[i]]) / xSS[variables[i]] * 100,
            marker="o",
            linestyle="-",
            label="baseline",
            color="rebeccapurple",
            alpha=0.9,
        )
        ax.plot(
            (y[0:30, inds_dnwr[i]] - ySS[variables[i]]) / ySS[variables[i]] * 100,
            marker="d",
            linestyle="-",
            label="dwnr 2% inflation target",
            color="darkred",
            alpha=0.9,
        )
        ax.plot(
            (z[0:30, inds_dnwr2[i]] - zSS[variables[i]]) / zSS[variables[i]] * 100,
            marker="^",
            linestyle="-",
            label="dwnr 0% inflation target",
            color="darkslategrey",
            alpha=0.9,
        )
        ax.set_title(var_names[i], size="18")
        ax.set_xlabel("Quarters")
        ax.set_ylabel("Percent")
        ax.legend()
        plt.savefig(os.path.join("bld", new_shockname, var_names[i] + ".pdf"))
        plt.close()

###################################################################
# Stronger Demand shock to hit the ZLB
_create_directory(ZLB)

shk = ("e_beta", 0.1)

# Load Again to reset parameters
# Load baseline
mod_b = ep.load(baseline)
_ = mod_b.solve_stst()
xSS = mod_b["stst"].copy()

# Load new model
mod_dnwr = ep.load(dnwr)
_ = mod_dnwr.solve_stst()
ySS = mod_dnwr["stst"].copy()

# Change Steady State inflation (CB's target)
dnwr_dict0 = ep.parse(dnwr)
dnwr_dict1 = copy.deepcopy(dnwr_dict0)
dnwr_dict1["steady_state"]["fixed_values"]["pi"] = 1
mod_dnwr2 = ep.load(dnwr_dict1)
_ = mod_dnwr2.solve_stst()
zSS = mod_dnwr2["stst"].copy()

# Find IRFs
x, flag_x = mod_b.find_path(shock=shk)
y, flag_y = mod_dnwr.find_path(shock=shk)
z, flag_z = mod_dnwr2.find_path(shock=shk)

# Find variables index
inds_b = [mod_b["variables"].index(v) for v in variables]
inds_dnwr = [mod_dnwr["variables"].index(v) for v in variables]
inds_dnwr2 = [mod_dnwr2["variables"].index(v) for v in variables]

# Plot #

# Produce the IRFs

for i in range(len(variables)):
    # We dont want interest rate as a percentage deviation, only as deviation
    if variables[i] == "R" or variables[i] == "hhdf" or variables[i] == "beta":
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(
            (x[0:30, inds_b[i]] - xSS[variables[i]]) * 100,
            marker="o",
            linestyle="-",
            label="baseline",
            color="rebeccapurple",
            alpha=0.9,
        )
        ax.plot(
            (y[0:30, inds_dnwr[i]] - ySS[variables[i]]) * 100,
            marker="d",
            linestyle="-",
            label="dwnr 2% inflation target",
            color="darkred",
            alpha=0.9,
        )
        ax.plot(
            (z[0:30, inds_dnwr2[i]] - zSS[variables[i]]) * 100,
            marker="^",
            linestyle="-",
            label="dwnr 0% inflation target",
            color="darkslategrey",
            alpha=0.9,
        )
        ax.set_title(var_names[i], size="18")
        ax.set_xlabel("Quarters")
        ax.set_ylabel("Absolute Deviation")
        ax.legend()
        plt.savefig(os.path.join("bld", ZLB, var_names[i] + ".pdf"))
        plt.close()
    else:
        fig, ax = plt.subplots(figsize=(8, 5))
        # plot as % deviation from Steady State
        ax.plot(
            (x[0:30, inds_b[i]] - xSS[variables[i]]) / xSS[variables[i]] * 100,
            marker="o",
            linestyle="-",
            label="baseline",
            color="rebeccapurple",
            alpha=0.9,
        )
        ax.plot(
            (y[0:30, inds_dnwr[i]] - ySS[variables[i]]) / ySS[variables[i]] * 100,
            marker="d",
            linestyle="-",
            label="dwnr 2% inflation target",
            color="darkred",
            alpha=0.9,
        )
        ax.plot(
            (z[0:30, inds_dnwr2[i]] - zSS[variables[i]]) / zSS[variables[i]] * 100,
            marker="^",
            linestyle="-",
            label="dwnr 0% inflation target",
            color="darkslategrey",
            alpha=0.9,
        )
        ax.set_title(var_names[i], size="18")
        ax.set_xlabel("Quarters")
        ax.set_ylabel("Percent")
        ax.legend()
        plt.savefig(os.path.join("bld", ZLB, var_names[i] + ".pdf"))
        plt.close()
