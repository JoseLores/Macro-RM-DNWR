# Macro-RM-DNWR
Repository for creating the figures included in my Final Project for the 'Research Module in Macroeconomics and Public Economics (University of Bonn, winter term 22/23). The paper is available [here](https://www.dropbox.com/s/3y4rdeay6jpu7n4/DNWR.pdf?dl=0)

## models
Contains the .yaml with the baseline medium-scale NK model and the model with downward nominal wage rigidity.

## main.py
Run this file to generate all the output. A "bld" folder will be created if needed with 4 subdirectories:
- positive_beta_shock: creates IRFs for Figure 1. Note that we generate several IRFs, only the most relevant are included in the paper.
- positive_beta_shock_tight: same shock but with a stronger response to inflation by the Central Bank
- Simulation: Creates the Phillips Curve from 1,000 normally distributed shocks to the discount factor. Saves the generated data from the simulation in a .csv file
- stronger_shock_hit_ZLB: creates IRFs that hit the Zero Lower Bound
