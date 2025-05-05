
# Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Markdown, display
import warnings
warnings.simplefilter('ignore')

# To run in the terminal:
##python data-code/_BuildFinalData.py

# Load data
git_path = "https://github.com/imccart/Insurance-Access/raw/refs/heads/master/data/output/"
final_data = pd.read_csv(git_path + "acs_medicaid.txt", sep="\t")

# Create percentage variables
final_data = (
    final_data.assign(
        perc_private=(final_data["ins_employer"] + final_data["ins_direct"]) / final_data["adult_pop"],
        perc_public=(final_data["ins_medicare"] + final_data["ins_medicaid"]) / final_data["adult_pop"],
        perc_ins=(final_data["adult_pop"] - final_data["uninsured"]) / final_data["adult_pop"],
        perc_unins=final_data["uninsured"] / final_data["adult_pop"],
        perc_employer=final_data["ins_employer"] / final_data["adult_pop"],
        perc_medicaid=final_data["ins_medicaid"] / final_data["adult_pop"],
        perc_medicare=final_data["ins_medicare"] / final_data["adult_pop"],
        perc_direct=final_data["ins_direct"] / final_data["adult_pop"],
    )
    .loc[~final_data["State"].isin(["Puerto Rico", "District of Columbia"])]
)

