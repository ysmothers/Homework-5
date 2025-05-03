import pandas as pd

# Load datasets
insurance = pd.read_csv('data/input/acs_insurance.txt', sep='\t')
medicaid = pd.read_csv('data/input/acs_medicaid.txt', sep='\t')
expansion = pd.read_csv('data/input/medicaid_expansion.txt', sep='\t')

# Preview
print(insurance.head())
print(medicaid.head())
print(expansion.head())
