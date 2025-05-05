
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


# Plot
direct_purchase_by_year = final_data.groupby('year')['perc_direct'].mean()

# Plot
direct_purchase_by_year.plot(title='Direct Purchase Health Insurance (Share of Adult Population)')
plt.xlabel('Year')
plt.ylabel('Share with Direct Purchase Insurance')
plt.show()

#question 3
medicaid_by_year = final_data.groupby('year')['perc_medicaid'].mean()
medicaid_by_year.plot(title='Medicaid Coverage (Share of Adult Population)', color='green')
plt.xlabel('Year')
plt.ylabel('Share with Medicaid')
plt.show()

##question 4
# Create expansion groups
expansion['expansion_2014'] = (expansion['expanded']) & (expansion['date_adopted'].str.startswith('2014'))

# Filter states
exp_2014_states = expansion[expansion['expansion_2014']]['State']
never_expanded_states = expansion[~expansion['expanded']]['State']

# Label each state in insurance data
insurance['expansion_group'] = insurance['state'].apply(
    lambda x: '2014 Expansion' if x in exp_2014_states.values
    else ('Never Expanded' if x in never_expanded_states.values else 'Drop')
)

# Drop "Drop" states
insurance_filtered = insurance[insurance['expansion_group'] != 'Drop']

# Group and plot
uninsured_by_year = insurance_filtered.groupby(['year', 'expansion_group'])['uninsured'].mean().unstack()

uninsured_by_year.plot(title='Uninsurance Rate by Medicaid Expansion Status')
plt.xlabel('Year')
plt.ylabel('Uninsured Share')
plt.show()

# question 5
# Filter only states that expanded in 2014 or never expanded
never_expanded = expansion[~expansion['expanded']]['State']
expanded_2014 = expansion[expansion['expansion_2014']]['State']

insurance['group'] = insurance['state'].apply(
    lambda s: 'Expanded 2014' if s in expanded_2014.values else (
        'Never Expanded' if s in never_expanded.values else 'Other'
    )
)
insurance_plot = insurance[insurance['group'].isin(['Expanded 2014', 'Never Expanded'])]

uninsured_trends = insurance_plot.groupby(['year', 'group'])['uninsured'].mean().unstack()
uninsured_trends.plot(title='Uninsurance Rate by Expansion Status')
plt.ylabel('Uninsured Share')
plt.xlabel('Year')
plt.grid(True)
plt.show()

# question 6

avg_uninsured = insurance_plot[insurance_plot['year'].isin([2012, 2015])]
dd_table = avg_uninsured.groupby(['group', 'year'])['uninsured'].mean().unstack()
print(dd_table)

dd_effect = (dd_table.loc['Expanded 2014', 2015] - dd_table.loc['Expanded 2014', 2012]) - \
            (dd_table.loc['Never Expanded', 2015] - dd_table.loc['Never Expanded', 2012])
print(f"DiD estimate: {dd_effect:.4f}")

#question 7
# Filter again
dd_df = insurance_plot.copy()
dd_df['post'] = (dd_df['year'] >= 2014).astype(int)
dd_df['treated'] = (dd_df['group'] == 'Expanded 2014').astype(int)
dd_df['post_treated'] = dd_df['post'] * dd_df['treated']

model = sm.OLS.from_formula("uninsured ~ post + treated + post_treated + C(state) + C(year)", data=dd_df)
result = model.fit()
print(result.summary())

#question 8
# Recode with all expansion status
full_expansion_map = expansion.set_index('State')['expanded'].to_dict()
insurance['treated_all'] = insurance['state'].map(full_expansion_map).fillna(False).astype(int)
insurance['post'] = (insurance['year'] >= 2014).astype(int)
insurance['post_treated'] = insurance['post'] * insurance['treated_all']

model2 = sm.OLS.from_formula("uninsured ~ post + treated_all + post_treated + C(state) + C(year)", data=insurance)
result2 = model2.fit()
print(result2.summary())


#question 9
event_df = insurance_plot.copy()
expansion_dates = expansion.set_index('State')['date_adopted'].dropna()

# Define event time
def get_event_time(row):
    if row['state'] in expansion_dates:
        expansion_year = int(expansion_dates[row['state']][:4])
        return row['year'] - expansion_year
    else:
        return row['year'] - 2014  # never-expanded comparison

event_df['event_time'] = event_df.apply(get_event_time, axis=1)

# Exclude extreme event times
event_df = event_df[(event_df['event_time'] >= -5) & (event_df['event_time'] <= 5)]

# Estimate event study regression
dummies = pd.get_dummies(event_df['event_time'], prefix='et')
event_study_data = pd.concat([event_df, dummies], axis=1)

formula = 'uninsured ~ ' + ' + '.join(dummies.columns.difference(['et0'])) + ' + C(state) + C(year)'
model3 = sm.OLS.from_formula(formula, data=event_study_data)
result3 = model3.fit()

# Extract and plot coefficients
coef = result3.params.filter(like='et')
conf = result3.conf_int().loc[coef.index]
years = [int(x.split('et')[-1]) for x in coef.index]

plt.errorbar(years, coef.values, 
             yerr=[coef.values - conf[0].values, conf[1].values - coef.values], fmt='o')
plt.axhline(0, color='gray', linestyle='--')
plt.title("Event Study: Effect of Medicaid Expansion on Uninsurance")
plt.xlabel("Event Time (Years Since Expansion)")
plt.ylabel("Change in Uninsurance Rate")
plt.grid(True)
plt.show()


#question 10

# ATE Q6: Event Study with time-varying treatment
reg_data2["relative_year"] = (reg_data2["year"] - reg_data2["expand_year"]).fillna(np.inf)
reg_data2["relative_year"] = reg_data2["relative_year"].clip(lower=-4)

dynamic_twfe2 = pf.feols("perc_unins ~ i(relative_year, ref=-1) | State + year",
                  data=reg_data2, vcov={"CRV1": "State"})

plt.figure(figsize=(8, 5))
joint_ci2 = dynamic_twfe2.coef() - dynamic_twfe2.confint(joint=True).T.iloc[0, :]
plt.errorbar(np.delete(np.arange(-4, 6), 3), dynamic_twfe2.coef(), 
             yerr=joint_ci2, fmt='o', color=blue, capsize=5)
plt.axvline(x=-1, color="gold", linestyle="--")
plt.axhline(y=0, color="black", linestyle="-")
plt.title("Event Study with Staggered Treatment", fontsize=16)
plt.ylabel("Coefficient", fontsize=12)
plt.xlabel("Years Relative to Expansion", fontsize=12)
plt.grid(axis='y', color='gray', linestyle='--', alpha=0.5)
plt.show()
