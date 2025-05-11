import pandas as pd
from scipy import stats
import math
import numpy as np
from statsmodels.stats.multitest import multipletests,fdrcorrection
import  matplotlib_venn
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

def select_fea(link,fea):
    import pandas as pd
    df=pd.read_csv(link,sep=',')
    select='(SampleID)|(Class)|(Label)|('+ fea +')'
    df=df.filter(regex=select)
    return df

df=select_fea(r"Data_feature_Discovery cohort_Breast Cancer.csv",'flen')
df

groups = df['Class'].unique()
groups

## Cancer vs Benign
results = []

# Iterate through each feature
for feature in df.columns[3:]:  # Exclude the first two columns which are sample and group identifiers
    for group_C, group_B in zip(groups[0:1], groups[1:-1]):
        group_C_data = df[df['Class'] == group_C][feature]
        group_B_data = df[df['Class'] == group_B][feature]

        # Calculate mean for each group
        mean_group_C = group_C_data.mean()
        mean_group_B = group_B_data.mean()

        # Calculate log2 fold change
        log2_fold_change = np.log2(mean_group_C / mean_group_B)

        # Perform Mann-Whitney U test
        stat, p_value = mannwhitneyu(group_C_data, group_B_data, alternative='two-sided')


        # Append results to the list
        results.append({
            'Feature': feature,
            'Breast Cancer': mean_group_C,
            'Benign': mean_group_B,
            'Log2(BC_B)': log2_fold_change,
            'p-value_BC_B': p_value
        })

# Convert results list to DataFrame
results_df_C_B = pd.DataFrame(results)

results_df_C_B

# Perform multiple testing correction using Benjamini-Hochberg method
reject, corrected_p_values, _, _ = multipletests(results_df_C_B['p-value_BC_B'], method='fdr_bh')

# Add corrected p-values to the DataFrame
results_df_C_B['Corrected p-value_BC_B'] = corrected_p_values
results_df_C_B

## Cancer vs Healthy
results = []

# Iterate through each feature
for feature in df.columns[3:]:  # Exclude the first two columns which are sample and group identifiers
    for group_C, group_H in zip(groups[0:1], groups[-1:]):
        group_C_data = df[df['Class'] == group_C][feature]
        group_H_data = df[df['Class'] == group_H][feature]

        # Calculate mean for each group
        mean_group_C = group_C_data.mean()
        mean_group_H = group_H_data.mean()

        # Calculate log2 fold change
        log2_fold_change = np.log2(mean_group_C / mean_group_H)

        # Perform Mann-Whitney U test
        stat, p_value = mannwhitneyu(group_C_data, group_H_data, alternative='two-sided')

        # Append results to the list
        results.append({
            'Healthy': mean_group_H,
            'Log2(BC_H)': log2_fold_change,
            'p-value_BC_H': p_value
        })

# Convert results list to DataFrame
results_df_C_H = pd.DataFrame(results)

results_df_C_H 

# Perform multiple testing correction using Benjamini-Hochberg method
reject, corrected_p_values, _, _ = multipletests(results_df_C_H['p-value_BC_H'], method='fdr_bh')

# Add corrected p-values to the DataFrame
results_df_C_H['Corrected p-value_BC_H'] = corrected_p_values
results_df_C_H

stat = pd.concat([results_df_C_B, results_df_C_H], axis=1)
stat

stat.to_csv("Pvalue_TMD.csv",index=False)

##Venn diagram
list_region_sig_BC_BR,list_region_sig_BR_HC,list_region_sig_BC_HC=[],[],[]
for i in range(len(stat)):
    if stat['Corrected p-value_BC_B'][i] <= 0.05:
        list_region_sig_BC_BR.append(stat['Feature'][i])
    if stat['Corrected p-value_BC_H'][i] <= 0.05:
        list_region_sig_BR_HC.append(stat['Feature'][i])
else:
    set1 = set(list_region_sig_BC_BR)
    set2 = set(list_region_sig_BR_HC)
    venn=matplotlib_venn.venn2_unweighted([set1, set2], ('Breast Cancer - Benign', 'Breast Cancer - Healthy'))
    # Customize the subset label font size
    for text in venn.set_labels:
        text.set_fontsize(16)
    for text in venn.subset_labels:
        text.set_fontsize(14)
    plt.title('TMD_Number of overlapping regions',fontsize=20)
    