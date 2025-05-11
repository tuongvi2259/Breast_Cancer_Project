import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedKFold
from collections import Counter

train=pd.read_csv(r"Data_feature_Discovery cohort_Breast Cancer.csv",sep=',')
train.info()
train.head()
####################
## Cancer vs Benign
filtered_train =train[(train['Cohort'] == 'train')] and train[(train['Class'] == 'BC') | (train['Class'] =='Bengin')]
filtered_train

y = filtered_train['Label']
X = filtered_train.drop(["SampleID","Cohort","Class","Label"],axis=1)

# Kendall correlation
Correlation=X.corr(method="kendall")
columns = np.full((Correlation.shape[0],), True, dtype=bool)
for i in range(Correlation.shape[0]):
    for j in range(i+1, Correlation.shape[0]):
        if Correlation.iloc[i,j] >= 0.7:
            if columns[j]:
                columns[j] = False
selected_columns = X.columns[columns]
selected_columns.shape

X=X[columns]
X

feature_names = X.columns

# Number of top features to select (assuming 3 here)
k = 500

# Set up 10-Fold Cross-Validation
kf =  StratifiedKFold(n_splits=10, shuffle=True, random_state=1)

# Initialize a list to store results
results = []

# Iterate through each fold
    
for fold, (train_index, test_index) in enumerate(kf.split(X, y)):
    # Use iloc for both X and y to ensure proper indexing
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    # Apply SelectKBest to select the top k features
    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit(X_train, y_train)
    
    # Get the selected feature names
    selected_features = X.columns[selector.get_support(indices=True)]
    
    # Create a DataFrame for the selected features
    importance_df = pd.DataFrame({
        'Feature': selected_features,
        'Fold': fold + 1
    })
    
    # Append to results list
    results.append(importance_df)

# Combine all fold results into a single DataFrame
results_df = pd.concat(results, ignore_index=True)
# Create an empty DataFrame to hold the results
folds =results_df['Fold'].unique()
data = pd.DataFrame()

# Populate the new DataFrame with each fold's data in separate columns
for fold in folds:
    # Filter the DataFrame for the current fold
    fold_data = results_df[results_df['Fold'] == fold][['Feature']]
    
    # Rename columns to reflect the current fold
    fold_data.columns = [f'Feature_Fold_{fold}']
    
    # Reset the index to avoid issues during concatenation
    fold_data = fold_data.reset_index(drop=True)
    
    # Concatenate the data into the new DataFrame
    data = pd.concat([data, fold_data], axis=1)

data.to_csv("Feature_Kbest_10Fold_BC_B.csv", index=False)
###################
##Cancer vs Helathy
filtered_train =train[(train['Cohort'] == 'train')] and train[(train['Class'] == 'BC') | (train['Class'] =='Healthy')]
filtered_train

y = filtered_train['Label']
X = filtered_train.drop(["SampleID","Cohort","Class","Label"],axis=1)

# Kendall correlation
Correlation=X.corr(method="kendall")
columns = np.full((Correlation.shape[0],), True, dtype=bool)
for i in range(Correlation.shape[0]):
    for j in range(i+1, Correlation.shape[0]):
        if Correlation.iloc[i,j] >= 0.7:
            if columns[j]:
                columns[j] = False
selected_columns = X.columns[columns]
selected_columns.shape

X=X[columns]
X

feature_names = X.columns

# Number of top features to select (assuming 3 here)
k = 500

# Set up 10-Fold Cross-Validation
kf =  StratifiedKFold(n_splits=10, shuffle=True, random_state=1)

# Initialize a list to store results
results = []

# Iterate through each fold
    
for fold, (train_index, test_index) in enumerate(kf.split(X, y)):
    # Use iloc for both X and y to ensure proper indexing
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    # Apply SelectKBest to select the top k features
    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit(X_train, y_train)
    
    # Get the selected feature names
    selected_features = X.columns[selector.get_support(indices=True)]
    
    # Create a DataFrame for the selected features
    importance_df = pd.DataFrame({
        'Feature': selected_features,
        'Fold': fold + 1
    })
    
    # Append to results list
    results.append(importance_df)

# Combine all fold results into a single DataFrame
results_df = pd.concat(results, ignore_index=True)
# Create an empty DataFrame to hold the results
folds =results_df['Fold'].unique()
data = pd.DataFrame()

# Populate the new DataFrame with each fold's data in separate columns
for fold in folds:
    # Filter the DataFrame for the current fold
    fold_data = results_df[results_df['Fold'] == fold][['Feature']]
    
    # Rename columns to reflect the current fold
    fold_data.columns = [f'Feature_Fold_{fold}']
    
    # Reset the index to avoid issues during concatenation
    fold_data = fold_data.reset_index(drop=True)
    
    # Concatenate the data into the new DataFrame
    data = pd.concat([data, fold_data], axis=1)

data.to_csv("Feature_Kbest_10Fold_BC_H.csv", index=False)