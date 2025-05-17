import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import SelectFromModel
from HyperclassifierSearch import HyperclassifierSearch
from sklearn.datasets import make_blobs
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import shap

#Import data

Discovery=pd.read_csv(r"Data_feature_Discovery cohort_Breast Cancer.csv",sep=',')

train =Discovery[(Discovery['Cohort'] == 'train')] 
test =Discovery[(Discovery['Cohort'] == 'test')] 

Validation=pd.read_csv(r"Data_feature_Validation Cohort_Breast Cancer.csv",sep=',')

lst_fea=pd.read_csv(r"List 439ft_Model Breast Cancer.csv",sep=',')
lst_fea

array = lst_fea['Feature_Combine'].dropna().to_numpy()

X_train= train[array]
X_train = X_train.values

X_test= test[array]

X_val= Validation[array]

y_train = train['Label']
y_test= test['Label']
y_val = Validation['Label']

####HyperClassifierSearch_default
## Set algorithm
models = {
    'LogisticRegression': LogisticRegression(max_iter=200),
    'RandomForestClassifier': RandomForestClassifier(),
    'SVM': SVC(max_iter=-1),
    'DT': DecisionTreeClassifier(),
    'XGB':XGBClassifier()
}
params = { 
    'LogisticRegression': { 'C': [1.0], 'class_weight':[None] },
    'RandomForestClassifier': { 'n_estimators': [100],'class_weight':[None]},
    'SVM' : {'kernel':['rbf']},
    'DT' : {'criterion':['gini'],'class_weight':[None]},
    'XGB': {'n_estimators':[100],'max_depth':[3],'learning_rate':[0.01]}
}


cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
search = HyperclassifierSearch(models,params)
best_model = search.train_model(X_train, y_train, cv=cv)
import matplotlib.pyplot as plt

# Assuming result is a DataFrame from search.evaluate_model()
result = search.evaluate_model()

# Extract algorithm names for labels
result['label'] = result['Estimator'].astype(str)

std = result['std_test_score'].to_list()
mean_ = result['mean_test_score'].to_list()
x = result['label'].to_list()

# Set up the matplotlib figure
plt.figure(figsize=(15, 4))

# Create error bar plot
plt.errorbar(x, mean_, yerr=std, fmt='o', ecolor='r', elinewidth=2, capsize=5, capthick=2, markersize=8, markerfacecolor='blue', markeredgewidth=1.5, markeredgecolor='black')

# Annotate each point with its accuracy and standard deviation, positioned slightly above the error bar
for i in range(len(x)):
    plt.annotate(f'{mean_[i]:.2f} ± {std[i]:.2f}',(x[i], mean_[i]), textcoords="offset points", xytext=(10,-5), ha='left', fontsize=14)

# Add title and labels
# plt.ylim(0.6,0.95)
plt.title('Error Bar Plot of 10-Fold Validation Accuracy', fontsize=16)
plt.xlabel('Algorithm', fontsize=16)
plt.ylabel('Mean Validation Accuracy', fontsize=16)

# Adjust tick parameters
plt.tick_params(axis='both', which='major', labelsize=14)

# Tight layout for better spacing
plt.tight_layout()

# Remove top and right spines
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# Show plot
plt.show()

####GridSearchCV
# Tunning hyperparameters for XGBClassifier

model = XGBClassifier(objective='binary:logistic',random_state=1) 
n_estimators = [100, 1000]
max_depth = [3, 4, 5]
learning_rate=[0.1, 0.01, 0.001]
# define grid search
grid = dict(n_estimators=n_estimators,max_depth=max_depth,learning_rate=learning_rate)
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='roc_auc',error_score=0)
grid_result = grid_search.fit(X_train, y_train)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

####Build model
#Create function
def compute_specificity(y_true, y_pred):
    true_negatives = sum((y_true == 0) & (y_pred == 0))
    actual_negatives = sum(y_true == 0)
    specificity = true_negatives / actual_negatives if actual_negatives != 0 else 0
    return specificity

def compute_sensitivity(y_true, y_pred):
    true_positives = sum((y_true == 1) & (y_pred == 1))
    actual_positives = sum(y_true == 1)
    sensitivity = true_positives / actual_positives if actual_positives != 0 else 0
    return sensitivity

from sklearn.utils import resample
from scipy.stats import norm
def compute_metrics_with_ci(y_true, y_pred, num_bootstraps=1000, confidence_level=0.95):
    specificity_scores = []
    sensitivity_scores = []

    # Bootstrap resampling
    for _ in range(num_bootstraps):
        boot_true, boot_pred = resample(y_true, y_pred)
        specificity = compute_specificity(boot_true, boot_pred)
        sensitivity = compute_sensitivity(boot_true, boot_pred)
        specificity_scores.append(specificity)
        sensitivity_scores.append(sensitivity)

    # Compute mean and confidence interval
    specificity_mean = np.mean(specificity_scores)
    sensitivity_mean = np.mean(sensitivity_scores)
    specificity_ci = norm.interval(confidence_level, loc=specificity_mean, scale=np.std(specificity_scores))
    sensitivity_ci = norm.interval(confidence_level, loc=sensitivity_mean, scale=np.std(sensitivity_scores))

    return {
        'specificity_mean': specificity_mean,
        'specificity_ci': specificity_ci,
        'sensitivity_mean': sensitivity_mean,
        'sensitivity_ci': sensitivity_ci
    }

def create_ml_model(model_type, **kwargs):
    '''
    Create Machine Learning model
    '''
    models = {
        'xgb': XGBClassifier
    }
    
    # Retrieve the model class based on the model_type argument
    model = models.get(model_type.lower())
    
    # Check if the model type exists
    if model:
        return model(**kwargs)
    else:
        supported_models = ", ".join(models.keys())
        print(f"Error: Model type '{model_type}' is not supported.\n"
              f"Supported models are: {supported_models}.\n"
              f"Please add your perferred model to create_model function.")
        return None
def kfold_evaluation(model, X, y, kfold):
    '''
    1. Train ML model on KFold
    
    2. Return metrics for each KFold and all KFolds
    '''
    fold_results = [] # average validation data results on all Kfolds
    fold_result_each_kfold = [] # training and validation results on each Kfold

    for train_index, val_index in cv.split(X, y):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # Train the model
        model.fit(X_train, y_train)

        # Predict on both sets
        train_predictions = model.predict(X_train)
        val_predictions = model.predict(X_val)
        
        # Calculate metrics for both sets
        train_metrics = calculate_results(y_train, train_predictions)
        val_metrics = calculate_results(y_val, val_predictions)

        fold_result = {
            'fold': len(fold_results) + 1,
            'train_metrics': train_metrics,
            'validation_metrics': val_metrics
        }

        fold_results.append(val_metrics)
        fold_result_each_kfold.append(fold_result)

    # Compute the average metrics over all folds
    model_results = {
        "accuracy": np.mean([result["accuracy"] for result in fold_results]),
        "precision": np.mean([result["precision"] for result in fold_results]),
        "recall": np.mean([result["recall"] for result in fold_results]),
        "f1": np.mean([result["f1"] for result in fold_results])
    }

    # Retrain the model on the entire dataset
    model.fit(X, y)

    return model, fold_result_each_kfold, model_results
from sklearn.metrics import precision_recall_fscore_support
def calculate_results(y_true, y_pred):
    '''
    Calculate accuracy, precision, recall, f1 score for a model
    '''
    model_accuracy = accuracy_score(y_true, y_pred) * 100  # Scale to 1-100
    model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
    
    model_results = {
        "accuracy": model_accuracy,
        "precision": model_precision,
        "recall": model_recall,
        "f1": model_f1
    }
    
    return model_results
def format_metrics_compact(metrics):
    '''
    Helper function to format metrics compactly
    '''
    return ", ".join([f"{key.capitalize()}: {value:.2f}" for key, value in metrics.items()])

def display_all_folds_results(model_name, model_results):
    '''
    Print average metrics of validation data on all KFolds
    '''
    print(f"Average {model_name} Performance Metrics Across All Folds (on validation data):")
    print(f"- Accuracy: {model_results['accuracy']:.2f}%")
    print(f"- Precision: {model_results['precision']:.4f}")
    print(f"- Recall: {model_results['recall']:.4f}")
    print(f"- F1 Score: {model_results['f1']:.4f}")

def display_each_fold_result(model_name, fold_results):
    '''
    Print metrics of training and validation data on each KFold and plot the accuracy graph using Seaborn.
    '''
    # Set up the seaborn style and palette
    sns.set(style="whitegrid", palette="muted")
    colors = ["#3498db", "#e74c3c"]  # Blue for Training, Red for Validation

    # Header
    print(f"{model_name} Performance Metrics Across Each Fold (on training and validation data):")
    print()
    print(f"{'Fold':<5} | {'Training Metrics':<70} | {'Validation Metrics':<70}")
    print("-" * 150)

    # Prepare data for plotting
    train_accuracies = []
    val_accuracies = []
    folds = []

    # Display results for each fold and collect data for graph
    for result in fold_results:
        train_metrics_formatted = format_metrics_compact(result['train_metrics'])
        val_metrics_formatted = format_metrics_compact(result['validation_metrics'])
        print(f"{result['fold']:<5} | {train_metrics_formatted:<70} | {val_metrics_formatted:<70}")

        # Append data for graph
        folds.append(result['fold'])
        train_accuracies.append(result['train_metrics']['accuracy'])
        val_accuracies.append(result['validation_metrics']['accuracy'])

    # Plotting the accuracies using Seaborn
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.lineplot(x=folds, y=train_accuracies, label='Training Accuracy', marker='o', color=colors[0])
    sns.lineplot(x=folds, y=val_accuracies, label='Validation Accuracy', marker='o', color=colors[1])
    ax.set_facecolor('#f8f9fa')  # Light gray background

    # Adding percentages on markers
    label_offset = 0.25  # Increased vertical offset for the labels
    for i, (tr_acc, val_acc) in enumerate(zip(train_accuracies, val_accuracies)):
        ax.text(folds[i], tr_acc + label_offset, f"{tr_acc:.2f}%", ha='center', va='bottom')
        ax.text(folds[i], val_acc + label_offset, f"{val_acc:.2f}%", ha='center', va='bottom')

    plt.title(f"{model_name} model: Training vs Validation Accuracy per Fold")
    plt.xlabel('Fold')
    plt.ylabel('Accuracy (%)')
    plt.xticks(folds)
    plt.legend()
    ax.margins(y=0.2)  # Add 10% padding to the top and bottom
    fig.tight_layout()
    plt.show()
#XGBClassifier
xgb_model= create_ml_model('xgb', learning_rate=0.1, max_depth=3, n_estimators=1000,objective='binary:logistic',random_state=1)

xgb_model, xgb_each_fold_results, xgb_results = kfold_evaluation(xgb_model, X_train, y_train, cv)

display_all_folds_results("XGB", xgb_results)

display_each_fold_result("XGB",xgb_each_fold_results)

#Fit model
xgb_model.fit(X_train, y_train)
proba_train = cross_val_predict(xgb_model, X_train, y_train, cv=cv, method='predict_proba')
[print(x) for x in proba_train[:,1]]

#Evaluate test set 
y_pred = xgb_model.predict_proba(X_test)
print(y_pred)

#Evaluate validation cohort
y_pred1 = xgb_model.predict_proba(X_val)
print(y_pred1)

####ROC curve plot
plt.figure(figsize = (10,10))
sns.set_theme(context='notebook', style='ticks', palette='deep', font='sans-serif', font_scale=1.4, color_codes=True, rc=None)

fpr_TRAIN, tpr_TRAIN, thresholds_TRAIN = metrics.roc_curve(y_train,  proba_train[:, 1])
auc_TRAIN = metrics.roc_auc_score(y_train,  proba_train[:, 1])
plt.plot(fpr_TRAIN,tpr_TRAIN,label="AUC train = "+str(round(auc_TRAIN, 4)), linewidth = 3, color='forestgreen')


fpr_TEST, tpr_TEST, thresholds_TEST = metrics.roc_curve(y_test,  y_pred[:, 1])
auc_TEST = metrics.roc_auc_score(y_test,  y_pred[:, 1])
plt.plot(fpr_TEST,tpr_TEST,label="AUC test = "+str(round(auc_TEST, 4)), linewidth = 3, color='royalblue')


plt.plot([0, 1], ls="--")
plt.title('Cancer vs Noncancer_Kbest_439ft_XGB')
plt.ylabel('Sensitivity', labelpad=30)
plt.xlabel('1-Specificity', labelpad=30)
plt.legend(loc=4)
plt.show()
####Save model
filename = 'Model Breast Cancer 439ft.sav'
pickle.dump(xgb_model, open(filename, 'wb'))

#load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))


####SHAP analysis


# Create SHAP explainer
explainer = shap.Explainer(xgb_model)

# Compute SHAP values
shap_values = explainer(X_train)

# Summary plot
shap.summary_plot(shap_values, X_train, feature_names=X_train.columns)
import numpy as np
import matplotlib.pyplot as plt

# Extract SHAP values and feature names
shap_array = shap_values.values
feature_names = np.array(shap_values.feature_names)

# Compute mean absolute SHAP values
mean_abs_shap = np.abs(shap_array).mean(axis=0)

# Get top 20 features
top_20_idx = np.argsort(mean_abs_shap)[-20:]
top_20_features = feature_names[top_20_idx]
top_20_importance = mean_abs_shap[top_20_idx]

# Define function to categorize feature type
def get_feature_type(name):
    if name.startswith("ME"):
        return "ME"
    elif name.startswith("CNA"):
        return "CNA"
    elif name.startswith("T_CNA"):
        return "T_CNA"
    elif name.startswith("SNAI1_UBE2V1"):
        return "SNAI1_UBE2V1"
    elif name.startswith("MOTIF"):
        return "MOTIF"

# Get feature types and assign colors
feature_types = [get_feature_type(f) for f in top_20_features]

# Optional: define your custom color palette
color_map = {
    "ME": "#1f77b4",        # blue
    "CNA": "#ff7f0e",       # orange
    "T_CNA": "#2ca02c",     # green
    "SNAI1_UBE2V1": "#d62728",       # red
    "MOTIF": "#9467bd", # purple
}
bar_colors = [color_map[ft] for ft in feature_types]

# Sort for clean plotting
sorted_idx = np.argsort(top_20_importance)
top_20_features = top_20_features[sorted_idx]
top_20_importance = top_20_importance[sorted_idx]
bar_colors = [bar_colors[i] for i in sorted_idx]
# Customize how feature types appear in the legend
legend_labels = {
    "ME": "ME21",
    "CNA": "CNA",
    "T_CNA": "T_CNA",
    "SNAI1_UBE2V1": "TMD",
    "MOTIF": "EM"
}

# Plot
plt.figure(figsize=(8, 8))
plt.barh(top_20_features, top_20_importance, color=bar_colors)
plt.xlabel("Mean |SHAP value|", fontsize=14)
plt.title("Top 20 Most Important Features (SHAP)", fontsize=16)

# Create custom legend elements
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=color_map[k], label=legend_labels[k]) for k in color_map
]

# Add legend with custom font sizes and no frame
legend = plt.legend(
    handles=legend_elements,
    title="Feature Type",
    loc="lower right",
    fontsize=12,
    title_fontsize=14,
    frameon=False  # <--- disables legend box
)

# Remove top and right spines from the plot
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# Adjust tick parameters
plt.tick_params(axis='x', which='major', labelsize=12)
plt.yticks(fontsize=12)

# Show plot
plt.tight_layout()
plt.show()
