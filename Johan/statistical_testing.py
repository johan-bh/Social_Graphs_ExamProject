import os
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle

# Create output directory for tables if it doesn't exist
os.makedirs("Tables", exist_ok=True)

combined_20_df = pd.read_csv(r"Data\statistics\combined_100_df.csv")

# ------------------------------------------------------------
# STEP 1: Mann-Whitney Tests with Multiple Comparison Correction
# ------------------------------------------------------------
metrics_to_test = [
    'NumNodes', 'NumEdges', 'AvgDegree', 'Density', 'AvgClustering', 'Modularity',
    'MeanEdgeSentiment', 'StdEdgeSentiment', 'SkewEdgeSentiment', 'KurtEdgeSentiment',
    'AvgShortestPath', 'Diameter', 'AvgBetweenness', 'AvgCloseness', 'AvgEigenvector',
    'Assortativity'
]

# Compute the mean of each metric for the top and bottom groups
group_means = combined_20_df.groupby('Group')[metrics_to_test].mean()

# Save group means to a LaTeX table
group_means_latex = group_means.T
group_means_latex.to_latex("Tables/means_by_group.tex", float_format="%.4f")

top_values_dict = {}
bottom_values_dict = {}
p_values = []

# Perform Mann-Whitney tests
for metric in metrics_to_test:
    top_values = combined_20_df.loc[combined_20_df['Group'] == 'Top', metric].dropna()
    bottom_values = combined_20_df.loc[combined_20_df['Group'] == 'Bottom', metric].dropna()
    if len(top_values) > 0 and len(bottom_values) > 0:
        stat, p_value = mannwhitneyu(top_values, bottom_values, alternative='two-sided')
        p_values.append(p_value)
    else:
        p_values.append(np.nan)

# Multiple comparison correction (Benjamini-Hochberg)
valid_indices = [i for i, v in enumerate(p_values) if not np.isnan(v)]
valid_p_values = [p_values[i] for i in valid_indices]

reject, p_adjusted, _, _ = multipletests(valid_p_values, method='fdr_bh')

# Prepare a DataFrame for Mann-Whitney test results
results_data = []
for i, metric in enumerate(metrics_to_test):
    if i in valid_indices:
        adj_idx = valid_indices.index(i)
        results_data.append({
            'Metric': metric,
            'Original p-value': p_values[i],
            'Adjusted p-value': p_adjusted[adj_idx],
            'Reject H0 (FDR)': reject[adj_idx]
        })
    else:
        # Missing values scenario
        results_data.append({
            'Metric': metric,
            'Original p-value': np.nan,
            'Adjusted p-value': np.nan,
            'Reject H0 (FDR)': np.nan
        })

mw_results_df = pd.DataFrame(results_data)
mw_results_df.set_index('Metric', inplace=True)

# Save Mann-Whitney results to a LaTeX table
mw_results_df.to_latex("Tables/mannwhitney_results.tex", float_format="%.4e")

# ------------------------------------------------------------
# STEP 2: Classification with Cross-Validation
# ------------------------------------------------------------
# Prepare features and target for classification
data = combined_20_df.dropna(subset=metrics_to_test)  # Drop rows with NaNs in metrics
X = data[metrics_to_test].values
y = (data['Group'] == 'Top').astype(int).values  # Top=1, Bottom=0

clf = LogisticRegression(max_iter=1000)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')

mean_acc = np.mean(scores)
std_acc = np.std(scores, ddof=1)

# Create a DataFrame for classification results
classification_results_df = pd.DataFrame({
    'Mean Accuracy': [mean_acc],
    'Std Accuracy': [std_acc]
})

# Save classification results to a LaTeX table
classification_results_df.to_latex("Tables/classification_accuracy.tex", float_format="%.4f", index=False)

# ------------------------------------------------------------
# STEP 3: Permutation Test for Classifier
# ------------------------------------------------------------
n_permutations = 1000
permutation_accuracies = []

for _ in range(n_permutations):
    y_perm = shuffle(y, random_state=None)
    perm_score = np.mean(cross_val_score(clf, X, y_perm, cv=cv, scoring='accuracy'))
    permutation_accuracies.append(perm_score)

permutation_accuracies = np.array(permutation_accuracies)
p_value_permutation = np.mean(permutation_accuracies >= mean_acc)

# Create a DataFrame for permutation test results
perm_test_results_df = pd.DataFrame({
    'Observed Accuracy': [mean_acc],
    'p-value (Permutation)': [p_value_permutation]
})

# Save permutation test results to a LaTeX table
perm_test_results_df.to_latex("Tables/permutation_results.tex", float_format="%.4f", index=False)

# Print summaries to console
print("\nMean values for each metric by group:")
print(group_means.T)

print("\nMann-Whitney Tests with Benjamini-Hochberg correction:")
print(mw_results_df)

print("\nClassification Accuracy (Logistic Regression):")
print(classification_results_df)

print("\nPermutation Test Results:")
print(perm_test_results_df)
