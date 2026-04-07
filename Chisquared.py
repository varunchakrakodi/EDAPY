import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from itertools import combinations

def calculate_cramers_v(contingency_table):
    """ Calculates Cramer's V statistic for categorical-categorical association. """
    chi2 = chi2_contingency(contingency_table)[0]
    n = contingency_table.sum().sum()
    phi2 = chi2 / n
    r, k = contingency_table.shape
    
    # Cramer's V formula: sqrt( (chi2/n) / min(k-1, r-1) )
    # Adding a small check for the denominator to avoid division by zero
    denominator = min(k - 1, r - 1)
    if denominator == 0:
        return 0
    return np.sqrt(phi2 / denominator)

file = input("Path to csv file containing data: ")
df = pd.read_csv(file)

cat_cols = df.select_dtypes(include=['object', 'category']).columns

# Initialize matrix and a list for significant pairs
p_matrix = pd.DataFrame(np.ones((len(cat_cols), len(cat_cols))), 
                        index=cat_cols, 
                        columns=cat_cols)
significant_pairs = []

for col1, col2 in combinations(cat_cols, 2):
    contingency_table = pd.crosstab(df[col1], df[col2])
    chi2, p, _, _ = chi2_contingency(contingency_table)
    
    # Fill the p-value matrix
    p_matrix.loc[col1, col2] = p
    p_matrix.loc[col2, col1] = p
    
    # Check for significance and calculate Cramer's V
    if p < 0.05:
        v_score = calculate_cramers_v(contingency_table)
        significant_pairs.append((col1, col2, p, v_score))

print("\n--- Significant Variable Pairs (p < 0.05) ---")
if not significant_pairs:
    print("No significant pairs found.")
else:
    # Header for clarity
    print(f"{'Variable 1':<20} | {'Variable 2':<20} | {'p-value':<10} | {'Cramer\'s V':<10}")
    print("-" * 70)
    for v1, v2, p_val, v_val in significant_pairs:
        print(f"{v1:<20} | {v2:<20} | {p_val:.4f}     | {v_val:.4f}")

plt.figure(figsize=(10, 8))
sns.heatmap(p_matrix, 
            annot=False, 
            cmap='inferno', 
            vmax=0.05, 
            linewidths=0.1,
            cbar_kws={'label': 'p-value'})

plt.title('Chi-Squared Test: Variable Dependencies\n(Color represents p-value significance)')
plt.tight_layout()
plt.show()
