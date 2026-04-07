import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from itertools import combinations

file = input("Path to csv file containing data: ")

df = pd.read_csv(file)

# 2. Identify categorical columns
cat_cols = df.select_dtypes(include=['object', 'category']).columns

# 3. Initialize matrix and a list for significant pairs
p_matrix = pd.DataFrame(np.ones((len(cat_cols), len(cat_cols))), 
                        index=cat_cols, 
                        columns=cat_cols)
significant_pairs = []

# 4. Compute Chi-squared p-values
for col1, col2 in combinations(cat_cols, 2):
    contingency_table = pd.crosstab(df[col1], df[col2])
    _, p, _, _ = chi2_contingency(contingency_table)
    
    # Fill the matrix
    p_matrix.loc[col1, col2] = p
    p_matrix.loc[col2, col1] = p
    
    # Check for significance
    if p < 0.05:
        significant_pairs.append((col1, col2, p))

# 5. Print significant results
print("--- Significant Variable Pairs (p < 0.05) ---")
if not significant_pairs:
    print("No significant pairs found.")
else:
    for v1, v2, p_val in significant_pairs:
        print(f"Significant Relationship: {v1} and {v2} (p = {p_val:.4f})")

# 6. Generate Heatmap (Color Only)
plt.figure(figsize=(10, 8))

# Using 'viridis' colormap
# annot=False removes the numbers from the cells
sns.heatmap(p_matrix, 
            annot=False, 
            cmap='magma', # change to viridis, plasma, inferno, magma, cividis if needed
            vmax=0.05, 
            linewidths=0.1,
            cbar_kws={'label': 'p-value'})

plt.title('Chi-Squared Test: Variable Dependencies\n(Color represents p-value significance)')
plt.tight_layout()
plt.show()
