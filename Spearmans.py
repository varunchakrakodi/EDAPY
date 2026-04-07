import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from itertools import combinations

encoded_csv = "Encoded.csv"
corr_csv = "spearman_correlation_matrix.csv"
pval_csv = "spearman_pvalue_matrix.csv"
plot_file = "spearman_matrix_plot.png"

file = input("Data File: ")
df = pd.read_csv(file)

# Clean column names
df.columns = [c.strip().replace(" ", "_") for c in df.columns]

# Standardize missing values and trim whitespace
for col in df.columns:
    if df[col].dtype == object:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace({"": np.nan, "NA": np.nan, "nan": np.nan})

#create a encoded df using factorise
encoded_df = df.copy()
encoding_maps = {}

for col in encoded_df.columns:
    if pd.api.types.is_numeric_dtype(encoded_df[col]):
        continue

    codes, uniques = pd.factorize(encoded_df[col], sort=True)

    # pd.factorize gives -1 for missing values, convert those back to NaN
    codes = pd.Series(codes, index=encoded_df.index).replace(-1, np.nan)

    encoded_df[col] = codes

    mapping = {str(cat): int(code) for code, cat in enumerate(uniques)}
    encoding_maps[col] = mapping

# Try converting remaining columns to numeric if possible
for col in encoded_df.columns:
    encoded_df[col] = pd.to_numeric(encoded_df[col], errors="coerce")

# Drop columns that became entirely missing
encoded_df = encoded_df.dropna(axis=1, how="all")

# Save encoded dataset
encoded_df.to_csv(encoded_csv, index=False)

# =========================
# SPEARMAN FOR ALL PAIRS
# =========================
cols = encoded_df.columns.tolist()
n = len(cols)

corr_matrix = pd.DataFrame(np.eye(n), index=cols, columns=cols, dtype=float)
p_matrix = pd.DataFrame(np.zeros((n, n)), index=cols, columns=cols, dtype=float)

for c1, c2 in combinations(cols, 2):
    pair = encoded_df[[c1, c2]].dropna()

    if len(pair) < 3:
        rho, pval = np.nan, np.nan
    else:
        result = spearmanr(pair[c1], pair[c2], nan_policy="omit")
        rho, pval = result.statistic, result.pvalue

    corr_matrix.loc[c1, c2] = rho
    corr_matrix.loc[c2, c1] = rho
    p_matrix.loc[c1, c2] = pval
    p_matrix.loc[c2, c1] = pval

np.fill_diagonal(corr_matrix.values, 1.0)
np.fill_diagonal(p_matrix.values, 0.0)

corr_matrix.to_csv(corr_csv)
p_matrix.to_csv(pval_csv)

#Create a matrix figure
fig, ax = plt.subplots(figsize=(max(12, 0.55 * n + 6), max(10, 0.55 * n + 4)))

im = ax.imshow(p_matrix.values, cmap="magma", vmin=0, vmax=0.05)

ax.set_xticks(np.arange(n))
ax.set_yticks(np.arange(n))
ax.set_xticklabels(cols, rotation=90, fontsize=8)
ax.set_yticklabels(cols, fontsize=8)

for i in range(n):
    for j in range(n):
        rho = corr_matrix.iloc[i, j]
        pval = p_matrix.iloc[i, j]

        txt = "NA" if pd.isna(rho) else f"{rho:.2f}"
        txt_color = "white" if pd.notna(pval) and pval < 0.025 else "black"

        ax.text(j, i, txt, ha="center", va="center", color=txt_color, fontsize=7)

ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.3)
ax.tick_params(which="minor", bottom=False, left=False)

cbar = plt.colorbar(im, ax=ax)
cbar.set_label("p-value")

ax.set_title("Spearman Correlation Matrix\nText = rho, Cell color = p-value")
plt.tight_layout()
plt.show()

#Find interesting pairs

threshold = float(input("Select significance: ").strip())
use_absolute = False      # True = filter by |rho| >= threshold
include_pvalue = True     # print p-value too

print("\nPairs meeting threshold condition:\n")

found = False

for c1, c2 in combinations(corr_matrix.columns, 2):
    rho = corr_matrix.loc[c1, c2]
    pval = p_matrix.loc[c1, c2]

    if pd.isna(rho):
        continue

    if use_absolute:
        condition = abs(rho) >= threshold
    else:
        condition = (rho >= threshold) or (rho <= -threshold)

    if condition:
        found = True
        if include_pvalue:
            print(f"{c1} vs {c2}: rho = {rho:.3f}, p = {pval:.4g}")
        else:
            print(f"{c1} vs {c2}: rho = {rho:.3f}")

if not found:
    print(f"No variable pairs found with correlation meeting threshold = {threshold}")
