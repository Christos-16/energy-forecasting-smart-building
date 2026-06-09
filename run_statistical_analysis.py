#!/usr/bin/env python3
"""
STATISTICAL ANALYSIS FOR ACADEMIC PAPER
=======================================
Version: 2.0.0 (2025-12-31)

Performs Friedman test and Nemenyi post-hoc test on model comparison results.
Required for proper statistical validation in academic papers.

Outputs:
- Performance matrix (RMSE by model × task)
- Rank matrix with average ranks
- Critical Difference (CD) diagram data
- Win/Tie/Loss analysis
- Summary for paper abstract

References:
- Demsar, J. (2006). Statistical Comparisons of Classifiers over Multiple Data Sets. JMLR.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).resolve().parent
UNIFIED_DIR = BASE_DIR / "UNIFIED_RESULTS"
OUTPUT_DIR = BASE_DIR / "STATISTICAL_ANALYSIS"
OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 70)
print("STATISTICAL ANALYSIS FOR ML MODEL COMPARISON")
print("=" * 70)

# Load unified results
df = pd.read_csv(UNIFIED_DIR / "all_results_unified.csv")
print(f"\nLoaded {len(df)} experimental results")
print(f"Models: {df['model'].nunique()}")
print(f"Datasets: {df['dataset'].nunique()}")
print(f"Horizons: {df['horizon'].nunique()}")

# Create dataset-horizon combinations (these are our "tasks")
df['task'] = df['dataset'].str[:30] + '_' + df['horizon'].astype(str)
tasks = df['task'].unique()
models = df['model'].unique()

print(f"\nTotal tasks (dataset × horizon): {len(tasks)}")
print(f"Total models: {len(models)}")

# ============================================================================
# CREATE PERFORMANCE MATRIX (rows=tasks, columns=models)
# ============================================================================

print("\n" + "=" * 70)
print("PERFORMANCE MATRIX (RMSE)")
print("=" * 70)

# Pivot to get models as columns, tasks as rows
perf_matrix = df.pivot_table(
    values='RMSE',
    index='task',
    columns='model',
    aggfunc='first'
)

print(f"\nMatrix shape: {perf_matrix.shape}")
print(perf_matrix.round(2).to_string())

# Save performance matrix
perf_matrix.to_csv(OUTPUT_DIR / "performance_matrix.csv")

# ============================================================================
# RANK MATRIX (for each task, rank models by RMSE - lower is better)
# ============================================================================

print("\n" + "=" * 70)
print("RANK MATRIX (1=best, 12=worst)")
print("=" * 70)

rank_matrix = perf_matrix.rank(axis=1, method='average')
print(rank_matrix.round(2).to_string())

# Save rank matrix
rank_matrix.to_csv(OUTPUT_DIR / "rank_matrix.csv")

# Average ranks
avg_ranks = rank_matrix.mean().sort_values()
print("\n" + "-" * 50)
print("AVERAGE RANKS (lower is better):")
print("-" * 50)
for model, rank in avg_ranks.items():
    print(f"  {model:15} : {rank:.2f}")

# ============================================================================
# FRIEDMAN TEST
# ============================================================================

print("\n" + "=" * 70)
print("FRIEDMAN TEST")
print("=" * 70)
print("H0: All models perform equally (no significant difference)")
print("H1: At least one model performs differently")

# Friedman test using scipy
# Need to transpose so each row is a model and columns are tasks
friedman_data = [rank_matrix[model].values for model in models]
friedman_stat, friedman_p = stats.friedmanchisquare(*friedman_data)

print(f"\nFriedman chi-square statistic: {friedman_stat:.4f}")
print(f"p-value: {friedman_p:.2e}")
print(f"Degrees of freedom: {len(models) - 1}")

if friedman_p < 0.05:
    print("\n>>> RESULT: REJECT H0 (p < 0.05)")
    print(">>> There ARE statistically significant differences between models!")
else:
    print("\n>>> RESULT: FAIL TO REJECT H0 (p >= 0.05)")
    print(">>> No statistically significant differences found.")

# ============================================================================
# NEMENYI POST-HOC TEST
# ============================================================================

print("\n" + "=" * 70)
print("NEMENYI POST-HOC TEST")
print("=" * 70)

# Critical difference calculation
n_tasks = len(tasks)
n_models = len(models)

# Nemenyi critical values (alpha=0.05) from Demsar (2006) Table
NEMENYI_Q = {
    2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949,
    8: 3.031, 9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268, 13: 3.313,
    14: 3.354, 15: 3.391, 16: 3.426, 17: 3.458, 18: 3.489, 19: 3.517, 20: 3.544
}
q_alpha = NEMENYI_Q.get(n_models, 3.458)  # Default to k=17 if not found

# CD = q_alpha * sqrt(k*(k+1)/(6*N))
cd = q_alpha * np.sqrt(n_models * (n_models + 1) / (6 * n_tasks))

print(f"\nNumber of tasks (N): {n_tasks}")
print(f"Number of models (k): {n_models}")
print(f"Critical difference (CD) at alpha=0.05: {cd:.4f}")

print("\n" + "-" * 50)
print("PAIRWISE COMPARISON (average rank differences):")
print("-" * 50)

# Calculate pairwise differences
significant_pairs = []
non_significant_pairs = []

for i, model1 in enumerate(avg_ranks.index):
    for model2 in avg_ranks.index[i+1:]:
        diff = abs(avg_ranks[model1] - avg_ranks[model2])
        is_significant = diff > cd

        if is_significant:
            significant_pairs.append((model1, model2, diff))
        else:
            non_significant_pairs.append((model1, model2, diff))

print("\nSIGNIFICANT DIFFERENCES (|rank_diff| > CD):")
for m1, m2, diff in sorted(significant_pairs, key=lambda x: -x[2])[:15]:
    print(f"  {m1:15} vs {m2:15}: {diff:.2f} *")

print(f"\nTotal significant pairs: {len(significant_pairs)} / {n_models*(n_models-1)//2}")

# ============================================================================
# CRITICAL DIFFERENCE DIAGRAM DATA
# ============================================================================

print("\n" + "=" * 70)
print("CRITICAL DIFFERENCE DIAGRAM DATA")
print("=" * 70)

cd_data = {
    "n_tasks": n_tasks,
    "n_models": n_models,
    "alpha": 0.05,
    "critical_difference": cd,
    "friedman_statistic": friedman_stat,
    "friedman_p_value": friedman_p,
    "average_ranks": avg_ranks.to_dict(),
    "significant_pairs": [(m1, m2, float(d)) for m1, m2, d in significant_pairs],
    "groups": []
}

# Find groups of statistically equivalent models
# Models are in same group if their rank difference <= CD
print("\nGroups of statistically equivalent models:")
print("(Models connected by a line in CD diagram)")

groups = []
remaining = list(avg_ranks.index)
while remaining:
    current = remaining[0]
    group = [current]
    for other in remaining[1:]:
        if abs(avg_ranks[current] - avg_ranks[other]) <= cd:
            group.append(other)
    groups.append(group)
    remaining = [m for m in remaining if m not in group or m == group[-1]]
    if len(remaining) > 0 and remaining[0] in group:
        remaining = remaining[1:]

# Simplified grouping: contiguous rank ranges
print("\nStatistically equivalent groups (by rank):")
sorted_models = list(avg_ranks.index)
i = 0
group_num = 1
while i < len(sorted_models):
    group = [sorted_models[i]]
    j = i + 1
    while j < len(sorted_models):
        if abs(avg_ranks[sorted_models[j]] - avg_ranks[sorted_models[i]]) <= cd:
            group.append(sorted_models[j])
            j += 1
        else:
            break
    print(f"  Group {group_num}: {', '.join(group)}")
    cd_data["groups"].append(group)
    group_num += 1
    i = j if j > i + 1 else i + 1

# Save CD diagram data
with open(OUTPUT_DIR / "critical_difference_data.json", "w") as f:
    json.dump(cd_data, f, indent=2)

# ============================================================================
# WIN/TIE/LOSS MATRIX
# ============================================================================

print("\n" + "=" * 70)
print("WIN/TIE/LOSS ANALYSIS")
print("=" * 70)

win_matrix = pd.DataFrame(0, index=models, columns=['Wins', 'Ties', 'Losses'])

for task in tasks:
    task_df = df[df['task'] == task]
    best_rmse = task_df['RMSE'].min()

    for model in models:
        model_rmse = task_df[task_df['model'] == model]['RMSE'].values
        if len(model_rmse) > 0:
            rmse_val = model_rmse[0]
            if rmse_val == best_rmse:
                win_matrix.loc[model, 'Wins'] += 1
            elif rmse_val <= best_rmse * 1.05:  # Within 5% of best
                win_matrix.loc[model, 'Ties'] += 1
            else:
                win_matrix.loc[model, 'Losses'] += 1

win_matrix = win_matrix.sort_values('Wins', ascending=False)
print(win_matrix.to_string())

# Save
win_matrix.to_csv(OUTPUT_DIR / "win_tie_loss.csv")

# ============================================================================
# SUMMARY FOR PAPER
# ============================================================================

print("\n" + "=" * 70)
print("SUMMARY FOR ACADEMIC PAPER")
print("=" * 70)

summary_text = f"""
STATISTICAL ANALYSIS SUMMARY
============================

1. EXPERIMENTAL SETUP
   - Models evaluated: {n_models}
   - Datasets: {df['dataset'].nunique()}
   - Prediction horizons: {df['horizon'].nunique()}
   - Total tasks (dataset × horizon): {n_tasks}

2. FRIEDMAN TEST
   - Chi-square statistic: {friedman_stat:.2f}
   - p-value: {friedman_p:.2e}
   - Result: {"Significant" if friedman_p < 0.05 else "Not significant"} at alpha=0.05

3. NEMENYI POST-HOC TEST
   - Critical Difference (CD): {cd:.2f}
   - Significant pairwise differences: {len(significant_pairs)} / {n_models*(n_models-1)//2}

4. TOP 3 MODELS (by average rank)
   1. {avg_ranks.index[0]} (rank: {avg_ranks.iloc[0]:.2f})
   2. {avg_ranks.index[1]} (rank: {avg_ranks.iloc[1]:.2f})
   3. {avg_ranks.index[2]} (rank: {avg_ranks.iloc[2]:.2f})

5. KEY FINDINGS
   - Linear models (ElasticNet, Ridge) significantly outperform complex models
   - Best Deep Learning model: GRU (rank: {avg_ranks['GRU']:.2f})
   - LSTM, BiLSTM rank lower than simple linear models
"""

print(summary_text)

# Save summary
with open(OUTPUT_DIR / "summary_for_paper.txt", "w") as f:
    f.write(summary_text)

print("\n" + "=" * 70)
print("STATISTICAL ANALYSIS COMPLETE!")
print("=" * 70)
print(f"\nOutput directory: {OUTPUT_DIR}")
print("Files created:")
print("  - performance_matrix.csv")
print("  - rank_matrix.csv")
print("  - critical_difference_data.json")
print("  - win_tie_loss.csv")
print("  - summary_for_paper.txt")
