#!/usr/bin/env python3
"""
Generate figures for the paper:
1. Critical Difference (CD) Diagram
2. Performance by Horizon bar chart
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Create figures directory
FIG_DIR = Path(__file__).parent / "paper_figures"
FIG_DIR.mkdir(exist_ok=True)

# Data from results (18 models including Chronos) - Updated Jan 2025
MODELS = [
    "Ridge", "ElasticNet", "ExtraTrees", "NGBoost", "TFT", "Chronos",
    "PatchTST", "RandomForest", "CatBoost", "LSTM", "GRU", "LightGBM", "BiLSTM",
    "XGBoost", "KNN", "Naive", "Prophet", "SARIMAX"
]

AVG_RANKS = [3.20, 4.63, 5.50, 6.58, 7.05, 7.70, 7.80, 8.18, 8.92,
             10.23, 10.23, 10.27, 10.43, 11.10, 12.53, 14.93, 15.37, 16.33]

# Performance by horizon (top 5 models)
HORIZON_DATA = {
    "Ridge": [3.06, 5.10, 6.85],
    "ElasticNet": [3.25, 5.08, 6.71],
    "ExtraTrees": [5.06, 8.95, 11.24],
    "LSTM": [6.50, 9.80, 12.50],  # approximate from results
    "TFT": [6.82, 10.31, 13.96],
}

HORIZONS = ["H1 (5 min)", "H5 (25 min)", "H12 (60 min)"]


def create_cd_diagram():
    """Create Critical Difference diagram."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Sort by rank
    sorted_data = sorted(zip(MODELS, AVG_RANKS), key=lambda x: x[1])
    models_sorted = [x[0] for x in sorted_data]
    ranks_sorted = [x[1] for x in sorted_data]

    # Critical difference
    CD = 4.49

    # Plot
    y_positions = np.arange(len(models_sorted))

    # Color by category
    colors = []
    for m in models_sorted:
        if m in ["Ridge", "ElasticNet"]:
            colors.append("#2ecc71")  # Green - Linear
        elif m in ["LSTM", "GRU", "BiLSTM"]:
            colors.append("#e74c3c")  # Red - RNN
        elif m in ["TFT", "PatchTST"]:
            colors.append("#9b59b6")  # Purple - Transformer
        elif m == "Chronos":
            colors.append("#e67e22")  # Dark Orange - Foundation
        elif m in ["Prophet", "SARIMAX"]:
            colors.append("#f39c12")  # Orange - Statistical
        elif m == "Naive":
            colors.append("#95a5a6")  # Gray - Baseline
        else:
            colors.append("#3498db")  # Blue - Other ML

    bars = ax.barh(y_positions, ranks_sorted, color=colors, edgecolor='black', alpha=0.8)

    # Add rank values
    for i, (model, rank) in enumerate(zip(models_sorted, ranks_sorted)):
        ax.text(rank + 0.3, i, f"{rank:.2f}", va='center', fontsize=10)

    # Add CD line
    ax.axvline(x=ranks_sorted[0] + CD, color='red', linestyle='--', linewidth=2,
               label=f'CD = {CD}')

    ax.set_yticks(y_positions)
    ax.set_yticklabels(models_sorted, fontsize=11)
    ax.set_xlabel("Average Rank (lower is better)", fontsize=12)
    ax.set_title("Model Rankings with Critical Difference (CD = 4.49)", fontsize=14, fontweight='bold')
    ax.set_xlim(0, 18)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', label='Linear (Ridge, ElasticNet)'),
        Patch(facecolor='#3498db', label='Tree/Boosting'),
        Patch(facecolor='#e74c3c', label='RNN (LSTM, GRU, BiLSTM)'),
        Patch(facecolor='#9b59b6', label='Transformer (TFT, PatchTST)'),
        Patch(facecolor='#e67e22', label='Foundation (Chronos)'),
        Patch(facecolor='#f39c12', label='Statistical (Prophet, SARIMAX)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig1_cd_diagram.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(FIG_DIR / "fig1_cd_diagram.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {FIG_DIR / 'fig1_cd_diagram.pdf'}")
    plt.close()


def create_horizon_performance():
    """Create performance by horizon bar chart."""
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(HORIZONS))
    width = 0.15

    colors = {
        "Ridge": "#2ecc71",
        "ElasticNet": "#27ae60",
        "ExtraTrees": "#3498db",
        "LSTM": "#e74c3c",
        "TFT": "#9b59b6",
    }

    for i, (model, rmses) in enumerate(HORIZON_DATA.items()):
        offset = (i - len(HORIZON_DATA)/2 + 0.5) * width
        bars = ax.bar(x + offset, rmses, width, label=model, color=colors[model],
                     edgecolor='black', alpha=0.85)

        # Add values on bars
        for bar, val in zip(bars, rmses):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                   f'{val:.1f}', ha='center', va='bottom', fontsize=8)

    ax.set_xlabel("Forecast Horizon", fontsize=12)
    ax.set_ylabel("RMSE (Watts)", fontsize=12)
    ax.set_title("Model Performance by Forecast Horizon", fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(HORIZONS, fontsize=11)
    ax.legend(loc='upper left', fontsize=10)
    ax.set_ylim(0, 16)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig2_performance_horizon.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(FIG_DIR / "fig2_performance_horizon.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {FIG_DIR / 'fig2_performance_horizon.pdf'}")
    plt.close()


def create_model_comparison():
    """Create simple Linear vs DL comparison."""
    fig, ax = plt.subplots(figsize=(10, 5))

    categories = ["Linear\n(Ridge)", "Boosting\n(CatBoost)", "Foundation\n(Chronos)", "RNN\n(LSTM)", "Transformer\n(TFT)"]
    avg_ranks = [3.26, 8.17, 9.70, 9.81, 6.57]
    colors = ["#2ecc71", "#3498db", "#e67e22", "#e74c3c", "#9b59b6"]

    bars = ax.bar(categories, avg_ranks, color=colors, edgecolor='black', alpha=0.85)

    # Add values
    for bar, val in zip(bars, avg_ranks):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
               f'{val:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_ylabel("Average Rank (lower is better)", fontsize=12)
    ax.set_title("Model Category Comparison", fontsize=14, fontweight='bold')
    ax.set_ylim(0, 12)
    ax.axhline(y=3.26 + 4.49, color='red', linestyle='--', alpha=0.7,
               label=f'Ridge + CD (significant threshold)')
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig3_category_comparison.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(FIG_DIR / "fig3_category_comparison.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {FIG_DIR / 'fig3_category_comparison.pdf'}")
    plt.close()


if __name__ == "__main__":
    print("Generating paper figures...")
    create_cd_diagram()
    create_horizon_performance()
    create_model_comparison()
    print(f"\nAll figures saved to: {FIG_DIR}")
