"""
Generate Comparison Plots for Pure DDQN vs Hybrid DDQN + A*.

Creates publication-ready visualizations comparing Pure DDQN and Hybrid DDQN + A*
across OPEN, AISLE, and DENSE dynamic obstacle scenarios.
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def generate_ddqn_vs_hybrid_plots():
    summary_path = "outputs/results/ddqn_vs_hybrid_dynamic_summary.csv"
    if not os.path.exists(summary_path):
        print(f"Error: {summary_path} not found. Run run_ddqn_vs_hybrid.py first.")
        return

    df = pd.read_csv(summary_path)
    os.makedirs("outputs/figures", exist_ok=True)

    # Set aesthetic style
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial"],
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "figure.titlesize": 16,
    })

    scenarios = ["open", "aisle", "dense"]
    colors = {"Pure DDQN": "#e74c3c", "Hybrid DDQN+A*": "#2ecc71"}

    # Figure: 2x2 Subplots Comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Success Rate
    ax1 = axes[0, 0]
    sns.barplot(
        data=df,
        x="scenario",
        y="success_rate",
        hue="method",
        palette=colors,
        ax=ax1,
        edgecolor="black",
        linewidth=1,
    )
    ax1.set_title("Success Rate Comparison (Dynamic Obstacles)", fontweight="bold")
    ax1.set_xlabel("Warehouse Scenario")
    ax1.set_ylabel("Success Rate (%)")
    ax1.set_ylim(0, 1.15)
    for p in ax1.patches:
        height = p.get_height()
        if height > 0:
            ax1.annotate(
                f"{height*100:.0f}%",
                (p.get_x() + p.get_width() / 2.0, height + 0.03),
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=10,
            )
    ax1.set_xticklabels(["Open", "Aisle", "Dense"])
    ax1.get_legend().remove()

    # 2. Average Steps
    ax2 = axes[0, 1]
    sns.barplot(
        data=df,
        x="scenario",
        y="average_steps",
        hue="method",
        palette=colors,
        ax=ax2,
        edgecolor="black",
        linewidth=1,
    )
    ax2.set_title("Average Navigation Steps (Lower is Better)", fontweight="bold")
    ax2.set_xlabel("Warehouse Scenario")
    ax2.set_ylabel("Average Steps per Episode")
    for p in ax2.patches:
        height = p.get_height()
        if height > 0:
            ax2.annotate(
                f"{height:.1f}",
                (p.get_x() + p.get_width() / 2.0, height + 2),
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=10,
            )
    ax2.set_xticklabels(["Open", "Aisle", "Dense"])
    ax2.get_legend().remove()

    # 3. Average Reward
    ax3 = axes[1, 0]
    sns.barplot(
        data=df,
        x="scenario",
        y="average_reward",
        hue="method",
        palette=colors,
        ax=ax3,
        edgecolor="black",
        linewidth=1,
    )
    ax3.set_title("Average Episode Reward (Higher is Better)", fontweight="bold")
    ax3.set_xlabel("Warehouse Scenario")
    ax3.set_ylabel("Cumulative Reward")
    for p in ax3.patches:
        height = p.get_height()
        va = "bottom" if height >= 0 else "top"
        offset = 50 if height >= 0 else -150
        ax3.annotate(
            f"{height:.1f}",
            (p.get_x() + p.get_width() / 2.0, height + offset),
            ha="center",
            va=va,
            fontweight="bold",
            fontsize=9,
        )
    ax3.set_xticklabels(["Open", "Aisle", "Dense"])
    ax3.get_legend().remove()

    # 4. Average Collisions
    ax4 = axes[1, 1]
    sns.barplot(
        data=df,
        x="scenario",
        y="average_collisions",
        hue="method",
        palette=colors,
        ax=ax4,
        edgecolor="black",
        linewidth=1,
    )
    ax4.set_title("Average Collisions (Lower is Better)", fontweight="bold")
    ax4.set_xlabel("Warehouse Scenario")
    ax4.set_ylabel("Collisions per Episode")
    for p in ax4.patches:
        height = p.get_height()
        if height > 0:
            ax4.annotate(
                f"{height:.1f}",
                (p.get_x() + p.get_width() / 2.0, height + 1),
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=10,
            )
    ax4.set_xticklabels(["Open", "Aisle", "Dense"])

    # Shared Legend
    handles, labels = ax4.get_legend_handles_labels()
    ax4.get_legend().remove()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=True,
        fontsize=12,
        title="Navigation Architecture",
        title_fontsize=13,
    )

    plt.suptitle(
        "Pure DDQN vs Hybrid DDQN+A* Performance Comparison across Warehouse Scenarios",
        fontsize=16,
        fontweight="bold",
        y=1.06,
    )

    plt.tight_layout()
    output_png = "outputs/figures/ddqn_vs_hybrid_comparison.png"
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Comparison plots successfully generated and saved to:\n  - {output_png}")


if __name__ == "__main__":
    generate_ddqn_vs_hybrid_plots()
