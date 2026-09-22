"""Generate comparison graphs from controlled experiment results."""

import os

import pandas as pd
import matplotlib.pyplot as plt


INPUT_FILE = "outputs/results/controlled_results.csv"
OUTPUT_DIR = "outputs/figures"


def load_results():
    """Load controlled experiment results."""
    return pd.read_csv(INPUT_FILE)


def create_summary(df):
    """Create scenario-method summary."""

    return (
        df.groupby(["scenario", "method"])
        .agg(
            success_rate=("success", "mean"),
            average_reward=("total_reward", "mean"),
            average_steps=("steps", "mean"),
            average_path_length=("path_length", "mean"),
            average_collisions=("collisions", "mean"),
            average_replans=("replans", "mean"),
        )
        .reset_index()
    )


def plot_metric(summary, metric, title, ylabel, filename):
    """Create grouped bar chart for one metric."""

    scenarios = ["open", "aisle", "dense"]

    methods = [
        "A*",
        "Pure DDQN",
        "Hybrid DDQN+A*"
    ]

    fig, ax = plt.subplots(figsize=(9, 5))

    x = range(len(scenarios))
    width = 0.25

    for i, method in enumerate(methods):

        values = []

        for scenario in scenarios:

            row = summary[
                (summary["scenario"] == scenario)
                & (summary["method"] == method)
            ]

            if row.empty:
                values.append(0)
            else:
                values.append(row.iloc[0][metric])

        positions = [
            value + (i - 1) * width
            for value in x
        ]

        ax.bar(
            positions,
            values,
            width=width,
            label=method
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        ["Open", "Aisle", "Dense"]
    )

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend()

    plt.tight_layout()

    output_path = os.path.join(
        OUTPUT_DIR,
        filename
    )

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()

    print(f"Created: {output_path}")


def main():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    df = load_results()

    summary = create_summary(df)

    # Save summary table.
    summary_file = (
        "outputs/results/controlled_summary.csv"
    )

    summary.to_csv(
        summary_file,
        index=False
    )

    print(f"Created: {summary_file}")

    # Success rate.
    plot_metric(
        summary,
        "success_rate",
        "Success Rate by Scenario and Method",
        "Success Rate",
        "success_rate.png"
    )

    # Reward.
    plot_metric(
        summary,
        "average_reward",
        "Average Episode Reward by Scenario",
        "Average Reward",
        "reward_comparison.png"
    )

    # Collision count.
    plot_metric(
        summary,
        "average_collisions",
        "Average Collisions by Scenario and Method",
        "Average Collisions",
        "collision_comparison.png"
    )

    # Path length.
    plot_metric(
        summary,
        "average_path_length",
        "Average Path Length by Scenario and Method",
        "Path Length",
        "path_length_comparison.png"
    )

    print()
    print("All evaluation graphs generated.")


if __name__ == "__main__":
    main()