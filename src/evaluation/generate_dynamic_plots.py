"""Generate summary and graphs for dynamic evaluation."""

import os

import pandas as pd
import matplotlib.pyplot as plt


INPUT_FILE = "outputs/results/dynamic_results.csv"
OUTPUT_DIR = "outputs/figures"


def load_results():
    """Load dynamic experiment results."""
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
            collision_rate=("collisions", lambda x: (x > 0).mean()),
            average_replans=("replans", "mean"),
        )
        .reset_index()
    )


def plot_metric(summary, metric, title, ylabel, filename):
    """Create grouped bar chart."""
    scenarios = ["open", "aisle", "dense"]
    methods = ["A*", "Pure DDQN", "Hybrid DDQN+A*"]

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

            values.append(row.iloc[0][metric] if not row.empty else 0)

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
    ax.set_xticklabels(["Open", "Aisle", "Dense"])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend()

    plt.tight_layout()

    output_path = os.path.join(
        OUTPUT_DIR,
        filename
    )

    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Created: {output_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = load_results()
    summary = create_summary(df)

    summary_file = (
        "outputs/results/dynamic_summary.csv"
    )

    summary.to_csv(
        summary_file,
        index=False
    )

    print(f"Created: {summary_file}")

    plot_metric(
        summary,
        "success_rate",
        "Dynamic Success Rate",
        "Success Rate",
        "dynamic_success_rate.png"
    )

    plot_metric(
        summary,
        "average_steps",
        "Dynamic Average Steps",
        "Average Steps",
        "dynamic_average_steps.png"
    )

    plot_metric(
        summary,
        "average_reward",
        "Dynamic Average Reward",
        "Average Reward",
        "dynamic_reward_comparison.png"
    )

    plot_metric(
        summary,
        "average_path_length",
        "Dynamic Path Length",
        "Average Path Length",
        "dynamic_path_length_comparison.png"
    )

    plot_metric(
        summary,
        "collision_rate",
        "Dynamic Collision Rate",
        "Collision Rate",
        "dynamic_collision_comparison.png"
    )

    plot_metric(
        summary,
        "average_replans",
        "Dynamic Replanning",
        "Average Replans",
        "dynamic_replanning_comparison.png"
    )

    print()
    print("All dynamic evaluation graphs generated.")


if __name__ == "__main__":
    main()