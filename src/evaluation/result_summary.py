"""Creates a comparison summary for navigation methods."""

import csv
import os

from src.evaluation.aggregator import aggregate_results


def create_summary(results_by_method,
                   output_file="results/summary.csv"):
    """Create and save an overall comparison summary."""

    summary = {}

    for method, results in results_by_method.items():
        summary[method] = aggregate_results(results)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    if not summary:
        return summary

    fields = [
        "method",
        "success_rate",
        "average_steps",
        "average_reward",
        "average_path_length",
        "average_path_efficiency",
        "collision_rate",
        "average_collisions",
        "average_replans"
    ]

    with open(output_file, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        for method, metrics in summary.items():
            row = {"method": method, **metrics}
            writer.writerow(row)

    return summary
