"""Runs evaluation experiments and stores episode results."""

import csv
import os

from src.evaluation.evaluator import Evaluator


def run_experiments(env, agent, method_name, episodes=10,
                    output_file="results/evaluation_results.csv"):
    """Run multiple episodes for one navigation method."""

    evaluator = Evaluator(env)
    results = []

    for episode in range(episodes):
        # Evaluate one navigation episode.
        metrics = evaluator.evaluate_episode(agent)

        results.append({
            "method": method_name,
            "episode": episode + 1,
            "success": metrics.success,
            "steps": metrics.steps,
            "total_reward": metrics.total_reward,
            "path_length": metrics.path_length,
            "optimal_path_length": metrics.optimal_path_length,
            "path_efficiency": metrics.path_efficiency,
            "collisions": metrics.collisions,
            "replans": metrics.replans
        })

    # Create the results folder if it does not exist.
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Save all episode results to a CSV file.
    with open(output_file, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    return results