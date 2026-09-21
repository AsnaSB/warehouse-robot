"""Aggregates episode results into overall evaluation metrics."""

from statistics import mean


def aggregate_results(results):
    """Calculate summary metrics from episode results."""

    if not results:
        return {}

    return {
        "success_rate": mean(r["success"] for r in results),
        "average_steps": mean(r["steps"] for r in results),
        "average_reward": mean(r["total_reward"] for r in results),
        "average_path_length": mean(r["path_length"] for r in results),
        "average_path_efficiency": mean(
            r["path_efficiency"] for r in results
        ),
        "collision_rate": mean(
            r["collisions"] > 0 for r in results
        ),
        "average_collisions": mean(
            r["collisions"] for r in results
        ),
        "average_replans": mean(
            r["replans"] for r in results
        )
    }