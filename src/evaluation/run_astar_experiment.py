"""Runs evaluation episodes for the A* baseline."""

from src.environment.warehouse_env import WarehouseEnv
from src.evaluation.baselines import AStarBaseline


def run_astar_episode(env):
    """Evaluate one A* navigation episode."""

    baseline = AStarBaseline(env)

    path = baseline.get_path()

    if path is None:
        return {
            "success": False,
            "steps": 0,
            "path_length": 0.0
        }

    # A* path contains the sequence of positions.
    steps = len(path) - 1

    return {
        "success": True,
        "steps": steps,
        "path_length": float(steps)
    }


if __name__ == "__main__":
    env = WarehouseEnv()
    env.reset()
    result = run_astar_episode(env)

    print("A* Experiment Result")
    print("--------------------")
    print("Success:", result["success"])
    print("Steps:", result["steps"])
    print("Path length:", result["path_length"])