"""Evaluate A*, Pure DDQN and Hybrid DDQN+A* with moving obstacles."""

import math
import os
import random
import sys

import pandas as pd
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.astar.astar_planner import AStarPlanner
from src.astar.replanner import DynamicReplanner
from src.ddqn.agent import DDQNAgent
from src.environment.obstacles import Worker, DynamicRobot
from src.environment.state_augmentation import StateAugmenter
from src.environment.warehouse_env import WarehouseEnv
from src.hybrid.hybrid_agent import HybridAgent
from src.hybrid.waypoint_manager import WaypointManager


GRID_SIZE = 12
MAX_STEPS = 200
EPISODES = 10
SCENARIOS = ["open", "aisle", "dense"]

PURE_CHECKPOINT = "experiments/checkpoints/pure_ddqn_ep2500.pth"
HYBRID_CHECKPOINT = "experiments/checkpoints/hybrid_ddqn_ep2500.pth"
OUTPUT_FILE = "outputs/results/dynamic_results.csv"


def load_agent(model_path):
    """Load a trained DDQN checkpoint."""
    agent = DDQNAgent(state_dim=51, action_dim=8)
    weights = torch.load(model_path, map_location=agent.device)

    if isinstance(weights, dict) and "online_state" in weights:
        weights = weights["online_state"]

    agent.online_net.load_state_dict(weights)
    agent.target_net.load_state_dict(weights)
    agent.online_net.eval()
    agent.target_net.eval()
    return agent


def greedy_action(agent, state):
    """Select the greedy action."""
    return agent.select_action(state)


def blocked_cells(env):
    """Return active dynamic-obstacle positions."""
    obstacles = env.workers + env.dynamic_robots
    return {
        obstacle.position
        for obstacle in obstacles
        if obstacle.active
    }


def move_dynamic_obstacles(env):
    """Move workers and dynamic robots by one simulation tick."""
    occupied = {env.robot_pos, env.goal_pos}

    for worker in env.workers:
        if not worker.active:
            continue
        old_position = worker.position
        if worker.patrol_route:
            next_index = worker.current_route_index + worker.direction
            if next_index >= len(worker.patrol_route) or next_index < 0:
                worker.direction *= -1
                next_index = worker.current_route_index + worker.direction
            candidate = worker.patrol_route[next_index]
            if candidate not in occupied:
                worker.update()
        else:
            worker.update()
        occupied.add(worker.position)

    for robot in env.dynamic_robots:
        if robot.active:
            robot.update(list(occupied))
            occupied.add(robot.position)


def make_dynamic_obstacles(env, episode):
    """Create deterministic moving obstacles on static free cells."""
    free = [
        (r, c)
        for r in range(GRID_SIZE)
        for c in range(GRID_SIZE)
        if env._static_grid[r][c] == 0
    ]

    rng = random.Random(episode + sum(ord(ch) for ch in env.config_name))
    rng.shuffle(free)

    planner = AStarPlanner(env._static_grid)
    routes = []

    for i in range(len(free)):
        for j in range(i + 1, len(free)):
            path = planner.find_path(free[i], free[j])
            if path and len(path) >= 5:
                routes.append(path)
                break
        if len(routes) >= 2:
            break

    workers = []
    for path in routes[:2]:
        workers.append(
            Worker(
                position=path[0],
                movement_pattern="predefined_patrol",
                patrol_route=path,
                grid_size=(GRID_SIZE, GRID_SIZE),
            )
        )

    env.set_dynamic_obstacles(workers=workers, dynamic_robots=[])
    return workers


def build_state(state_builder, env, waypoint, scenario):
    """Build the trained 51-D state."""
    return state_builder.build_state(
        robot_position=env.robot_pos,
        goal_position=env.goal_pos,
        workers=env.workers,
        dynamic_robots=env.dynamic_robots,
        waypoint=waypoint,
        context=scenario,
        risk_level="low",
    )


def path_length(path):
    """Calculate Euclidean path length."""
    if not path or len(path) < 2:
        return 0.0
    return sum(
        math.sqrt(
            (b[0] - a[0]) ** 2 +
            (b[1] - a[1]) ** 2
        )
        for a, b in zip(path, path[1:])
    )


def execute_astar(env):
    """Run A* with dynamic replanning."""
    planner = AStarPlanner(env._static_grid)
    replanner = DynamicReplanner(planner)
    path = planner.find_path(env.robot_pos, env.goal_pos, blocked_cells(env))
    replans = 0
    total_reward = 0.0
    total_path_length = 0.0
    collisions = 0
    steps = 0
    action_map = {
        (-1, 0): 0, (-1, 1): 1, (0, 1): 2, (1, 1): 3,
        (1, 0): 4, (1, -1): 5, (0, -1): 6, (-1, -1): 7,
    }

    while steps < MAX_STEPS and env.robot_pos != env.goal_pos:
        if path is None:
            break

        blocked = blocked_cells(env)
        if replanner.is_path_blocked(path, blocked):
            new_path = replanner.replan(
                current_position=env.robot_pos,
                goal=env.goal_pos,
                current_path=path,
                blocked_cells=blocked,
            )
            if new_path != path:
                replans += 1
            path = new_path

        if not path or env.robot_pos not in path:
            break

        index = path.index(env.robot_pos)
        if index + 1 >= len(path):
            break

        target = path[index + 1]
        current = env.robot_pos
        action = action_map.get((target[0] - current[0], target[1] - current[1]))
        if action is None:
            break

        move_dynamic_obstacles(env)
        previous = env.robot_pos
        _, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        total_path_length += math.sqrt(
            (env.robot_pos[0] - previous[0]) ** 2 +
            (env.robot_pos[1] - previous[1]) ** 2
        )
        collisions += int(info.get("collided", False))
        steps += 1
        if terminated or truncated:
            break

    return {
        "success": env.robot_pos == env.goal_pos,
        "steps": steps,
        "reward": total_reward,
        "path_length": total_path_length,
        "collisions": collisions,
        "replans": replans,
    }


def execute_ddqn(env, agent, state_builder, scenario):
    """Run Pure DDQN while obstacles move."""
    total_reward = 0.0
    total_path_length = 0.0
    collisions = 0
    steps = 0

    while steps < MAX_STEPS and env.robot_pos != env.goal_pos:
        move_dynamic_obstacles(env)
        state = build_state(state_builder, env, env.goal_pos, scenario)
        action = greedy_action(agent, state)
        previous = env.robot_pos
        _, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        total_path_length += math.sqrt(
            (env.robot_pos[0] - previous[0]) ** 2 +
            (env.robot_pos[1] - previous[1]) ** 2
        )
        collisions += int(info.get("collided", False))
        steps += 1
        if terminated or truncated:
            break

    return {
        "success": env.robot_pos == env.goal_pos,
        "steps": steps,
        "reward": total_reward,
        "path_length": total_path_length,
        "collisions": collisions,
        "replans": 0,
    }


def execute_hybrid(env, agent, state_builder, scenario):
    """Run Hybrid DDQN+A* with dynamic replanning."""
    planner = AStarPlanner(env._static_grid)
    replanner = DynamicReplanner(planner)
    waypoint_manager = WaypointManager()
    hybrid = HybridAgent(agent, waypoint_manager, replanner, planner)
    replans = 0

    original_replan = replanner.replan

    def counted_replan(*args, **kwargs):
        nonlocal replans
        replans += 1
        return original_replan(*args, **kwargs)

    replanner.replan = counted_replan
    waypoint_manager.set_path([])

    total_reward = 0.0
    total_path_length = 0.0
    collisions = 0
    steps = 0

    while steps < MAX_STEPS and env.robot_pos != env.goal_pos:
        move_dynamic_obstacles(env)
        blocked = blocked_cells(env)
        waypoint = hybrid.update_route_and_waypoint(
            env.robot_pos,
            env.goal_pos,
            blocked,
        )

        state = build_state(state_builder, env, waypoint, scenario)
        action = hybrid.ddqn.select_action(state)
        previous = env.robot_pos
        _, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        total_path_length += math.sqrt(
            (env.robot_pos[0] - previous[0]) ** 2 +
            (env.robot_pos[1] - previous[1]) ** 2
        )
        collisions += int(info.get("collided", False))
        steps += 1
        if terminated or truncated:
            break

    return {
        "success": env.robot_pos == env.goal_pos,
        "steps": steps,
        "reward": total_reward,
        "path_length": total_path_length,
        "collisions": collisions,
        "replans": replans,
    }


def main():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    print("\nDYNAMIC WAREHOUSE ROBOT EVALUATION")
    print("Loading trained DDQN models...")
    pure_agent = load_agent(PURE_CHECKPOINT)
    hybrid_agent = load_agent(HYBRID_CHECKPOINT)
    print("Models loaded successfully.\n")

    rows = []

    for scenario in SCENARIOS:
        print(f"SCENARIO: {scenario.upper()}")

        for method in ["A*", "Pure DDQN", "Hybrid DDQN+A*"]:
            print(f"{method}:")

            for episode in range(1, EPISODES + 1):
                env = WarehouseEnv(
                    config=scenario,
                    grid_size=GRID_SIZE,
                    max_steps=MAX_STEPS,
                    seed=episode,
                )
                env.reset(seed=episode)
                make_dynamic_obstacles(env, episode)

                state_builder = StateAugmenter(
                    env._static_grid,
                    local_radius=2,
                    max_dynamic_obstacles=4,
                )

                if method == "A*":
                    result = execute_astar(env)
                elif method == "Pure DDQN":
                    result = execute_ddqn(
                        env, pure_agent, state_builder, scenario
                    )
                else:
                    result = execute_hybrid(
                        env, hybrid_agent, state_builder, scenario
                    )

                rows.append({
                    "method": method,
                    "scenario": scenario,
                    "episode": episode,
                    "success": result["success"],
                    "steps": result["steps"],
                    "total_reward": result["reward"],
                    "path_length": result["path_length"],
                    "collisions": result["collisions"],
                    "replans": result["replans"],
                })

                print(
                    f"  ep{episode}: success={result['success']}, "
                    f"steps={result['steps']}, "
                    f"collisions={result['collisions']}, "
                    f"replans={result['replans']}"
                )

                env.close()

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_FILE, index=False)

    print("\nDynamic experiments completed.")
    print(f"Results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
