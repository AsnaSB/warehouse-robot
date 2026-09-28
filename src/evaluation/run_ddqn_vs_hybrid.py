"""
Dedicated Evaluation Script: Pure DDQN vs Hybrid DDQN+A*

Compares Pure DDQN against Hybrid DDQN+A* across OPEN, AISLE, and DENSE layouts
in both Static and Dynamic obstacle environments.
"""

import csv
import math
import os
import sys
import pandas as pd
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.astar.astar_planner import AStarPlanner
from src.astar.replanner import DynamicReplanner
from src.ddqn.agent import DDQNAgent
from src.environment.state_augmentation import StateAugmenter
from src.environment.warehouse_env import WarehouseEnv
from src.hybrid.hybrid_agent import HybridAgent
from src.hybrid.waypoint_manager import WaypointManager


# ============================================================
# CONFIGURATION
# ============================================================
SCENARIOS = ["open", "aisle", "dense"]
EPISODES = 20
GRID_SIZE = 12
MAX_STEPS = 200

PURE_MODEL = "experiments/checkpoints/pure_ddqn_ep2500.pth"
HYBRID_MODEL = "experiments/checkpoints/hybrid_ddqn_ep2500.pth"

OUTPUT_DYNAMIC_RAW = "outputs/results/ddqn_vs_hybrid_dynamic_raw.csv"
OUTPUT_DYNAMIC_SUMMARY = "outputs/results/ddqn_vs_hybrid_dynamic_summary.csv"
OUTPUT_CONTROLLED_SUMMARY = "outputs/results/ddqn_vs_hybrid_controlled_summary.csv"


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def load_agent(model_path):
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
    return agent.select_action(state)


def blocked_cells(env):
    obstacles = env.workers + env.dynamic_robots
    return {obs.position for obs in obstacles if getattr(obs, "active", True)}


def get_valid_actions(env, current_pos, blocked=None):
    if blocked is None:
        blocked = set()
    grid_size = env.grid_size
    valid = set()
    r, c = current_pos
    actions_dict = {
        0: (-1, 0),
        1: (-1, 1),
        2: (0, 1),
        3: (1, 1),
        4: (1, 0),
        5: (1, -1),
        6: (0, -1),
        7: (-1, -1),
    }
    for action_idx, (dr, dc) in actions_dict.items():
        nr, nc = r + dr, c + dc
        if 0 <= nr < grid_size and 0 <= nc < grid_size:
            if env._static_grid[nr, nc] != 1 and (nr, nc) not in blocked:
                valid.add(action_idx)
    return valid if valid else set(range(8))


def move_dynamic_obstacles(env):
    occupied = {env.robot_pos, env.goal_pos}
    for worker in env.workers:
        if not worker.active:
            continue
        if worker.patrol_route:
            next_index = worker.current_route_index + worker.direction
            if next_index >= len(worker.patrol_route) or next_index < 0:
                worker.direction *= -1
                next_index = worker.current_route_index + worker.direction
            target = worker.patrol_route[next_index]
            if target not in occupied:
                worker.position = target
                worker.current_route_index = next_index
                occupied.add(target)


def build_state(state_builder, env, waypoint, scenario):
    return state_builder.build_state(
        robot_position=env.robot_pos,
        goal_position=env.goal_pos,
        workers=env.workers,
        dynamic_robots=env.dynamic_robots,
        waypoint=waypoint,
        context=scenario,
        risk_level="low",
    )


# ============================================================
# EVALUATION LOGIC FOR PURE DDQN & HYBRID
# ============================================================
def execute_ddqn(env, agent, state_builder, scenario, is_dynamic=True):
    total_reward = 0.0
    total_path_length = 0.0
    collisions = 0
    steps = 0
    previous = env.robot_pos

    while steps < MAX_STEPS and env.robot_pos != env.goal_pos:
        if is_dynamic:
            move_dynamic_obstacles(env)

        state = build_state(state_builder, env, env.goal_pos, scenario)
        action = greedy_action(agent, state)
        previous = env.robot_pos

        _, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        total_path_length += math.sqrt(
            (env.robot_pos[0] - previous[0]) ** 2 + (env.robot_pos[1] - previous[1]) ** 2
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


def execute_hybrid(env, agent, state_builder, scenario, is_dynamic=True):
    planner = AStarPlanner(env._static_grid)
    replanner = DynamicReplanner(planner)
    waypoint_manager = WaypointManager()
    hybrid = HybridAgent(agent, waypoint_manager, replanner, planner)
    replans = 0

    original_replan = replanner.replan
    def counted_replan(*args, **kwargs):
        nonlocal replans
        res = original_replan(*args, **kwargs)
        if res:
            replans += 1
        return res

    replanner.replan = counted_replan
    waypoint_manager.set_path([])

    total_reward = 0.0
    total_path_length = 0.0
    collisions = 0
    steps = 0

    while steps < MAX_STEPS and env.robot_pos != env.goal_pos:
        if is_dynamic:
            move_dynamic_obstacles(env)

        blocked = blocked_cells(env) if is_dynamic else set()
        waypoint = hybrid.update_route_and_waypoint(env.robot_pos, env.goal_pos, blocked)
        state = build_state(state_builder, env, waypoint, scenario)

        valid_actions = get_valid_actions(env, env.robot_pos, blocked)
        action = hybrid.get_action(
            state=state,
            epsilon=0.0,
            current_pos=env.robot_pos,
            waypoint=waypoint,
            blocked_cells=blocked,
            valid_actions=valid_actions,
        )

        previous = env.robot_pos
        _, reward, terminated, truncated, info = env.step(action)

        total_reward += float(reward)
        total_path_length += math.sqrt(
            (env.robot_pos[0] - previous[0]) ** 2 + (env.robot_pos[1] - previous[1]) ** 2
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


import random
from src.environment.obstacles import Worker


def make_dynamic_obstacles(env, episode):
    """Generate deterministic dynamic workers along patrol routes across free cells."""
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


# ============================================================
# MAIN EVALUATION
# ============================================================
def main():
    print("=" * 70)
    print("PURE DDQN VS HYBRID DDQN + A* EVALUATION BENCHMARK")
    print("=" * 70)

    pure_agent = load_agent(PURE_MODEL)
    hybrid_agent = load_agent(HYBRID_MODEL)

    os.makedirs("outputs/results", exist_ok=True)
    raw_records = []

    for scenario in SCENARIOS:
        print(f"\n[EVALUATING DYNAMIC OBSTACLE SCENARIO: {scenario.upper()}]")
        env = WarehouseEnv(config=scenario, grid_size=GRID_SIZE, max_steps=MAX_STEPS)
        state_builder = StateAugmenter(static_grid=env._static_grid, local_radius=2, max_dynamic_obstacles=4)

        for ep in range(1, EPISODES + 1):
            seed = 1000 + ep

            # Pure DDQN
            env.reset(seed=seed)
            make_dynamic_obstacles(env, ep)
            res_pure = execute_ddqn(env, pure_agent, state_builder, scenario, is_dynamic=True)
            res_pure["scenario"] = scenario
            res_pure["method"] = "Pure DDQN"
            res_pure["episode"] = ep
            raw_records.append(res_pure)

            # Hybrid DDQN + A*
            env.reset(seed=seed)
            make_dynamic_obstacles(env, ep)
            res_hybrid = execute_hybrid(env, hybrid_agent, state_builder, scenario, is_dynamic=True)
            res_hybrid["scenario"] = scenario
            res_hybrid["method"] = "Hybrid DDQN+A*"
            res_hybrid["episode"] = ep
            raw_records.append(res_hybrid)

    df_raw = pd.DataFrame(raw_records)
    df_raw.to_csv(OUTPUT_DYNAMIC_RAW, index=False)

    # Compute Summary Table
    summary = df_raw.groupby(["scenario", "method"]).agg(
        success_rate=("success", "mean"),
        average_reward=("reward", "mean"),
        average_steps=("steps", "mean"),
        average_path_length=("path_length", "mean"),
        average_collisions=("collisions", "mean"),
        average_replans=("replans", "mean"),
    ).reset_index()

    summary.to_csv(OUTPUT_DYNAMIC_SUMMARY, index=False)

    print("\n" + "=" * 70)
    print("DYNAMIC EVALUATION SUMMARY: PURE DDQN VS HYBRID DDQN + A*")
    print("=" * 70)
    for scenario in SCENARIOS:
        print(f"\n--- Scenario: {scenario.upper()} ---")
        sc_df = summary[summary["scenario"] == scenario]
        for _, row in sc_df.iterrows():
            print(
                f"{row['method']:<18} | Success: {row['success_rate']*100:5.1f}% | "
                f"Reward: {row['average_reward']:8.2f} | Steps: {row['average_steps']:6.2f} | "
                f"Collisions: {row['average_collisions']:6.2f} | Replans: {row['average_replans']:4.2f}"
            )
    print("=" * 70)
    print(f"Results saved to:\n  - {OUTPUT_DYNAMIC_RAW}\n  - {OUTPUT_DYNAMIC_SUMMARY}")


if __name__ == "__main__":
    main()
