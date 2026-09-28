"""Evaluate A*, Pure DDQN and Hybrid DDQN+A* with moving obstacles."""

import math
import os
import random
import sys

import pandas as pd
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.astar.astar_planner import AStarPlanner
from src.astar.replanner import DynamicReplanner
from src.ddqn.agent import DDQNAgent
from src.environment.obstacles import Worker
from src.environment.state_augmentation import StateAugmenter
from src.environment.warehouse_env import WarehouseEnv
from src.hybrid.hybrid_agent import HybridAgent
from src.hybrid.waypoint_manager import WaypointManager


# ============================================================
# CONFIGURATION
# ============================================================

GRID_SIZE = 12
MAX_STEPS = 200

# Number of evaluation episodes per scenario and method.
EPISODES = 20

SCENARIOS = [
    "open",
    "aisle",
    "dense",
]

PURE_CHECKPOINT = (
    "experiments/checkpoints/pure_ddqn_ep2500.pth"
)

HYBRID_CHECKPOINT = (
    "experiments/checkpoints/hybrid_ddqn_ep2500.pth"
)

OUTPUT_FILE = (
    "outputs/results/dynamic_results.csv"
)

SUMMARY_FILE = (
    "outputs/results/dynamic_summary.csv"
)


# ============================================================
# MODEL LOADING
# ============================================================

def load_agent(model_path):
    """Load a trained DDQN checkpoint."""

    agent = DDQNAgent(
        state_dim=51,
        action_dim=8,
    )

    weights = torch.load(
        model_path,
        map_location=agent.device,
    )

    # Support checkpoints that contain "online_state".
    if isinstance(weights, dict) and "online_state" in weights:
        weights = weights["online_state"]

    agent.online_net.load_state_dict(weights)
    agent.target_net.load_state_dict(weights)

    agent.online_net.eval()
    agent.target_net.eval()

    return agent


def greedy_action(agent, state):
    """
    Select the DDQN action during evaluation.

    The current DDQNAgent.select_action() implementation
    does not accept an epsilon keyword argument, so the
    agent's existing evaluation behaviour is used here.
    """

    return agent.select_action(state)


# ============================================================
# DYNAMIC OBSTACLES
# ============================================================

def blocked_cells(env):
    """Return active dynamic-obstacle positions."""

    obstacles = (
        env.workers +
        env.dynamic_robots
    )

    return {
        obstacle.position
        for obstacle in obstacles
        if obstacle.active
    }


def move_dynamic_obstacles(env):
    """
    Move workers and dynamic robots by one simulation tick.
    """

    occupied = {
        env.robot_pos,
        env.goal_pos,
    }

    # --------------------------------------------------------
    # Move workers
    # --------------------------------------------------------

    for worker in env.workers:

        if not worker.active:
            continue

        if worker.patrol_route:

            next_index = (
                worker.current_route_index
                + worker.direction
            )

            # Reverse patrol direction at route boundaries.
            if (
                next_index >= len(worker.patrol_route)
                or next_index < 0
            ):
                worker.direction *= -1

                next_index = (
                    worker.current_route_index
                    + worker.direction
                )

            candidate = worker.patrol_route[next_index]

            if candidate not in occupied:
                worker.update()

        else:
            worker.update()

        occupied.add(worker.position)

    # --------------------------------------------------------
    # Move dynamic robots
    # --------------------------------------------------------

    for robot in env.dynamic_robots:

        if robot.active:
            robot.update(list(occupied))
            occupied.add(robot.position)


def make_dynamic_obstacles(env, episode):
    """
    Create deterministic moving workers on static free cells.
    """

    free = [
        (r, c)
        for r in range(GRID_SIZE)
        for c in range(GRID_SIZE)
        if env._static_grid[r][c] == 0
    ]

    # Deterministic obstacle generation for reproducibility.
    rng = random.Random(
        episode
        + sum(
            ord(ch)
            for ch in env.config_name
        )
    )

    rng.shuffle(free)

    planner = AStarPlanner(
        env._static_grid
    )

    routes = []

    # --------------------------------------------------------
    # Generate patrol routes
    # --------------------------------------------------------

    for i in range(len(free)):

        for j in range(i + 1, len(free)):

            path = planner.find_path(
                free[i],
                free[j],
            )

            if path and len(path) >= 5:
                routes.append(path)
                break

        if len(routes) >= 2:
            break

    # --------------------------------------------------------
    # Create workers
    # --------------------------------------------------------

    workers = []

    for path in routes[:2]:

        workers.append(
            Worker(
                position=path[0],
                movement_pattern="predefined_patrol",
                patrol_route=path,
                grid_size=(
                    GRID_SIZE,
                    GRID_SIZE,
                ),
            )
        )

    env.set_dynamic_obstacles(
        workers=workers,
        dynamic_robots=[],
    )

    return workers


def get_valid_actions(env, current_pos, blocked_cells=None):
    """Compute legal action indices from current_pos avoiding shelves and blocked cells."""
    if blocked_cells is None:
        blocked_cells = set()
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
            if env._static_grid[nr, nc] != 1 and (nr, nc) not in blocked_cells:
                valid.add(action_idx)
    return valid if valid else set(range(8))


# ============================================================
# STATE CONSTRUCTION
# ============================================================

def build_state(
    state_builder,
    env,
    waypoint,
    scenario,
):
    """
    Build the trained 51-dimensional state.
    """

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
# PATH LENGTH
# ============================================================

def path_length(path):
    """Calculate Euclidean path length."""

    if not path or len(path) < 2:
        return 0.0

    return sum(
        math.sqrt(
            (b[0] - a[0]) ** 2
            + (b[1] - a[1]) ** 2
        )
        for a, b in zip(path, path[1:])
    )


# ============================================================
# A* EVALUATION
# ============================================================

def execute_astar(env):
    """Run A* with dynamic replanning."""

    planner = AStarPlanner(
        env._static_grid
    )

    replanner = DynamicReplanner(
        planner
    )

    path = planner.find_path(
        env.robot_pos,
        env.goal_pos,
        blocked_cells(env),
    )

    replans = 0
    total_reward = 0.0
    total_path_length = 0.0
    collisions = 0
    steps = 0

    # Action mapping:
    #
    # 0 = Up
    # 1 = Up-right
    # 2 = Right
    # 3 = Down-right
    # 4 = Down
    # 5 = Down-left
    # 6 = Left
    # 7 = Up-left

    action_map = {
        (-1, 0): 0,
        (-1, 1): 1,
        (0, 1): 2,
        (1, 1): 3,
        (1, 0): 4,
        (1, -1): 5,
        (0, -1): 6,
        (-1, -1): 7,
    }

    while (
        steps < MAX_STEPS
        and env.robot_pos != env.goal_pos
    ):

        if path is None:
            break

        # ----------------------------------------------------
        # Check whether the current path is blocked.
        # ----------------------------------------------------

        blocked = blocked_cells(env)

        if replanner.is_path_blocked(
            path,
            blocked,
        ):

            new_path = replanner.replan(
                current_position=env.robot_pos,
                goal=env.goal_pos,
                current_path=path,
                blocked_cells=blocked,
            )

            if new_path != path:
                replans += 1

            path = new_path

        if (
            not path
            or env.robot_pos not in path
        ):
            break

        index = path.index(
            env.robot_pos
        )

        if index + 1 >= len(path):
            break

        target = path[index + 1]
        current = env.robot_pos

        action = action_map.get(
            (
                target[0] - current[0],
                target[1] - current[1],
            )
        )

        if action is None:
            break

        # ----------------------------------------------------
        # Move dynamic obstacles.
        # ----------------------------------------------------

        move_dynamic_obstacles(env)

        previous = env.robot_pos

        _, reward, terminated, truncated, info = env.step(
            action
        )

        total_reward += float(reward)

        total_path_length += math.sqrt(
            (env.robot_pos[0] - previous[0]) ** 2
            + (env.robot_pos[1] - previous[1]) ** 2
        )

        collisions += int(
            info.get("collided", False)
        )

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


# ============================================================
# PURE DDQN EVALUATION
# ============================================================

def execute_ddqn(
    env,
    agent,
    state_builder,
    scenario,
):
    """Run Pure DDQN while obstacles move."""

    total_reward = 0.0
    total_path_length = 0.0
    collisions = 0
    steps = 0

    while (
        steps < MAX_STEPS
        and env.robot_pos != env.goal_pos
    ):

        # ----------------------------------------------------
        # Move dynamic obstacles.
        # ----------------------------------------------------

        move_dynamic_obstacles(env)

        # ----------------------------------------------------
        # Build state.
        #
        # Pure DDQN uses the goal as its waypoint input.
        # ----------------------------------------------------

        state = build_state(
            state_builder,
            env,
            env.goal_pos,
            scenario,
        )

        # ----------------------------------------------------
        # Select DDQN action.
        # ----------------------------------------------------

        action = greedy_action(
            agent,
            state,
        )

        previous = env.robot_pos

        _, reward, terminated, truncated, info = env.step(
            action
        )

        total_reward += float(reward)

        total_path_length += math.sqrt(
            (env.robot_pos[0] - previous[0]) ** 2
            + (env.robot_pos[1] - previous[1]) ** 2
        )

        collisions += int(
            info.get("collided", False)
        )

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


# ============================================================
# HYBRID DDQN + A* EVALUATION
# ============================================================

def execute_hybrid(
    env,
    agent,
    state_builder,
    scenario,
):
    """Run Hybrid DDQN+A* with dynamic replanning."""

    planner = AStarPlanner(
        env._static_grid
    )

    replanner = DynamicReplanner(
        planner
    )

    waypoint_manager = WaypointManager()

    hybrid = HybridAgent(
        agent,
        waypoint_manager,
        replanner,
        planner,
    )

    replans = 0

    # --------------------------------------------------------
    # Keep track of actual calls to replanning.
    # --------------------------------------------------------

    original_replan = replanner.replan

    def counted_replan(*args, **kwargs):
        nonlocal replans

        result = original_replan(
            *args,
            **kwargs,
        )

        if result:
            replans += 1

        return result

    replanner.replan = counted_replan

    waypoint_manager.set_path([])

    total_reward = 0.0
    total_path_length = 0.0
    collisions = 0
    steps = 0

    while (
        steps < MAX_STEPS
        and env.robot_pos != env.goal_pos
    ):

        # ----------------------------------------------------
        # 1. Move dynamic obstacles.
        # ----------------------------------------------------

        move_dynamic_obstacles(env)

        blocked = blocked_cells(env)

        # ----------------------------------------------------
        # 2. A* updates route and gives waypoint.
        # ----------------------------------------------------

        waypoint = hybrid.update_route_and_waypoint(
            env.robot_pos,
            env.goal_pos,
            blocked,
        )

        # ----------------------------------------------------
        # 3. Build 51-dimensional Hybrid state.
        # ----------------------------------------------------

        state = build_state(
            state_builder,
            env,
            waypoint,
            scenario,
        )

        # ----------------------------------------------------
        # 4. Hybrid chooses action (with guidance & masking).
        # ----------------------------------------------------

        valid_actions = get_valid_actions(
            env,
            env.robot_pos,
            blocked,
        )

        action = hybrid.get_action(
            state=state,
            epsilon=0.0,
            current_pos=env.robot_pos,
            waypoint=waypoint,
            blocked_cells=blocked,
            valid_actions=valid_actions,
        )

        previous = env.robot_pos

        # ----------------------------------------------------
        # 5. Execute action.
        # ----------------------------------------------------

        _, reward, terminated, truncated, info = env.step(
            action
        )

        total_reward += float(reward)

        total_path_length += math.sqrt(
            (env.robot_pos[0] - previous[0]) ** 2
            + (env.robot_pos[1] - previous[1]) ** 2
        )

        collisions += int(
            info.get("collided", False)
        )

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


# ============================================================
# FINAL SUMMARY
# ============================================================

def print_final_summary(df):
    """Calculate and print final evaluation metrics."""

    summary = (
        df.groupby(
            [
                "scenario",
                "method",
            ],
            sort=False,
        )
        .agg(
            success_rate=(
                "success",
                "mean",
            ),

            average_reward=(
                "total_reward",
                "mean",
            ),

            average_steps=(
                "steps",
                "mean",
            ),

            average_path_length=(
                "path_length",
                "mean",
            ),

            average_collisions=(
                "collisions",
                "mean",
            ),

            collision_rate=(
                "collisions",
                lambda x: (x > 0).mean(),
            ),

            average_replans=(
                "replans",
                "mean",
            ),
        )
        .reset_index()
    )

    print("\n")

    print(
        "=" * 110
    )

    print(
        "                 FINAL DYNAMIC EVALUATION RESULTS"
    )

    print(
        "=" * 110
    )

    for scenario in SCENARIOS:

        print(
            f"\nSCENARIO: {scenario.upper()}"
        )

        print(
            "-" * 110
        )

        scenario_df = summary[
            summary["scenario"] == scenario
        ]

        print(
            f"{'Method':<20}"
            f"{'Success':>10}"
            f"{'Reward':>12}"
            f"{'Steps':>12}"
            f"{'Path':>12}"
            f"{'Collisions':>14}"
            f"{'Collision %':>14}"
            f"{'Replans':>12}"
        )

        print(
            "-" * 110
        )

        for _, row in scenario_df.iterrows():

            print(
                f"{row['method']:<20}"
                f"{row['success_rate'] * 100:>9.1f}%"
                f"{row['average_reward']:>12.2f}"
                f"{row['average_steps']:>12.2f}"
                f"{row['average_path_length']:>12.2f}"
                f"{row['average_collisions']:>14.2f}"
                f"{row['collision_rate'] * 100:>13.1f}%"
                f"{row['average_replans']:>12.2f}"
            )

    print(
        "=" * 110
    )

    return summary


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Create output directory.
    # --------------------------------------------------------

    os.makedirs(
        os.path.dirname(OUTPUT_FILE),
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Header.
    # --------------------------------------------------------

    print("\n")

    print(
        "=" * 70
    )

    print(
        "       DYNAMIC WAREHOUSE ROBOT EVALUATION"
    )

    print(
        "=" * 70
    )

    print(
        f"Grid size : {GRID_SIZE} x {GRID_SIZE}"
    )

    print(
        f"Episodes  : {EPISODES} per scenario/method"
    )

    print(
        f"Max steps : {MAX_STEPS}"
    )

    # --------------------------------------------------------
    # Load models.
    # --------------------------------------------------------

    print(
        "\nLoading trained DDQN models..."
    )

    pure_agent = load_agent(
        PURE_CHECKPOINT
    )

    hybrid_agent = load_agent(
        HYBRID_CHECKPOINT
    )

    print(
        "Models loaded successfully.\n"
    )

    rows = []

    # ========================================================
    # RUN ALL EXPERIMENTS
    # ========================================================

    for scenario in SCENARIOS:

        print(
            f"\n{'=' * 70}"
        )

        print(
            f"SCENARIO: {scenario.upper()}"
        )

        print(
            f"{'=' * 70}"
        )

        methods = [
            "A*",
            "Pure DDQN",
            "Hybrid DDQN+A*",
        ]

        for method in methods:

            print(
                f"\n{method}"
            )

            print(
                "-" * 50
            )

            for episode in range(
                1,
                EPISODES + 1,
            ):

                # ------------------------------------------------
                # Create environment.
                # ------------------------------------------------

                env = WarehouseEnv(
                    config=scenario,
                    grid_size=GRID_SIZE,
                    max_steps=MAX_STEPS,
                    seed=episode,
                )

                env.reset(
                    seed=episode
                )

                # ------------------------------------------------
                # Add moving obstacles.
                # ------------------------------------------------

                make_dynamic_obstacles(
                    env,
                    episode,
                )

                # ------------------------------------------------
                # Build state representation.
                # ------------------------------------------------

                state_builder = StateAugmenter(
                    env._static_grid,
                    local_radius=2,
                    max_dynamic_obstacles=4,
                )

                # ------------------------------------------------
                # Run selected method.
                # ------------------------------------------------

                if method == "A*":

                    result = execute_astar(
                        env
                    )

                elif method == "Pure DDQN":

                    result = execute_ddqn(
                        env,
                        pure_agent,
                        state_builder,
                        scenario,
                    )

                else:

                    result = execute_hybrid(
                        env,
                        hybrid_agent,
                        state_builder,
                        scenario,
                    )

                # ------------------------------------------------
                # Store episode result.
                # ------------------------------------------------

                rows.append(
                    {
                        "method": method,
                        "scenario": scenario,
                        "episode": episode,
                        "success": result["success"],
                        "steps": result["steps"],
                        "total_reward": result["reward"],
                        "path_length": result["path_length"],
                        "collisions": result["collisions"],
                        "replans": result["replans"],
                    }
                )

                # ------------------------------------------------
                # Episode result.
                # ------------------------------------------------

                print(
                    f"  Episode {episode:02d} | "
                    f"Success={result['success']} | "
                    f"Steps={result['steps']:3d} | "
                    f"Collisions={result['collisions']:3d} | "
                    f"Replans={result['replans']:2d}"
                )

                env.close()

    # ========================================================
    # SAVE RAW RESULTS
    # ========================================================

    df = pd.DataFrame(
        rows
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # ========================================================
    # PRINT FINAL SUMMARY
    # ========================================================

    summary = print_final_summary(
        df
    )

    # ========================================================
    # SAVE SUMMARY
    # ========================================================

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    print(
        f"\nRaw results saved to:"
        f"\n{OUTPUT_FILE}"
    )

    print(
        f"\nSummary saved to:"
        f"\n{SUMMARY_FILE}"
    )

    print(
        "\nDynamic evaluation completed."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()