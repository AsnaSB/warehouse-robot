"""Run controlled experiments for A*, Pure DDQN and Hybrid DDQN+A*."""

import csv
import math
import os

import torch

from src.astar.astar_planner import AStarPlanner
from src.astar.replanner import DynamicReplanner
from src.ddqn.agent import DDQNAgent
from src.environment.state_augmentation import StateAugmenter
from src.environment.warehouse_env import WarehouseEnv
from src.hybrid.hybrid_agent import HybridAgent
from src.hybrid.waypoint_manager import WaypointManager


# ---------------------------------------------------------
# Experiment settings
# ---------------------------------------------------------

SCENARIOS = [
    "open",
    "aisle",
    "dense"
]

EPISODES = 10
GRID_SIZE = 12
MAX_STEPS = 200

OUTPUT_FILE = (
    "outputs/results/controlled_results.csv"
)

PURE_DDQN_MODEL = (
    "experiments/checkpoints/pure_ddqn_ep2500.pth"
)

HYBRID_MODEL = (
    "experiments/checkpoints/hybrid_ddqn_ep2500.pth"
)


# ---------------------------------------------------------
# Load trained DDQN
# ---------------------------------------------------------

def load_agent(model_path):
    """Create DDQN and load trained weights."""

    agent = DDQNAgent(
        state_dim=51,
        action_dim=8
    )

    weights = torch.load(
        model_path,
        map_location=agent.device
    )

    # Support checkpoints stored directly as state_dict.
    # Also support checkpoints containing online_state.
    if isinstance(weights, dict):
        if "online_state" in weights:
            weights = weights["online_state"]

    agent.online_net.load_state_dict(weights)

    agent.target_net.load_state_dict(
        agent.online_net.state_dict()
    )

    agent.online_net.eval()
    agent.target_net.eval()

    return agent


# ---------------------------------------------------------
# Greedy action
# ---------------------------------------------------------

def greedy_action(agent, state):
    """Select the highest-Q action."""

    with torch.no_grad():

        state_tensor = (
            torch.FloatTensor(state)
            .unsqueeze(0)
            .to(agent.device)
        )

        q_values = agent.online_net(
            state_tensor
        )

        return q_values.argmax().item()


# ---------------------------------------------------------
# Path length
# ---------------------------------------------------------

def calculate_path_length(path):
    """Calculate Euclidean grid path length."""

    if not path or len(path) < 2:
        return 0.0

    length = 0.0

    for current, next_pos in zip(
        path,
        path[1:]
    ):

        dr = next_pos[0] - current[0]
        dc = next_pos[1] - current[1]

        length += math.sqrt(
            dr ** 2 + dc ** 2
        )

    return length


# ---------------------------------------------------------
# State builder
# ---------------------------------------------------------

def build_state(
    state_builder,
    env,
    waypoint,
    scenario
):
    """Build the 51-feature state."""

    return state_builder.build_state(
        robot_position=env.robot_pos,
        goal_position=env.goal_pos,
        workers=env.workers,
        dynamic_robots=env.dynamic_robots,
        waypoint=waypoint,
        context=scenario,
        risk_level="low"
    )


# ---------------------------------------------------------
# A* evaluation
# ---------------------------------------------------------

def evaluate_astar(env):
    """Evaluate one A* episode by executing the planned path."""

    planner = AStarPlanner(
        env._static_grid
    )

    path = planner.find_path(
        env.robot_pos,
        env.goal_pos
    )

    if path is None:

        return {
            "success": False,
            "steps": 0,
            "reward": 0.0,
            "path_length": 0.0,
            "collisions": 0,
            "replans": 0
        }

    # A* action mapping.
    action_map = {
        (-1, 0): 0,
        (-1, 1): 1,
        (0, 1): 2,
        (1, 1): 3,
        (1, 0): 4,
        (1, -1): 5,
        (0, -1): 6,
        (-1, -1): 7
    }

    # Store path if supported by the environment.
    if hasattr(env, "set_astar_path"):
        env.set_astar_path(path)

    total_reward = 0.0
    path_length = 0.0
    collisions = 0
    steps = 0

    previous_position = env.robot_pos

    path_index = 1

    terminated = False
    truncated = False

    while not (terminated or truncated):

        if env.robot_pos == env.goal_pos:
            break

        if path_index >= len(path):
            break

        current_position = env.robot_pos
        target_position = path[path_index]

        dr = target_position[0] - current_position[0]
        dc = target_position[1] - current_position[1]

        action = action_map.get(
            (dr, dc)
        )

        if action is None:
            break

        (
            _,
            reward,
            terminated,
            truncated,
            info
        ) = env.step(action)

        total_reward += float(reward)

        current_position = env.robot_pos

        move_dr = (
            current_position[0]
            - previous_position[0]
        )

        move_dc = (
            current_position[1]
            - previous_position[1]
        )

        path_length += math.sqrt(
            move_dr ** 2 + move_dc ** 2
        )

        if info.get(
            "collided",
            False
        ):
            collisions += 1

        previous_position = current_position

        steps += 1

        # Move to the next A* waypoint only when
        # the current target position was reached.
        if current_position == target_position:
            path_index += 1

    return {
        "success": (
            env.robot_pos == env.goal_pos
        ),
        "steps": steps,
        "reward": total_reward,
        "path_length": path_length,
        "collisions": collisions,
        "replans": 0
    }


# ---------------------------------------------------------
# Pure DDQN evaluation
# ---------------------------------------------------------

def evaluate_ddqn(
    env,
    agent,
    state_builder,
    scenario
):
    """Evaluate Pure DDQN."""

    env.reset()

    total_reward = 0.0
    path_length = 0.0
    collisions = 0

    previous_position = env.robot_pos

    terminated = False
    truncated = False
    steps = 0

    while not (terminated or truncated):

        state = build_state(
            state_builder,
            env,
            env.goal_pos,
            scenario
        )

        action = greedy_action(
            agent,
            state
        )

        (
            _,
            reward,
            terminated,
            truncated,
            info
        ) = env.step(action)

        total_reward += float(reward)

        current_position = env.robot_pos

        dr = (
            current_position[0]
            - previous_position[0]
        )

        dc = (
            current_position[1]
            - previous_position[1]
        )

        path_length += math.sqrt(
            dr ** 2 + dc ** 2
        )

        if info.get(
            "collided",
            False
        ):
            collisions += 1

        previous_position = current_position
        steps += 1

    return {
        "success": (
            env.robot_pos == env.goal_pos
        ),
        "steps": steps,
        "reward": total_reward,
        "path_length": path_length,
        "collisions": collisions,
        "replans": 0
    }


# ---------------------------------------------------------
# Hybrid evaluation
# ---------------------------------------------------------

def evaluate_hybrid(
    env,
    agent,
    state_builder,
    scenario
):
    """Evaluate Hybrid DDQN+A*."""

    env.reset()

    planner = AStarPlanner(
        env._static_grid
    )

    replanner = DynamicReplanner(
        planner
    )

    waypoint_manager = WaypointManager()

    hybrid_agent = HybridAgent(
        agent,
        waypoint_manager,
        replanner,
        planner
    )

    # Count actual calls to DynamicReplanner.replan().
    replan_counter = {
        "count": 0
    }

    original_replan = replanner.replan

    def counted_replan(*args, **kwargs):
        """Count a replan and execute the original method."""

        replan_counter["count"] += 1

        return original_replan(
            *args,
            **kwargs
        )

    replanner.replan = counted_replan

    waypoint_manager.set_path([])

    total_reward = 0.0
    path_length = 0.0
    collisions = 0

    previous_position = env.robot_pos

    terminated = False
    truncated = False
    steps = 0

    while not (terminated or truncated):

        current_position = env.robot_pos

        blocked_cells = set()

        for obstacle in (
            list(env.workers)
            + list(env.dynamic_robots)
        ):

            if getattr(
                obstacle,
                "active",
                True
            ):

                if hasattr(
                    obstacle,
                    "position"
                ):

                    blocked_cells.add(
                        obstacle.position
                    )

        waypoint = (
            hybrid_agent.update_route_and_waypoint(
                current_position,
                env.goal_pos,
                blocked_cells
            )
        )

        state = build_state(
            state_builder,
            env,
            waypoint,
            scenario
        )

        action = greedy_action(
            hybrid_agent.ddqn,
            state
        )

        (
            _,
            reward,
            terminated,
            truncated,
            info
        ) = env.step(action)

        total_reward += float(reward)

        current_position = env.robot_pos

        dr = (
            current_position[0]
            - previous_position[0]
        )

        dc = (
            current_position[1]
            - previous_position[1]
        )

        path_length += math.sqrt(
            dr ** 2 + dc ** 2
        )

        if info.get(
            "collided",
            False
        ):
            collisions += 1

        previous_position = current_position
        steps += 1

    return {
        "success": (
            env.robot_pos == env.goal_pos
        ),
        "steps": steps,
        "reward": total_reward,
        "path_length": path_length,
        "collisions": collisions,
        "replans": replan_counter["count"]
    }


# ---------------------------------------------------------
# Main experiment
# ---------------------------------------------------------

def run_controlled_experiments():

    os.makedirs(
        os.path.dirname(OUTPUT_FILE),
        exist_ok=True
    )

    results = []

    print("=" * 70)
    print(
        "CONTROLLED WAREHOUSE ROBOT EVALUATION"
    )
    print("=" * 70)
    print()

    print(
        "Loading trained DDQN models..."
    )

    # Load each model only once.
    pure_ddqn = load_agent(
        PURE_DDQN_MODEL
    )

    hybrid_ddqn = load_agent(
        HYBRID_MODEL
    )

    print(
        "Models loaded successfully."
    )

    for scenario in SCENARIOS:

        print()
        print(
            "=" * 70
        )
        print(
            f"SCENARIO: {scenario.upper()}"
        )
        print(
            "=" * 70
        )

        for method in [
            "A*",
            "Pure DDQN",
            "Hybrid DDQN+A*"
        ]:

            print()
            print(
                f"Evaluating {method}..."
            )

            for episode in range(
                1,
                EPISODES + 1
            ):

                # Same seed for every method
                # within the same scenario.
                env = WarehouseEnv(
                    config=scenario,
                    grid_size=GRID_SIZE,
                    max_steps=MAX_STEPS,
                    seed=episode
                )

                env.reset()

                # Use the original project's
                # StateAugmenter configuration.
                state_builder = (
                    StateAugmenter(
                        env._static_grid,
                        local_radius=2,
                        max_dynamic_obstacles=4
                    )
                )

                # -----------------------------
                # A*
                # -----------------------------

                if method == "A*":

                    metrics = evaluate_astar(
                        env
                    )

                # -----------------------------
                # Pure DDQN
                # -----------------------------

                elif method == "Pure DDQN":

                    metrics = evaluate_ddqn(
                        env,
                        pure_ddqn,
                        state_builder,
                        scenario
                    )

                # -----------------------------
                # Hybrid
                # -----------------------------

                else:

                    metrics = evaluate_hybrid(
                        env,
                        hybrid_ddqn,
                        state_builder,
                        scenario
                    )

                results.append({
                    "method": method,
                    "scenario": scenario,
                    "episode": episode,
                    "success": metrics["success"],
                    "steps": metrics["steps"],
                    "total_reward": metrics["reward"],
                    "path_length": metrics["path_length"],
                    "collisions": metrics["collisions"],
                    "replans": metrics["replans"]
                })

                print(
                    f"Episode {episode}: "
                    f"success={metrics['success']}, "
                    f"steps={metrics['steps']}, "
                    f"collisions={metrics['collisions']}, "
                    f"replans={metrics['replans']}"
                )

    # -----------------------------------------------------
    # Save CSV
    # -----------------------------------------------------

    with open(
        OUTPUT_FILE,
        "w",
        newline=""
    ) as file:

        fieldnames = [
            "method",
            "scenario",
            "episode",
            "success",
            "steps",
            "total_reward",
            "path_length",
            "collisions",
            "replans"
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(results)

    print()
    print(
        "=" * 70
    )
    print(
        "Controlled experiments completed."
    )
    print(
        f"Results saved to: {OUTPUT_FILE}"
    )
    print(
        "=" * 70
    )


if __name__ == "__main__":
    run_controlled_experiments()