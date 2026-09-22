"""Run A*, Pure DDQN and Hybrid DDQN+A* evaluation experiments."""

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

EPISODES = 10
MAX_STEPS = 200
GRID_SIZE = 12
ENV_CONFIG = "open"

PURE_DDQN_MODEL = (
    "experiments/checkpoints/pure_ddqn_ep2500.pth"
)

HYBRID_MODEL = (
    "experiments/checkpoints/hybrid_ddqn_ep2500.pth"
)

OUTPUT_FILE = (
    "outputs/results/evaluation_results.csv"
)


# ---------------------------------------------------------
# Load trained DDQN
# ---------------------------------------------------------

def load_agent(model_path):
    """Create DDQN and load trained network weights."""

    agent = DDQNAgent(
        state_dim=51,
        action_dim=8
    )

    weights = torch.load(
        model_path,
        map_location=agent.device
    )

    agent.online_net.load_state_dict(weights)
    agent.target_net.load_state_dict(weights)

    agent.online_net.eval()
    agent.target_net.eval()

    return agent


# ---------------------------------------------------------
# Greedy action
# ---------------------------------------------------------

def greedy_action(agent, state):
    """Select the highest-Q action without exploration."""

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
    """Calculate Euclidean length of a grid path."""

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
# A* baseline
# ---------------------------------------------------------

def evaluate_astar(env):
    """Evaluate one A* episode."""

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

    return {
        "success": True,
        "steps": len(path) - 1,
        "reward": 0.0,
        "path_length": calculate_path_length(path),
        "collisions": 0,
        "replans": 0
    }


# ---------------------------------------------------------
# State construction
# ---------------------------------------------------------

def build_state(
    state_builder,
    env,
    waypoint
):
    """Build the 51-feature DDQN state."""

    return state_builder.build_state(
        robot_position=env.robot_pos,
        goal_position=env.goal_pos,
        workers=env.workers,
        dynamic_robots=env.dynamic_robots,
        waypoint=waypoint,
        context=ENV_CONFIG,
        risk_level="low"
    )


# ---------------------------------------------------------
# Pure DDQN
# ---------------------------------------------------------

def evaluate_ddqn(
    env,
    agent,
    state_builder
):
    """Evaluate Pure DDQN navigation."""

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
            env.goal_pos
        )

        action = greedy_action(
            agent,
            state
        )

        (
            next_state,
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

        if info.get("collided", False):
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
# Hybrid DDQN + A*
# ---------------------------------------------------------

def evaluate_hybrid(
    env,
    hybrid_agent,
    state_builder
):
    """Evaluate Hybrid DDQN+A* navigation."""

    env.reset()

    hybrid_agent.waypoint_manager.set_path([])

    total_reward = 0.0
    path_length = 0.0
    collisions = 0

    previous_position = env.robot_pos

    terminated = False
    truncated = False
    steps = 0

    while not (terminated or truncated):

        current_position = env.robot_pos
        goal_position = env.goal_pos

        blocked_cells = {
            obstacle.position
            for obstacle in (
                list(env.workers)
                + list(env.dynamic_robots)
            )
            if getattr(
                obstacle,
                "active",
                True
            )
        }

        waypoint = (
            hybrid_agent.update_route_and_waypoint(
                current_position,
                goal_position,
                blocked_cells
            )
        )

        state = build_state(
            state_builder,
            env,
            waypoint
        )

        action = greedy_action(
            hybrid_agent.ddqn,
            state
        )

        (
            next_state,
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

        if info.get("collided", False):
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
# Main evaluation
# ---------------------------------------------------------

def run_evaluation():

    os.makedirs(
        os.path.dirname(OUTPUT_FILE),
        exist_ok=True
    )

    results = []

    methods = [
        "A*",
        "Pure DDQN",
        "Hybrid DDQN+A*"
    ]

    for method in methods:

        print()
        print(
            f"Evaluating {method}..."
        )

        # Load model only once for each method.
        agent = None
        hybrid_agent = None

        if method == "Pure DDQN":

            agent = load_agent(
                PURE_DDQN_MODEL
            )

        elif method == "Hybrid DDQN+A*":

            agent = load_agent(
                HYBRID_MODEL
            )

        for episode in range(
            1,
            EPISODES + 1
        ):

            env = WarehouseEnv(
                config=ENV_CONFIG,
                grid_size=GRID_SIZE,
                max_steps=MAX_STEPS,
                seed=episode
            )

            env.reset()

            state_builder = StateAugmenter(
                env._static_grid,
                local_radius=2,
                max_dynamic_obstacles=4
            )

            # ---------------------------------------------
            # A*
            # ---------------------------------------------

            if method == "A*":

                metrics = evaluate_astar(
                    env
                )

            # ---------------------------------------------
            # Pure DDQN
            # ---------------------------------------------

            elif method == "Pure DDQN":

                metrics = evaluate_ddqn(
                    env,
                    agent,
                    state_builder
                )

            # ---------------------------------------------
            # Hybrid
            # ---------------------------------------------

            else:

                planner = AStarPlanner(
                    env._static_grid
                )

                replanner = DynamicReplanner(
                    planner
                )

                waypoint_manager = (
                    WaypointManager()
                )

                hybrid_agent = HybridAgent(
                    agent,
                    waypoint_manager,
                    replanner,
                    planner
                )

                metrics = evaluate_hybrid(
                    env,
                    hybrid_agent,
                    state_builder
                )

            results.append({
                "method": method,
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
                f"reward={metrics['reward']:.2f}"
            )

    # -----------------------------------------------------
    # Save results
    # -----------------------------------------------------

    with open(
        OUTPUT_FILE,
        "w",
        newline=""
    ) as file:

        fieldnames = [
            "method",
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
        "======================================"
    )
    print(
        "Evaluation completed successfully."
    )
    print(
        f"Results saved to: {OUTPUT_FILE}"
    )
    print(
        "======================================"
    )


if __name__ == "__main__":
    run_evaluation()