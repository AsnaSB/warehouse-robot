"""
Warehouse simulation runner.

Uses:
    - WarehouseEnv
    - Dynamic workers
    - Dynamic robots
    - A* planner
    - Existing WarehouseEnv renderer

This is currently an A*-controlled simulation.
DDQN / HybridAgent can be connected later.
"""

from __future__ import annotations

import time
from typing import Optional, Tuple

import matplotlib.pyplot as plt

from src.environment.warehouse_env import WarehouseEnv
from src.environment.obstacles import Worker, DynamicRobot
from src.environment.constants import ACTIONS
from src.astar.astar_planner import AStarPlanner


Position = Tuple[int, int]


class WarehouseSimulation:

    def __init__(
        self,
        config: str = "open",
        grid_size: int = 12,
        max_steps: int = 200,
        seed: int = 42,
        delay: float = 0.15,
    ):
        self.config = config
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.seed = seed
        self.delay = delay

        # Create the existing environment
        self.env = WarehouseEnv(
            config=config,
            grid_size=grid_size,
            max_steps=max_steps,
            seed=seed,
        )

        # Create A* planner using the environment's static grid
        self.planner = AStarPlanner(
            self.env._static_grid.tolist()
        )

        self.running = True

        self.last_reward = 0.0
        self.last_info = {}

        self.current_path: Optional[list] = None

    # ---------------------------------------------------------
    # DYNAMIC OBSTACLE SETUP
    # ---------------------------------------------------------

    def setup_dynamic_obstacles(self):
        """Create and register dynamic obstacles."""

        grid = (self.env.grid_size, self.env.grid_size)

        # Moving worker
        worker = Worker(
            position=(2, 2),
            movement_pattern="predefined_patrol",
            patrol_route=[
                (2, 2),
                (2, 3),
                (2, 4),
                (2, 5),
                (2, 6),
            ],
            speed=1.0,
            grid_size=grid,
        )

        # Moving dynamic robot
        dynamic_robot = DynamicRobot(
            position=(8, 2),
            movement_pattern="linear",
            movement_direction=(0, 1),
            speed=1.0,
            grid_size=grid,
        )

        self.env.set_dynamic_obstacles(
            workers=[worker],
            dynamic_robots=[dynamic_robot],
        )

    # ---------------------------------------------------------
    # RESET
    # ---------------------------------------------------------

    def reset(self):
        """Reset the warehouse environment."""

        _, info = self.env.reset(seed=self.seed)

        self.env.set_astar_path(None)

        self.current_path = None
        self.last_reward = 0.0
        self.last_info = info

        print("\n" + "=" * 50)
        print("WAREHOUSE SIMULATION")
        print("=" * 50)

        print(f"Configuration : {self.config}")
        print(f"Grid size     : {self.grid_size} x {self.grid_size}")
        print(f"Robot         : {self.env.robot_pos}")
        print(f"Goal          : {self.env.goal_pos}")
        print(f"Context       : {info.get('context')}")
        print(f"Risk level    : {info.get('risk_level')}")
        print("=" * 50)

    # ---------------------------------------------------------
    # UPDATE DYNAMIC OBSTACLES
    # ---------------------------------------------------------

    def update_dynamic_obstacles(self):
        """Move workers and dynamic robots."""

        # Move workers
        for worker in self.env.workers:

            if worker.active:
                worker.update()

        # Track occupied positions
        occupied_positions = [
            worker.position
            for worker in self.env.workers
            if worker.active
        ]

        # Robot itself should not be occupied by dynamic obstacles
        occupied_positions.append(self.env.robot_pos)

        # Move dynamic robots
        for robot in self.env.dynamic_robots:

            if robot.active:
                robot.update(
                    occupied_positions=occupied_positions
                )

                occupied_positions.append(robot.position)

    # ---------------------------------------------------------
    # GET DYNAMIC BLOCKED CELLS
    # ---------------------------------------------------------

    def get_blocked_cells(self):
        """Return positions occupied by dynamic obstacles."""

        blocked = set()

        for worker in self.env.workers:

            if worker.active:
                blocked.add(worker.position)

        for robot in self.env.dynamic_robots:

            if robot.active:
                blocked.add(robot.position)

        # The robot's current position must remain available
        blocked.discard(self.env.robot_pos)

        return blocked

    # ---------------------------------------------------------
    # A* PATH PLANNING
    # ---------------------------------------------------------

    def plan_path(self):
        """Calculate a new A* path using current obstacle positions."""

        blocked_cells = self.get_blocked_cells()

        path = self.planner.find_path(
            start=self.env.robot_pos,
            goal=self.env.goal_pos,
            blocked_cells=blocked_cells,
        )

        self.current_path = path

        # Give the path to WarehouseEnv for visualization
        self.env.set_astar_path(path)

        if path is None:

            print("A* could not find a path.")

            return False

        return True

    # ---------------------------------------------------------
    # CONVERT MOVEMENT TO ACTION
    # ---------------------------------------------------------

    @staticmethod
    def movement_to_action(
        current: Position,
        target: Position,
    ):
        """
        Convert a movement such as (-1, 0)
        into the corresponding action number.
        """

        dr = target[0] - current[0]
        dc = target[1] - current[1]

        for action, movement in ACTIONS.items():

            if movement == (dr, dc):
                return action

        return None

    # ---------------------------------------------------------
    # SELECT NEXT A* ACTION
    # ---------------------------------------------------------

    def select_astar_action(self):
        """Select the next movement from the current A* path."""

        if not self.current_path:
            return None

        current = self.env.robot_pos

        try:
            current_index = self.current_path.index(current)

        except ValueError:
            return None

        next_index = current_index + 1

        if next_index >= len(self.current_path):
            return None

        next_position = self.current_path[next_index]

        return self.movement_to_action(
            current,
            next_position,
        )

    # ---------------------------------------------------------
    # SIMULATION STEP
    # ---------------------------------------------------------

    def step(self):
        """
        Perform one complete simulation step.

        Order:
            1. Move dynamic obstacles
            2. Recalculate A* path
            3. Select next action
            4. Move warehouse robot
            5. Store reward/info
        """

        # 1. Move dynamic obstacles
        self.update_dynamic_obstacles()

        # 2. Replan using A*
        if not self.plan_path():
            return False

        # 3. Select next action
        action = self.select_astar_action()

        if action is None:

            if self.env.robot_pos == self.env.goal_pos:

                print("Goal reached.")

                return False

            print("No valid next action. Replanning...")

            return True

        # 4. Execute action in environment
        (
            _,
            reward,
            terminated,
            truncated,
            info,
        ) = self.env.step(action)

        # 5. Store results
        self.last_reward = reward
        self.last_info = info

        print(
            f"Step {info.get('step_count', 0):3d} | "
            f"Robot {self.env.robot_pos} | "
            f"Action {action} | "
            f"Reward {reward:.2f} | "
            f"Collision {info.get('collided', False)}"
        )

        # Goal reached
        if terminated:

            print("\nSUCCESS: Robot reached the goal.")

            return False

        # Maximum steps reached
        if truncated:

            print("\nSimulation stopped: maximum steps reached.")

            return False

        return True

    # ---------------------------------------------------------
    # RUN SIMULATION
    # ---------------------------------------------------------

    def run(self):
        """Run the complete warehouse simulation."""

        # Create dynamic obstacles
        self.setup_dynamic_obstacles()

        # Reset environment
        self.reset()

        # Enable interactive matplotlib
        plt.ion()

        # Initial rendering
        self.env.render(mode="human")

        time.sleep(0.5)

        self.running = True

        while self.running:

            # Execute one simulation step
            self.running = self.step()

            # Render updated environment
            self.env.render(mode="human")

            # Control simulation speed
            plt.pause(self.delay)

        plt.ioff()

        print("\nSimulation finished.")

        plt.show()

    # ---------------------------------------------------------
    # CLOSE
    # ---------------------------------------------------------

    def close(self):
        """Close the environment and matplotlib window."""

        self.running = False

        self.env.close()


# =============================================================
# MAIN
# =============================================================

def main():

    simulation = WarehouseSimulation(
        config="open",
        grid_size=12,
        max_steps=200,
        seed=42,
        delay=0.15,
    )

    try:

        simulation.run()

    except KeyboardInterrupt:

        print("\nSimulation interrupted by user.")

    finally:

        simulation.close()


if __name__ == "__main__":
    main()