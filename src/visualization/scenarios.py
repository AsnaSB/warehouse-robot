"""
Reproducible warehouse simulation scenarios.

Member 3:
    Scenario 1 - Open Warehouse
    Scenario 2 - Static Obstacles
    Scenario 3 - Dynamic Obstacle
    Scenario 4 - Replanning
"""

from __future__ import annotations

import time
from typing import Optional, Tuple

import matplotlib.pyplot as plt

from src.environment.warehouse_env import WarehouseEnv
from src.environment.obstacles import Worker, DynamicRobot
from src.astar.astar_planner import AStarPlanner


Position = Tuple[int, int]


class ScenarioRunner:
    """Runs reproducible warehouse simulation scenarios."""

    def __init__(
        self,
        grid_size: int = 12,
        max_steps: int = 100,
        seed: int = 42,
        delay: float = 0.15,
    ):
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.seed = seed
        self.delay = delay

        self.env: Optional[WarehouseEnv] = None
        self.planner: Optional[AStarPlanner] = None

        self.current_path = None
        self.previous_path = None

    # ---------------------------------------------------------
    # ENVIRONMENT CREATION
    # ---------------------------------------------------------

    def create_environment(self, config: str):
        """Create a fresh environment for a scenario."""

        self.env = WarehouseEnv(
            config=config,
            grid_size=self.grid_size,
            max_steps=self.max_steps,
            seed=self.seed,
        )

        self.planner = AStarPlanner(
            self.env._static_grid.tolist()
        )

        self.current_path = None
        self.previous_path = None

    # ---------------------------------------------------------
    # RESET WITH FIXED START / GOAL
    # ---------------------------------------------------------

    def reset_environment(
        self,
        start: Position,
        goal: Position,
    ):
        """Reset the environment and use reproducible positions."""

        if self.env is None:
            raise RuntimeError("Environment has not been created.")

        self.env.reset(seed=self.seed)

        self.env.robot_pos = start
        self.env.start_pos = start
        self.env.goal_pos = goal

        self.env.set_astar_path(None)

    # ---------------------------------------------------------
    # DYNAMIC OBSTACLE POSITIONS
    # ---------------------------------------------------------

    def get_blocked_cells(self):
        """Return currently occupied dynamic obstacle cells."""

        blocked = set()

        if self.env is None:
            return blocked

        for worker in self.env.workers:
            if worker.active:
                blocked.add(worker.position)

        for robot in self.env.dynamic_robots:
            if robot.active:
                blocked.add(robot.position)

        # Never treat the robot's current cell as blocked.
        blocked.discard(self.env.robot_pos)

        return blocked

    # ---------------------------------------------------------
    # A* PLANNING
    # ---------------------------------------------------------

    def plan_path(self):
        """Generate an A* path using current dynamic obstacles."""

        if self.env is None or self.planner is None:
            raise RuntimeError("Environment has not been created.")

        blocked_cells = self.get_blocked_cells()

        path = self.planner.find_path(
            start=self.env.robot_pos,
            goal=self.env.goal_pos,
            blocked_cells=blocked_cells,
        )

        self.current_path = path
        self.env.set_astar_path(path)

        return path

    # ---------------------------------------------------------
    # PATH -> ACTION
    # ---------------------------------------------------------

    @staticmethod
    def movement_to_action(
        current: Position,
        target: Position,
    ):
        """Convert a neighbouring cell movement to an action."""

        dr = target[0] - current[0]
        dc = target[1] - current[1]

        movements = {
            (-1, 0): 0,   # N
            (-1, 1): 1,   # NE
            (0, 1): 2,    # E
            (1, 1): 3,    # SE
            (1, 0): 4,    # S
            (1, -1): 5,   # SW
            (0, -1): 6,   # W
            (-1, -1): 7,  # NW
        }

        return movements.get((dr, dc))

    # ---------------------------------------------------------
    # NEXT ACTION
    # ---------------------------------------------------------

    def get_next_action(self):
        """Return the action corresponding to the next A* waypoint."""

        if not self.current_path:
            return None

        current = self.env.robot_pos

        try:
            index = self.current_path.index(current)
        except ValueError:
            return None

        if index + 1 >= len(self.current_path):
            return None

        next_position = self.current_path[index + 1]

        return self.movement_to_action(
            current,
            next_position,
        )

    # ---------------------------------------------------------
    # MOVE DYNAMIC OBSTACLES
    # ---------------------------------------------------------

    def update_dynamic_obstacles(self):
        """Advance all active dynamic obstacles."""

        if self.env is None:
            return

        for worker in self.env.workers:

            if worker.active:
                worker.update()

        occupied_positions = [
            worker.position
            for worker in self.env.workers
            if worker.active
        ]

        occupied_positions.append(self.env.robot_pos)

        for robot in self.env.dynamic_robots:

            if robot.active:

                robot.update(
                    occupied_positions=occupied_positions
                )

                occupied_positions.append(robot.position)

    # ---------------------------------------------------------
    # DISPLAY
    # ---------------------------------------------------------

    def render(self, title: str):
        """Render the current environment."""

        if self.env is None:
            return

        self.env.render(mode="human")

        if self.env._ax is not None:
            self.env._ax.set_title(title)

        plt.pause(self.delay)

    # =========================================================
    # SCENARIO 1
    # =========================================================

    def scenario_1_open_warehouse(self):
        """
        Scenario 1 - Open Warehouse

        Robot moves from a fixed start position to a fixed goal
        with minimal obstruction.
        """

        print("\n" + "=" * 60)
        print("SCENARIO 1 - OPEN WAREHOUSE")
        print("=" * 60)

        self.create_environment("open")

        self.reset_environment(
            start=(1, 1),
            goal=(10, 10),
        )

        return self.run_robot(
            "Scenario 1 - Open Warehouse"
        )

    # =========================================================
    # SCENARIO 2
    # =========================================================

    def scenario_2_static_obstacles(self):
        """
        Scenario 2 - Static Obstacles

        Robot navigates through a warehouse containing fixed
        obstacles.
        """

        print("\n" + "=" * 60)
        print("SCENARIO 2 - STATIC OBSTACLES")
        print("=" * 60)

        self.create_environment("aisle")

        self.reset_environment(
            start=(1, 1),
            goal=(10, 10),
        )

        return self.run_robot(
            "Scenario 2 - Static Obstacles"
        )

    # =========================================================
    # SCENARIO 3
    # =========================================================

    def scenario_3_dynamic_obstacle(self):
        """
        Scenario 3 - Dynamic Obstacle

        A moving worker crosses the warehouse while the robot
        follows an A* route.
        """

        print("\n" + "=" * 60)
        print("SCENARIO 3 - DYNAMIC OBSTACLE")
        print("=" * 60)

        self.create_environment("open")

        worker = Worker(
            position=(5, 2),
            movement_pattern="predefined_patrol",
            patrol_route=[
                (5, 2),
                (5, 3),
                (5, 4),
                (5, 5),
                (5, 6),
                (5, 7),
                (5, 8),
            ],
            speed=1.0,
            grid_size=(self.grid_size, self.grid_size),
        )

        self.env.set_dynamic_obstacles(
            workers=[worker],
            dynamic_robots=[],
        )

        self.reset_environment(
            start=(5, 1),
            goal=(5, 10),
        )

        return self.run_dynamic_scenario(
            "Scenario 3 - Dynamic Obstacle"
        )

    # =========================================================
    # SCENARIO 4
    # =========================================================

    def scenario_4_replanning(self):
        """
        Scenario 4 - Replanning

        The robot initially receives an A* route. A dynamic
        obstacle is then introduced onto/near the route,
        forcing A* to generate a new route.
        """

        print("\n" + "=" * 60)
        print("SCENARIO 4 - REPLANNING")
        print("=" * 60)

        self.create_environment("open")

        self.reset_environment(
            start=(6, 1),
            goal=(6, 10),
        )

        # First path without dynamic obstacle
        initial_path = self.plan_path()

        if initial_path is None:
            print("Initial A* path could not be generated.")
            return False

        print(
            f"Initial path generated: "
            f"{len(initial_path)} cells"
        )

        self.render(
            "Scenario 4 - Initial A* Route"
        )

        # Place a worker directly on the route.
        route_index = min(
            4,
            len(initial_path) - 2,
        )

        blocking_position = initial_path[route_index]

        worker = Worker(
            position=blocking_position,
            movement_pattern="predefined_patrol",
            patrol_route=[
                blocking_position,
            ],
            speed=1.0,
            grid_size=(self.grid_size, self.grid_size),
        )

        self.env.set_dynamic_obstacles(
            workers=[worker],
            dynamic_robots=[],
        )

        print(
            f"Dynamic obstacle introduced at "
            f"{blocking_position}"
        )

        # Replan immediately
        new_path = self.plan_path()

        if new_path is None:
            print("No safe path exists after obstruction.")
            return False

        print(
            f"Replanned path generated: "
            f"{len(new_path)} cells"
        )

        if new_path != initial_path:
            print("REPLANNING SUCCESS: path changed.")
        else:
            print(
                "Path did not change; obstacle did not "
                "affect the selected route."
            )

        return self.run_robot(
            "Scenario 4 - Replanning"
        )

    # =========================================================
    # STANDARD ROBOT RUN
    # =========================================================

    def run_robot(self, title: str):
        """Run robot movement using repeated A* planning."""

        if self.env is None:
            return False

        self.env.set_astar_path(None)

        for step in range(self.max_steps):

            path = self.plan_path()

            if path is None:
                print("No path available.")
                return False

            if self.env.robot_pos == self.env.goal_pos:
                print("Goal reached.")
                return True

            action = self.get_next_action()

            if action is None:
                print("No valid action.")
                return False

            (
                _,
                reward,
                terminated,
                truncated,
                info,
            ) = self.env.step(action)

            print(
                f"Step {step + 1:3d} | "
                f"Robot={self.env.robot_pos} | "
                f"Goal={self.env.goal_pos} | "
                f"Reward={reward:.2f} | "
                f"Collision={info.get('collided', False)}"
            )

            self.render(title)

            if terminated:
                print("Goal reached.")
                return True

            if truncated:
                print("Maximum steps reached.")
                return False

        return False

    # =========================================================
    # DYNAMIC SCENARIO RUN
    # =========================================================

    def run_dynamic_scenario(self, title: str):
        """Run a scenario where dynamic obstacles move."""

        if self.env is None:
            return False

        for step in range(self.max_steps):

            # Move dynamic obstacles first.
            self.update_dynamic_obstacles()

            # Recalculate route based on current obstacle positions.
            path = self.plan_path()

            if path is None:
                print(
                    "A* could not find a safe path."
                )

                self.render(title)

                return False

            if self.env.robot_pos == self.env.goal_pos:
                print("Goal reached.")
                return True

            action = self.get_next_action()

            if action is None:
                print("No valid action.")
                return False

            (
                _,
                reward,
                terminated,
                truncated,
                info,
            ) = self.env.step(action)

            print(
                f"Step {step + 1:3d} | "
                f"Robot={self.env.robot_pos} | "
                f"Reward={reward:.2f} | "
                f"Collision={info.get('collided', False)}"
            )

            self.render(title)

            if terminated:
                print("Goal reached.")
                return True

            if truncated:
                print("Maximum steps reached.")
                return False

        return False

    # =========================================================
    # RUN ALL SCENARIOS
    # =========================================================

    def run_all(self):
        """Run all four reproducible scenarios."""

        results = {}

        results["scenario_1"] = (
            self.scenario_1_open_warehouse()
        )

        time.sleep(1)

        results["scenario_2"] = (
            self.scenario_2_static_obstacles()
        )

        time.sleep(1)

        results["scenario_3"] = (
            self.scenario_3_dynamic_obstacle()
        )

        time.sleep(1)

        results["scenario_4"] = (
            self.scenario_4_replanning()
        )

        print("\n" + "=" * 60)
        print("SCENARIO RESULTS")
        print("=" * 60)

        for name, result in results.items():
            print(
                f"{name}: "
                f"{'SUCCESS' if result else 'FAILED'}"
            )

        return results

    # ---------------------------------------------------------
    # CLOSE
    # ---------------------------------------------------------

    def close(self):
        """Close the simulation."""

        if self.env is not None:
            self.env.close()


# =============================================================
# MAIN
# =============================================================

def main():

    runner = ScenarioRunner(
        grid_size=12,
        max_steps=100,
        seed=42,
        delay=0.15,
    )

    try:
        runner.run_all()

        plt.ioff()
        plt.show()

    except KeyboardInterrupt:
        print("\nSimulation interrupted.")

    finally:
        runner.close()


if __name__ == "__main__":
    main()