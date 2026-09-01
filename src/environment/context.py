"""
Context classification for the warehouse environment.

Day 3 scope:
- Local obstacle density
- Nearest static obstacle distance
- Nearest dynamic obstacle distance
- Available movement directions
- Basic aisle-structure detection

The classifier returns one of the existing project contexts:
    open
    aisle
    dense

Dynamic-heavy scenarios are intentionally not introduced here because
they belong to the later dynamic-risk/scenario stage.
"""

from typing import Iterable, Optional, Tuple

import numpy as np

from src.environment.constants import (
    SHELF,
    CONTEXT_OPEN,
    CONTEXT_AISLE,
    CONTEXT_DENSE,
)


Position = Tuple[int, int]


class ContextClassifier:
    """
    Analyze the local warehouse environment around the robot.

    Parameters
    ----------
    static_grid:
        2D numpy array containing FREE/SHELF cells.

    workers:
        Optional collection of Worker objects.

    dynamic_robots:
        Optional collection of DynamicRobot objects.

    local_radius:
        Radius of the local sensing window.
        A radius of 2 gives a 5x5 local area.
    """

    def __init__(
        self,
        static_grid: np.ndarray,
        workers: Optional[Iterable] = None,
        dynamic_robots: Optional[Iterable] = None,
        local_radius: int = 2,
    ):
        if static_grid.ndim != 2:
            raise ValueError("static_grid must be a 2D array.")

        if static_grid.shape[0] != static_grid.shape[1]:
            raise ValueError("static_grid must be square.")

        if local_radius < 1:
            raise ValueError("local_radius must be at least 1.")

        self.static_grid = static_grid
        self.grid_size = static_grid.shape[0]

        self.workers = list(workers or [])
        self.dynamic_robots = list(dynamic_robots or [])

        self.local_radius = local_radius

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def classify(self, robot_position: Position) -> dict:
        """
        Analyze the environment around the robot.

        Returns
        -------
        dict
            Context label and supporting measurements.
        """

        return {
            "context": self._classify_context(robot_position),
            "obstacle_density": self.local_obstacle_density(
                robot_position
            ),
            "nearest_static_distance": self.nearest_static_distance(
                robot_position
            ),
            "nearest_dynamic_distance": self.nearest_dynamic_distance(
                robot_position
            ),
            "available_directions": self.available_directions(
                robot_position
            ),
            "aisle_structure": self.detect_aisle_structure(
                robot_position
            ),
        }

    # ------------------------------------------------------------------
    # LOCAL OBSTACLE DENSITY
    # ------------------------------------------------------------------

    def local_obstacle_density(
        self,
        robot_position: Position,
    ) -> float:
        """
        Calculate the fraction of shelf cells inside the local
        sensing window.

        The robot's own cell is included in the window but normally
        contains FREE in the static grid.
        """

        row, col = robot_position

        shelf_count = 0
        total_cells = 0

        for r in range(
            row - self.local_radius,
            row + self.local_radius + 1,
        ):
            for c in range(
                col - self.local_radius,
                col + self.local_radius + 1,
            ):
                if not self._in_bounds(r, c):
                    continue

                total_cells += 1

                if self.static_grid[r, c] == SHELF:
                    shelf_count += 1

        if total_cells == 0:
            return 0.0

        return shelf_count / total_cells

    # ------------------------------------------------------------------
    # NEAREST STATIC OBSTACLE
    # ------------------------------------------------------------------

    def nearest_static_distance(
        self,
        robot_position: Position,
    ) -> Optional[int]:
        """
        Return Manhattan distance to the nearest shelf.

        Returns None if there are no shelves.
        """

        shelf_positions = np.argwhere(
            self.static_grid == SHELF
        )

        if len(shelf_positions) == 0:
            return None

        row, col = robot_position

        return min(
            abs(int(r) - row) + abs(int(c) - col)
            for r, c in shelf_positions
        )

    # ------------------------------------------------------------------
    # NEAREST DYNAMIC OBSTACLE
    # ------------------------------------------------------------------

    def nearest_dynamic_distance(
        self,
        robot_position: Position,
    ) -> Optional[int]:
        """
        Return Manhattan distance to the nearest active dynamic
        obstacle.

        Returns None if there are no active dynamic obstacles.
        """

        dynamic_positions = []

        for obstacle in self.workers + self.dynamic_robots:
            if obstacle.active:
                dynamic_positions.append(
                    obstacle.position
                )

        if not dynamic_positions:
            return None

        row, col = robot_position

        return min(
            abs(position[0] - row)
            + abs(position[1] - col)
            for position in dynamic_positions
        )

    # ------------------------------------------------------------------
    # AVAILABLE DIRECTIONS
    # ------------------------------------------------------------------

    def available_directions(
        self,
        robot_position: Position,
    ) -> int:
        """
        Count valid neighboring cells based on static obstacles
        and warehouse boundaries.

        Eight neighboring directions are considered.
        """

        row, col = robot_position

        count = 0

        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):

                if dr == 0 and dc == 0:
                    continue

                new_r = row + dr
                new_c = col + dc

                if not self._in_bounds(new_r, new_c):
                    continue

                if self.static_grid[new_r, new_c] == SHELF:
                    continue

                count += 1

        return count

    # ------------------------------------------------------------------
    # AISLE STRUCTURE
    # ------------------------------------------------------------------

    def detect_aisle_structure(
        self,
        robot_position: Position,
    ) -> bool:
        """
        Detect a simple aisle-like local structure.

        The heuristic identifies situations where shelves constrain
        the robot along opposite sides while leaving a corridor
        direction open.

        This is intentionally lightweight for Day 3. The classifier
        will be refined during the later context/risk stage.
        """

        row, col = robot_position

        north = self._is_shelf(row - 1, col)
        south = self._is_shelf(row + 1, col)
        east = self._is_shelf(row, col + 1)
        west = self._is_shelf(row, col - 1)

        vertical_constraint = bool(north and south)
        horizontal_constraint = bool(east and west)

        return bool(
            vertical_constraint
            or horizontal_constraint
        )

    # ------------------------------------------------------------------
    # CONTEXT CLASSIFICATION
    # ------------------------------------------------------------------

    def _classify_context(
        self,
        robot_position: Position,
    ) -> str:
        """
        Assign the base warehouse context.

        Dense takes priority over aisle because a highly obstructed
        area should be treated as dense even if it also exhibits
        aisle-like structure.
        """

        density = self.local_obstacle_density(
            robot_position
        )

        if density >= 0.50:
            return CONTEXT_DENSE

        if self.detect_aisle_structure(
            robot_position
        ):
            return CONTEXT_AISLE

        return CONTEXT_OPEN

    # ------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------

    def _in_bounds(
        self,
        row: int,
        col: int,
    ) -> bool:
        return (
            0 <= row < self.grid_size
            and
            0 <= col < self.grid_size
        )

    def _is_shelf(
        self,
        row: int,
        col: int,
    ) -> bool:
        """
        Out-of-bounds is treated as constrained for structure
        detection.
        """

        if not self._in_bounds(row, col):
            return True

        return self.static_grid[row, col] == SHELF