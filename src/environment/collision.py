from typing import Iterable, Optional, Tuple


Position = Tuple[int, int]


class CollisionModel:
    """
    Collision and movement validation for the warehouse environment.

    Handles:

    - Warehouse boundaries
    - Static shelves
    - Dynamic workers
    - Dynamic robots
    - Diagonal movement safety

    This module does not control obstacle movement.
    It only determines whether a proposed robot movement is valid.
    """

    def __init__(
        self,
        grid_size: int,
        static_grid,
        workers: Optional[Iterable] = None,
        dynamic_robots: Optional[Iterable] = None,
    ):
        self.grid_size = grid_size
        self.static_grid = static_grid

        self.workers = list(workers or [])
        self.dynamic_robots = list(dynamic_robots or [])

    # ------------------------------------------------------------------
    # PUBLIC MOVEMENT CHECKS
    # ------------------------------------------------------------------

    def can_move(
        self,
        current_position: Position,
        new_position: Position,
    ) -> bool:
        """
        Check whether the robot can move from current_position
        to new_position.

        Checks:

        1. Boundary
        2. Static shelf
        3. Worker
        4. Dynamic robot
        """

        if not self._in_bounds(new_position):
            return False

        if self._is_shelf(new_position):
            return False

        if self._occupied_by_worker(new_position):
            return False

        if self._occupied_by_dynamic_robot(new_position):
            return False

        return True

    def can_move_diagonal(
        self,
        current_position: Position,
        new_position: Position,
    ) -> bool:
        """
        Check whether a diagonal robot movement is safe.

        A diagonal movement is allowed only when:

        1. The destination is valid.
        2. The horizontal intermediate cell is valid.
        3. The vertical intermediate cell is valid.

        This prevents the robot from cutting through the
        corner of two adjacent obstacles.
        """

        current_row, current_col = current_position
        new_row, new_col = new_position

        row_change = new_row - current_row
        col_change = new_col - current_col

        # First validate the destination.
        if not self.can_move(
            current_position,
            new_position,
        ):
            return False

        # Only apply corner safety to diagonal movement.
        if abs(row_change) != 1 or abs(col_change) != 1:
            return True

        horizontal_position = (
            current_row,
            new_col,
        )

        vertical_position = (
            new_row,
            current_col,
        )

        if not self.can_move(
            current_position,
            horizontal_position,
        ):
            return False

        if not self.can_move(
            current_position,
            vertical_position,
        ):
            return False

        return True

    # ------------------------------------------------------------------
    # SETTERS
    # ------------------------------------------------------------------

    def set_dynamic_obstacles(
        self,
        workers: Optional[Iterable] = None,
        dynamic_robots: Optional[Iterable] = None,
    ) -> None:
        """
        Update the currently active dynamic obstacles.
        """

        self.workers = list(workers or [])
        self.dynamic_robots = list(dynamic_robots or [])

    # ------------------------------------------------------------------
    # BOUNDARY
    # ------------------------------------------------------------------

    def _in_bounds(self, position: Position) -> bool:
        """Return True when position lies inside the grid."""

        row, col = position

        return (
            0 <= row < self.grid_size
            and
            0 <= col < self.grid_size
        )

    # ------------------------------------------------------------------
    # STATIC OBSTACLES
    # ------------------------------------------------------------------

    def _is_shelf(self, position: Position) -> bool:
        """Return True when the position contains a static shelf."""

        row, col = position

        return bool(self.static_grid[row, col] == 1)

    # ------------------------------------------------------------------
    # DYNAMIC OBSTACLES
    # ------------------------------------------------------------------

    def _occupied_by_worker(
        self,
        position: Position,
    ) -> bool:
        """Return True when a worker occupies the position."""

        return any(
            obstacle.active
            and obstacle.position == position
            for obstacle in self.workers
        )

    def _occupied_by_dynamic_robot(
        self,
        position: Position,
    ) -> bool:
        """Return True when a dynamic robot occupies the position."""

        return any(
            obstacle.active
            and obstacle.position == position
            for obstacle in self.dynamic_robots
        )