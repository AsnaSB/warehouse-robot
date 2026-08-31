from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import random


Position = Tuple[int, int]


@dataclass
class DynamicObstacle:
    """
    Base class for dynamic obstacles in the warehouse.

    Day 1:
        - position
        - previous_position
        - movement_pattern
        - speed
        - active state

    Day 2:
        Worker and DynamicRobot extend this base class
        with their respective movement behaviours.
    """

    position: Position
    movement_pattern: str
    speed: float = 1.0
    active: bool = True
    previous_position: Optional[Position] = None

    def __post_init__(self):
        """Initialize previous position."""

        if self.previous_position is None:
            self.previous_position = self.position

    def update_position(self, new_position: Position) -> None:
        """
        Update the current position while preserving
        the previous position.
        """

        self.previous_position = self.position
        self.position = new_position

    def activate(self) -> None:
        """Activate the dynamic obstacle."""

        self.active = True

    def deactivate(self) -> None:
        """Deactivate the dynamic obstacle."""

        self.active = False

    def get_position(self) -> Position:
        """Return the current position."""

        return self.position

    def get_previous_position(self) -> Position:
        """Return the previous position."""

        return self.previous_position


@dataclass
class Worker(DynamicObstacle):
    """
    Dynamic warehouse worker.

    Supported movement modes:
        - predefined patrol route
        - random patrol

    Additional behaviour:
        - configurable speed
        - direction changes
        - boundary handling
        - optional stopping/pausing
    """

    patrol_route: List[Position] = field(default_factory=list)
    random_patrol: bool = False
    pause_steps: int = 0
    current_route_index: int = 0
    direction: int = 1
    remaining_pause_steps: int = 0
    grid_size: Optional[Tuple[int, int]] = None

    def __post_init__(self):
        super().__post_init__()

        if self.random_patrol:
            self.movement_pattern = "random_patrol"
        elif self.movement_pattern == "":
            self.movement_pattern = "predefined_patrol"

        if self.patrol_route:
            if self.position not in self.patrol_route:
                self.patrol_route.insert(0, self.position)

            self.current_route_index = self.patrol_route.index(
                self.position
            )

    def update(self) -> Position:
        """
        Perform one movement update.

        Returns
        -------
        Position
            The worker's new position.
        """

        if not self.active:
            return self.position

        if self.remaining_pause_steps > 0:
            self.remaining_pause_steps -= 1
            return self.position

        if self.random_patrol:
            return self._random_patrol_step()

        return self._predefined_patrol_step()

    def _predefined_patrol_step(self) -> Position:
        """Move along a predefined patrol route."""

        if len(self.patrol_route) <= 1:
            self._start_pause()
            return self.position

        next_index = self.current_route_index + self.direction

        # Reverse direction at either end of the route.
        if next_index >= len(self.patrol_route):
            self.direction = -1
            next_index = self.current_route_index + self.direction

        elif next_index < 0:
            self.direction = 1
            next_index = self.current_route_index + self.direction

        new_position = self.patrol_route[next_index]

        if self.grid_size is not None:
            if not self._inside_grid(new_position):
                self.direction *= -1
                next_index = self.current_route_index + self.direction

                if 0 <= next_index < len(self.patrol_route):
                    new_position = self.patrol_route[next_index]
                else:
                    new_position = self.position

        self.current_route_index = next_index

        self.update_position(new_position)

        self._start_pause()

        return self.position

    def _random_patrol_step(self) -> Position:
        """
        Move randomly to a neighbouring valid cell.

        Movement is restricted to the four cardinal directions
        for the worker patrol behaviour.
        """

        row, col = self.position

        candidates = [
            (row - 1, col),
            (row + 1, col),
            (row, col - 1),
            (row, col + 1),
        ]

        if self.grid_size is not None:
            candidates = [
                position
                for position in candidates
                if self._inside_grid(position)
            ]

        if not candidates:
            return self.position

        new_position = random.choice(candidates)

        self.update_position(new_position)

        self._start_pause()

        return self.position

    def _inside_grid(self, position: Position) -> bool:
        """Check whether a position is inside the warehouse grid."""

        if self.grid_size is None:
            return True

        rows, cols = self.grid_size
        row, col = position

        return (
            0 <= row < rows
            and
            0 <= col < cols
        )

    def _start_pause(self) -> None:
        """Start an optional pause after movement."""

        if self.pause_steps > 0:
            self.remaining_pause_steps = self.pause_steps


@dataclass
class DynamicRobot(DynamicObstacle):
    """
    Dynamic warehouse robot.

    Supports:
        - linear movement
        - forward/backward movement
        - path reversal
        - configurable speed
        - boundary handling
        - collision detection

    Collision detection is kept generic so that the
    WarehouseEnv can provide occupied positions during
    later environment integration.
    """

    movement_direction: Position = (0, 1)
    allow_reverse: bool = True
    grid_size: Optional[Tuple[int, int]] = None
    movement_path: List[Position] = field(default_factory=list)
    current_path_index: int = 0
    forward: bool = True

    # Day 2 collision state
    last_collision: bool = False

    def __post_init__(self):
        super().__post_init__()

        if self.movement_pattern == "":
            self.movement_pattern = "linear"

        if self.movement_path:
            if self.position not in self.movement_path:
                self.movement_path.insert(0, self.position)

            self.current_path_index = self.movement_path.index(
                self.position
            )

    def update(
        self,
        occupied_positions: Optional[List[Position]] = None
    ) -> Position:
        """
        Perform one movement update.

        Parameters
        ----------
        occupied_positions:
            Positions occupied by other entities or obstacles.
            If a candidate position is occupied, the robot does
            not move and last_collision is set to True.

        Returns
        -------
        Position
            The robot's current position after the update.
        """

        self.last_collision = False

        if not self.active:
            return self.position

        if occupied_positions is None:
            occupied_positions = []

        if self.movement_path:
            return self._path_movement(occupied_positions)

        return self._linear_movement(occupied_positions)

    def _linear_movement(
        self,
        occupied_positions: List[Position]
    ) -> Position:
        """
        Move linearly according to movement_direction while
        checking boundaries and collisions.
        """

        row, col = self.position
        dr, dc = self.movement_direction

        new_position = (
            row + dr,
            col + dc
        )

        # Boundary handling
        if self.grid_size is not None:
            if not self._inside_grid(new_position):

                if self.allow_reverse:
                    self._reverse_direction()

                    dr, dc = self.movement_direction

                    new_position = (
                        row + dr,
                        col + dc
                    )

                    if not self._inside_grid(new_position):
                        return self.position

                else:
                    return self.position

        # Collision detection
        if self.check_collision(
            new_position,
            occupied_positions
        ):
            return self.position

        self.update_position(new_position)

        return self.position

    def _path_movement(
        self,
        occupied_positions: List[Position]
    ) -> Position:
        """
        Move forward/backward along a predefined path while
        checking collisions.
        """

        if len(self.movement_path) <= 1:
            return self.position

        if self.forward:
            next_index = self.current_path_index + 1

            if next_index >= len(self.movement_path):
                if self.allow_reverse:
                    self.forward = False
                    next_index = self.current_path_index - 1
                else:
                    return self.position

        else:
            next_index = self.current_path_index - 1

            if next_index < 0:
                if self.allow_reverse:
                    self.forward = True
                    next_index = self.current_path_index + 1
                else:
                    return self.position

        new_position = self.movement_path[next_index]

        # Boundary handling
        if self.grid_size is not None:
            if not self._inside_grid(new_position):
                if self.allow_reverse:
                    self.forward = not self.forward
                return self.position

        # Collision detection
        if self.check_collision(
            new_position,
            occupied_positions
        ):
            return self.position

        self.current_path_index = next_index
        self.update_position(new_position)

        return self.position

    def check_collision(
        self,
        position: Position,
        occupied_positions: List[Position]
    ) -> bool:
        """
        Check whether a candidate position is occupied.

        Parameters
        ----------
        position:
            Candidate position the robot wants to enter.

        occupied_positions:
            Positions occupied by other objects.

        Returns
        -------
        bool
            True if the position is occupied, otherwise False.
        """

        collision = position in occupied_positions

        self.last_collision = collision

        return collision

    def _reverse_direction(self) -> None:
        """Reverse the current linear movement direction."""

        dr, dc = self.movement_direction

        self.movement_direction = (
            -dr,
            -dc
        )

    def _inside_grid(self, position: Position) -> bool:
        """Check whether a position is inside the warehouse grid."""

        if self.grid_size is None:
            return True

        rows, cols = self.grid_size
        row, col = position

        return (
            0 <= row < rows
            and
            0 <= col < cols
        )