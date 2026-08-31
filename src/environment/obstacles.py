from dataclasses import dataclass
from typing import Optional, Tuple


Position = Tuple[int, int]


@dataclass
class DynamicObstacle:
    """
    Base representation of a dynamic obstacle.

    Day 1 responsibility:
    Store the state and configuration of a dynamic obstacle.

    Movement behaviour will be implemented on Day 2.
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
        Update the obstacle position.

        The previous position is retained before the update.
        """

        self.previous_position = self.position
        self.position = new_position

    def activate(self) -> None:
        """Activate the obstacle."""

        self.active = True

    def deactivate(self) -> None:
        """Deactivate the obstacle."""

        self.active = False

    def get_position(self) -> Position:
        """Return current position."""

        return self.position

    def get_previous_position(self) -> Position:
        """Return previous position."""

        return self.previous_position