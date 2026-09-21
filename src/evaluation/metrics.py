from dataclasses import dataclass


@dataclass
class EpisodeMetrics:
    """Stores evaluation results for one navigation episode."""

    total_reward: float = 0.0
    steps: int = 0
    path_length: float = 0.0
    optimal_path_length: float = 0.0
    collisions: int = 0
    goal_reached: bool = False
    replans: int = 0

    @property
    def success(self) -> bool:
        """Whether the robot successfully reached the goal."""
        return self.goal_reached

    @property
    def path_efficiency(self) -> float:
        """
        Ratio between executed path length and optimal path length.

        A value closer to 1.0 means the executed path is closer
        to the reference optimal path.
        """
        if self.optimal_path_length <= 0:
            return 0.0

        return self.path_length / self.optimal_path_length