"""Baseline methods used for comparing navigation approaches."""

from src.astar.astar_planner import AStarPlanner


class AStarBaseline:
    """A* only baseline for reference path planning."""

    def __init__(self, env):
        self.env = env

    def get_path(self):
        """Find a path from the robot to the goal using A*."""
        planner = AStarPlanner(self.env._static_grid)
        return planner.find_path(
            self.env.robot_pos,
            self.env.goal_pos
        )


class DDQNBaseline:
    """DDQN only baseline using the trained DDQN agent."""

    def __init__(self, agent):
        self.agent = agent

    def select_action(self, state):
        """Select an action using DDQN."""
        return self.agent.select_action(state)


class HybridBaseline:
    """Hybrid DDQN+A* baseline using the hybrid agent."""

    def __init__(self, agent):
        self.agent = agent

    def reset(self):
        """Reset the hybrid agent."""
        return self.agent.reset()

    def detect_context(self, *args, **kwargs):
        """Detect the current warehouse context."""
        return self.agent.detect_context(*args, **kwargs)

    def should_replan(self, *args, **kwargs):
        """Check whether replanning is required."""
        return self.agent.should_replan(*args, **kwargs)

    def calculate_reward(self, *args, **kwargs):
        """Calculate the adaptive reward."""
        return self.agent.calculate_reward(*args, **kwargs)
    