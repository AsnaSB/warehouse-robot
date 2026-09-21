import math

from src.astar.astar_planner import AStarPlanner
from src.evaluation.metrics import EpisodeMetrics


class Evaluator:
    """Runs navigation episodes and collects evaluation metrics."""

    def __init__(self, env):
        self.env = env

    def _get_optimal_path_length(self):
        """Calculate reference A* path length."""

        planner = AStarPlanner(self.env._static_grid)

        path = planner.find_path(
            self.env.robot_pos,
            self.env.goal_pos
        )

        if path is None or len(path) < 2:
            return 0.0

        length = 0.0

        for current, next_pos in zip(path, path[1:]):
            dr = next_pos[0] - current[0]
            dc = next_pos[1] - current[1]

            length += math.sqrt(dr ** 2 + dc ** 2)

        return length

    def evaluate_episode(self, agent):
        """Run one episode using the given agent."""

        state, _ = self.env.reset()

        metrics = EpisodeMetrics()

        # Reference A* path
        metrics.optimal_path_length = self._get_optimal_path_length()

        previous_pos = self.env.robot_pos

        terminated = False
        truncated = False

        while not (terminated or truncated):

            action = agent.select_action(state)

            next_state, reward, terminated, truncated, info = self.env.step(action)

            metrics.total_reward += float(reward)
            metrics.steps += 1

            # Calculate actual movement distance
            current_pos = self.env.robot_pos

            dr = current_pos[0] - previous_pos[0]
            dc = current_pos[1] - previous_pos[1]

            metrics.path_length += math.sqrt(dr ** 2 + dc ** 2)

            previous_pos = current_pos

            # Collision
            if info.get("collided", False):
                metrics.collisions += 1

            # Goal
            if current_pos == self.env.goal_pos:
                metrics.goal_reached = True

            state = next_state

        return metrics