import time
import numpy as np

class MetricsTracker:
    def __init__(self):
        # Navigation & Safety
        self.successes = 0
        self.total_episodes = 0
        self.collisions = 0
        self.near_misses = 0
        self.min_obstacle_distances = []
        
        # Efficiency & Hybrid
        self.path_lengths = []
        self.astar_path_lengths = []
        self.replans = 0
        
        # Timing
        self.planning_times = []
        self.inference_times = []

        # Learning
        self.episode_rewards = []
        self.losses = []
        self.epsilons = []

    def record_step(self, reward, min_dist, q_value, inference_time):
        """Called every single time the robot takes a step."""
        if min_dist < 2.0:  # Define your own threshold for a near miss
            self.near_misses += 1
        self.min_obstacle_distances.append(min_dist)
        self.inference_times.append(inference_time)

    def record_replan(self, planning_time):
        """Called when the Dynamic Replanner is triggered."""
        self.replans += 1
        self.planning_times.append(planning_time)

    def record_episode(self, success, collision, steps, actual_length, astar_length, total_reward, loss, epsilon):
        """Called at the end of every episode."""
        self.total_episodes += 1
        if success:
            self.successes += 1
        if collision:
            self.collisions += 1
            
        self.path_lengths.append(actual_length)
        self.astar_path_lengths.append(astar_length)
        self.episode_rewards.append(total_reward)
        self.losses.append(loss)
        self.epsilons.append(epsilon)

    def get_summary(self):
        """Generates the dashboard metrics."""
        return {
            "Success Rate": self.successes / max(1, self.total_episodes),
            "Collision Rate": self.collisions / max(1, self.total_episodes),
            "Avg Episode Reward": np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0,
            "Total Replans": self.replans,
            "Path Efficiency": np.mean(self.astar_path_lengths) / max(1, np.mean(self.path_lengths)) if self.path_lengths else 0
        }