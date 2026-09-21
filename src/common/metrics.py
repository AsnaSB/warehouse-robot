"""
metrics.py

Tracks training and evaluation metrics for the warehouse robot.
"""

class MetricsTracker:
    def __init__(self):
        # Step-level metrics
        self.near_misses = 0
        
        # Episode-level metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.successes = []
        self.collisions = []
        self.losses = []
        self.epsilons = []

    def record_step(self, reward, min_dist, q_value, inference_time):
        """Records data for a single step in the environment."""
        # Safely handle cases where min_dist is None (no obstacles nearby)
        if min_dist is not None and min_dist < 2.0:  # Define your own threshold for a near miss
            self.near_misses += 1

    def record_episode(
        self, 
        success, 
        collision, 
        steps, 
        actual_length, 
        astar_length, 
        total_reward, 
        loss, 
        epsilon
    ):
        """Records aggregate data at the end of an episode."""
        self.successes.append(success)
        self.collisions.append(collision)
        self.episode_lengths.append(steps)
        self.episode_rewards.append(total_reward)
        self.losses.append(loss)
        self.epsilons.append(epsilon)