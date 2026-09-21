import sys
import os

# Force the project root to the absolute front of Python's path list
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import numpy as np
from src.environment.warehouse_env import WarehouseEnv
from src.environment.state_augmentation import StateAugmenter
from src.ddqn.agent import DDQNAgent
from src.common.metrics import MetricsTracker

def train_pure_ddqn():
    num_episodes = 2500
    max_steps = 200
    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.998
    save_interval = 250
    
    os.makedirs("experiments/checkpoints", exist_ok=True)
    
    # Use 'open' layout to avoid config crashes
    env = WarehouseEnv(config="open")
    
    state_builder = StateAugmenter(
        static_grid=env._static_grid, 
        local_radius=2, 
        max_dynamic_obstacles=4
    )
    
    agent = DDQNAgent(state_dim=51, action_dim=8)
    tracker = MetricsTracker()

    print("--- Starting Pure DDQN Baseline Training ---")

    for episode in range(1, num_episodes + 1):
        obs, info = env.reset()
        total_reward = 0
        done = False
        step_count = 0
        episode_loss = []
        
        while not done and step_count < max_steps:
            current_pos = env.robot_pos
            goal_pos = env.goal_pos
            
            # Pure DDQN doesn't use A* waypoints, so we just pass the final goal
            state = state_builder.build_state(
                robot_position=current_pos,
                goal_position=goal_pos,
                workers=env.workers,
                dynamic_robots=env.dynamic_robots,
                waypoint=goal_pos, 
                context=info.get("context", "open"),
                risk_level=info.get("risk_level", "low")
            )
            
            action = agent.select_action(state, epsilon=epsilon)
            next_obs, reward, terminated, truncated, next_info = env.step(action)
            done = terminated or truncated
            
            next_pos = env.robot_pos
            next_state = state_builder.build_state(
                robot_position=next_pos,
                goal_position=goal_pos,
                workers=env.workers,
                dynamic_robots=env.dynamic_robots,
                waypoint=goal_pos,
                context=next_info.get("context", "open"),
                risk_level=next_info.get("risk_level", "low")
            )
            
            agent.store_transition(state, action, reward, next_state, done)
            loss = agent.train_step()
            
            if loss is not None:
                episode_loss.append(loss)
                
            total_reward += reward
            step_count += 1
            info = next_info

        agent.update_target_network()
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        
        if episode % save_interval == 0:
            agent.save(f"experiments/checkpoints/pure_ddqn_ep{episode}.pth")
            success = (env.robot_pos == env.goal_pos)
            print(f"Pure DDQN Ep {episode} | Reward: {total_reward:.2f} | Eps: {epsilon:.3f} | Reached Goal: {success}")

if __name__ == "__main__":
    train_pure_ddqn()