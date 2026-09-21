import sys
import os
import numpy as np

# Force the project root to the absolute front of Python's path list
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.environment.warehouse_env import WarehouseEnv
from src.environment.state_augmentation import StateAugmenter
from src.astar.astar_planner import AStarPlanner
from src.astar.replanner import DynamicReplanner
from src.hybrid.waypoint_manager import WaypointManager
from src.ddqn.agent import DDQNAgent
from src.hybrid.hybrid_agent import HybridAgent

def tune_hybrid_agent():
    print("--- Initiating Day 13: Hyperparameter Fine-Tuning ---")
    
    env = WarehouseEnv(config="open")
    state_builder = StateAugmenter(env._static_grid, local_radius=2, max_dynamic_obstacles=4)
    
    agent = DDQNAgent(state_dim=51, action_dim=8)
    model_path = "experiments/checkpoints/hybrid_ddqn_ep2500.pth"
    
    # 1. Load the perfectly safe, collision-free weights
    if os.path.exists(model_path):
        agent.load(model_path)
        print(f"Loaded base weights from {model_path}")
    else:
        print("Error: Base model not found.")
        return

    # 2. Drop the learning rate (e.g., 1e-4) so it fine-tunes rather than overwrites
    for param_group in agent.optimizer.param_groups:
        param_group['lr'] = 1e-4  

    planner = AStarPlanner(env._static_grid)
    replanner = DynamicReplanner(planner)
    hybrid_agent = HybridAgent(agent, WaypointManager(), replanner, planner)

    num_episodes = 500
    epsilon = 0.20        # Just enough randomness to explore bolder, faster moves
    epsilon_decay = 0.99
    epsilon_min = 0.01

    for episode in range(1, num_episodes + 1):
        obs, info = env.reset()
        hybrid_agent.waypoint_manager.set_path([])
        
        total_reward = 0
        done = False
        step_count = 0
        
        while not done and step_count < 200:
            current_pos = env.robot_pos
            goal_pos = env.goal_pos
            
            blocked_cells = {
                obs_obj.position for obs_obj in (env.workers + env.dynamic_robots) if getattr(obs_obj, 'active', True)
            }
            
            active_waypoint = hybrid_agent.update_route_and_waypoint(current_pos, goal_pos, blocked_cells)
            
            state = state_builder.build_state(
                robot_position=current_pos, goal_position=goal_pos,
                workers=env.workers, dynamic_robots=env.dynamic_robots,
                waypoint=active_waypoint, context=info.get("context", "open"), risk_level="low"
            )
            
            action = hybrid_agent.get_action(state, epsilon=epsilon)
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            # --- DAY 13 REWARD SHAPING ---
            # Inject a heavier penalty for every step taken.
            # This makes "freezing" painful, forcing the AI to move toward the goal.
            reward -= 0.5 
            
            done = terminated or truncated
            
            next_pos = env.robot_pos
            next_blocked = {
                o.position for o in (env.workers + env.dynamic_robots) if getattr(o, 'active', True)
            }
            next_waypoint = hybrid_agent.update_route_and_waypoint(next_pos, goal_pos, next_blocked)
            next_state = state_builder.build_state(
                robot_position=next_pos, goal_position=goal_pos,
                workers=env.workers, dynamic_robots=env.dynamic_robots,
                waypoint=next_waypoint, context=info.get("context", "open"), risk_level="low"
            )
            
            # Store the heavily penalized transition and train
            hybrid_agent.ddqn.store_transition(state, action, reward, next_state, done)
            hybrid_agent.ddqn.train_step()
            
            total_reward += reward
            step_count += 1
            
        hybrid_agent.ddqn.update_target_network()
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        
        if episode % 50 == 0:
            success = "Success" if env.robot_pos == env.goal_pos else "Failed"
            print(f"Fine-Tune Ep {episode:03d} | Status: {success:<7} | Steps: {step_count:<4} | Eps: {epsilon:.3f}")

    # 3. Save the newly optimized brain
    save_path = "experiments/checkpoints/best_hybrid_tuned.pth"
    hybrid_agent.ddqn.save(save_path)
    print(f"\nOptimization complete! Aggressive weights saved to {save_path}")

if __name__ == "__main__":
    tune_hybrid_agent()