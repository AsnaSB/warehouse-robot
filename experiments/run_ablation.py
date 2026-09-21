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

def evaluate_ablation(env_config, architecture_type, episodes=20):
    env = WarehouseEnv(config=env_config)
    state_builder = StateAugmenter(env._static_grid, local_radius=2, max_dynamic_obstacles=4)
    
    ddqn = DDQNAgent(state_dim=51, action_dim=8)
    planner = AStarPlanner(env._static_grid)
    replanner = DynamicReplanner(planner)
    waypoint_manager = WaypointManager()
    
    # Map architectures to their respective trained weights
    weight_map = {
        "Full System (A*+DDQN+Adaptive)": "experiments/checkpoints/hybrid_ddqn_ep2500.pth",
        "Hybrid with Static Reward": "experiments/checkpoints/hybrid_static_ep2500.pth",
        "DDQN+Adaptive (No A*)": "experiments/checkpoints/pure_ddqn_ep2500.pth"
    }
    
    model_path = weight_map.get(architecture_type)
    if os.path.exists(model_path):
        ddqn.load(model_path)
    else:
        raise FileNotFoundError(f"Missing weights for {architecture_type} at {model_path}")
            
    hybrid_agent = HybridAgent(ddqn, waypoint_manager, replanner, planner)

    successes = 0
    collisions = 0
    cumulative_rewards = []

    for _ in range(episodes):
        obs, info = env.reset()
        hybrid_agent.waypoint_manager.set_path([])
        done = False
        steps = 0
        ep_reward = 0
        
        while not done and steps < 200:
            current_pos = env.robot_pos
            goal_pos = env.goal_pos
            blocked_cells = {o.position for o in (env.workers + env.dynamic_robots) if getattr(o, 'active', True)}
            
            if architecture_type == "DDQN+Adaptive (No A*)":
                # Ablation: Remove global A* planning, force waypoint to goal
                waypoint = goal_pos
            else:
                waypoint = hybrid_agent.update_route_and_waypoint(current_pos, goal_pos, blocked_cells)
                
            state = state_builder.build_state(
                robot_position=current_pos, 
                goal_position=goal_pos, 
                workers=env.workers, 
                dynamic_robots=env.dynamic_robots, 
                waypoint=waypoint, 
                context=env_config, 
                risk_level="low"
            )
            
            action = hybrid_agent.get_action(state, epsilon=0.0)
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            # Simulated reward tracking for ablation comparison
            if architecture_type == "Hybrid with Static Reward":
                reward = -0.1 if not terminated else (10 if env.robot_pos == env.goal_pos else -10)
                
            ep_reward += reward
            done = terminated or truncated
            steps += 1
            
        if env.robot_pos == env.goal_pos:
            successes += 1
        if info.get("collided", False):
            collisions += 1
        cumulative_rewards.append(ep_reward)

    return (successes / episodes) * 100, (collisions / episodes) * 100, np.mean(cumulative_rewards)

def run_ablation_study():
    print("--- Day 10: Architectural Ablation Study ---")
    architectures = [
        "Full System (A*+DDQN+Adaptive)", 
        "Hybrid with Static Reward", 
        "DDQN+Adaptive (No A*)"
    ]
    config = "maze" # Testing in a complex environment where A* matters
    
    print(f"{'Architecture':<32} | {'Success':<8} | {'Collision':<9} | {'Avg Reward'}")
    print("-" * 70)
    
    for arch in architectures:
        try:
            success, collision, avg_reward = evaluate_ablation(config, arch, episodes=15)
            print(f"{arch:<32} | {success:>7.1f}% | {collision:>8.1f}% | {avg_reward:>9.1f}")
        except Exception as e:
            print(f"{arch:<32} | ERROR: {str(e)}")

if __name__ == "__main__":
    run_ablation_study()