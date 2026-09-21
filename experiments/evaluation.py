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

def evaluate_model(model_path, is_hybrid=True, num_episodes=50):
    print(f"--- Evaluating {'Hybrid' if is_hybrid else 'Pure DDQN'} Model ---")
    print(f"Loading weights from: {model_path}")
    
    env = WarehouseEnv(config="open")
    state_builder = StateAugmenter(
        static_grid=env._static_grid, 
        local_radius=2, 
        max_dynamic_obstacles=4
    )
    
    agent = DDQNAgent(state_dim=51, action_dim=8)
    
    # Load the trained weights
    if os.path.exists(model_path):
        agent.load(model_path)
    else:
        print(f"Error: Checkpoint not found at {model_path}")
        return

    if is_hybrid:
        planner = AStarPlanner(env._static_grid)
        replanner = DynamicReplanner(planner)
        waypoint_manager = WaypointManager()
        hybrid_agent = HybridAgent(agent, waypoint_manager, replanner, planner)

    successes = 0
    collisions = 0
    total_steps = []

    for episode in range(1, num_episodes + 1):
        obs, info = env.reset()
        if is_hybrid:
            hybrid_agent.waypoint_manager.set_path([])
            
        done = False
        step_count = 0
        
        while not done and step_count < 200:
            current_pos = env.robot_pos
            goal_pos = env.goal_pos
            
            if is_hybrid:
                dynamic_obstacles = env.workers + env.dynamic_robots
                blocked_cells = {
                    obs_obj.position 
                    for obs_obj in dynamic_obstacles 
                    if getattr(obs_obj, 'active', True)
                }
                active_waypoint = hybrid_agent.update_route_and_waypoint(
                    current_pos, goal_pos, blocked_cells
                )
                env.set_astar_path(hybrid_agent.waypoint_manager.path)
            else:
                active_waypoint = goal_pos
                
            state = state_builder.build_state(
                robot_position=current_pos,
                goal_position=goal_pos,
                workers=env.workers,
                dynamic_robots=env.dynamic_robots,
                waypoint=active_waypoint,
                context=info.get("context", "open"),
                risk_level=info.get("risk_level", "low")
            )
            
            # Epsilon is forced to 0.0 for pure exploitation (testing)
            if is_hybrid:
                action = hybrid_agent.get_action(state, epsilon=0.0)
            else:
                action = agent.select_action(state, epsilon=0.0)
            
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step_count += 1
            
        if env.robot_pos == env.goal_pos:
            successes += 1
        if info.get("collided", False):
            collisions += 1
        total_steps.append(step_count)
        
        status = "Success" if env.robot_pos == env.goal_pos else "Failed"
        print(f"Test Ep {episode:02d}: {status} | Steps: {step_count}")

    print("\n--- Final Evaluation Results ---")
    print(f"Success Rate:   {(successes / num_episodes) * 100:.1f}%")
    print(f"Collision Rate: {(collisions / num_episodes) * 100:.1f}%")
    print(f"Avg Steps:      {np.mean(total_steps):.1f}")

if __name__ == "__main__":
    # Point these to the highest episode checkpoints you just generated
    hybrid_checkpoint = "experiments/checkpoints/hybrid_ddqn_ep2500.pth"
    pure_checkpoint = "experiments/checkpoints/pure_ddqn_ep2500.pth"
    
    # 1. Test Hybrid Agent (25 Episodes)
    evaluate_model(hybrid_checkpoint, is_hybrid=True, num_episodes=25)
    
    print("\n" + "="*50 + "\n")
    
    # 2. Test Pure DDQN Agent (25 Episodes)
    evaluate_model(pure_checkpoint, is_hybrid=False, num_episodes=25)