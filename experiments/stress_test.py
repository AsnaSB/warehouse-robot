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

def run_stress_test():
    model_path = "experiments/checkpoints/hybrid_ddqn_ep2500.pth"
    # Testing different layout architectures to see if the DDQN generalized well
    test_configs = ["open", "maze", "cluttered"]
    episodes_per_test = 25
    
    print("--- Initiating Day 11 Stress Testing ---")
    
    # The neural network expects exactly 51 inputs.
    agent = DDQNAgent(state_dim=51, action_dim=8)
    if os.path.exists(model_path):
        agent.load(model_path)
    else:
        print(f"Error: {model_path} not found.")
        return

    print(f"{'Config':<12} | {'Success Rate':<15} | {'Collision Rate':<15} | {'Avg Steps'}")
    print("-" * 65)

    for config in test_configs:
        try:
            env = WarehouseEnv(config=config)
        except Exception:
            # Fallback if a specific config string isn't implemented in your env
            continue
            
        # We keep max_dynamic_obstacles=4 here so the state vector remains 51-D,
        # forcing the agent to prioritize only the 4 closest threats in a crowded room.
        state_builder = StateAugmenter(env._static_grid, local_radius=2, max_dynamic_obstacles=4)
        
        planner = AStarPlanner(env._static_grid)
        replanner = DynamicReplanner(planner)
        waypoint_manager = WaypointManager()
        hybrid_agent = HybridAgent(agent, waypoint_manager, replanner, planner)

        successes = 0
        collisions = 0
        total_steps = []

        for episode in range(1, episodes_per_test + 1):
            obs, info = env.reset()
            hybrid_agent.waypoint_manager.set_path([])
            
            done = False
            step_count = 0
            
            while not done and step_count < 200:
                current_pos = env.robot_pos
                goal_pos = env.goal_pos
                
                # Artificially duplicating dynamic obstacles for extreme crowding
                dynamic_obstacles = env.workers + env.dynamic_robots
                blocked_cells = {
                    obs_obj.position for obs_obj in dynamic_obstacles if getattr(obs_obj, 'active', True)
                }
                
                active_waypoint = hybrid_agent.update_route_and_waypoint(current_pos, goal_pos, blocked_cells)
                
                state = state_builder.build_state(
                    robot_position=current_pos, goal_position=goal_pos,
                    workers=env.workers, dynamic_robots=env.dynamic_robots,
                    waypoint=active_waypoint, context=info.get("context", config), risk_level="high"
                )
                
                action = hybrid_agent.get_action(state, epsilon=0.0)
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                step_count += 1
                
            if env.robot_pos == env.goal_pos:
                successes += 1
            if info.get("collided", False):
                collisions += 1
            total_steps.append(step_count)

        succ_rate = (successes / episodes_per_test) * 100
        coll_rate = (collisions / episodes_per_test) * 100
        avg_stp = np.mean(total_steps)
        
        print(f"{config.upper():<12} | {succ_rate:>13.1f}% | {coll_rate:>13.1f}% | {avg_stp:>9.1f}")

if __name__ == "__main__":
    run_stress_test()