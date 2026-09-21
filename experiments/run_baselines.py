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

def evaluate_agent(env_config, agent_type, episodes=20):
    env = WarehouseEnv(config=env_config)
    state_builder = StateAugmenter(env._static_grid, local_radius=2, max_dynamic_obstacles=4)
    
    # Initialize components
    ddqn = DDQNAgent(state_dim=51, action_dim=8)
    planner = AStarPlanner(env._static_grid)
    replanner = DynamicReplanner(planner)
    waypoint_manager = WaypointManager()
    
    # Load respective weights based on agent type
    if agent_type == "Pure DDQN":
        model_path = "experiments/checkpoints/pure_ddqn_ep2500.pth"
        if os.path.exists(model_path):
            ddqn.load(model_path)
        else:
            raise FileNotFoundError(f"Missing weights at {model_path}")
            
    elif agent_type == "Hybrid (A* + DDQN)":
        model_path = "experiments/checkpoints/hybrid_ddqn_ep2500.pth"
        if os.path.exists(model_path):
            ddqn.load(model_path)
        else:
            raise FileNotFoundError(f"Missing weights at {model_path}")
            
    hybrid_agent = HybridAgent(ddqn, waypoint_manager, replanner, planner)

    successes = 0
    collisions = 0
    steps_list = []

    for _ in range(episodes):
        obs, info = env.reset()
        hybrid_agent.waypoint_manager.set_path([])
        done = False
        steps = 0
        
        while not done and steps < 200:
            current_pos = env.robot_pos
            goal_pos = env.goal_pos
            blocked_cells = {o.position for o in (env.workers + env.dynamic_robots) if getattr(o, 'active', True)}
            
            if agent_type == "Pure A*":
                env.set_astar_path(planner.find_path(current_pos, goal_pos, blocked_cells))
                action = 0 
                if current_pos in blocked_cells:
                    info["collided"] = True
                    done = True
            elif agent_type == "Pure DDQN":
                state = state_builder.build_state(
                    robot_position=current_pos, 
                    goal_position=goal_pos, 
                    workers=env.workers, 
                    dynamic_robots=env.dynamic_robots, 
                    waypoint=goal_pos, 
                    context=env_config, 
                    risk_level="low"
                )
                # Use the working hybrid wrapper to fetch the action safely
                action = hybrid_agent.get_action(state, epsilon=0.0)
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
                
            if not done:
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            steps += 1
            
        if env.robot_pos == env.goal_pos:
            successes += 1
        if info.get("collided", False):
            collisions += 1
        steps_list.append(steps)

    return (successes / episodes) * 100, (collisions / episodes) * 100, np.mean(steps_list)

def run_baselines():
    print("--- Day 10: Baseline Execution Framework ---")
    baselines = ["Pure A*", "Pure DDQN", "Hybrid (A* + DDQN)"]
    config = "open" 
    
    print(f"{'Method':<20} | {'Success Rate':<15} | {'Collision Rate':<15} | {'Avg Steps'}")
    print("-" * 75)
    
    for baseline in baselines:
        try:
            success, collision, avg_steps = evaluate_agent(config, baseline, episodes=20)
            print(f"{baseline:<20} | {success:>13.1f}% | {collision:>13.1f}% | {avg_steps:>9.1f}")
        except Exception as e:
            print(f"{baseline:<20} | ERROR: {str(e)}")

if __name__ == "__main__":
    run_baselines()