import sys
import os
import numpy as np

# Force the project root to the absolute front of Python's path list
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from src.environment.warehouse_env import WarehouseEnv
from src.environment.state_augmentation import StateAugmenter
from src.astar.astar_planner import AStarPlanner
from src.astar.replanner import DynamicReplanner
from src.hybrid.waypoint_manager import WaypointManager
from src.ddqn.agent import DDQNAgent
from src.hybrid.hybrid_agent import HybridAgent

def test_end_to_end_pipeline():
    print("--- Running Integration Test: End-to-End Hybrid Pipeline ---")
    
    env = WarehouseEnv(config="open")
    state_builder = StateAugmenter(env._static_grid, local_radius=2, max_dynamic_obstacles=4)
    
    ddqn = DDQNAgent(state_dim=51, action_dim=8)
    planner = AStarPlanner(env._static_grid)
    replanner = DynamicReplanner(planner)
    waypoint_manager = WaypointManager()
    
    hybrid_agent = HybridAgent(ddqn, waypoint_manager, replanner, planner)
    
    obs, info = env.reset()
    hybrid_agent.waypoint_manager.set_path([])
    
    done = False
    steps = 0
    
    # Run a short multi-step integration loop to verify component handoff
    while not done and steps < 50:
        current_pos = env.robot_pos
        goal_pos = env.goal_pos
        
        blocked_cells = {o.position for o in (env.workers + env.dynamic_robots) if getattr(o, 'active', True)}
        waypoint = hybrid_agent.update_route_and_waypoint(current_pos, goal_pos, blocked_cells)
        
        state = state_builder.build_state(
            robot_position=current_pos,
            goal_position=goal_pos,
            workers=env.workers,
            dynamic_robots=env.dynamic_robots,
            waypoint=waypoint,
            context=info.get("context", "open"),
            risk_level="low"
        )
        
        action = hybrid_agent.get_action(state, epsilon=0.0)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1
        
    print(f"End-to-end integration test completed successfully over {steps} steps!")
    assert steps > 0, "Pipeline failed to execute steps!"

if __name__ == "__main__":
    test_end_to_end_pipeline()