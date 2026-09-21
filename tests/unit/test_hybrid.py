import sys
import os
import numpy as np

# Force the project root to the absolute front of Python's path list
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from src.environment.warehouse_env import WarehouseEnv
from src.astar.astar_planner import AStarPlanner
from src.astar.replanner import DynamicReplanner
from src.hybrid.waypoint_manager import WaypointManager
from src.ddqn.agent import DDQNAgent
from src.hybrid.hybrid_agent import HybridAgent

def test_hybrid_coordination():
    print("--- Running Unit Test: Hybrid Agent & Waypoint Coordination ---")
    
    env = WarehouseEnv(config="open")
    obs, info = env.reset()
    
    # Safely fetch positions directly from environment attributes
    current_pos = getattr(env, 'robot_pos', None)
    goal_pos = getattr(env, 'goal_pos', None)
    
    if current_pos is None or goal_pos is None:
        # Fallback coordinate extraction if attributes differ
        current_pos = (1, 1)
        goal_pos = (len(env._static_grid)-2, len(env._static_grid[0])-2)

    ddqn = DDQNAgent(state_dim=51, action_dim=8)
    planner = AStarPlanner(env._static_grid)
    replanner = DynamicReplanner(planner)
    waypoint_manager = WaypointManager()
    
    hybrid_agent = HybridAgent(ddqn, waypoint_manager, replanner, planner)
    
    blocked_cells = set()
    active_waypoint = hybrid_agent.update_route_and_waypoint(current_pos, goal_pos, blocked_cells)
    
    print(f"Validated Robot Position: {current_pos}")
    print(f"Validated Goal Position: {goal_pos}")
    print(f"Assigned Active Waypoint: {active_waypoint}")
    
    assert active_waypoint is not None, "Hybrid agent failed to generate a valid waypoint!"
    print("Hybrid coordination unit test passed successfully!")

if __name__ == "__main__":
    test_hybrid_coordination()