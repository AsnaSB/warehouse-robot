import time
from src.environment.warehouse_env import WarehouseEnv
from src.astar.astar_planner import AStarPlanner
from src.hybrid.waypoint_manager import WaypointManager
from src.hybrid.state_builder import HybridStateBuilder
from src.hybrid.hybrid_agent import DDQNAgent
from src.astar.replanner import DynamicReplanner

def run_day8_end_to_end_test():
    # 1. Initialize all components
    env = WarehouseEnv(layout="Dynamic-Heavy") 
    planner = AStarPlanner(env.grid)
    replanner = DynamicReplanner(planner)
    waypoint_manager = WaypointManager()
    state_builder = HybridStateBuilder()
    
    # DDQN initialized but explicitly NOT optimized yet
    agent = DDQNAgent(state_dim=51, action_dim=5) 

    # 2. Reset environment and establish initial global route
    obs = env.reset()
    start_pos = env.robot_pos
    goal_pos = env.goal_pos
    
    global_path = planner.find_path(start_pos, goal_pos, env.get_static_obstacles())
    waypoint_manager.set_path(global_path)

    done = False
    step_count = 0

    print("--- Starting Day 8 Hybrid Execution Test ---")

    # 3. Execution Loop: A* -> DDQN -> obstacle movement -> replanning -> DDQN -> goal
    while not done:
        current_pos = env.robot_pos
        dynamic_obstacles = env.get_dynamic_obstacles() # Track moving workers

        # TEST CASE: Blocked waypoint & Replanning 
        if replanner.is_path_blocked(global_path, current_pos, dynamic_obstacles):
            print(f"[Step {step_count}] TEST TRIGGER: Worker blocking path! Replanning...")
            global_path = replanner.replan(current_pos, goal_pos, env.get_all_obstacles())
            waypoint_manager.set_path(global_path)

        # Get immediate local target
        next_waypoint = waypoint_manager.get_next_waypoint(current_pos)

        # Build 51-D state vector
        state = state_builder.build_state(obs, next_waypoint)

        # TEST CASE: DDQN local decision (untrained, utilizing high epsilon randomness)
        action = agent.select_action(state)

        # TEST CASE: Moving robot & moving worker (env.step processes physics for both)
        obs, reward, done, info = env.step(action)
        
        step_count += 1
        
        # Monitor for static obstacle and worker collisions
        if info.get('collision'):
            print(f"[Step {step_count}] TEST COLLISION: Robot hit a hazard.")
            break

    # TEST CASE: Goal reaching
    if current_pos == goal_pos:
        print("TEST SUCCESS: Robot reached the final goal coordinate!")
    else:
        print("TEST ENDED: Loop terminated before reaching goal.")

if __name__ == "__main__":
    run_day8_end_to_end_test()