import time
from src.environment.warehouse_env import WarehouseEnv
from src.astar.astar_planner import AStarPlanner
from src.hybrid.waypoint_manager import WaypointManager
from src.environment.state_augmentation import StateAugmenter  # Updated import
from src.hybrid.hybrid_agent import DDQNAgent
from src.astar.replanner import DynamicReplanner
from src.common.metrics import MetricsTracker 

def run_day8_end_to_end_test():
    # 1. Initialize components
    env = WarehouseEnv(config="dynamic-heavy") 
    
    planner = AStarPlanner(env._static_grid)
    replanner = DynamicReplanner(planner)
    waypoint_manager = WaypointManager()
    
    # Updated: Using Member 1's exact StateAugmenter signature
    state_builder = StateAugmenter(
        static_grid=env._static_grid,
        local_radius=2,
        max_dynamic_obstacles=4
    )
    
    agent = DDQNAgent(state_dim=51, action_dim=8) 
    tracker = MetricsTracker()

    # 2. Reset environment and establish initial global route
    obs, info = env.reset()
    start_pos = env.robot_pos
    goal_pos = env.goal_pos
    
    global_path = planner.find_path(start_pos, goal_pos, env._static_grid)
    waypoint_manager.set_path(global_path)
    env.set_astar_path(global_path)

    done = False
    step_count = 0
    total_reward = 0

    print("--- Starting Hybrid Execution with Metrics Tracking ---")

    # 3. Execution Loop
    while not done:
        current_pos = env.robot_pos
        dynamic_obstacles = env.workers + env.dynamic_robots 

        # Replanning Block
        if replanner.is_path_blocked(global_path, current_pos, dynamic_obstacles):
            print(f"[Step {step_count}] Worker blocking path! Replanning...")
            replan_start = time.time()
            
            global_path = replanner.replan(current_pos, goal_pos, env._static_grid)
            waypoint_manager.set_path(global_path)
            env.set_astar_path(global_path)
            
            tracker.record_replan(time.time() - replan_start)

        next_waypoint = waypoint_manager.get_next_waypoint(current_pos)
        
        # Updated: Build 51-D state using Member 1's specific signature
        state = state_builder.build_state(
            robot_position=current_pos,
            goal_position=goal_pos,
            workers=env.workers,
            dynamic_robots=env.dynamic_robots,
            waypoint=next_waypoint,
            context=info.get("context", "open"),
            risk_level=info.get("risk_level", "low")
        )
        
        # Action Selection
        action_start = time.time()
        action = agent.select_action(state, epsilon=1.0) 
        inference_time = time.time() - action_start

        # Environment Step
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        step_count += 1
        total_reward += reward
        
        tracker.record_step(
            reward=reward, 
            min_dist=info.get("nearest_dynamic_distance", 5.0), 
            q_value=0.0, 
            inference_time=inference_time
        )
        
        if info.get("collided"):
            print(f"[Step {step_count}] COLLISION: Robot hit a hazard.")
            break

    # 4. Final Episode Wrap-up
    success = (current_pos == goal_pos)
    if success:
        print("SUCCESS: Robot reached the final goal coordinate!")

    tracker.record_episode(
        success=success,
        collision=info.get("collided", False),
        steps=step_count,
        actual_length=step_count,
        astar_length=len(global_path) if global_path else 0,
        total_reward=total_reward,
        loss=0.0, 
        epsilon=1.0
    )

    print("\n--- Day 9 Metrics Summary ---")
    summary = tracker.get_summary()
    for key, value in summary.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    run_day8_end_to_end_test()