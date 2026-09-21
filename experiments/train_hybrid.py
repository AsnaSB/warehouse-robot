import sys
import os

# Force the project root to the absolute front of Python's path list
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import numpy as np
from src.environment.warehouse_env import WarehouseEnv
from src.environment.state_augmentation import StateAugmenter
from src.astar.astar_planner import AStarPlanner
from src.astar.replanner import DynamicReplanner
from src.hybrid.waypoint_manager import WaypointManager
from src.ddqn.agent import DDQNAgent
from src.hybrid.hybrid_agent import HybridAgent
from src.common.metrics import MetricsTracker

def train_hybrid_agent():
    # 1. Hyperparameters & Initialization
    num_episodes = 2500
    max_steps = 200
    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.998
    save_interval = 250
    
    os.makedirs("experiments/checkpoints", exist_ok=True)
    
    # Initialize Environment and State Builder
    env = WarehouseEnv(config="open")
    state_builder = StateAugmenter(
        static_grid=env._static_grid, 
        local_radius=2, 
        max_dynamic_obstacles=4
    )
    
    # Initialize Hybrid Components
    planner = AStarPlanner(env._static_grid)
    replanner = DynamicReplanner(planner)
    waypoint_manager = WaypointManager()
    ddqn_agent = DDQNAgent(state_dim=51, action_dim=8)
    
    # The central brain wrapping everything
    hybrid_agent = HybridAgent(
        ddqn_agent=ddqn_agent,
        waypoint_manager=waypoint_manager,
        replanner=replanner,
        astar_planner=planner
    )
    
    tracker = MetricsTracker()

    print(f"--- Starting Hybrid DDQN Training for {num_episodes} Episodes ---")

    # 2. Main Training Loop
    for episode in range(1, num_episodes + 1):
        obs, info = env.reset()
        hybrid_agent.waypoint_manager.set_path([]) # Reset global path for new episode
        
        total_reward = 0
        done = False
        step_count = 0
        episode_loss = []
        
        while not done and step_count < max_steps:
            current_pos = env.robot_pos
            goal_pos = env.goal_pos
            
            # Extract dynamic obstacles as a set of (row, col) tuples for the A* planner
            dynamic_obstacles = env.workers + env.dynamic_robots
            blocked_cells = {
                obstacle.position 
                for obstacle in dynamic_obstacles 
                if getattr(obstacle, 'active', True)
            }
            
            # Sync route, replan if needed, and get the immediate A* waypoint
            active_waypoint = hybrid_agent.update_route_and_waypoint(
                current_pos=current_pos,
                goal_pos=goal_pos,
                blocked_cells=blocked_cells
            )
            
            # Sync Member 1's visualizer route
            env.set_astar_path(hybrid_agent.waypoint_manager.path)
            
            # Build the 51-D state incorporating the A* waypoint
            state = state_builder.build_state(
                robot_position=current_pos,
                goal_position=goal_pos,
                workers=env.workers,
                dynamic_robots=env.dynamic_robots,
                waypoint=active_waypoint,
                context=info.get("context", "open"),
                risk_level=info.get("risk_level", "low")
            )
            
            # DDQN Action Selection (Exploration vs Exploitation)
            action = hybrid_agent.get_action(state, epsilon=epsilon)
            
            # Environment Physics Step
            next_obs, reward, terminated, truncated, next_info = env.step(action)
            done = terminated or truncated
            
            # Calculate next state for the Replay Buffer
            next_pos = env.robot_pos
            next_blocked = {
                obs.position for obs in (env.workers + env.dynamic_robots) if getattr(obs, 'active', True)
            }
            
            # Get the expected next waypoint to build an accurate next_state target
            next_waypoint = hybrid_agent.update_route_and_waypoint(next_pos, goal_pos, next_blocked)
            
            next_state = state_builder.build_state(
                robot_position=next_pos,
                goal_position=goal_pos,
                workers=env.workers,
                dynamic_robots=env.dynamic_robots,
                waypoint=next_waypoint,
                context=next_info.get("context", "open"),
                risk_level=next_info.get("risk_level", "low")
            )
            
            # 3. Experience Replay & Backpropagation
            hybrid_agent.ddqn.store_transition(state, action, reward, next_state, done)
            loss = hybrid_agent.ddqn.train_step()
            
            if loss is not None:
                episode_loss.append(loss)
                
            total_reward += reward
            step_count += 1
            info = next_info
            
            tracker.record_step(
                reward=reward, 
                min_dist=info.get("nearest_dynamic_distance", 5.0), 
                q_value=0.0, 
                inference_time=0.0
            )

        # 4. Network Synchronization & Epsilon Decay
        hybrid_agent.ddqn.update_target_network()
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        
        avg_loss = np.mean(episode_loss) if episode_loss else 0.0
        success = (env.robot_pos == env.goal_pos)
        
        tracker.record_episode(
            success=success,
            collision=info.get("collided", False),
            steps=step_count,
            actual_length=step_count,
            astar_length=len(hybrid_agent.waypoint_manager.path) if hybrid_agent.waypoint_manager.path else 0,
            total_reward=total_reward,
            loss=avg_loss, 
            epsilon=epsilon
        )

        # 5. Checkpointing
        if episode % save_interval == 0:
            checkpoint_path = f"experiments/checkpoints/hybrid_ddqn_ep{episode}.pth"
            hybrid_agent.ddqn.save(checkpoint_path)
            print(f"Hybrid Ep {episode} | Reward: {total_reward:.2f} | Eps: {epsilon:.3f} | Reached Goal: {success}")

if __name__ == "__main__":
    train_hybrid_agent()