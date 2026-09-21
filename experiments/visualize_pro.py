import sys
import os
import pygame
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

# --- Advanced Pygame Configuration ---
CELL_SIZE = 45
FPS = 12

COLORS = {
    "concrete_light": (224, 224, 224),
    "concrete_dark": (200, 200, 200),
    "crate_base": (139, 94, 60),
    "crate_top": (160, 110, 75),
    "crate_shadow": (100, 65, 40),
    "agv_body": (45, 52, 54),
    "agv_tread": (0, 0, 0),
    "agv_led": (0, 229, 255),       # Cyan glow
    "worker_vest": (255, 112, 67),  # High-vis orange
    "worker_hat": (253, 216, 53),   # Hardhat yellow
    "goal_pad": (102, 187, 106),
    "path_neon": (128, 216, 255),
    "dashboard": (33, 33, 33),
    "text": (255, 255, 255)
}

def draw_warehouse_floor(screen, rows, cols):
    """Draws a checkered concrete warehouse floor."""
    for r in range(rows):
        for c in range(cols):
            color = COLORS["concrete_light"] if (r + c) % 2 == 0 else COLORS["concrete_dark"]
            rect = pygame.Rect(c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, color, rect)

def draw_crate(screen, x, y):
    """Draws a 3D-looking wooden shipping crate for the aisles."""
    rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
    # Shadow/Border
    pygame.draw.rect(screen, COLORS["crate_shadow"], rect)
    # Main box
    pygame.draw.rect(screen, COLORS["crate_base"], rect.inflate(-4, -4))
    # Top lid
    pygame.draw.rect(screen, COLORS["crate_top"], rect.inflate(-12, -12))
    # Crossbeam lines
    pygame.draw.line(screen, COLORS["crate_shadow"], (x+6, y+6), (x+CELL_SIZE-6, y+CELL_SIZE-6), 2)
    pygame.draw.line(screen, COLORS["crate_shadow"], (x+CELL_SIZE-6, y+6), (x+6, y+CELL_SIZE-6), 2)

def draw_agv_robot(screen, cx, cy):
    """Draws an industrial Automated Guided Vehicle (AGV)."""
    size = CELL_SIZE - 10
    # Left and Right Treads (Wheels)
    pygame.draw.rect(screen, COLORS["agv_tread"], (cx - size//2, cy - size//2, 6, size), border_radius=2)
    pygame.draw.rect(screen, COLORS["agv_tread"], (cx + size//2 - 6, cy - size//2, 6, size), border_radius=2)
    # Main Chassis
    pygame.draw.rect(screen, COLORS["agv_body"], (cx - size//2 + 4, cy - size//2 + 2, size - 8, size - 4), border_radius=4)
    # Center LED Status Indicator
    pygame.draw.circle(screen, COLORS["agv_led"], (cx, cy), 5)

def draw_worker(screen, cx, cy):
    """Draws a top-down view of a warehouse worker."""
    # Shoulders (High-vis vest)
    pygame.draw.circle(screen, COLORS["worker_vest"], (cx, cy), CELL_SIZE//2 - 6)
    # Head (Hardhat)
    pygame.draw.circle(screen, COLORS["worker_hat"], (cx, cy), CELL_SIZE//2 - 12)

def draw_goal(screen, cx, cy):
    """Draws a pulsing landing pad for the goal."""
    pygame.draw.circle(screen, COLORS["goal_pad"], (cx, cy), CELL_SIZE//2 - 4, 3)
    pygame.draw.circle(screen, COLORS["goal_pad"], (cx, cy), CELL_SIZE//2 - 12)

def run_pro_visualizer():
    model_path = "experiments/checkpoints/hybrid_ddqn_ep2500.pth"
    
    env = WarehouseEnv(config="open")
    state_builder = StateAugmenter(env._static_grid, local_radius=2, max_dynamic_obstacles=4)
    agent = DDQNAgent(state_dim=51, action_dim=8)
    
    if os.path.exists(model_path):
        agent.load(model_path)
    else:
        print(f"Error: {model_path} not found.")
        return

    hybrid_agent = HybridAgent(
        agent, 
        WaypointManager(), 
        DynamicReplanner(AStarPlanner(env._static_grid)), 
        AStarPlanner(env._static_grid)
    )

    pygame.init()
    pygame.display.set_caption("Pro AGV Warehouse Simulator")
    
    grid_rows, grid_cols = len(env._static_grid), len(env._static_grid[0])
    screen_width = grid_cols * CELL_SIZE
    screen_height = grid_rows * CELL_SIZE + 70 # Bottom dashboard
    screen = pygame.display.set_mode((screen_width, screen_height))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 18, bold=True)

    obs, info = env.reset()
    hybrid_agent.waypoint_manager.set_path([])
    
    done = False
    step_count = 0
    running = True

    while running and not done and step_count < 200:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
        current_pos = env.robot_pos
        goal_pos = env.goal_pos
        
        dynamic_obstacles = env.workers + env.dynamic_robots
        blocked_cells = {
            obs_obj.position for obs_obj in dynamic_obstacles if getattr(obs_obj, 'active', True)
        }
        
        active_waypoint = hybrid_agent.update_route_and_waypoint(current_pos, goal_pos, blocked_cells)
        env.set_astar_path(hybrid_agent.waypoint_manager.path)
        
        state = state_builder.build_state(
            robot_position=current_pos, goal_position=goal_pos,
            workers=env.workers, dynamic_robots=env.dynamic_robots,
            waypoint=active_waypoint, context=info.get("context", "open"), risk_level=info.get("risk_level", "low")
        )
        
        action = hybrid_agent.get_action(state, epsilon=0.0)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # --- 1. Draw Floor & Static Racks ---
        draw_warehouse_floor(screen, grid_rows, grid_cols)
        for r in range(grid_rows):
            for c in range(grid_cols):
                if env._static_grid[r][c] == 1:
                    draw_crate(screen, c * CELL_SIZE, r * CELL_SIZE)
        
        # --- 2. Draw Navigation Path ---
        path = hybrid_agent.waypoint_manager.path
        if path and len(path) > 1:
            points = [(c * CELL_SIZE + CELL_SIZE//2, r * CELL_SIZE + CELL_SIZE//2) for r, c in path]
            pygame.draw.lines(screen, COLORS["path_neon"], False, points, 4)

        # --- 3. Draw Goal, Obstacles, and Robot ---
        goal_center = (goal_pos[1] * CELL_SIZE + CELL_SIZE//2, goal_pos[0] * CELL_SIZE + CELL_SIZE//2)
        draw_goal(screen, goal_center[0], goal_center[1])

        for obs_cell in blocked_cells:
            obs_center = (obs_cell[1] * CELL_SIZE + CELL_SIZE//2, obs_cell[0] * CELL_SIZE + CELL_SIZE//2)
            draw_worker(screen, obs_center[0], obs_center[1])

        robot_center = (current_pos[1] * CELL_SIZE + CELL_SIZE//2, current_pos[0] * CELL_SIZE + CELL_SIZE//2)
        draw_agv_robot(screen, robot_center[0], robot_center[1])

        # --- 4. Draw Dashboard UI ---
        pygame.draw.rect(screen, COLORS["dashboard"], (0, screen_height - 70, screen_width, 70))
        status = "NAVIGATING ROUTE..." if not done else "GOAL REACHED!" if env.robot_pos == env.goal_pos else "COLLISION DETECTED!"
        color = COLORS["goal_pad"] if status == "GOAL REACHED!" else COLORS["worker_vest"] if "COLLISION" in status else COLORS["agv_led"]
        
        txt_step = font.render(f"SYSTEM STEP: {step_count:03d}", True, COLORS["text"])
        txt_status = font.render(f"STATUS: {status}", True, color)
        screen.blit(txt_step, (20, screen_height - 50))
        screen.blit(txt_status, (250, screen_height - 50))

        pygame.display.flip()
        clock.tick(FPS)
        step_count += 1

    pygame.time.delay(3000)
    pygame.quit()

if __name__ == "__main__":
    run_pro_visualizer()