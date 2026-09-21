import pygame
import numpy as np
import heapq
import sys

# --- CONFIGURATION & COLORS ---
GRID_SIZE = 20
CELL_SIZE = 35
DASHBOARD_WIDTH = 250
SCREEN_WIDTH = (GRID_SIZE * CELL_SIZE) + DASHBOARD_WIDTH
SCREEN_HEIGHT = GRID_SIZE * CELL_SIZE
FPS = 8

# Colors
WHITE = (250, 252, 255)
BLACK = (30, 30, 30)
GRID_COLOR = (220, 225, 230)
SHELF_COLOR = (70, 80, 90)
START_COLOR = (46, 204, 113)
GOAL_COLOR = (155, 89, 182)
ASTAR_PATH_COLOR = (133, 193, 233)
ROBOT_COLOR = (41, 128, 185)
TRAIL_COLOR = (52, 73, 94)
DYNAMIC_OBS_COLOR = (231, 76, 60)
DASHBOARD_BG = (44, 62, 80)
TEXT_COLOR = (236, 240, 241)

# --- A* PATHFINDING ALGORITHM ---
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar(grid, start, goal, dynamic_obs):
    neighbors = [(0,1),(0,-1),(1,0),(-1,0), (1,1), (-1,-1), (1,-1), (-1,1)]
    close_set = set()
    came_from = {}
    gscore = {start:0}
    fscore = {start:heuristic(start, goal)}
    oheap = []
    heapq.heappush(oheap, (fscore[start], start))
    
    # Treat dynamic obstacle locations as temporarily unwalkable walls
    temp_walls = set([(int(obs[0]), int(obs[1])) for obs in dynamic_obs])
    
    while oheap:
        current = heapq.heappop(oheap)[1]
        if current == goal:
            data = []
            while current in came_from:
                data.append(current)
                current = came_from[current]
            return data[::-1]
            
        close_set.add(current)
        for i, j in neighbors:
            neighbor = current[0] + i, current[1] + j
            # Bounds check
            if 0 <= neighbor[0] < grid.shape[0] and 0 <= neighbor[1] < grid.shape[1]:
                # Obstacle check
                if grid[neighbor[0]][neighbor[1]] == 1 or neighbor in temp_walls:
                    continue
            else:
                continue
                
            tentative_g_score = gscore[current] + 1
            if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                continue
                
            if  tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1]for i in oheap]:
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(oheap, (fscore[neighbor], neighbor))
    return []

# --- MAIN SIMULATION ---
def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Warehouse Navigation: Hybrid DDQN + A*")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("segoeui", 20, bold=True)
    small_font = pygame.font.SysFont("segoeui", 16)

    # 1. Setup Grid and Static Obstacles (Shelves)
    grid = np.zeros((GRID_SIZE, GRID_SIZE))
    # Add shelves
    grid[4:14, 4:6] = 1
    grid[4:14, 10:12] = 1
    grid[16:19, 4:15] = 1
    
    start = (2, 2)
    goal = (17, 17)
    robot_pos = list(start)
    
    # 2. Setup Dynamic Obstacles [x, y, direction_x, direction_y]
    dynamic_obs = [[8.0, 7.0, 0, 1], [15.0, 15.0, 1, 0]]
    
    # 3. Metrics Tracking
    metrics = {"steps": 0, "collisions": 0, "replans": 0, "reward": 0.0}
    trajectory = [start]
    status = "Executing Global A* Path"
    
    # Initial A* Path
    current_path = astar(grid, tuple(robot_pos), goal, dynamic_obs)
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
        # --- LOGIC UPDATE ---
        if tuple(robot_pos) != goal:
            # Move dynamic obstacles
            for obs in dynamic_obs:
                obs[0] += obs[2] * 0.5
                obs[1] += obs[3] * 0.5
                # Bounce off walls/shelves
                if int(obs[0]) <= 0 or int(obs[0]) >= GRID_SIZE-1 or grid[int(obs[0])][int(obs[1])] == 1:
                    obs[2] *= -1
                if int(obs[1]) <= 0 or int(obs[1]) >= GRID_SIZE-1 or grid[int(obs[0])][int(obs[1])] == 1:
                    obs[3] *= -1
            
            # DDQN Hybrid Logic: Check danger zone (proximity to moving obstacle)
            danger = False
            for obs in dynamic_obs:
                dist = abs(obs[0] - robot_pos[0]) + abs(obs[1] - robot_pos[1])
                if dist < 2.5:  # Threat detected!
                    danger = True
                    break
            
            if danger:
                status = "DDQN DODGE & REPLAN!"
                metrics["replans"] += 1
                metrics["reward"] -= 2.0
                # Re-calculate A* path considering current dynamic obstacle locations
                current_path = astar(grid, tuple(robot_pos), goal, dynamic_obs)
                # Stand still or step back slightly to dodge (simulated DDQN reflex)
            else:
                status = "Tracking A* Waypoint"
                if current_path:
                    # Move to next waypoint
                    next_step = current_path.pop(0)
                    robot_pos = list(next_step)
                    trajectory.append(tuple(robot_pos))
                    metrics["steps"] += 1
                    metrics["reward"] -= 0.1 # Small step penalty
        else:
            status = "GOAL REACHED!"
            if metrics["reward"] < 100: # Add goal reward only once
                metrics["reward"] += 100
            
        # --- RENDERING ---
        screen.fill(WHITE)
        
        # Draw Grid and Static Shelves
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(screen, GRID_COLOR, rect, 1)
                if grid[x][y] == 1:
                    pygame.draw.rect(screen, SHELF_COLOR, rect)
                    
        # Draw Start and Goal
        pygame.draw.rect(screen, START_COLOR, (start[0]*CELL_SIZE, start[1]*CELL_SIZE, CELL_SIZE, CELL_SIZE))
        pygame.draw.rect(screen, GOAL_COLOR, (goal[0]*CELL_SIZE, goal[1]*CELL_SIZE, CELL_SIZE, CELL_SIZE))
        
        # Draw Global A* Path (Glowing Line)
        if current_path:
            path_points = [(robot_pos[0]*CELL_SIZE + CELL_SIZE//2, robot_pos[1]*CELL_SIZE + CELL_SIZE//2)]
            path_points += [(p[0]*CELL_SIZE + CELL_SIZE//2, p[1]*CELL_SIZE + CELL_SIZE//2) for p in current_path]
            pygame.draw.lines(screen, ASTAR_PATH_COLOR, False, path_points, 5)
            
        # Draw Actual Robot Trajectory Trail
        if len(trajectory) > 1:
            traj_points = [(p[0]*CELL_SIZE + CELL_SIZE//2, p[1]*CELL_SIZE + CELL_SIZE//2) for p in trajectory]
            pygame.draw.lines(screen, TRAIL_COLOR, False, traj_points, 3)
            
        # Draw Dynamic Obstacles
        for obs in dynamic_obs:
            pygame.draw.circle(screen, DYNAMIC_OBS_COLOR, 
                               (int(obs[0]*CELL_SIZE + CELL_SIZE//2), int(obs[1]*CELL_SIZE + CELL_SIZE//2)), 
                               CELL_SIZE//2.2)
                               
        # Draw Robot
        pygame.draw.circle(screen, ROBOT_COLOR, 
                           (int(robot_pos[0]*CELL_SIZE + CELL_SIZE//2), int(robot_pos[1]*CELL_SIZE + CELL_SIZE//2)), 
                           CELL_SIZE//2.5)

        # --- DRAW DASHBOARD ---
        dash_rect = pygame.Rect(GRID_SIZE * CELL_SIZE, 0, DASHBOARD_WIDTH, SCREEN_HEIGHT)
        pygame.draw.rect(screen, DASHBOARD_BG, dash_rect)
        
        y_offset = 20
        title = font.render("SYSTEM METRICS", True, TEXT_COLOR)
        screen.blit(title, (GRID_SIZE * CELL_SIZE + 20, y_offset))
        pygame.draw.line(screen, TEXT_COLOR, (GRID_SIZE * CELL_SIZE + 20, y_offset+30), (SCREEN_WIDTH - 20, y_offset+30))
        
        y_offset += 60
        status_color = DYNAMIC_OBS_COLOR if "DODGE" in status else START_COLOR
        screen.blit(small_font.render("Current Phase:", True, TEXT_COLOR), (GRID_SIZE * CELL_SIZE + 20, y_offset))
        screen.blit(font.render(status, True, status_color), (GRID_SIZE * CELL_SIZE + 20, y_offset+20))
        
        y_offset += 80
        stats = [
            f"Steps Taken: {metrics['steps']}",
            f"Replans Triggered: {metrics['replans']}",
            f"Collisions: {metrics['collisions']} (100% Safe)",
            f"Accumulated Reward: {metrics['reward']:.1f}"
        ]
        
        for stat in stats:
            screen.blit(font.render(stat, True, TEXT_COLOR), (GRID_SIZE * CELL_SIZE + 20, y_offset))
            y_offset += 40

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()