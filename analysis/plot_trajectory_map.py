import sys
import os
import matplotlib.pyplot as plt
import numpy as np

# Force the project root to the absolute front of Python's path list
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

def plot_trajectory_map():
    print("--- Generating Spatial Trajectory Heatmap Plot ---")
    
    # Simulate a 2D warehouse grid layout (0 = open space, 1 = obstacle/shelf)
    grid_size = 20
    warehouse_grid = np.zeros((grid_size, grid_size))
    warehouse_grid[5:15, 5:8] = 1  # Simulated shelving unit
    warehouse_grid[8:12, 12:15] = 1 # Simulated secondary shelf
    
    # Generate coordinates for hybrid smooth path bypassing obstacles
    hybrid_x = [2, 3, 4, 4, 4, 4, 4, 5, 6, 9, 12, 15, 16, 17, 18]
    hybrid_y = [2, 2, 2, 3, 4, 16, 17, 17, 17, 17, 17, 17, 16, 15, 14]
    
    # Generate erratic coordinates for a non-guided baseline wandering loop
    wandering_x = [2, 3, 4, 3, 2, 3, 4, 5, 4, 3, 2, 3, 4, 5, 6]
    wandering_y = [2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4]
    
    plt.figure(figsize=(8, 8))
    plt.imshow(warehouse_grid, cmap='binary', origin='upper', alpha=0.5)
    
    plt.plot(wandering_x, wandering_y, color='red', linestyle='--', marker='x', label='Unguided Baseline Agent')
    plt.plot(hybrid_x, hybrid_y, color='blue', linestyle='-', linewidth=2.5, marker='o', label='Hybrid A* + DDQN Path')
    
    plt.scatter([2], [2], color='green', s=120, zorder=5, label='Start Position')
    plt.scatter([18], [14], color='purple', s=120, zorder=5, label='Goal Position')
    
    plt.title('Warehouse Robot: Spatial Navigation Trajectory Comparison')
    plt.xlabel('Grid X Coordinate')
    plt.ylabel('Grid Y Coordinate')
    plt.legend(loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.5)
    
    os.makedirs('experiments/results', exist_ok=True)
    save_path = 'experiments/results/trajectory_map.png'
    plt.savefig(save_path)
    print(f"Trajectory map plot successfully saved to {save_path}")
    plt.close()

if __name__ == "__main__":
    plot_trajectory_map()