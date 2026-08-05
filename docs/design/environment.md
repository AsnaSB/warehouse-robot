# Member 1 Design Document: Environment & Simulation

## 1. Overview
The Environment module (`src/environment/`) provides a custom Gymnasium-compliant 2D Warehouse grid environment (`WarehouseEnv`) designed for training reinforcement learning agents and executing A* pathfinding.

## 2. Core Components

### 2.1 Warehouse Environments & Layouts (`src/environment/configs.py`)
Provides grid generation presets for $12 \times 12$ warehouse maps:
- **Open**: Clear floor with minimal static shelves.
- **Aisle**: Parallel shelving units forming narrow 1-cell or 2-cell corridors.
- **Dense**: High-density obstacle layout with obstacles and tight squeezes.
- **Dynamic-Heavy**: Mixed layout with multiple active moving workers and dynamic robots.

### 2.2 Dynamic Obstacles (`src/environment/obstacles.py`)
- **Worker**: Moves along predefined or random patrol routes at 1 cell / 2 steps.
- **Dynamic Robot**: Moves along linear pathways, reversing direction at wall boundaries.
- Collision detection checks bounding box intersections and cell occupancy.

### 2.3 Reward Functions (`src/environment/rewards.py`)
Calculates base reward and context-adaptive rewards:
- Step Penalty ($R_{\text{step}} = -1$)
- Goal Reward ($R_{\text{goal}} = +100$)
- Collision Penalty ($R_{\text{collision}} = -20$)
- Progress Reward ($R_{\text{progress}} = +5 \times \Delta d$)
- Waypoint Progress ($R_{\text{waypoint}} = +2$)

### 2.4 Context Classifier (`src/environment/context.py`)
Classifies the robot's current local vicinity into one of three zone contexts:
- **Aisle**: Wall/shelf obstacles on opposite parallel sides of the robot.
- **Dense**: High local obstacle density ($> 35\%$ of local $5\times5$ window occupied).
- **Open**: Low local obstacle density ($< 15\%$ occupied) with clear sight lines.

### 2.5 Renderer & Visualizer (`src/common/visualizer.py`)
Renders the warehouse environment using Matplotlib. Supports:
- Real-time rendering frame by frame.
- Saving trajectory animations as `.gif` images.
