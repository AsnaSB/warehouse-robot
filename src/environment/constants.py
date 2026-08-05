"""
Shared constants for the warehouse environment.

Keeping these in one place means Member 2 (A*), Member 3 (DDQN), and
Member 4 (hybrid) all import the same values instead of hardcoding
magic numbers that can drift out of sync.
"""

# --- Cell types (stored in the grid as int8) ---
FREE = 0
SHELF = 1
DYNAMIC_OBSTACLE = 2
ROBOT = 3
GOAL = 4

CELL_NAMES = {
    FREE: "free",
    SHELF: "shelf",
    DYNAMIC_OBSTACLE: "dynamic_obstacle",
    ROBOT: "robot",
    GOAL: "goal",
}

# --- Actions: 8-directional movement ---
# (row_delta, col_delta) for each action index.
# Matches the DDQN's 8-action output layer (see docs/design/ddqn.md).
ACTIONS = {
    0: (-1, 0),   # N
    1: (-1, 1),   # NE
    2: (0, 1),    # E
    3: (1, 1),    # SE
    4: (1, 0),    # S
    5: (1, -1),   # SW
    6: (0, -1),   # W
    7: (-1, -1),  # NW
}
ACTION_NAMES = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
NUM_ACTIONS = len(ACTIONS)

# --- Context zones (used by context.py, Week 5) ---
CONTEXT_OPEN = "open"
CONTEXT_AISLE = "aisle"
CONTEXT_DENSE = "dense"
CONTEXTS = [CONTEXT_OPEN, CONTEXT_AISLE, CONTEXT_DENSE]

DEFAULT_GRID_SIZE = 12
