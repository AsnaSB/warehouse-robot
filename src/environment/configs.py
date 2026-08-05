"""
Warehouse layout presets.

Each function returns a (grid_size, grid_size) numpy array of shelf
placements only (FREE / SHELF) — robot and goal positions are placed
separately at reset() time so every episode starts from a randomized
position. Full scenario variety (Dynamic-Heavy, etc.) gets fleshed out
in Week 6; these are the base three from the design doc.
"""

import numpy as np

from src.environment.constants import FREE, SHELF, DEFAULT_GRID_SIZE


def open_layout(grid_size: int = DEFAULT_GRID_SIZE) -> np.ndarray:
    """Mostly clear floor with a few scattered shelves."""
    grid = np.full((grid_size, grid_size), FREE, dtype=np.int8)
    rng = np.random.default_rng(seed=1)
    num_shelves = max(1, (grid_size * grid_size) // 20)  # ~5% coverage
    for _ in range(num_shelves):
        r, c = rng.integers(0, grid_size, size=2)
        grid[r, c] = SHELF
    return grid


def aisle_layout(grid_size: int = DEFAULT_GRID_SIZE) -> np.ndarray:
    """Parallel shelf rows forming 1-cell-wide corridors."""
    grid = np.full((grid_size, grid_size), FREE, dtype=np.int8)
    # Shelf rows every 3rd row, leaving a 2-row-wide aisle between them.
    # Leave the first and last columns clear so aisles connect end to end.
    for r in range(1, grid_size - 1, 3):
        grid[r, 1:grid_size - 1] = SHELF
    return grid


def dense_layout(grid_size: int = DEFAULT_GRID_SIZE) -> np.ndarray:
    """Tightly packed shelving with narrow gaps."""
    grid = np.full((grid_size, grid_size), FREE, dtype=np.int8)
    # Shelf blocks every other row and column, leaving single-cell gaps.
    for r in range(1, grid_size - 1, 2):
        for c in range(1, grid_size - 1, 2):
            grid[r, c] = SHELF
            if c + 1 < grid_size - 1:
                grid[r, c + 1] = SHELF
    return grid


LAYOUTS = {
    "open": open_layout,
    "aisle": aisle_layout,
    "dense": dense_layout,
}


def get_layout(name: str, grid_size: int = DEFAULT_GRID_SIZE) -> np.ndarray:
    if name not in LAYOUTS:
        raise ValueError(
            f"Unknown layout '{name}'. Choose from: {list(LAYOUTS.keys())}"
        )
    return LAYOUTS[name](grid_size)
