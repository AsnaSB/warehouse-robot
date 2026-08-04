"""
Shared pytest fixtures for the warehouse robot navigation project.

NOTE: Several fixtures below use lightweight STUBS in place of real classes
(WarehouseEnv, AStarPlanner, etc.) that Members 1-4 haven't built yet.
Once their real modules land, replace the stub import with the real one --
the fixture names and return shapes are designed to stay the same so
tests written against the stub keep working.

Assumptions here match docs/design/metrics.md. If Member 1/2/3/4's actual
API differs, update the stub classes below (not the individual tests).
"""

import pytest


# ---------------------------------------------------------------------------
# STUB CLASSES -- delete / replace once real modules exist in src/
# ---------------------------------------------------------------------------

class StubWarehouseEnv:
    """
    Placeholder for src.environment.WarehouseEnv.
    Assumed interface (confirm with Member 1):
        env.reset() -> state
        env.step(action) -> (next_state, reward, done, info)
        env.robot_pos, env.goal_pos -> (row, col) tuples
    """
    def __init__(self, grid_size=12, obstacles=None):
        self.grid_size = grid_size
        self.obstacles = obstacles or []
        self.robot_pos = (0, 0)
        self.goal_pos = (grid_size - 1, grid_size - 1)

    def reset(self):
        self.robot_pos = (0, 0)
        return self._get_state()

    def step(self, action):
        # dummy transition -- replace with real logic later
        next_state = self._get_state()
        reward = -1
        done = False
        info = {"collision": False}
        return next_state, reward, done, info

    def _get_state(self):
        return {"robot_pos": self.robot_pos, "goal_pos": self.goal_pos}


class StubAStarPlanner:
    """
    Placeholder for src.astar.AStarPlanner.
    Assumed interface (confirm with Member 2):
        planner.plan(start, goal, grid) -> list of waypoints [(r,c), ...]
    """
    def plan(self, start, goal, grid=None):
        # dummy straight-line path -- replace with real logic later
        return [start, goal]


# ---------------------------------------------------------------------------
# FIXTURES
# ---------------------------------------------------------------------------

@pytest.fixture
def small_grid_env():
    """5x5 grid, no obstacles -- fast sanity checks."""
    return StubWarehouseEnv(grid_size=5, obstacles=[])


@pytest.fixture
def standard_grid_env():
    """12x12 grid, default config -- matches project spec."""
    return StubWarehouseEnv(grid_size=12, obstacles=[])


@pytest.fixture
def narrow_aisle_env():
    """Aisle-context scenario -- confirm obstacle layout with Member 1."""
    obstacles = [(r, 5) for r in range(0, 4)] + [(r, 5) for r in range(6, 12)]
    return StubWarehouseEnv(grid_size=12, obstacles=obstacles)


@pytest.fixture
def dense_obstacle_env():
    """Dense-context scenario -- confirm obstacle density with Member 1."""
    obstacles = [(r, c) for r in range(3, 9) for c in range(3, 9) if (r + c) % 2 == 0]
    return StubWarehouseEnv(grid_size=12, obstacles=obstacles)


@pytest.fixture
def astar_planner():
    """Stub A* planner -- swap for real AStarPlanner once Member 2 pushes code."""
    return StubAStarPlanner()


@pytest.fixture
def sample_waypoints():
    """Canned waypoint list, for testing hybrid/DDQN code independent of A*."""
    return [(0, 0), (2, 1), (4, 3), (6, 6), (9, 9), (11, 11)]