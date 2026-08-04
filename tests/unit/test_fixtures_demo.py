"""
Demonstrates how to use the stub fixtures from conftest.py.
Delete or repurpose this file once real unit tests replace it in Week 2.
"""


def test_small_grid_env_resets(small_grid_env):
    state = small_grid_env.reset()
    assert state["robot_pos"] == (0, 0)


def test_standard_grid_env_size(standard_grid_env):
    assert standard_grid_env.grid_size == 12


def test_astar_planner_returns_path(astar_planner):
    path = astar_planner.plan(start=(0, 0), goal=(11, 11))
    assert path[0] == (0, 0)
    assert path[-1] == (11, 11)


def test_sample_waypoints_shape(sample_waypoints):
    assert isinstance(sample_waypoints, list)
    assert all(isinstance(wp, tuple) and len(wp) == 2 for wp in sample_waypoints)