from src.environment.obstacles import (
    DynamicRobot,
    Worker,
)
from src.environment.warehouse_env import WarehouseEnv


def test_robot_worker_collision():

    env = WarehouseEnv(
        config="open",
        grid_size=12,
        seed=42,
    )

    env.reset(seed=42)

    worker = Worker(
        position=(5, 6),
        movement_pattern="predefined_patrol",
        patrol_route=[(5, 6)],
        active=True,
    )

    env.robot_pos = (5, 5)

    env.set_dynamic_obstacles(
        workers=[worker]
    )

    _, _, _, _, info = env.step(2)

    assert env.robot_pos == (5, 5)
    assert info["collided"] is True

    env.close()


def test_robot_dynamic_robot_collision():

    env = WarehouseEnv(
        config="open",
        grid_size=12,
        seed=42,
    )

    env.reset(seed=42)

    dynamic_robot = DynamicRobot(
        position=(5, 6),
        movement_pattern="linear",
        movement_direction=(0, 1),
        active=True,
    )

    env.robot_pos = (5, 5)

    env.set_dynamic_obstacles(
        dynamic_robots=[dynamic_robot]
    )

    _, _, _, _, info = env.step(2)

    assert env.robot_pos == (5, 5)
    assert info["collided"] is True

    env.close()


def test_robot_boundary_collision():

    env = WarehouseEnv(
        config="open",
        grid_size=12,
        seed=42,
    )

    env.reset(seed=42)

    env.robot_pos = (0, 0)

    # Action 0 = North.
    _, _, _, _, info = env.step(0)

    assert env.robot_pos == (0, 0)
    assert info["collided"] is True

    env.close()


def test_robot_shelf_collision():

    env = WarehouseEnv(
        config="open",
        grid_size=12,
        seed=42,
    )

    env.reset(seed=42)

    # Place a shelf directly east of the robot.
    env._static_grid[5, 6] = 1

    env.robot_pos = (5, 5)

    _, _, _, _, info = env.step(2)

    assert env.robot_pos == (5, 5)
    assert info["collided"] is True

    env.close()


def test_safe_diagonal_movement():

    env = WarehouseEnv(
        config="open",
        grid_size=12,
        seed=42,
    )

    env.reset(seed=42)

    # (0,2) -> (1,3) is a verified free diagonal
    # in the actual Open layout.
    env.robot_pos = (0, 2)

    # Action 3 = South-East (1, 1).
    _, _, _, _, info = env.step(3)

    assert env.robot_pos == (1, 3)
    assert info["collided"] is False

    env.close()


def test_diagonal_corner_cutting_is_blocked():

    env = WarehouseEnv(
        config="open",
        grid_size=12,
        seed=42,
    )

    env.reset(seed=42)

    # Controlled 2x2 area:
    #
    # FREE   SHELF
    # SHELF  FREE
    #
    # The destination is free, but both orthogonal
    # intermediate cells are blocked.

    env._static_grid[5, 5] = 0
    env._static_grid[5, 6] = 1
    env._static_grid[6, 5] = 1
    env._static_grid[6, 6] = 0

    env.robot_pos = (5, 5)

    # Action 3 = South-East.
    _, _, _, _, info = env.step(3)

    # Diagonal corner cutting must be blocked.
    assert env.robot_pos == (5, 5)
    assert info["collided"] is True

    env.close()