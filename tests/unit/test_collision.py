import unittest

import numpy as np

from src.environment.collision import CollisionModel
from src.environment.obstacles import (
    DynamicRobot,
    Worker,
)


class TestCollisionModel(unittest.TestCase):

    def setUp(self):
        self.grid = np.zeros(
            (12, 12),
            dtype=np.int8,
        )

    def test_boundary_collision(self):
        collision = CollisionModel(
            grid_size=12,
            static_grid=self.grid,
        )

        self.assertFalse(
            collision.can_move(
                (0, 0),
                (-1, 0),
            )
        )

        self.assertFalse(
            collision.can_move(
                (11, 11),
                (11, 12),
            )
        )

    def test_free_movement(self):
        collision = CollisionModel(
            grid_size=12,
            static_grid=self.grid,
        )

        self.assertTrue(
            collision.can_move(
                (5, 5),
                (5, 6),
            )
        )

    def test_shelf_collision(self):
        self.grid[5, 6] = 1

        collision = CollisionModel(
            grid_size=12,
            static_grid=self.grid,
        )

        self.assertFalse(
            collision.can_move(
                (5, 5),
                (5, 6),
            )
        )

    def test_worker_collision(self):
        worker = Worker(
            position=(5, 6),
            movement_pattern="predefined_patrol",
            patrol_route=[(5, 6)],
            active=True,
        )

        collision = CollisionModel(
            grid_size=12,
            static_grid=self.grid,
            workers=[worker],
        )

        self.assertFalse(
            collision.can_move(
                (5, 5),
                (5, 6),
            )
        )

    def test_inactive_worker_does_not_block(self):
        worker = Worker(
            position=(5, 6),
            movement_pattern="predefined_patrol",
            patrol_route=[(5, 6)],
            active=False,
        )

        collision = CollisionModel(
            grid_size=12,
            static_grid=self.grid,
            workers=[worker],
        )

        self.assertTrue(
            collision.can_move(
                (5, 5),
                (5, 6),
            )
        )

    def test_dynamic_robot_collision(self):
        robot = DynamicRobot(
            position=(5, 6),
            movement_pattern="linear",
            movement_direction=(0, 1),
            active=True,
        )

        collision = CollisionModel(
            grid_size=12,
            static_grid=self.grid,
            dynamic_robots=[robot],
        )

        self.assertFalse(
            collision.can_move(
                (5, 5),
                (5, 6),
            )
        )

    def test_inactive_dynamic_robot_does_not_block(self):
        robot = DynamicRobot(
            position=(5, 6),
            movement_pattern="linear",
            movement_direction=(0, 1),
            active=False,
        )

        collision = CollisionModel(
            grid_size=12,
            static_grid=self.grid,
            dynamic_robots=[robot],
        )

        self.assertTrue(
            collision.can_move(
                (5, 5),
                (5, 6),
            )
        )

    def test_safe_diagonal_movement(self):
        collision = CollisionModel(
            grid_size=12,
            static_grid=self.grid,
        )

        self.assertTrue(
            collision.can_move_diagonal(
                (5, 5),
                (6, 6),
            )
        )

    def test_diagonal_corner_collision(self):
        self.grid[5, 6] = 1
        self.grid[6, 5] = 1

        collision = CollisionModel(
            grid_size=12,
            static_grid=self.grid,
        )

        self.assertFalse(
            collision.can_move_diagonal(
                (5, 5),
                (6, 6),
            )
        )


if __name__ == "__main__":
    unittest.main()