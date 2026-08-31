import unittest

from src.environment.obstacles import (
    DynamicObstacle,
    Worker,
    DynamicRobot,
)


class TestDynamicObstacle(unittest.TestCase):

    def test_initialization(self):
        obstacle = DynamicObstacle(
            position=(3, 4),
            movement_pattern="patrol",
            speed=1.0,
            active=True,
        )

        self.assertEqual(obstacle.position, (3, 4))
        self.assertEqual(
            obstacle.previous_position,
            (3, 4)
        )
        self.assertEqual(
            obstacle.movement_pattern,
            "patrol"
        )
        self.assertEqual(obstacle.speed, 1.0)
        self.assertTrue(obstacle.active)

    def test_position_update(self):
        obstacle = DynamicObstacle(
            position=(3, 4),
            movement_pattern="patrol",
        )

        obstacle.update_position((3, 5))

        self.assertEqual(
            obstacle.previous_position,
            (3, 4)
        )

        self.assertEqual(
            obstacle.position,
            (3, 5)
        )

    def test_activation(self):
        obstacle = DynamicObstacle(
            position=(3, 4),
            movement_pattern="patrol",
            active=False,
        )

        obstacle.activate()

        self.assertTrue(obstacle.active)

    def test_deactivation(self):
        obstacle = DynamicObstacle(
            position=(3, 4),
            movement_pattern="patrol",
            active=True,
        )

        obstacle.deactivate()

        self.assertFalse(obstacle.active)


class TestWorker(unittest.TestCase):

    def test_predefined_patrol(self):
        worker = Worker(
            position=(2, 2),
            movement_pattern="predefined_patrol",
            patrol_route=[
                (2, 2),
                (2, 3),
                (2, 4),
            ],
            speed=1.0,
            grid_size=(12, 12),
        )

        worker.update()

        self.assertEqual(
            worker.position,
            (2, 3)
        )

        worker.update()

        self.assertEqual(
            worker.position,
            (2, 4)
        )

    def test_predefined_patrol_reversal(self):
        worker = Worker(
            position=(2, 2),
            movement_pattern="predefined_patrol",
            patrol_route=[
                (2, 2),
                (2, 3),
                (2, 4),
            ],
            grid_size=(12, 12),
        )

        worker.update()
        worker.update()
        worker.update()

        self.assertEqual(
            worker.position,
            (2, 3)
        )

    def test_random_patrol_stays_inside_grid(self):
        worker = Worker(
            position=(5, 5),
            movement_pattern="random_patrol",
            random_patrol=True,
            grid_size=(12, 12),
        )

        for _ in range(20):
            worker.update()

            row, col = worker.position

            self.assertTrue(
                0 <= row < 12
            )

            self.assertTrue(
                0 <= col < 12
            )

    def test_worker_pause(self):
        worker = Worker(
            position=(3, 3),
            movement_pattern="predefined_patrol",
            patrol_route=[
                (3, 3),
                (3, 4),
                (3, 5),
            ],
            pause_steps=2,
            grid_size=(12, 12),
        )

        worker.update()

        self.assertEqual(
            worker.position,
            (3, 4)
        )

        worker.update()

        self.assertEqual(
            worker.position,
            (3, 4)
        )

        worker.update()

        self.assertEqual(
            worker.position,
            (3, 4)
        )


class TestDynamicRobot(unittest.TestCase):

    def test_linear_movement(self):
        robot = DynamicRobot(
            position=(5, 2),
            movement_pattern="linear",
            movement_direction=(0, 1),
            grid_size=(12, 12),
        )

        robot.update()

        self.assertEqual(
            robot.position,
            (5, 3)
        )

        self.assertEqual(
            robot.previous_position,
            (5, 2)
        )

    def test_boundary_reversal(self):
        robot = DynamicRobot(
            position=(5, 11),
            movement_pattern="linear",
            movement_direction=(0, 1),
            grid_size=(12, 12),
        )

        robot.update()

        self.assertEqual(
            robot.position,
            (5, 10)
        )

        self.assertEqual(
            robot.movement_direction,
            (0, -1)
        )

    def test_path_reversal(self):
        robot = DynamicRobot(
            position=(6, 2),
            movement_pattern="linear",
            movement_path=[
                (6, 2),
                (6, 3),
                (6, 4),
            ],
            allow_reverse=True,
            grid_size=(12, 12),
        )

        robot.update()
        self.assertEqual(
            robot.position,
            (6, 3)
        )

        robot.update()
        self.assertEqual(
            robot.position,
            (6, 4)
        )

        robot.update()
        self.assertEqual(
            robot.position,
            (6, 3)
        )

    def test_dynamic_robot_stays_inside_grid(self):
        robot = DynamicRobot(
            position=(6, 10),
            movement_pattern="linear",
            movement_direction=(0, 1),
            grid_size=(12, 12),
        )

        for _ in range(20):
            robot.update()

            row, col = robot.position

            self.assertTrue(
                0 <= row < 12
            )

            self.assertTrue(
                0 <= col < 12
            )

    def test_collision_detection(self):
        robot = DynamicRobot(
            position=(5, 5),
            movement_pattern="linear",
            movement_direction=(0, 1),
            grid_size=(12, 12),
        )

        occupied_positions = [
            (5, 6)
        ]

        robot.update(occupied_positions)

        self.assertEqual(
            robot.position,
            (5, 5)
        )

        self.assertTrue(
            robot.last_collision
        )

    def test_no_collision_allows_movement(self):
        robot = DynamicRobot(
            position=(5, 5),
            movement_pattern="linear",
            movement_direction=(0, 1),
            grid_size=(12, 12),
        )

        occupied_positions = [
            (2, 2),
            (8, 8)
        ]

        robot.update(occupied_positions)

        self.assertEqual(
            robot.position,
            (5, 6)
        )

        self.assertFalse(
            robot.last_collision
        )

    def test_collision_does_not_move_robot(self):
        robot = DynamicRobot(
            position=(5, 5),
            movement_pattern="linear",
            movement_direction=(0, 1),
            grid_size=(12, 12),
        )

        occupied_positions = [
            (5, 6)
        ]

        robot.update(occupied_positions)

        self.assertEqual(
            robot.previous_position,
            (5, 5)
        )

        self.assertEqual(
            robot.position,
            (5, 5)
        )

    def test_path_collision(self):
        robot = DynamicRobot(
            position=(6, 2),
            movement_pattern="linear",
            movement_path=[
                (6, 2),
                (6, 3),
                (6, 4),
            ],
            allow_reverse=True,
            grid_size=(12, 12),
        )

        occupied_positions = [
            (6, 3)
        ]

        robot.update(occupied_positions)

        self.assertEqual(
            robot.position,
            (6, 2)
        )

        self.assertTrue(
            robot.last_collision
        )
if __name__ == "__main__":
    unittest.main()