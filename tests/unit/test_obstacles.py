import unittest

from src.environment.obstacles import DynamicObstacle


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


if __name__ == "__main__":
    unittest.main()