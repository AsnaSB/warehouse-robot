"""
Day 2 dynamic obstacle verification.

Verifies:
- Worker predefined patrol
- Worker random patrol
- Worker boundary handling
- Worker pause behaviour
- Dynamic robot linear movement
- Dynamic robot path reversal
- Dynamic robot boundary handling
"""

import random

from src.environment.obstacles import Worker, DynamicRobot


GRID_SIZE = (12, 12)


def verify_worker_predefined_patrol():
    print("\n=== Worker: Predefined Patrol ===")

    worker = Worker(
        position=(2, 2),
        movement_pattern="predefined_patrol",
        patrol_route=[
            (2, 2),
            (2, 3),
            (2, 4),
            (2, 5),
            (2, 6),
        ],
        speed=1.0,
        grid_size=GRID_SIZE,
    )

    positions = [worker.position]

    for step in range(10):
        previous = worker.position
        current = worker.update()

        positions.append(current)

        print(
            f"Step {step + 1:02d}: "
            f"{previous} -> {current}"
        )

    assert len(set(positions)) > 1, (
        "Worker did not move."
    )

    assert all(
        0 <= row < GRID_SIZE[0]
        and 0 <= col < GRID_SIZE[1]
        for row, col in positions
    ), "Worker left the grid."

    print("PASS: predefined patrol movement works.")


def verify_worker_random_patrol():
    print("\n=== Worker: Random Patrol ===")

    random.seed(42)

    worker = Worker(
        position=(5, 5),
        movement_pattern="random_patrol",
        random_patrol=True,
        speed=1.0,
        grid_size=GRID_SIZE,
    )

    positions = [worker.position]

    for step in range(15):
        previous = worker.position
        current = worker.update()

        positions.append(current)

        print(
            f"Step {step + 1:02d}: "
            f"{previous} -> {current}"
        )

    assert len(set(positions)) > 1, (
        "Random worker did not move."
    )

    assert all(
        0 <= row < GRID_SIZE[0]
        and 0 <= col < GRID_SIZE[1]
        for row, col in positions
    ), "Random worker left the grid."

    print("PASS: random patrol movement works.")


def verify_worker_pause():
    print("\n=== Worker: Pause Behaviour ===")

    worker = Worker(
        position=(3, 3),
        movement_pattern="predefined_patrol",
        patrol_route=[
            (3, 3),
            (3, 4),
            (3, 5),
        ],
        pause_steps=2,
        grid_size=GRID_SIZE,
    )

    first_position = worker.position

    worker.update()

    moved_position = worker.position

    assert moved_position != first_position, (
        "Worker did not make the initial movement."
    )

    worker.update()
    paused_position_1 = worker.position

    worker.update()
    paused_position_2 = worker.position

    assert paused_position_1 == moved_position
    assert paused_position_2 == moved_position

    print(
        f"Initial: {first_position}"
    )
    print(
        f"After movement: {moved_position}"
    )
    print(
        f"Pause 1: {paused_position_1}"
    )
    print(
        f"Pause 2: {paused_position_2}"
    )

    print("PASS: worker pause behaviour works.")


def verify_dynamic_robot_linear():
    print("\n=== Dynamic Robot: Linear Movement ===")

    robot = DynamicRobot(
        position=(8, 2),
        movement_pattern="linear",
        movement_direction=(0, 1),
        speed=1.0,
        grid_size=GRID_SIZE,
    )

    positions = [robot.position]

    for step in range(15):
        previous = robot.position
        current = robot.update()

        positions.append(current)

        print(
            f"Step {step + 1:02d}: "
            f"{previous} -> {current}"
        )

    assert len(set(positions)) > 1, (
        "Dynamic robot did not move."
    )

    assert all(
        0 <= row < GRID_SIZE[0]
        and 0 <= col < GRID_SIZE[1]
        for row, col in positions
    ), "Dynamic robot left the grid."

    print("PASS: linear movement works.")


def verify_dynamic_robot_boundary_reversal():
    print("\n=== Dynamic Robot: Boundary Reversal ===")

    robot = DynamicRobot(
        position=(6, 10),
        movement_pattern="linear",
        movement_direction=(0, 1),
        speed=1.0,
        grid_size=GRID_SIZE,
    )

    positions = [robot.position]

    for step in range(5):
        previous = robot.position
        current = robot.update()

        positions.append(current)

        print(
            f"Step {step + 1:02d}: "
            f"{previous} -> {current}"
        )

    assert all(
        0 <= row < GRID_SIZE[0]
        and 0 <= col < GRID_SIZE[1]
        for row, col in positions
    ), "Dynamic robot violated boundary."

    assert robot.position != (6, 12), (
        "Dynamic robot moved outside the grid."
    )

    print("PASS: boundary handling works.")


def verify_dynamic_robot_path_reversal():
    print("\n=== Dynamic Robot: Path Reversal ===")

    path = [
        (9, 2),
        (9, 3),
        (9, 4),
        (9, 5),
        (9, 6),
    ]

    robot = DynamicRobot(
        position=(9, 2),
        movement_pattern="linear",
        movement_path=path,
        allow_reverse=True,
        speed=1.0,
        grid_size=GRID_SIZE,
    )

    positions = [robot.position]

    for step in range(12):
        previous = robot.position
        current = robot.update()

        positions.append(current)

        print(
            f"Step {step + 1:02d}: "
            f"{previous} -> {current}"
        )

    assert positions[:5] == path, (
        "Robot did not correctly follow the forward path."
    )

    assert positions[5] == (9, 5), (
        "Robot did not reverse correctly."
    )

    assert all(
        position in path
        for position in positions
    ), "Robot left the predefined path."

    print("PASS: path reversal works.")
def verify_collision_detection():
    print("\n=== Dynamic Robot: Collision Detection ===")

    robot = DynamicRobot(
        position=(5, 5),
        movement_pattern="linear",
        movement_direction=(0, 1),
        speed=1.0,
        grid_size=GRID_SIZE,
    )

    occupied_positions = [
        (5, 6)
    ]

    previous_position = robot.position

    current_position = robot.update(
        occupied_positions
    )

    print(
        f"Attempted movement: "
        f"{previous_position} -> {current_position}"
    )

    assert current_position == previous_position, (
        "Robot moved into an occupied position."
    )

    assert robot.last_collision is True, (
        "Collision was not detected."
    )

    print(
        f"Collision detected at "
        f"{occupied_positions[0]}"
    )

    print("PASS: collision detection works.")
def run_episode(episode_number: int):
    print(
        f"\n=== Episode {episode_number} ==="
    )

    worker = Worker(
        position=(2, 2),
        movement_pattern="predefined_patrol",
        patrol_route=[
            (2, 2),
            (2, 3),
            (2, 4),
            (2, 5),
            (2, 6),
        ],
        speed=1.0,
        grid_size=GRID_SIZE,
    )

    robot = DynamicRobot(
        position=(8, 2),
        movement_pattern="linear",
        movement_direction=(0, 1),
        speed=1.0,
        grid_size=GRID_SIZE,
    )

    worker_initial = worker.position
    robot_initial = robot.position

    worker_positions = [worker.position]
    robot_positions = [robot.position]

    for step in range(20):

        worker.update()

        robot.update(
            occupied_positions=[
                worker.position
            ]
        )

        worker_positions.append(
            worker.position
        )

        robot_positions.append(
            robot.position
        )

        # Verify worker boundary safety
        worker_row, worker_col = worker.position

        assert (
            0 <= worker_row < GRID_SIZE[0]
        )

        assert (
            0 <= worker_col < GRID_SIZE[1]
        )

        # Verify robot boundary safety
        robot_row, robot_col = robot.position

        assert (
            0 <= robot_row < GRID_SIZE[0]
        )

        assert (
            0 <= robot_col < GRID_SIZE[1]
        )

        # Verify collision safety
        assert robot.position != worker.position

    worker_moved = (
        len(set(worker_positions)) > 1
    )

    robot_moved = (
        len(set(robot_positions)) > 1
    )

    assert worker_moved, (
        f"Worker did not move in episode "
        f"{episode_number}."
    )

    assert robot_moved, (
        f"Dynamic robot did not move in episode "
        f"{episode_number}."
    )

    assert worker.position != worker_initial or worker_moved

    assert robot.position != robot_initial or robot_moved

    print(
        f"Worker: "
        f"{worker_initial} -> {worker.position}"
    )

    print(
        f"Dynamic robot: "
        f"{robot_initial} -> {robot.position}"
    )

    print(
        "PASS: movement, position updates, "
        "collision safety, and boundaries verified."
    )


def verify_multiple_episodes():
    print("\n" + "=" * 60)
    print("MULTI-EPISODE VERIFICATION")
    print("=" * 60)

    for episode in range(1, 4):
        run_episode(episode)

    print("\nPASS: all episodes completed successfully.")

def main():
    print("=" * 60)
    print("DAY 2 — DYNAMIC OBSTACLE VERIFICATION")
    print("=" * 60)

    verify_worker_predefined_patrol()
    verify_worker_random_patrol()
    verify_worker_pause()

    verify_dynamic_robot_linear()
    verify_dynamic_robot_boundary_reversal()
    verify_dynamic_robot_path_reversal()
    verify_collision_detection()
    verify_multiple_episodes()
    
    print("\n" + "=" * 60)
    print("ALL DAY 2 DYNAMIC OBSTACLE CHECKS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()