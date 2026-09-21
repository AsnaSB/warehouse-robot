import matplotlib.pyplot as plt
import numpy as np

from src.environment.constants import (
    FREE,
    SHELF,
    ROBOT,
    GOAL,
)


class WarehouseRenderer:
    """
    Visualization layer for WarehouseEnv.

    This class only reads the environment state.
    It does not implement robot movement,
    collision detection, rewards, or obstacle logic.
    """

    def __init__(self, env, figsize=(8, 8)):

        self.env = env

        self.fig, self.ax = plt.subplots(
            figsize=figsize
        )

        self.fig.canvas.manager.set_window_title(
            "Autonomous Warehouse Simulation"
        )

    def draw(self):

        env = self.env

        self.ax.clear()

        # --------------------------------------------------
        # Base warehouse
        # --------------------------------------------------

        grid = env._static_grid.copy()

        # --------------------------------------------------
        # Draw shelves / floor
        # --------------------------------------------------

        self.ax.imshow(
            grid,
            cmap="Greys",
            vmin=0,
            vmax=1,
        )

        # --------------------------------------------------
        # Start position
        # --------------------------------------------------

        if env.start_pos is not None:

            row, col = env.start_pos

            self.ax.scatter(
                col,
                row,
                marker="o",
                s=180,
                facecolors="none",
                edgecolors="green",
                linewidths=2,
                label="Start",
            )

        # --------------------------------------------------
        # Goal
        # --------------------------------------------------

        if env.goal_pos is not None:

            row, col = env.goal_pos

            self.ax.scatter(
                col,
                row,
                marker="*",
                s=250,
                color="red",
                label="Goal",
            )

        # --------------------------------------------------
        # Robot
        # --------------------------------------------------

        if env.robot_pos is not None:

            row, col = env.robot_pos

            self.ax.scatter(
                col,
                row,
                marker="o",
                s=180,
                color="blue",
                label="Robot",
                zorder=5,
            )

        # --------------------------------------------------
        # Workers
        # --------------------------------------------------

        for worker in env.workers:

            if not worker.active:
                continue

            row, col = worker.position

            self.ax.scatter(
                col,
                row,
                marker="s",
                s=130,
                color="orange",
                label="Worker",
                zorder=4,
            )

        # --------------------------------------------------
        # Dynamic robots
        # --------------------------------------------------

        for robot in env.dynamic_robots:

            if not robot.active:
                continue

            row, col = robot.position

            self.ax.scatter(
                col,
                row,
                marker="D",
                s=130,
                color="purple",
                label="Dynamic Robot",
                zorder=4,
            )

        # --------------------------------------------------
        # A* path
        # --------------------------------------------------

        if env.current_astar_path:

            rows = [
                position[0]
                for position in env.current_astar_path
            ]

            cols = [
                position[1]
                for position in env.current_astar_path
            ]

            self.ax.plot(
                cols,
                rows,
                linestyle="--",
                linewidth=2,
                marker=".",
                markersize=5,
                color="cyan",
                label="A* Path",
                zorder=3,
            )

        # --------------------------------------------------
        # Grid
        # --------------------------------------------------

        self.ax.set_xticks(
            np.arange(env.grid_size)
        )

        self.ax.set_yticks(
            np.arange(env.grid_size)
        )

        self.ax.grid(
            True,
            linewidth=0.5,
            alpha=0.4,
        )

        # --------------------------------------------------
        # Labels
        # --------------------------------------------------

        self.ax.set_xlabel("Column")

        self.ax.set_ylabel("Row")

        self.ax.set_title(
            f"Warehouse Simulation | "
            f"Scenario: {env.config_name.capitalize()} | "
            f"Step: {env._step_count}"
        )

        # --------------------------------------------------
        # Remove duplicate legend entries
        # --------------------------------------------------

        handles, labels = (
            self.ax.get_legend_handles_labels()
        )

        unique = dict(
            zip(labels, handles)
        )

        if unique:

            self.ax.legend(
                unique.values(),
                unique.keys(),
                loc="upper right",
            )

        self.fig.tight_layout()

        self.fig.canvas.draw()

        self.fig.canvas.flush_events()

    def show(self):

        self.draw()

        plt.show()

    def close(self):

        plt.close(self.fig)