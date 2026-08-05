"""
WarehouseEnv — core 2D grid simulation.

Week 2 scope: grid init, robot/goal placement, step()/reset(), rendering.
NOT in scope yet (later weeks, other files):
  - dynamic obstacles              -> obstacles.py (Week 3, Member 1)
  - real reward shaping            -> rewards.py (Week 4, Member 1)
  - context classification         -> context.py (Week 5, Member 1)
  - A* waypoints / hybrid wiring   -> src/hybrid/ (Member 4)

step() currently returns a placeholder reward (-1 per step, +100 on
goal, -20 on collision) just so the interface is usable end-to-end;
Member 1 will replace this with the real reward module in Week 4.

Built Gymnasium-style (reset/step signatures match the Gymnasium API)
so it plugs into standard RL tooling. Falls back to a plain base class
if gymnasium isn't installed in a given environment — install it with
`pip install gymnasium` for full compatibility.
"""

from typing import Optional

import numpy as np

from src.environment.constants import (
    FREE, SHELF, ROBOT, GOAL,
    ACTIONS, NUM_ACTIONS, DEFAULT_GRID_SIZE,
)
from src.environment.configs import get_layout

try:
    import gymnasium as gym
    from gymnasium import spaces
    _HAS_GYM = True
except ImportError:
    _HAS_GYM = False


_BaseEnv = gym.Env if _HAS_GYM else object


class WarehouseEnv(_BaseEnv):
    """2D grid warehouse environment.

    Parameters
    ----------
    config : str
        One of "open", "aisle", "dense" — selects the shelf layout.
    grid_size : int
        Grid is grid_size x grid_size. Default 12 (see design doc).
    max_steps : int
        Episode truncates after this many steps.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        config: str = "open",
        grid_size: int = DEFAULT_GRID_SIZE,
        max_steps: int = 200,
        seed: Optional[int] = None,
    ):
        self.config_name = config
        self.grid_size = grid_size
        self.max_steps = max_steps
        self._rng = np.random.default_rng(seed)

        # Static shelf layout — fixed per config, doesn't change per episode.
        self._static_grid = get_layout(config, grid_size)

        self.robot_pos = None
        self.goal_pos = None
        self._step_count = 0

        # For rendering
        self._fig = None
        self._ax = None

        if _HAS_GYM:
            self.observation_space = spaces.Box(
                low=0, high=max(grid_size, 4),
                shape=(4,), dtype=np.float32,
            )
            self.action_space = spaces.Discrete(NUM_ACTIONS)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._step_count = 0
        self.robot_pos = self._random_free_cell()
        self.goal_pos = self._random_free_cell(exclude={self.robot_pos})

        obs = self._get_obs()
        info = {"config": self.config_name}
        return obs, info

    def step(self, action: int):
        if action not in ACTIONS:
            raise ValueError(f"Invalid action {action}. Must be 0-{NUM_ACTIONS - 1}.")

        self._step_count += 1
        dr, dc = ACTIONS[action]
        new_r = self.robot_pos[0] + dr
        new_c = self.robot_pos[1] + dc

        collided = False
        if not self._in_bounds(new_r, new_c) or self._is_shelf(new_r, new_c):
            # Blocked move: robot stays in place, treated as a collision.
            collided = True
        else:
            self.robot_pos = (new_r, new_c)

        reached_goal = self.robot_pos == self.goal_pos

        # --- Placeholder reward; real version lands in rewards.py (Week 4) ---
        if reached_goal:
            reward = 100.0
        elif collided:
            reward = -20.0
        else:
            reward = -1.0

        terminated = reached_goal
        truncated = self._step_count >= self.max_steps

        obs = self._get_obs()
        info = {"collided": collided, "step_count": self._step_count}
        return obs, reward, terminated, truncated, info

    def render(self, mode: str = "human"):
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap

        if self._fig is None:
            self._fig, self._ax = plt.subplots(figsize=(5, 5))

        display_grid = self._static_grid.copy()
        gr, gc = self.goal_pos
        rr, rc = self.robot_pos
        display_grid[gr, gc] = GOAL
        display_grid[rr, rc] = ROBOT

        cmap = ListedColormap(
            ["white", "dimgray", "orange", "royalblue", "limegreen"]
        )  # FREE, SHELF, DYNAMIC_OBSTACLE, ROBOT, GOAL

        self._ax.clear()
        self._ax.imshow(display_grid, cmap=cmap, vmin=0, vmax=4)
        self._ax.set_xticks(range(self.grid_size))
        self._ax.set_yticks(range(self.grid_size))
        self._ax.grid(color="lightgray", linewidth=0.5)
        self._ax.set_title(f"{self.config_name.capitalize()} — step {self._step_count}")

        if mode == "human":
            plt.pause(0.001)
            return None
        elif mode == "rgb_array":
            self._fig.canvas.draw()
            buf = np.asarray(self._fig.canvas.buffer_rgba())
            return buf[:, :, :3]

    def close(self):
        if self._fig is not None:
            import matplotlib.pyplot as plt
            plt.close(self._fig)
            self._fig, self._ax = None, None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        """Minimal Week-2 observation: robot (r, c) + goal (r, c).

        This will be replaced by the full 32-dim state vector (local
        patch + normalized positions + context one-hot) once
        context.py (Week 5) and state_augmentation.py (Member 4, Week 3)
        land. Kept simple for now so step()/reset() are testable in
        isolation this week.
        """
        return np.array(
            [*self.robot_pos, *self.goal_pos], dtype=np.float32
        )

    def _in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.grid_size and 0 <= c < self.grid_size

    def _is_shelf(self, r: int, c: int) -> bool:
        return self._static_grid[r, c] == SHELF

    def _random_free_cell(self, exclude: Optional[set] = None) -> tuple:
        exclude = exclude or set()
        free_cells = [
            (r, c)
            for r in range(self.grid_size)
            for c in range(self.grid_size)
            if self._static_grid[r, c] == FREE and (r, c) not in exclude
        ]
        if not free_cells:
            raise RuntimeError("No free cells available for placement.")
        idx = self._rng.integers(0, len(free_cells))
        return free_cells[idx]
