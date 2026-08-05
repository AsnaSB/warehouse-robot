"""
Manual smoke test for WarehouseEnv — run this yourself to confirm Week 2
is working before you commit.

Usage:
    python test_env_manual.py
"""

import matplotlib.pyplot as plt
from src.environment.warehouse_env import WarehouseEnv


def test_config(config_name: str):
    print(f"\n--- Testing config: {config_name} ---")
    env = WarehouseEnv(config=config_name, seed=42)

    obs, info = env.reset(seed=42)
    print(f"reset() OK -> robot={env.robot_pos}, goal={env.goal_pos}, obs={obs}")
    assert obs.shape == (4,), "Unexpected observation shape"

    total_reward = 0
    collisions = 0
    for step in range(50):
        action = env.action_space.sample() if hasattr(env, "action_space") \
            else __import__("random").randint(0, 7)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if info["collided"]:
            collisions += 1
        if terminated or truncated:
            print(f"Episode ended at step {step+1} (terminated={terminated}, truncated={truncated})")
            break

    print(f"step() OK -> total_reward={total_reward}, collisions={collisions}")

    # Render and save an image so you can SEE it worked, not just trust the print statements
    env.render(mode="human")
    plt.savefig(f"render_check_{config_name}.png")
    print(f"Saved render_check_{config_name}.png — open it and look")
    env.close()


if __name__ == "__main__":
    for cfg in ["open", "aisle", "dense"]:
        test_config(cfg)
    print("\nALL CHECKS PASSED — Week 2 environment is working.")
