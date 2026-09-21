import sys
import os
import json
import torch

# Force the project root to the absolute front of Python's path list
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

def load_training_logs(log_path="experiments/results/training_metrics.json"):
    """Loads JSON-based training logs if available."""
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            data = json.load(f)
        print(f"Successfully loaded training logs from {log_path}")
        return data
    else:
        print(f"Notice: Log file not found at {log_path}. Returning default mock tracking structures.")
        return {
            "episodes": list(range(1, 101)),
            "rewards": [-20.0 + (i * 0.3) for i in range(100)],
            "success_rates": [min(100, i * 1.0) for i in range(100)]
        }

def load_checkpoint_metadata(checkpoint_path="experiments/checkpoints/hybrid_ddqn_ep2500.pth"):
    """Loads PyTorch model checkpoint safely for evaluation inspection."""
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        print(f"Successfully loaded checkpoint metadata from {checkpoint_path}")
        return checkpoint
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}")
        return None

if __name__ == "__main__":
    print("--- Testing Result Loader Utility ---")
    logs = load_training_logs()
    checkpoint = load_checkpoint_metadata()
    print("Result loader utility verification complete!")