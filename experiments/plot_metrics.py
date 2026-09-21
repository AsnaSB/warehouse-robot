import matplotlib.pyplot as plt
import numpy as np
import os

def plot_evaluation_metrics():
    # --- Data Setup ---
    # Update the Pure DDQN numbers based on your actual terminal output
    models = ['Pure DDQN', 'Hybrid (A* + DDQN)']
    
    # [Pure DDQN value, Hybrid value]
    success_rates = [45.0, 80.0]   # Percentage (%)
    collision_rates = [35.0, 0.0]  # Percentage (%)
    avg_steps = [150.0, 45.0]      # Step count

    # --- Figure Setup ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Warehouse Robot Navigation: Model Evaluation', fontsize=16, fontweight='bold', y=1.05)

    colors = ['#ff9999', '#66b3ff'] # Red for Pure DDQN, Blue for Hybrid

    # 1. Success Rate Plot
    axes[0].bar(models, success_rates, color=colors, edgecolor='black')
    axes[0].set_title('Success Rate (%)')
    axes[0].set_ylim(0, 100)
    for i, v in enumerate(success_rates):
        axes[0].text(i, v + 2, f"{v}%", ha='center', fontweight='bold')

    # 2. Collision Rate Plot
    axes[1].bar(models, collision_rates, color=colors, edgecolor='black')
    axes[1].set_title('Collision Rate (%)')
    axes[1].set_ylim(0, max(collision_rates) + 15)
    for i, v in enumerate(collision_rates):
        axes[1].text(i, v + 1, f"{v}%", ha='center', fontweight='bold')

    # 3. Average Steps (Path Length) Plot
    axes[2].bar(models, avg_steps, color=colors, edgecolor='black')
    axes[2].set_title('Average Steps to Goal')
    axes[2].set_ylim(0, max(avg_steps) + 20)
    for i, v in enumerate(avg_steps):
        axes[2].text(i, v + 2, f"{v}", ha='center', fontweight='bold')

    # Polish and display
    plt.tight_layout()
    
    # Save the plot to a file
    os.makedirs("experiments/results", exist_ok=True)
    save_path = "experiments/results/evaluation_comparison.png"
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Plot saved successfully to: {save_path}")
    
    plt.show()

if __name__ == "__main__":
    plot_evaluation_metrics()