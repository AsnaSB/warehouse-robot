import sys
import os
import matplotlib.pyplot as plt
import numpy as np

# Force the project root to the absolute front of Python's path list
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

def plot_architecture_comparison():
    print("--- Generating Architecture Comparison Plot ---")
    
    models = ['Pure A*', 'Pure DDQN', 'Hybrid (A*+DDQN)', 'Tuned Hybrid']
    success_rates = [0.0, 85.0, 85.0, 92.0]
    collision_rates = [100.0, 0.0, 0.0, 0.0]
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(9, 5))
    rects1 = ax.bar(x - width/2, success_rates, width, label='Success Rate (%)', color='#2ca02c')
    rects2 = ax.bar(x + width/2, collision_rates, width, label='Collision Rate (%)', color='#d62728')
    
    ax.set_ylabel('Percentage (%)')
    ax.set_title('Warehouse Robot: Architecture Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    
    ax.bar_label(rects1, padding=3, fmt='%.1f%%')
    ax.bar_label(rects2, padding=3, fmt='%.1f%%')
    
    fig.tight_layout()
    
    # Save the output plot to a results or figures directory
    os.makedirs('experiments/results', exist_ok=True)
    save_path = 'experiments/results/architecture_comparison.png'
    plt.savefig(save_path)
    print(f"Comparison plot successfully saved to {save_path}")
    plt.close()

if __name__ == "__main__":
    plot_architecture_comparison()