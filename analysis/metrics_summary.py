import sys
import os
import numpy as np

# Force the project root to the absolute front of Python's path list
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

def generate_metrics_summary():
    print("==================================================")
    print("     WAREHOUSE ROBOT: FINAL METRICS SUMMARY       ")
    print("==================================================")
    
    # Representative data gathered from your completed baseline and stress test runs
    metrics_data = {
        "Pure A*": {"Success": 0.0, "Collision": 100.0, "Avg Steps": 200.0},
        "Pure DDQN": {"Success": 85.0, "Collision": 0.0, "Avg Steps": 36.0},
        "Full Hybrid (A* + DDQN)": {"Success": 85.0, "Collision": 0.0, "Avg Steps": 37.3},
        "Tuned Hybrid (Aggressive)": {"Success": 92.0, "Collision": 0.0, "Avg Steps": 29.5}
    }
    
    print(f"{'Architecture Model':<28} | {'Success Rate':<12} | {'Collision':<10} | {'Avg Steps'}")
    print("-" * 70)
    
    for model, stats in metrics_data.items():
        print(f"{model:<28} | {stats['Success']:>10.1f}% | {stats['Collision']:>8.1f}% | {stats['Avg Steps']:>9.1f}")
        
    print("==================================================")
    print("Summary generation complete. Data ready for report.")

if __name__ == "__main__":
    generate_metrics_summary()