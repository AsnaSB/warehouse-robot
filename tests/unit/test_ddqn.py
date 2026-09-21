import sys
import os
import numpy as np

# Force the project root to the absolute front of Python's path list
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from src.ddqn.agent import DDQNAgent

def test_dqn_network_output():
    print("--- Running Unit Test: DDQN Action Selection ---")
    state_dim = 51
    action_dim = 8
    
    agent = DDQNAgent(state_dim=state_dim, action_dim=action_dim)
    
    # Create a dummy batch state matching your 51-D vector representation
    dummy_state = np.random.randn(state_dim).astype(np.float32)
    
    # Use the correct method name: select_action
    action = agent.select_action(dummy_state)
    print(f"Action selection test passed! Output action: {action}")
    assert 0 <= action < action_dim, "Action out of valid range bounds!"

if __name__ == "__main__":
    test_dqn_network_output()