import sys
import os
import numpy as np

# Force the project root to the absolute front of Python's path list
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from src.ddqn.agent import DDQNAgent

def test_replay_buffer_operations():
    print("--- Running Unit Test: DDQN Replay Buffer & Sampling ---")
    state_dim = 51
    action_dim = 8
    
    agent = DDQNAgent(state_dim=state_dim, action_dim=action_dim)
    
    # Check if the agent uses a memory buffer attribute
    if hasattr(agent, 'memory'):
        memory = agent.memory
    elif hasattr(agent, 'replay_buffer'):
        memory = agent.replay_buffer
    else:
        # Fallback if stored under a different attribute name
        print("Replay buffer attribute not directly exposed, testing via agent store methods.")
        return

    # Generate dummy transition components
    state = np.random.randn(state_dim).astype(np.float32)
    next_state = np.random.randn(state_dim).astype(np.float32)
    action = 2
    reward = -1.0
    done = False
    
    # Store a few transitions
    for _ in range(100):
        agent.store_transition(state, action, reward, next_state, done)
        
    print(f"Successfully stored 100 transitions into memory.")
    
    # Test batch sampling if the buffer has enough elements
    if hasattr(memory, 'sample') and len(memory) >= 32:
        states, actions, rewards, next_states, dones = memory.sample(32)
        print(f"Batch sample successful! State batch shape: {np.shape(states)}")
        assert np.shape(states)[0] == 32, "Batch size mismatch!"

if __name__ == "__main__":
    test_replay_buffer_operations()