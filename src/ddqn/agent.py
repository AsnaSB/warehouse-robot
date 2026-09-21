import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.ddqn.network import QNetwork
from src.ddqn.replay_buffer import ReplayBuffer

class DDQNAgent:
    def __init__(self, state_dim=51, action_dim=8, lr=1e-4, gamma=0.99, batch_size=64, buffer_capacity=50000):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        
        # Use GPU if available to speed up training, otherwise fallback to CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize Online and Target Networks
        self.online_net = QNetwork(state_dim, num_actions=action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, num_actions=action_dim).to(self.device)
        
        # Target network starts with identical weights
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval() # Target network is never explicitly trained
        
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        
        self.memory = ReplayBuffer(capacity=buffer_capacity)
        
    def select_action(self, state, epsilon=0.0):
        """Epsilon-greedy action selection."""
        if random.random() <= epsilon:
            return random.randint(0, self.action_dim - 1)
            
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.online_net(state_tensor)
            
        return torch.argmax(q_values).item()
        
    def store_transition(self, state, action, reward, next_state, done):
        """Passes the transition to the replay buffer."""
        self.memory.push(state, action, reward, next_state, done)
        
    def train_step(self):
        """Samples a batch and performs one Double DQN optimization step."""
        if len(self.memory) < self.batch_size:
            return None # Not enough data to train yet
            
        # 1. Sample from Replay Buffer
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # 2. Compute Current Q values
        current_q = self.online_net(states).gather(1, actions)
        
        # 3. Double DQN Logic for Target Q
        with torch.no_grad():
            # Online network selects the best action for the next state
            next_actions = self.online_net(next_states).argmax(1, keepdim=True)
            # Target network evaluates the Q-value of that chosen action
            next_q = self.target_net(next_states).gather(1, next_actions)
            
            # Bellman equation
            target_q = rewards + (1 - dones) * self.gamma * next_q
            
        # 4. Backpropagation
        loss = self.loss_fn(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping prevents exploding gradients during early training
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return loss.item()
        
    def update_target_network(self):
        """Syncs target network weights with online network."""
        self.target_net.load_state_dict(self.online_net.state_dict())
        
    def save(self, filepath):
        """Saves the online network weights."""
        torch.save(self.online_net.state_dict(), filepath)
        
    def load(self, filepath):
        """Loads weights into both networks."""
        self.online_net.load_state_dict(torch.load(filepath, map_location=self.device))
        self.target_net.load_state_dict(self.online_net.state_dict())