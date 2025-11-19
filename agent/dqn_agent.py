import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional
import os

from .neural_network import QNetwork
from .replay_buffer import ReplayBuffer


class DQNAgent:
    """
    Deep Q-Network Agent cho Flappy Bird.
    
    Sử dụng:
    - Experience Replay
    - Target Network
    - Epsilon-greedy exploration
    """
    
    def __init__(
        self,
        state_dim: int = 3,
        action_dim: int = 2,
        hidden_dim: int = 128,
        lr: float = 0.0001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: int = 100000,
        buffer_size: int = 100000,
        batch_size: int = 64,
        target_update_freq: int = 1000,
        device: Optional[str] = None
    ):
        """
        Args:
            state_dim: Số chiều của state (3: dx, dy, v)
            action_dim: Số actions (2: không flap, flap)
            hidden_dim: Số neurons trong hidden layers
            lr: Learning rate
            gamma: Discount factor
            epsilon_start: Epsilon ban đầu cho exploration
            epsilon_end: Epsilon cuối cùng
            epsilon_decay: Số steps để decay epsilon
            buffer_size: Kích thước replay buffer
            batch_size: Kích thước mini-batch
            target_update_freq: Tần suất update target network (steps)
            device: 'cuda' hoặc 'cpu', None = auto detect
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Q-Networks
        self.policy_net = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        
        # Replay buffer
        self.memory = ReplayBuffer(buffer_size)
        
        # Epsilon-greedy parameters
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0
        
        # Training stats
        self.loss_history = []
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Chọn action sử dụng epsilon-greedy policy.
        
        Args:
            state: State hiện tại [dx, dy, v]
            training: Nếu False, không exploration (epsilon = 0)
            
        Returns:
            action: 0 (không flap) hoặc 1 (flap)
        """
        # Epsilon decay
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                  np.exp(-1. * self.steps_done / self.epsilon_decay)
        
        if training:
            self.steps_done += 1
        
        # Epsilon-greedy
        if training and np.random.random() < epsilon:
            return np.random.randint(self.action_dim)
        
        # Greedy action
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            action = q_values.argmax(dim=1).item()
        
        return action
    
    def store_transition(self, state: np.ndarray, action: int, reward: float,
                        next_state: np.ndarray, done: bool):
        """
        Lưu transition vào replay buffer.
        """
        self.memory.push(state, action, reward, next_state, done)
    
    def train_step(self) -> Optional[float]:
        """
        Thực hiện một bước training.
        
        Returns:
            loss: Loss value nếu có training, None nếu buffer chưa đủ
        """
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample mini-batch
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # Current Q-values
        current_q_values = self.policy_net(states).gather(1, actions)
        
        # Target Q-values
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Loss
        loss = nn.MSELoss()(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10)
        self.optimizer.step()
        
        # Update target network
        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        loss_value = loss.item()
        self.loss_history.append(loss_value)
        
        return loss_value
    
    def save(self, path: str):
        """Lưu model weights."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'steps_done': self.steps_done,
            'loss_history': self.loss_history
        }, path)
        print(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model weights."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.steps_done = checkpoint['steps_done']
        self.loss_history = checkpoint.get('loss_history', [])
        print(f"Model loaded from {path}")
