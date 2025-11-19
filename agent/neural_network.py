import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """
    Q-Network cho DQN.
    
    Input: state vector [dx, dy, v] - 3 features
    Output: Q-values cho 2 actions [không flap, flap]
    """
    
    def __init__(self, state_dim: int = 3, action_dim: int = 2, hidden_dim: int = 128):
        """
        Args:
            state_dim: Số chiều của state (mặc định 3: dx, dy, v)
            action_dim: Số action có thể thực hiện (mặc định 2: 0=không flap, 1=flap)
            hidden_dim: Số neurons trong hidden layers
        """
        super(QNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state):
        """
        Forward pass.
        
        Args:
            state: tensor shape (batch_size, 3)
        
        Returns:
            Q-values: tensor shape (batch_size, 2)
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values


class DuelingQNetwork(nn.Module):
    """
    Dueling Q-Network - phiên bản nâng cao của Q-Network.
    Tách riêng Value và Advantage streams.
    """
    
    def __init__(self, state_dim: int = 3, action_dim: int = 2, hidden_dim: int = 128):
        super(DuelingQNetwork, self).__init__()
        
        # Shared feature extraction
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
    def forward(self, state):
        features = self.feature(state)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values
