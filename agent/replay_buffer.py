import numpy as np
import random
from collections import deque
from typing import Tuple


class ReplayBuffer:
    """
    Experience Replay Buffer cho DQN.
    Lưu trữ transitions (s, a, r, s', done) và sample random mini-batches.
    """
    
    def __init__(self, capacity: int = 100000):
        """
        Args:
            capacity: Kích thước tối đa của buffer
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool):
        """
        Thêm một transition vào buffer.
        
        Args:
            state: State hiện tại
            action: Action đã thực hiện (0 hoặc 1)
            reward: Reward nhận được
            next_state: State tiếp theo
            done: Episode đã kết thúc hay chưa
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """
        Sample một mini-batch ngẫu nhiên từ buffer.
        
        Args:
            batch_size: Kích thước batch
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
        """
        batch = random.sample(self.buffer, batch_size)
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32)
        )
    
    def __len__(self) -> int:
        """Trả về số lượng transitions trong buffer."""
        return len(self.buffer)


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Buffer - phiên bản nâng cao.
    Sample transitions dựa trên TD error (priority).
    """
    
    def __init__(self, capacity: int = 100000, alpha: float = 0.6):
        """
        Args:
            capacity: Kích thước tối đa
            alpha: Mức độ ưu tiên (0 = uniform, 1 = full prioritization)
        """
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        
    def push(self, state, action, reward, next_state, done):
        max_priority = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
        
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int, beta: float = 0.4):
        """
        Sample với prioritization.
        
        Args:
            batch_size: Kích thước batch
            beta: Compensation cho bias (tăng dần từ 0.4 -> 1.0)
        """
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.position]
        
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        # Importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        states, actions, rewards, next_states, dones = zip(*samples)
        
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
            indices,
            np.array(weights, dtype=np.float32)
        )
    
    def update_priorities(self, indices, priorities):
        """Cập nhật priorities sau khi train."""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
    
    def __len__(self):
        return len(self.buffer)
