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
