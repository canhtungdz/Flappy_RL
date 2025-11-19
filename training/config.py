from dataclasses import dataclass


@dataclass
class DQNConfig:
    """Cấu hình hyperparameters cho DQN training."""
    
    # Network architecture
    state_dim: int = 3
    action_dim: int = 2
    hidden_dim: int = 128
    
    # Training hyperparameters
    lr: float = 0.0001
    gamma: float = 0.99
    batch_size: int = 64
    buffer_size: int = 100000
    
    # Exploration
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: int = 100000
    
    # Target network
    target_update_freq: int = 1000
    
    # Training
    num_episodes: int = 10000
    max_steps_per_episode: int = 10000
    train_freq: int = 4  # Train mỗi 4 steps
    
    # Logging
    log_freq: int = 10  # Log mỗi 10 episodes
    save_freq: int = 100  # Save model mỗi 100 episodes
    
    # Paths
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    
    # Reward shaping
    reward_alive: float = 0.1
    reward_pass_pipe: float = 10.0
    reward_death: float = -100.0
