import os
import sys
import numpy as np
from typing import Tuple
import asyncio

# Add root to path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from agent.dqn_agent import DQNAgent
from FlapPyBird.src.flappy import Flappy
from .config import DQNConfig


class DQNTrainer:
    """
    Trainer class để train DQN agent với Flappy Bird environment.
    """
    
    def __init__(self, config: DQNConfig, resume_from: str = None):
        self.config = config
        self.agent = DQNAgent(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            hidden_dim=config.hidden_dim,
            lr=config.lr,
            gamma=config.gamma,
            epsilon_start=config.epsilon_start,
            epsilon_end=config.epsilon_end,
            epsilon_decay=config.epsilon_decay,
            buffer_size=config.buffer_size,
            batch_size=config.batch_size,
            target_update_freq=config.target_update_freq
        )
        
        # Create directories
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)
        
        # Stats với giới hạn để tránh memory leak
        self.episode_rewards = []
        self.episode_scores = []
        self.episode_steps = []
        self.best_score = 0
        self.start_episode = 0
        self.max_history = 10000  # Chỉ giữ 10k episodes
        
        # Resume from checkpoint if provided
        if resume_from and os.path.exists(resume_from):
            self.load_checkpoint(resume_from)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint để tiếp tục training."""
        print(f"Loading checkpoint from {checkpoint_path}...")
        
        # Load agent
        self.agent.load(checkpoint_path)
        
        # Try to load training stats
        stats_path = checkpoint_path.replace('.pth', '_stats.pth')
        if os.path.exists(stats_path):
            import torch
            stats = torch.load(stats_path)
            
            # Chỉ load history gần nhất
            self.episode_rewards = stats.get('episode_rewards', [])[-self.max_history:]
            self.episode_scores = stats.get('episode_scores', [])[-self.max_history:]
            self.episode_steps = stats.get('episode_steps', [])[-self.max_history:]
            self.best_score = stats.get('best_score', 0)
            self.start_episode = stats.get('episode', 0)
            print(f"✓ Resumed from episode {self.start_episode}")
            print(f"✓ Best score so far: {self.best_score}")
            print(f"✓ Loaded {len(self.episode_rewards)} episode history")
        else:
            print("⚠ Training stats not found. Starting with fresh stats.")
    
    def save_checkpoint(self, episode: int, checkpoint_path: str):
        """Save checkpoint cùng với training stats."""
        # Save agent
        self.agent.save(checkpoint_path)
        
        # Save training stats (chỉ lưu history gần nhất)
        import torch
        stats_path = checkpoint_path.replace('.pth', '_stats.pth')
        torch.save({
            'episode': episode + 1,
            'episode_rewards': self.episode_rewards[-self.max_history:],
            'episode_scores': self.episode_scores[-self.max_history:],
            'episode_steps': self.episode_steps[-self.max_history:],
            'best_score': self.best_score
        }, stats_path)
    
    async def train_episode(self) -> Tuple[float, int, int]:
        """Train một episode với cleanup."""
        game = Flappy()
        
        try:
            result = await game.play_training_episode(
                agent=self.agent,
                config=self.config,
                render=False
            )
            return result
        finally:
            # Cleanup để giải phóng memory
            del game
    
    async def train(self):
        """Main training loop."""
        print("=" * 50)
        if self.start_episode > 0:
            print(f"Resuming DQN Training from Episode {self.start_episode}")
        else:
            print("Starting DQN Training")
        print("=" * 50)
        print(f"Device: {self.agent.device}")
        print(f"Total episodes: {self.config.num_episodes}")
        print(f"Best score: {self.best_score}")
        print()
        
        for episode in range(self.start_episode, self.config.num_episodes):
            # Train one episode
            total_reward, score, steps = await self.train_episode()
            
            self.episode_rewards.append(total_reward)
            self.episode_scores.append(score)
            self.episode_steps.append(steps)
            
            # Giới hạn history
            if len(self.episode_rewards) > self.max_history:
                self.episode_rewards.pop(0)
                self.episode_scores.pop(0)
                self.episode_steps.pop(0)
            
            # Logging
            if (episode + 1) % self.config.log_freq == 0:
                self._log_progress(episode)
            
            # Save checkpoint với stats
            if (episode + 1) % self.config.save_freq == 0:
                checkpoint_path = os.path.join(
                    self.config.checkpoint_dir,
                    f"dqn_episode_{episode + 1}.pth"
                )
                self.save_checkpoint(episode, checkpoint_path)
            
            # Save best model với stats
            if score > self.best_score:
                self.best_score = score
                best_path = os.path.join(self.config.checkpoint_dir, "best_model.pth")
                self.save_checkpoint(episode, best_path)
        
        # Final save với stats
        final_path = os.path.join(self.config.checkpoint_dir, "final_model.pth")
        self.save_checkpoint(self.config.num_episodes - 1, final_path)
        
        print("=" * 50)
        print("Training completed!")
        print(f"Best score: {self.best_score}")
        print("=" * 50)
    
    def _log_progress(self, episode: int):
        """Log training progress."""
        n = self.config.log_freq
        avg_reward = np.mean(self.episode_rewards[-n:])
        avg_score = np.mean(self.episode_scores[-n:])
        avg_steps = np.mean(self.episode_steps[-n:])
        
        epsilon = self.agent.epsilon_end + (self.agent.epsilon_start - self.agent.epsilon_end) * \
                 np.exp(-1. * self.agent.steps_done / self.agent.epsilon_decay)
        
        print(f"Episode {episode + 1}/{self.config.num_episodes}")
        print(f"  Avg Reward: {avg_reward:.2f}")
        print(f"  Avg Score: {avg_score:.2f}")
        print(f"  Avg Steps: {avg_steps:.0f}")
        print(f"  Epsilon: {epsilon:.3f}")
        print(f"  Total Steps: {self.agent.steps_done}")
        print(f"  Buffer: {len(self.agent.memory)}")
        if self.agent.loss_history:
            print(f"  Avg Loss: {np.mean(self.agent.loss_history[-100:]):.4f}")
        print(f"  Best Score: {self.best_score}")
        print()
