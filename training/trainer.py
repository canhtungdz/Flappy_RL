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
    
    def __init__(self, config: DQNConfig):
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
        
        # Stats
        self.episode_rewards = []
        self.episode_scores = []
        self.episode_steps = []
        self.losses = []
        self.best_score = 0
        
    async def train_episode(self) -> Tuple[float, int, int]:
        """
        Train một episode.
        
        Returns:
            (total_reward, score, steps)
        """
        game = Flappy()
        
        # Run episode với modified game loop (render=False để train nhanh)
        result = await game.play_training_episode(
            agent=self.agent,
            config=self.config,
            render=False  # Tắt rendering để train nhanh hơn
        )
        
        return result
    
    async def train(self):
        """Main training loop."""
        print("=" * 50)
        print("Starting DQN Training")
        print("=" * 50)
        print(f"Device: {self.agent.device}")
        print(f"Total episodes: {self.config.num_episodes}")
        print()
        
        for episode in range(self.config.num_episodes):
            # Train one episode
            total_reward, score, steps = await self.train_episode()
            
            self.episode_rewards.append(total_reward)
            self.episode_scores.append(score)
            self.episode_steps.append(steps)
            
            # Logging
            if (episode + 1) % self.config.log_freq == 0:
                self._log_progress(episode)
            
            # Save checkpoint
            if (episode + 1) % self.config.save_freq == 0:
                checkpoint_path = os.path.join(
                    self.config.checkpoint_dir,
                    f"dqn_episode_{episode + 1}.pth"
                )
                self.agent.save(checkpoint_path)
            
            # Save best model
            if score > self.best_score:
                self.best_score = score
                best_path = os.path.join(self.config.checkpoint_dir, "best_model.pth")
                self.agent.save(best_path)
        
        # Final save
        final_path = os.path.join(self.config.checkpoint_dir, "final_model.pth")
        self.agent.save(final_path)
        
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
