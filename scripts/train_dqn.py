import os
import sys
import argparse
import numpy as np
from datetime import datetime

# Add root directory to path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from agent.dqn_agent import DQNAgent
from training.config import DQNConfig


def train_dqn(config: DQNConfig):
    """
    Training loop cho DQN agent.
    
    Note: Phần này cần integrate với Flappy game environment.
    Hiện tại là skeleton code để bạn tham khảo cấu trúc.
    """
    # Initialize agent
    agent = DQNAgent(
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
    
    # Training stats
    episode_rewards = []
    episode_scores = []
    best_score = 0
    
    print("=" * 50)
    print("Starting DQN Training")
    print("=" * 50)
    print(f"Device: {agent.device}")
    print(f"Total episodes: {config.num_episodes}")
    print()
    
    for episode in range(config.num_episodes):
        # TODO: Initialize game environment
        # game = Flappy()
        # state = game.reset()  # Cần implement reset() method
        
        episode_reward = 0
        episode_score = 0
        steps = 0
        done = False
        
        # Placeholder: Giả lập một episode
        state = np.random.randn(3).astype(np.float32)
        
        while not done and steps < config.max_steps_per_episode:
            # Select action
            action = agent.select_action(state, training=True)
            
            # TODO: Execute action in environment
            # next_state, reward, done, info = game.step(action)
            # episode_score = info.get('score', 0)
            
            # Placeholder
            next_state = np.random.randn(3).astype(np.float32)
            reward = config.reward_alive
            done = np.random.random() < 0.01  # 1% chance to end
            
            # Store transition
            agent.store_transition(state, action, reward, next_state, done)
            
            # Train
            if steps % config.train_freq == 0:
                loss = agent.train_step()
            
            state = next_state
            episode_reward += reward
            steps += 1
        
        episode_rewards.append(episode_reward)
        episode_scores.append(episode_score)
        
        # Logging
        if (episode + 1) % config.log_freq == 0:
            avg_reward = np.mean(episode_rewards[-config.log_freq:])
            avg_score = np.mean(episode_scores[-config.log_freq:])
            epsilon = agent.epsilon_end + (agent.epsilon_start - agent.epsilon_end) * \
                     np.exp(-1. * agent.steps_done / agent.epsilon_decay)
            
            print(f"Episode {episode + 1}/{config.num_episodes}")
            print(f"  Avg Reward: {avg_reward:.2f}")
            print(f"  Avg Score: {avg_score:.2f}")
            print(f"  Epsilon: {epsilon:.3f}")
            print(f"  Steps: {agent.steps_done}")
            print(f"  Buffer: {len(agent.memory)}")
            if agent.loss_history:
                print(f"  Avg Loss: {np.mean(agent.loss_history[-100:]):.4f}")
            print()
        
        # Save checkpoint
        if (episode + 1) % config.save_freq == 0:
            checkpoint_path = os.path.join(
                config.checkpoint_dir,
                f"dqn_episode_{episode + 1}.pth"
            )
            agent.save(checkpoint_path)
        
        # Save best model
        if episode_score > best_score:
            best_score = episode_score
            best_path = os.path.join(config.checkpoint_dir, "best_model.pth")
            agent.save(best_path)
    
    # Final save
    final_path = os.path.join(config.checkpoint_dir, "final_model.pth")
    agent.save(final_path)
    
    print("=" * 50)
    print("Training completed!")
    print(f"Best score: {best_score}")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="Train DQN agent for Flappy Bird")
    parser.add_argument("--episodes", type=int, default=10000, help="Number of episodes")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--buffer-size", type=int, default=100000, help="Replay buffer size")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Checkpoint directory")
    
    args = parser.parse_args()
    
    # Create config
    config = DQNConfig(
        num_episodes=args.episodes,
        lr=args.lr,
        gamma=args.gamma,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        checkpoint_dir=args.checkpoint_dir
    )
    
    # Train
    train_dqn(config)


if __name__ == "__main__":
    main()
