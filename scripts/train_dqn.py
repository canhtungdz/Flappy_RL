import os
import sys
import argparse
import asyncio

# Add root directory to path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from training.trainer import DQNTrainer
from training.config import DQNConfig


def main():
    parser = argparse.ArgumentParser(description="Train DQN agent for Flappy Bird")
    parser.add_argument("--episodes", type=int, default=10000, help="Number of episodes")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--buffer-size", type=int, default=100000, help="Replay buffer size")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden layer dimension")
    parser.add_argument("--epsilon-decay", type=int, default=100000, help="Epsilon decay steps")
    
    # Thêm resume argument
    parser.add_argument("--resume", type=str, default=None, 
                       help="Path to checkpoint to resume training (e.g., checkpoints/dqn_episode_1400.pth)")
    
    args = parser.parse_args()
    
    # Validate resume path
    if args.resume and not os.path.exists(args.resume):
        print(f"Error: Checkpoint not found: {args.resume}")
        sys.exit(1)
    
    # Create config
    config = DQNConfig(
        num_episodes=args.episodes,
        lr=args.lr,
        gamma=args.gamma,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        checkpoint_dir=args.checkpoint_dir,
        hidden_dim=args.hidden_dim,
        epsilon_decay=args.epsilon_decay
    )
    
    # Create trainer với resume option
    trainer = DQNTrainer(config, resume_from=args.resume)
    
    # Train
    asyncio.run(trainer.train())


if __name__ == "__main__":
    main()
