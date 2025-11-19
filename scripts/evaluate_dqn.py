import os
import sys
import argparse
import numpy as np
import asyncio

# Add root directory to path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from agent.dqn_agent import DQNAgent
from FlapPyBird.src.flappy import Flappy


async def evaluate_dqn(checkpoint_path: str, num_episodes: int = 10):
    """
    Đánh giá DQN agent đã train.
    
    Args:
        checkpoint_path: Đường dẫn đến model checkpoint
        num_episodes: Số episodes để evaluate
    """
    # Load agent
    agent = DQNAgent()
    agent.load(checkpoint_path)
    
    print("=" * 50)
    print(f"Evaluating DQN Agent")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Episodes: {num_episodes}")
    print("=" * 50)
    print()
    
    scores = []
    
    for episode in range(num_episodes):
        # Initialize game
        game = Flappy()
        
        # Play one episode
        score = await game.start(agent=agent)
        scores.append(score)
        
        print(f"Episode {episode + 1}: Score = {score}")
    
    # Statistics
    print()
    print("=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    print(f"Average Score: {np.mean(scores):.2f}")
    print(f"Std Dev: {np.std(scores):.2f}")
    print(f"Min Score: {np.min(scores)}")
    print(f"Max Score: {np.max(scores)}")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="Evaluate DQN agent")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to evaluate")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    
    asyncio.run(evaluate_dqn(args.checkpoint, args.episodes))


if __name__ == "__main__":
    main()
