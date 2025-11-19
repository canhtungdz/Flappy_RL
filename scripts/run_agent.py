# scripts/run_agent.py
import asyncio
import os
import sys

# Thêm thư mục gốc (flappy_rl) vào sys.path để import được
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from FlapPyBird.src.flappy import Flappy
from agent.random_agent import RandomAgent
from agent.rulebase_agent import RuleBaseAgent, AdvancedRuleBaseAgent
from agent.dqn_agent import DQNAgent

async def main():
    game = Flappy()
    
    # Chọn agent muốn test:
    
    # Option 1: Random Agent
    # agent = RandomAgent(flap_prob=0.10)
    
    # Option 2: Rule-based Agent
    # agent = RuleBaseAgent()
    # agent = AdvancedRuleBaseAgent()
    
    # Option 3: DQN Agent (TRAINED MODEL)
    agent = DQNAgent()
    agent.load("checkpoints/best_model.pth")  # Load model đã train
    
    print(f"Playing with DQN Agent (loaded from best_model.pth)")
    print("Press SPACE or Click to start")
    print("Press ESC to quit")
    
    await game.start(agent=agent)

if __name__ == "__main__":
    asyncio.run(main())
