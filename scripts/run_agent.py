# scripts/run_agent.py
import asyncio
import os
import sys

# Thêm thư mục gốc (flappy_rl) vào sys.path để import được
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

# Tuỳ project của bạn, import Flappy cho đúng:
# Nếu package name là FlapPyBird.src.flappy:
from FlapPyBird.src.flappy import Flappy
from agent.random_agent import RandomAgent
from agent.rulebase_agent import RuleBaseAgent
from agent.rulebase_agent import AdvancedRuleBaseAgent
async def main():
    game = Flappy()
    rand_agent = RandomAgent(flap_prob=0.10)  # điều chỉnh nhẹ nếu nhảy điên quá
    rulebase_agent = RuleBaseAgent()
    advrulebase_agent = AdvancedRuleBaseAgent()
    await game.start(agent=advrulebase_agent)

if __name__ == "__main__":
    asyncio.run(main())
