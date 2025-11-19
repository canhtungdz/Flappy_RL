import asyncio
import os
import sys
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from FlapPyBird.src.flappy import Flappy
from agent.random_agent import RandomAgent
from agent.rulebase_agent import RuleBaseAgent, AdvancedRuleBaseAgent
from agent.dqn_agent import DQNAgent


async def test_agent(agent, name, num_episodes=10):
    """Test một agent với nhiều episodes."""
    print(f"\n{'=' * 50}")
    print(f"Testing {name}")
    print(f"{'=' * 50}")
    
    scores = []
    
    for episode in range(num_episodes):
        game = Flappy()
        score = await game.start(agent=agent)
        scores.append(score)
        print(f"Episode {episode + 1}/{num_episodes}: Score = {score}")
    
    # Statistics
    avg_score = np.mean(scores)
    std_score = np.std(scores)
    min_score = np.min(scores)
    max_score = np.max(scores)
    
    print(f"\n{name} Results:")
    print(f"  Average Score: {avg_score:.2f}")
    print(f"  Std Dev: {std_score:.2f}")
    print(f"  Min Score: {min_score}")
    print(f"  Max Score: {max_score}")
    
    return {
        'name': name,
        'avg': avg_score,
        'std': std_score,
        'min': min_score,
        'max': max_score
    }


async def main():
    num_episodes = 10
    results = []
    
    # Test Random Agent
    print("\n" + "=" * 70)
    print("COMPARING AGENTS")
    print("=" * 70)
    
    # 1. Random Agent
    random_agent = RandomAgent(flap_prob=0.10)
    result = await test_agent(random_agent, "Random Agent", num_episodes)
    results.append(result)
    
    # 2. Rule-based Agent
    rule_agent = RuleBaseAgent()
    result = await test_agent(rule_agent, "Rule-based Agent", num_episodes)
    results.append(result)
    
    # 3. Advanced Rule-based Agent
    adv_rule_agent = AdvancedRuleBaseAgent()
    result = await test_agent(adv_rule_agent, "Advanced Rule-based Agent", num_episodes)
    results.append(result)
    
    # 4. DQN Agent (Best Model)
    dqn_agent = DQNAgent()
    if os.path.exists("checkpoints/best_model.pth"):
        dqn_agent.load("checkpoints/best_model.pth")
        result = await test_agent(dqn_agent, "DQN Agent (Best)", num_episodes)
        results.append(result)
    
    # 5. DQN Agent (Final Model)
    dqn_agent_final = DQNAgent()
    if os.path.exists("checkpoints/final_model.pth"):
        dqn_agent_final.load("checkpoints/final_model.pth")
        result = await test_agent(dqn_agent_final, "DQN Agent (Final)", num_episodes)
        results.append(result)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Agent':<30} {'Avg Score':<12} {'Max Score':<12} {'Std Dev':<12}")
    print("-" * 70)
    
    for r in results:
        print(f"{r['name']:<30} {r['avg']:<12.2f} {r['max']:<12} {r['std']:<12.2f}")
    
    print("=" * 70)
    
    # Find best agent
    best = max(results, key=lambda x: x['avg'])
    print(f"\n🏆 Best Agent: {best['name']} with avg score {best['avg']:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
