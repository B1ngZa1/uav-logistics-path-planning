"""
基线对比实验
项目：基于深度强化学习的多无人机协同物流配送路径规划
"""

import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class RandomBaseline:
    """随机策略基线"""

    def __init__(self, action_dim=3, a_max=3.0):
        self.action_dim = action_dim
        self.a_max = a_max

    def select_action(self, state):
        return np.random.uniform(-self.a_max, self.a_max, size=self.action_dim)


class GreedyBaseline:
    """贪心策略基线"""

    def __init__(self, a_max=3.0):
        self.a_max = a_max

    def select_action(self, state):
        # 朝目标方向加速
        target = state[8:11]  # 目标位置
        direction = target / (np.linalg.norm(target) + 1e-8)
        return direction * self.a_max


def run_baseline_comparison(num_episodes=100):
    """
    基线对比实验

    对比算法：
    1. Random：随机策略
    2. Greedy：贪心策略
    3. PPO（ours）：本文方法
    """
    print("=" * 60)
    print("基线对比实验")
    print("=" * 60)

    baselines = [
        ('Random', RandomBaseline()),
        ('Greedy', GreedyBaseline()),
        ('PPO (Ours)', None),  # 需要加载训练好的模型
    ]

    results = {}
    for name, baseline in baselines:
        print(f"\n测试算法：{name}")
        print("-" * 40)

        # TODO: 运行测试
        # result = test_algorithm(baseline, num_episodes)
        # results[name] = result

        print("待实现...")

    return results


if __name__ == '__main__':
    run_baseline_comparison()
