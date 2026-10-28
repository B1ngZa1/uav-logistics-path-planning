# -*- coding: utf-8 -*-
"""
================================================================================
algorithm/algorithm.py — 学生DIY算法 / Student DIY Algorithm
================================================================================
职责 / Responsibility:
    学生自行实现的强化学习算法骨架。
    学生的任务是选择合适的RL算法并完成训练逻辑。
    Skeleton for a reinforcement learning algorithm to be implemented by students.
    Students' task is to choose a suitable RL algorithm and implement the
    training logic.

参考 / Reference:
    - agent_dqn/algorithm/algorithm.py (Double Dueling DQN 完整实现)
    - agent_q_learning/algorithm/algorithm.py (Q-learning 完整实现)
    - agent_monte_carlo/algorithm/algorithm.py (Monte Carlo 完整实现)

领域适配 / Domain Adaptation:
    - 无人机配送: 50维连续观测, 8维离散动作
    - Drone delivery: 50-dim continuous observation, 8 discrete actions

提示 / Tips (可选算法) / Algorithm options:
    - DQN / Double DQN / Dueling DQN (异策略 / off-policy)
    - Policy Gradient / REINFORCE (同策略 / on-policy)
    - Actor-Critic / A2C / PPO (同策略 / on-policy)
    - 或自定义算法 / or custom algorithm
================================================================================
"""

from typing import List, Dict, Any


class Algorithm:
    """
    学生自行实现的算法 / Algorithm to be implemented by students

    提示 / Tips:
        - 应管理模型(网络)、优化器、经验回放缓冲区(如需要)等
        - learn() 是核心方法, 接收样本并更新模型
        - 可额外实现 save_model() / load_model() 支持持久化
        - Should manage model, optimizer, replay buffer (if needed), etc.
        - learn() is the core method — receives samples and updates model
        - May additionally implement save_model() / load_model() for persistence

    Usage / 用法::

        algo = Algorithm()
        stats = algo.learn(list_sample_data)   # 训练一步 / one training step
    """

    def __init__(self):
        """
        初始化算法 — 在此创建模型、优化器、缓冲区等。
        Initialize algorithm — create model, optimizer, buffer, etc. here.

        TODO: 在此初始化算法组件 / Initialize algorithm components here
        建议 / Suggestions:
            - 创建模型实例 / Create model instance: self.model = Model()
            - 创建优化器 / Create optimizer: self.optimizer = torch.optim.Adam(...)
            - 创建经验回放缓冲区 (如需) / Create replay buffer (if needed)
            - 设置超参数 / Set hyperparameters: self.gamma, self.lr, etc.
        """
        pass

    def learn(self, list_sample_data):
        """
        学习 — 从经验样本中更新模型参数。
        Learn — update model parameters from experience samples.

        TODO: 在此实现学习逻辑 / Implement learning logic here
        建议步骤 / Suggested steps:
            1. 从 list_sample_data 中提取 obs, act, rew, next_obs, done
               Extract obs, act, rew, next_obs, done from list_sample_data
            2. 根据选择的算法计算 loss
               Compute loss per chosen algorithm (TD error / policy grad / etc.)
               - DQN: TD error = r + gamma * max Q(s',a') - Q(s,a)
               - Policy Gradient: loss = -log(pi(a|s)) * G_t
               - PPO: clipped surrogate objective
            3. 反向传播并更新参数 / Backpropagate and update parameters
            4. 返回训练统计信息 / Return training statistics

        Args / 参数:
            list_sample_data (List): 样本数据列表 / List of sample data
                每个样本包含: obs, act, rew, next_obs, done

        Returns / 返回:
            dict: 学习统计数据 / Learning statistics
                {
                    "loss": float,       # 损失值 / Loss value
                    "learn_count": int,  # 学习迭代计数 / Learning iteration count
                    # ... 学生可添加更多字段 / students may add more fields
                }
        """
        # TODO: 在此实现学习逻辑 / Implement learning logic here
        pass

    def save_model(self, filepath: str) -> None:
        """
        保存模型权重到文件 / Save model weights to file.

        TODO: 在此实现模型保存 / Implement model saving here

        Args / 参数:
            filepath (str): 保存路径 / Save path
        """
        # TODO: 实现模型保存 / Implement model saving
        # 参考: torch.save(...) 或 np.save(...)
        pass

    def load_model(self, filepath: str) -> None:
        """
        从文件加载模型权重 / Load model weights from file.

        TODO: 在此实现模型加载 / Implement model loading here

        Args / 参数:
            filepath (str): 模型文件路径 / Model file path
        """
        # TODO: 实现模型加载 / Implement model loading
        # 参考: torch.load(...) 或 np.load(...)
        pass
