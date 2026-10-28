#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
无人机配送 — Q-learning算法模块
Drone Delivery — Q-learning Algorithm Module

职责 / Responsibility:
    实现核心Q-learning学习算法，包括Q值迭代更新和epsilon衰减。
    Implements the core Q-learning algorithm, including Q-value iterative
    update and epsilon decay.

参考 / Reference:
    kaiwu_gorge_walk_v2/agent_q_learning/algorithm/algorithm.py
    agent_sarsa/algorithm/algorithm.py (SparseQTable & Algorithm patterns)

领域适配 / Domain Adaptation:
    Gorge Walk treasure collection → Drone pickup-delivery:
    - hero_pos → drone_pos (x,y grid)
    - treasure → delivery_order
    - treasure_state (10-bit binary) → order_status_mask (5-bit)
    - endpoint → depot at [30,10]
    - max_step=1000 → max_steps=500
    - treasure_score=100 → delivery_reward=+100
    - end_score=150 → safe_return_bonus=+500

Q-learning 更新公式 (Off-Policy / 异策略/离线策略):
    Q(s,a) += lr * [ r + gamma * max_a' Q(s',a') * (1 - done) - Q(s,a) ]

与SARSA的区别 (Difference from SARSA):
    - Q-learning: 使用 max_a' Q(s',a') — 最大Q值 (异策略)
      Uses max_a' Q(s',a') — maximum Q-value (off-policy)
    - SARSA: 使用 Q(s', a') — 实际下一个动作的Q值 (同策略)
      Uses Q(s', a') — Q-value of actual next action (on-policy)
"""

import numpy as np
from typing import List, Dict, Any, Union

from agent_q_learning.conf.conf import Config
from agent_q_learning.model.model import SparseQTable


class Algorithm:
    """
    Q-learning算法 — 异策略时序差分控制 (Off-Policy TD Control).

    核心思想 / Core idea:
        通过迭代更新Q值来学习最优策略，使用下一状态的最大Q值作为自举目标。
        Learns optimal policy by iteratively updating Q-values, using the max Q-value
        of the next state as the bootstrapping target.

    Q-learning更新公式 / Q-learning update formula:
        Q(s,a) := Q(s,a) + lr * [ r + γ * max_a' Q(s',a') * (1 - done) - Q(s,a) ]

    其中 / Where:
        - Q(s,a): 当前状态s下执行动作a的Q值 / Q-value for taking action a in state s
        - lr:     学习率 / Learning rate
        - r:      即时奖励 / Immediate reward
        - γ:      折扣因子 / Discount factor (gamma)
        - s':     下一状态 / Next state
        - a':     下一状态中的动作 / Action in next state
        - done:   是否终止 (terminated or truncated)
    """

    def __init__(self,
                 gamma: float = None,
                 learning_rate: float = None,
                 action_size: int = None):
        """
        初始化Q-learning算法。
        Initialize Q-learning algorithm.

        Args:
            gamma: 折扣因子 / Discount factor (default: Config.GAMMA)
            learning_rate: 学习率 / Learning rate (default: Config.LEARNING_RATE)
            action_size: 动作空间大小 / Action space size (default: Config.ACTION_SIZE)
        """
        self.gamma = gamma if gamma is not None else Config.GAMMA
        self.learning_rate = learning_rate if learning_rate is not None else Config.LEARNING_RATE
        self.action_size = action_size if action_size is not None else Config.ACTION_SIZE

        # 探索率 (由Agent管理，Algorithm中仅为引用)
        # Exploration rate (managed by Agent; Algorithm only references it)
        self.epsilon = Config.EPSILON_START
        self.epsilon_min = Config.EPSILON_MIN
        self.epsilon_decay = Config.EPSILON_DECAY

        # 累计训练步数 (更新次数)
        # Cumulative training steps (update count)
        self.train_steps: int = 0

        # 初始化稀疏Q表 — 乐观初始化为0 (鼓励探索)
        # Initialize sparse Q-table — optimistic init to 0
        self.Q: SparseQTable = SparseQTable(action_size=self.action_size, default_q=0.0)

    # ============================================================
    # 学习 (Learning)
    # ============================================================

    def learn(self, list_sample_data: List[Any]) -> Dict[str, float]:
        """
        使用Q-learning算法更新Q表。
        Update the Q-table using Q-learning algorithm.

        遍历所有样本，对每个样本执行Q-learning更新:
        Iterates over all samples, performing Q-learning update for each:

            Q(s,a) += lr * [ r + gamma * max_a' Q(s',a') * (1-done) - Q(s,a) ]

        Args:
            list_sample_data: 样本数据列表，每个样本可以是:
                List of sample data; each sample can be:
                - SampleData 对象 / SampleData object
                - dict with keys: state, action, reward, next_state, done

        Returns:
            学习指标字典，包含:
            Learning metrics dict containing:
                - td_error: 平均TD误差 / Average TD error
                - lr: 当前学习率 / Current learning rate
                - epsilon: 当前探索率 / Current epsilon
        """
        total_td_error = 0.0
        num_samples = 0

        for sample in list_sample_data:
            if sample is None:
                continue

            # 提取样本数据 / Extract sample data
            state_key, action, reward, next_key, done = self._extract_sample(sample)

            if state_key is None or action is None:
                continue

            # ========================================================
            # Q-learning 核心更新 (Off-Policy / 异策略)
            # Q-learning core update
            # ========================================================

            # 当前Q值 / Current Q-value
            current_q = float(self.Q[state_key, action])

            if done:
                # 终止状态: 没有未来奖励, 目标 = 即时奖励
                # Terminal state: no future rewards, target = immediate reward
                target = float(reward)
            else:
                # 非终止状态: 使用 max Q(s', a') 作为自举目标 (异策略)
                # Non-terminal: use max Q(s', a') as bootstrap target (off-policy)
                max_next_q = self.Q.get_max_q(next_key)
                target = float(reward) + self.gamma * max_next_q

            # TD误差 / TD error: delta = target - Q(s,a)
            td_error = target - current_q

            # 应用更新 / Apply update:
            # Q(s,a) += lr * [r + gamma * max Q(s',a') * (1-done) - Q(s,a)]
            new_q = current_q + self.learning_rate * td_error
            self.Q[state_key, action] = new_q

            total_td_error += abs(td_error)
            num_samples += 1
            self.train_steps += 1

        # 衰减探索率 / Decay epsilon after processing episode
        self._decay_epsilon()

        # 计算并返回指标 / Compute and return metrics
        avg_td_error = total_td_error / max(num_samples, 1)

        return {
            "td_error": float(avg_td_error),
            "lr": float(self.learning_rate),
            "epsilon": float(self.epsilon),
            "train_steps": int(self.train_steps),
            "q_table_size": len(self.Q),
            "num_samples": int(num_samples),
        }

    # ============================================================
    # 辅助方法 (Helper methods)
    # ============================================================

    def _extract_sample(self, sample: Any) -> tuple:
        """
        从样本中提取 (state, action, reward, next_state, done) 元组。
        Extract (state, action, reward, next_state, done) tuple from a sample.

        支持 SampleData 对象和字典两种格式。
        Supports both SampleData objects and dict formats.

        Args:
            sample: 样本对象 (SampleData 或 dict) / Sample object

        Returns:
            (state_key, action, reward, next_state, done) 元组
        """
        # SampleData对象 / SampleData object
        if hasattr(sample, "obs") and hasattr(sample, "act"):
            return (
                int(sample.obs) if sample.obs is not None else None,
                int(sample.act) if sample.act is not None else None,
                float(getattr(sample, "rew", 0.0)),
                int(sample._obs) if sample._obs is not None else None,
                bool(getattr(sample, "done", False)),
            )
        # 字典格式 / Dict format
        elif isinstance(sample, dict):
            return (
                int(sample.get("state")) if sample.get("state") is not None else None,
                int(sample.get("action")) if sample.get("action") is not None else None,
                float(sample.get("reward", 0.0)),
                int(sample.get("next_state")) if sample.get("next_state") is not None else None,
                bool(sample.get("done", False)),
            )
        # 未知格式，尝试属性访问 / Unknown format, try attribute access
        else:
            return (
                int(getattr(sample, "state", None)) if getattr(sample, "state", None) is not None else None,
                int(getattr(sample, "action", None)) if getattr(sample, "action", None) is not None else None,
                float(getattr(sample, "reward", 0.0)),
                int(getattr(sample, "next_state", None)) if getattr(sample, "next_state", None) is not None else None,
                bool(getattr(sample, "done", False)),
            )

    def _decay_epsilon(self) -> None:
        """
        衰减探索率 (指数衰减)。
        Decay exploration rate (exponential decay).

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def reset(self) -> None:
        """
        重置算法状态 (用于新训练会话)。
        Reset algorithm state (for a new training session).
        """
        self.epsilon = Config.EPSILON_START
        self.train_steps = 0
        self.Q = SparseQTable(action_size=self.action_size, default_q=0.0)

    def __repr__(self) -> str:
        return (f"Algorithm(gamma={self.gamma}, lr={self.learning_rate}, "
                f"epsilon={self.epsilon:.4f}, Q_states={len(self.Q)}, "
                f"train_steps={self.train_steps})")
