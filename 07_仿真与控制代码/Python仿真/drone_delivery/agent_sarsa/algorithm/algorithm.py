#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
无人机配送 — SARSA算法模块
Drone Delivery — SARSA Algorithm Module

职责 / Responsibility:
    实现SARSA在线策略时序差分控制算法核心逻辑。
    Implements the core SARSA on-policy TD control algorithm logic.

SARSA关键特性 / SARSA Key Characteristics:
    - 在线策略(On-Policy): 使用当前策略实际选择的a'更新Q(s,a)
      On-Policy: uses the actual a' chosen by current policy to update Q(s,a)
    - 对探索敏感: SARSA的Q值"意识"到epsilon-greedy策略会偶尔选择次优/危险动作，
      因此学到的策略在探索期更保守、更安全
      Exploration-sensitive: SARSA's Q-values "account for" epsilon-greedy
      occasionally choosing suboptimal/risky actions, leading to safer policies

与Q-learning的关键区别 / Key Difference from Q-learning:
    Q-learning (离线策略/异策略):
        Q(s,a) += lr * [r + gamma * max_a' Q(s',a') - Q(s,a)]

    SARSA (在线策略/同策略):
        Q(s,a) += lr * [r + gamma * Q(s',a') * (1-done) - Q(s,a)]

    其中a'是策略实际选择的下一个动作，不是最大Q值动作。
    Where a' is the actual next action chosen by the policy, not max-Q action.

参考 / Reference:
    kaiwu_gorge_walk_v2/agent_sarsa/algorithm/algorithm.py



领域适配 / Domain Adaptation:
    峡谷行走(Gorge Walk) — small 64-state grid
      → 无人机配送(Drone Delivery) — large 102400-state grid + order tracking
"""

import numpy as np
from typing import Dict, List, Any

from agent_sarsa.model.model import SparseQTable


class Algorithm:
    """
    SARSA在线策略时序差分控制算法。
    SARSA on-policy temporal difference control algorithm.

    SARSA更新公式 / SARSA Update Formula:
        Q(s,a) += lr * [R(s,a) + gamma * Q(s',a') * (1-done) - Q(s,a)]

    其中 / Where:
        Q(s,a)  : 在状态s采取动作a的Q值 / Q-value for taking action a in state s
        lr      : 学习率 / Learning rate
        R(s,a)  : 即时奖励 / Immediate reward
        gamma   : 折扣因子 / Discount factor
        Q(s',a'): 下一状态s'中实际执行的动作a'的Q值（在线策略核心）
                  Q-value of the ACTUAL next action a' in next state s'
                  (this is the core of on-policy learning)
        done    : 回合是否终止 / Whether episode ended

    终止状态特殊处理 / Terminal State Special Case:
        当 done=True 时，不存在下一状态和下一动作，更新简化为：
        When done=True, there is no next state or action, update simplifies to:
            Q(s,a) += lr * [R(s,a) - Q(s,a)]
    """

    def __init__(self, gamma: float, learning_rate: float,
                 state_size: int, action_size: int):
        """
        初始化SARSA算法。
        Initialize SARSA algorithm.

        Args:
            gamma: 折扣因子 (0~1) / Discount factor (0~1)
            learning_rate: 学习率 / Learning rate
            state_size: 状态空间总大小 / Total state space size
            action_size: 动作空间大小 / Action space size
        """
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.state_size = state_size
        self.action_size = action_size

        # 使用稀疏Q表，乐观初始化为1.0（鼓励探索未知状态）
        # Use sparse Q-table, optimistic init at 1.0 (encourage exploring unseen states)
        self.Q = SparseQTable(action_size=self.action_size, default_value=1.0)

    def learn(self, list_sample_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        使用SARSA算法更新Q表。
        Update Q-table using the SARSA algorithm.

        SARSA在线策略更新 / SARSA On-Policy Update:
            Q(s,a) += lr * [r + gamma * Q(s',a') * (1-done) - Q(s,a)]

        关键点（Key points）:
        1. 使用 sample["next_action"] 获取策略实际选择的下一个动作a'
           Use sample["next_action"] to get the actual next action a'
        2. done=True时只使用即时奖励，不加上Q(s',a')项
           When done=True, only immediate reward is used, no Q(s',a') term
        3. next_action=-1时视为终止状态，同样只使用即时奖励
           next_action=-1 indicates terminal state, also only uses immediate reward

        Args:
            list_sample_data: 样本字典列表，每个样本包含:
                样本字典列表，每个样本包含:
                - state (int): 当前状态键 / Current state key
                - action (int): 在当前状态采取的动作 / Action taken in current state
                - reward (float): 获得的奖励 / Reward received
                - next_state (int): 下一状态键 / Next state key
                - next_action (int): 在下一状态实际采取的动作 (-1表示终止)
                                     Actual action taken in next state (-1 if terminal)
                - done (bool): 回合是否结束 / Whether episode ended

        Returns:
            学习统计字典 / Learning statistics dict:
                td_error: TD误差的绝对值 / Absolute TD error
                lr: 当前学习率 / Current learning rate
        """
        # 取第一个样本进行单步更新 / Take first sample for single-step update
        sample = list_sample_data[0]

        state = int(sample["state"])
        action = int(sample["action"])
        reward = float(sample["reward"])
        next_state = int(sample.get("next_state", state))
        next_action = sample.get("next_action", -1)
        done = bool(sample.get("done", False))

        # ============================================================
        # SARSA更新 — 在线策略/同策略核心
        # SARSA Update — On-Policy Core
        # ============================================================
        if next_action == -1 or done:
            # 终止状态: 没有下一状态和下一动作，只使用即时奖励
            # Terminal state: no next state or action, use only immediate reward
            # delta = R(s,a) - Q(s,a)
            delta = reward - self.Q[state, action]
        else:
            # 在线策略SARSA: 使用Q(s',a')，其中a'是策略实际选择的下一个动作
            # On-policy SARSA: use Q(s',a') where a' is the actual next action
            next_action = int(next_action)
            # delta = R(s,a) + gamma * Q(s',a') * (1-done) - Q(s,a)
            delta = reward + self.gamma * self.Q[next_state, next_action] - self.Q[state, action]

        # 应用Q值更新 / Apply Q-value update
        # Q(s,a) += lr * delta
        self.Q[state, action] += self.learning_rate * delta

        # 返回TD误差用于监控 / Return TD error for monitoring
        return {
            "td_error": float(abs(delta)),
            "lr": float(self.learning_rate),
        }
