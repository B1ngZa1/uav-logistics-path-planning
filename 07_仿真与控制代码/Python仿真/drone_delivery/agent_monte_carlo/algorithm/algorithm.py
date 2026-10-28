#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright (c) 2026 Drone Delivery RL Project. All Rights Reserved.
###########################################################################
"""
Monte Carlo Algorithm Implementation for Drone Food Delivery.

蒙特卡洛算法实现，用于无人机外卖配送任务。
参考实现：Kaiwu Gorge Walk v2 agent_monte_carlo/algorithm/algorithm.py

Monte Carlo (MC) is a model-free RL method that learns from complete episodes.
蒙特卡洛是一种无模型强化学习方法，从完整回合中学习。

Key distinction from TD methods (Q-learning/SARSA):
与TD方法（Q-learning/SARSA）的关键区别:
    TD update (every step, bootstrapping / 每步更新，自举):
        Q(s,a) += lr * [r + gamma * V(s') - Q(s,a)]

    MC update (episode end, full return / 回合结束，完整回报):
        Q(s,a) += lr * [G_t - Q(s,a)]

    where G_t = r_t + gamma*r_{t+1} + gamma^2*r_{t+2} + ... + gamma^{T-t}*r_T
    其中 G_t 是从时刻t到回合结束的累积折扣回报。

First-Visit Monte Carlo (首次访问蒙特卡洛):
    For each episode, only the FIRST occurrence of each (s,a) pair is used
    for updating. Subsequent visits to the same (s,a) in the same episode
    are ignored. This reduces bias in the return estimate.
    对于每个回合，只使用每个(s,a)对在该回合中的首次出现进行更新。
    同一回合中后续对同一(s,a)的访问被忽略。这减少了回报估计的偏差。

This module contains:
    - Algorithm: Core first-visit Monte Carlo learning algorithm

Domain mapping from Gorge Walk v2 to Drone Delivery:
    hero_pos       → drone_pos         (无人机网格位置)
    treasure/sub_type==1 → orders      (订单)
    organ status   → order_status_mask (订单完成状态)
    endpoint       → depot             (配送中心)
    start          → depot             (配送中心)
    4-direction    → same 4-direction  (相同4方向)
    treasure_score → delivery_reward   (配送奖励=+100)
    max_step=1000  → max_steps=500     (最大步数)
"""


import numpy as np
from typing import List, Dict, Any, Set, Tuple


class Algorithm:
    """
    First-Visit Monte Carlo algorithm for on-policy control.

    首次访问蒙特卡洛同策略控制算法。

    回合更新（Episode Update）:
    蒙特卡洛方法在完整回合结束后才进行学习。它使用实际获得的完整回报
    G_t 来更新Q值，而不像TD方法那样依赖自举（bootstrapping）。
    MC learns only after a complete episode finishes. It uses the actual
    full return G_t to update Q-values, without bootstrapping.

    First-Visit MC Update Formula (首次访问MC更新公式):
        For each (s_t, a_t) appearing for the FIRST time at timestep t:
            G_t = sum_{k=t}^{T} gamma^{k-t} * r_k  (cumulative discounted return)
            Q(s_t, a_t) += learning_rate * (G_t - Q(s_t, a_t))

    Why first-visit? (为什么使用首次访问？)
        - Reduces bias: multiple visits to the same (s,a) in one episode
          are correlated, and using all would bias the estimate.
          减少偏差：同一回合中多次访问同一个(s,a)是相关的，使用全部会引入偏差。
        - More data-efficient than every-visit in practice.
          在实践中比每次访问更数据高效。

    Why lower learning rate? (为什么使用更低的学习率？)
        - G_t has HIGH variance (sum of many random variables).
          G_t的方差很大（多个随机变量之和）。
        - TD targets have lower variance (only 1 step of noise).
          TD目标的方差较小（仅1步噪声）。
        - Therefore MC uses lr=0.01 while Q-learning uses lr=0.1.
          因此MC使用lr=0.01而Q-learning使用lr=0.1。
    """

    def __init__(self, gamma: float, learning_rate: float,
                 state_size: int, action_size: int):
        """
        Initialize First-Visit Monte Carlo algorithm.

        初始化首次访问蒙特卡洛算法。

        Args:
            gamma: Discount factor (0-1) / 折扣因子 (0-1)
            learning_rate: Learning rate for Q-value updates / Q值更新的学习率
            state_size: Total number of possible states / 可能状态的总数
            action_size: Number of discrete actions / 离散动作数量
        """
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.state_size = state_size
        self.action_size = action_size

        # Sparse Q-table: {state_id: np.array([Q(s,0), Q(s,1), Q(s,2), Q(s,3)])}
        # 稀疏Q表：{状态ID: np.array([Q(s,0), Q(s,1), Q(s,2), Q(s,3)])}
        self.Q: Dict[int, np.ndarray] = {}

        # Visit counter for incremental mean updates (optional tracking)
        # 访问计数器用于递增均值更新（可选追踪）
        self.visit_counts: Dict[int, np.ndarray] = {}

        # Training step counter
        # 训练步数计数器
        self.train_step: int = 0

    def _ensure_state(self, state: int) -> np.ndarray:
        """
        Ensure Q-values exist for a state. Lazily creates row if first visit.

        确保状态有Q值。如果是首次访问，惰性创建行。

        MC uses optimistic initialization (0.0) to avoid bias.
        Unlike Q-learning/SARSA which use 1.0 (optimistic), MC uses 0.0
        because G_t values can be any magnitude and 1.0 could skew learning.
        MC使用中性初始化(0.0)以避免偏差。与Q-learning/SARSA使用1.0(乐观)不同，
        MC使用0.0因为G_t值可以是任意大小，1.0会扭曲学习。

        Args:
            state: State key (integer) / 状态键（整数）

        Returns:
            Q-value array for the state / 该状态的Q值数组
        """
        state = int(state)
        if state not in self.Q:
            self.Q[state] = np.zeros(self.action_size, dtype=np.float64)
            self.visit_counts[state] = np.zeros(self.action_size, dtype=np.int32)
        return self.Q[state]

    def _compute_returns(self, samples: List[Dict[str, Any]]) -> List[float]:
        """
        Compute cumulative discounted return G_t for each timestep.

        为每个时间步计算累积折扣回报 G_t。

        G_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ... + gamma^{T-t} * r_T

        Computation is done backwards for O(T) efficiency:
        G_T = r_T
        G_{T-1} = r_{T-1} + gamma * G_T
        G_{T-2} = r_{T-2} + gamma * G_{T-1}
        ...

        Args:
            samples: List of sample dicts, each with keys:
                     state, action, reward, next_state, done
                     样本字典列表，每个包含键：state, action, reward, next_state, done

        Returns:
            List of G_t values, one per sample / G_t值列表，每个样本一个
        """
        T = len(samples)
        returns = [0.0] * T

        # Backward computation: G_T = r_T, then accumulate backwards
        # 反向计算：G_T = r_T，然后向后累积
        cumulative = 0.0
        for t in range(T - 1, -1, -1):
            reward = float(samples[t].get("reward", 0.0))
            cumulative = reward + self.gamma * cumulative
            returns[t] = cumulative

        return returns

    def learn(self, list_sample_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Update Q-table using First-Visit Monte Carlo with all episode samples.

        使用首次访问蒙特卡洛方法，使用所有回合样本更新Q表。

        这是MC与TD方法(如Q-learning/SARSA)的根本区别：
        - TD方法每步学习一次（增量式更新）
        - MC方法在回合结束时一次性学习整个回合（批量更新）
        This is the fundamental difference from TD methods:
        - TD methods learn once per step (incremental update)
        - MC methods learn the entire episode at once at episode end (batch update)

        First-Visit Logic (首次访问逻辑):
            1. Compute G_t for each timestep t (cumulative discounted return)
               为每个时间步t计算G_t（累积折扣回报）
            2. Iterate through samples chronologically
               按时间顺序遍历样本
            3. For each (s,a) pair: update ONLY if this is the FIRST time
               (s,a) appears in this episode
               对于每个(s,a)对：仅当这是该(s,a)对在此回合中的首次出现时才更新
            4. Q(s,a) += lr * (G_t - Q(s,a))
               使用学习率更新Q值

        Args:
            list_sample_data: List of sample dicts for the ENTIRE episode.
                Each dict has keys:
                    - state (int): Current state key / 当前状态键
                    - action (int): Action taken / 执行的动作
                    - reward (float): Reward received / 获得的奖励
                    - next_state (int): Next state key / 下一个状态键
                    - done (bool): Whether episode ended / 回合是否结束
                整个回合的样本字典列表。
                每个字典包含键：state, action, reward, next_state, done

        Returns:
            Dictionary with learning statistics:
            包含学习统计信息的字典：
                total_error: Sum of absolute TD errors / 绝对TD误差之和
                num_updates: Number of (s,a) pairs updated / 更新的(s,a)对数量
                lr: Current learning rate / 当前学习率
                mean_return: Mean G_t over all steps / 所有步骤的平均G_t
        """
        if not list_sample_data:
            return {
                "total_error": 0.0,
                "num_updates": 0,
                "lr": float(self.learning_rate),
                "mean_return": 0.0,
            }

        T = len(list_sample_data)

        # Step 1: Compute G_t for all timesteps (backward pass)
        # 步骤1：为所有时间步计算G_t（反向计算）
        returns = self._compute_returns(list_sample_data)

        # Step 2: First-visit MC update (forward pass)
        # 步骤2：首次访问MC更新（正向遍历）
        # Track which (s,a) pairs have been seen in this episode
        # 记录此回合中已经见过的(s,a)对
        visited_pairs: Set[Tuple[int, int]] = set()

        total_error = 0.0
        num_updates = 0

        for t, sample in enumerate(list_sample_data):
            state = int(sample.get("state", 0))
            action = int(sample.get("action", 0))
            G_t = returns[t]

            # First-visit check: skip if (s,a) already appeared in this episode
            # 首次访问检查：如果(s,a)在此回合中已经出现过，则跳过
            pair = (state, action)
            if pair in visited_pairs:
                continue

            # Mark this (s,a) pair as visited for this episode
            # 标记此(s,a)对在此回合中已访问
            visited_pairs.add(pair)

            # Ensure Q-values exist for this state
            # 确保此状态有Q值
            q_values = self._ensure_state(state)

            # Monte Carlo update: Q(s,a) += lr * (G_t - Q(s,a))
            # 蒙特卡洛更新：Q(s,a) += lr * (G_t - Q(s,a))
            old_q = q_values[action]
            delta = G_t - old_q
            q_values[action] += self.learning_rate * delta

            # Track visit count (for statistics only)
            # 追踪访问计数（仅用于统计）
            self.visit_counts[state][action] += 1

            total_error += abs(delta)
            num_updates += 1

        self.train_step += 1

        # Compute mean return for monitoring
        # 计算平均回报用于监控
        mean_return = float(np.mean(returns)) if returns else 0.0

        return {
            "total_error": float(total_error),
            "num_updates": num_updates,
            "lr": float(self.learning_rate),
            "mean_return": float(mean_return),
        }

    def get_q_value(self, state: int, action: int) -> float:
        """
        Get Q-value for a specific state-action pair.

        获取指定状态-动作对的Q值。

        Args:
            state: State key / 状态键
            action: Action index / 动作索引

        Returns:
            Q-value / Q值
        """
        state = int(state)
        action = int(action)
        if state in self.Q:
            return float(self.Q[state][action])
        return 0.0

    def get_q_values(self, state: int) -> np.ndarray:
        """
        Get all Q-values for a state.

        获取状态的所有Q值。

        Args:
            state: State key / 状态键

        Returns:
            Array of Q-values, shape (action_size,) / Q值数组，形状(action_size,)
        """
        state = int(state)
        if state in self.Q:
            return self.Q[state].copy()
        return np.zeros(self.action_size, dtype=np.float64)

    def get_best_action(self, state: int, legal_actions: List[int]) -> int:
        """
        Get the best action for a state from legal actions (greedy policy).

        从合法动作中获取状态的最佳动作（贪心策略）。

        Breaks ties randomly to avoid always selecting the first action.
        随机打破平局以避免总是选择第一个动作。

        Args:
            state: State key / 状态键
            legal_actions: List of legal action indices / 合法动作索引列表

        Returns:
            Best action index / 最佳动作索引
        """
        if not legal_actions:
            return 0

        q_values = self.get_q_values(state)
        legal_q = [(a, q_values[a]) for a in legal_actions]
        max_q = max(v for _, v in legal_q)
        best_actions = [a for a, v in legal_q if v == max_q]

        # Random tie-breaking
        # 随机打破平局
        return int(np.random.choice(best_actions))

    def num_visited_states(self) -> int:
        """
        Return the number of states that have been visited.

        返回已访问状态的数量。

        Returns:
            Number of visited states / 已访问状态数量
        """
        return len(self.Q)

    def to_serializable(self) -> Dict[str, Any]:
        """
        Convert Q-table to a serializable dictionary for saving.

        将Q表转换为可序列化的字典用于保存。

        Returns:
            Serializable dictionary / 可序列化的字典
        """
        return {
            "action_size": self.action_size,
            "state_size": self.state_size,
            "gamma": self.gamma,
            "learning_rate": self.learning_rate,
            "train_step": self.train_step,
            "q_table": {int(s): v.tolist() for s, v in self.Q.items()},
            "visit_counts": {int(s): v.tolist() for s, v in self.visit_counts.items()},
        }

    @classmethod
    def from_serializable(cls, payload: Dict[str, Any],
                          gamma: float = None,
                          learning_rate: float = None) -> "Algorithm":
        """
        Restore Algorithm from a serialized dictionary.

        从序列化字典恢复Algorithm。

        Args:
            payload: Serialized data dict / 序列化的数据字典
            gamma: Discount factor override / 折扣因子覆盖
            learning_rate: Learning rate override / 学习率覆盖

        Returns:
            Restored Algorithm instance / 恢复的Algorithm实例
        """
        action_size = payload.get("action_size", 4)
        state_size = payload.get("state_size", 102400)

        algo = cls(
            gamma=gamma if gamma is not None else payload.get("gamma", 0.99),
            learning_rate=learning_rate if learning_rate is not None else payload.get("learning_rate", 0.01),
            state_size=state_size,
            action_size=action_size,
        )

        algo.train_step = payload.get("train_step", 0)

        # Restore Q-table
        # 恢复Q表
        for state_str, q_list in payload.get("q_table", {}).items():
            state = int(state_str)
            algo.Q[state] = np.array(q_list, dtype=np.float64)

        # Restore visit counts
        # 恢复访问计数
        for state_str, v_list in payload.get("visit_counts", {}).items():
            state = int(state_str)
            algo.visit_counts[state] = np.array(v_list, dtype=np.int32)

        return algo
