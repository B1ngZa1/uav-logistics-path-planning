#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
无人机配送 — 蒙特卡洛模型模块
Drone Delivery — Monte Carlo Model Module

职责 / Responsibility:
    管理蒙特卡洛算法的Q表与策略表。
    Manages the Q-table and policy table for Monte Carlo algorithm.

蒙特卡洛特性 / Monte Carlo Characteristics:
    - 使用稀疏Q表(SparseQTable)：状态空间大(102400)，多数状态不会被访问
      Uses sparse Q-table: large state space (102400), most states are never visited
    - 维护访问计数N(s,a)用于递增均值更新
      Maintains visit count N(s,a) for incremental mean updates
    - 不包含神经网络(MC是表格型方法)
      No neural network (MC is a tabular method)

参考 / Reference:
    kaiwu_gorge_walk_v2/agent_monte_carlo/model/model.py
    kaiwu_gorge_walk_v2/agent_q_learning/agent.py (SparseQTable pattern)
"""

import numpy as np
from collections import defaultdict
from typing import Dict, Optional, Tuple


class SparseQTable:
    """
    稀疏Q表：使用字典存储状态-动作价值，避免为未访问状态分配内存
    Sparse Q-table: stores state-action values in a dict, avoiding memory
    allocation for unvisited states.

    蒙特卡洛版本额外维护访问计数 / Monte Carlo version additionally maintains visit counts.

    Key formula:
        Q(s,a) += (G - Q(s,a)) / N(s,a)    -- 递增均值 / incremental mean
        OR
        Q(s,a) += lr * (G - Q(s,a))        -- 固定学习率 / fixed learning rate
    """

    def __init__(self, action_size: int = 4, default_q: float = 0.0):
        """
        初始化稀疏Q表 / Initialize sparse Q-table

        Args:
            action_size: 动作空间大小 / Size of action space
            default_q: 未访问状态的默认Q值 / Default Q-value for unvisited states
        """
        self.action_size = action_size
        self.default_q = default_q

        # Q-values dict: state -> np.array of shape (action_size,)
        # Q值字典: 状态 → 形状为(action_size,)的数组
        self.data: Dict[int, np.ndarray] = {}

        # Visit counts for incremental mean: state -> np.array of shape (action_size,)
        # 访问计数用于递增均值: 状态 → 形状为(action_size,)的数组
        self.visit_counts: Dict[int, np.ndarray] = {}

        # Total training steps (updates performed)
        # 总训练步数（已执行的更新次数）
        self.train_steps: int = 0

    def get_q_values(self, state: int) -> np.ndarray:
        """
        获取指定状态的Q值向量 / Get Q-value vector for a given state

        Args:
            state: 状态键 / State key

        Returns:
            Q值向量，形状为(action_size,) / Q-value vector, shape (action_size,)
        """
        if state not in self.data:
            self.data[state] = np.full(self.action_size, self.default_q, dtype=np.float64)
            self.visit_counts[state] = np.zeros(self.action_size, dtype=np.int32)
        return self.data[state]

    def get_q_value(self, state: int, action: int) -> float:
        """
        获取指定状态-动作对的Q值 / Get Q-value for a state-action pair

        Args:
            state: 状态键 / State key
            action: 动作 / Action

        Returns:
            Q值 / Q-value
        """
        q_values = self.get_q_values(state)
        return float(q_values[action])

    def update(self, state: int, action: int, target: float,
               use_incremental_mean: bool = True, lr: float = 0.01) -> float:
        """
        更新指定状态-动作对的Q值 / Update Q-value for a state-action pair

        蒙特卡洛更新公式 / Monte Carlo update formula:
            递增均值 / Incremental mean:
                N(s,a) += 1
                Q(s,a) += (G - Q(s,a)) / N(s,a)

            固定学习率 / Fixed learning rate:
                Q(s,a) += lr * (G - Q(s,a))

        Args:
            state: 状态键 / State key
            action: 动作 / Action
            target: 目标值G(完整回报) / Target value G (full return)
            use_incremental_mean: 是否使用递增均值 / Whether to use incremental mean
            lr: 学习率（仅use_incremental_mean=False时使用） / Learning rate (only when use_incremental_mean=False)

        Returns:
            更新量(绝对值) / Update magnitude (absolute value)
        """
        q_values = self.get_q_values(state)
        visit = self.visit_counts[state]
        old_q = q_values[action]

        if use_incremental_mean:
            # 递增均值: Q(s,a) += (G - Q(s,a)) / N(s,a)
            # Incremental mean: Q(s,a) += (G - Q(s,a)) / N(s,a)
            visit[action] += 1
            q_values[action] += (target - old_q) / visit[action]
        else:
            # 固定学习率: Q(s,a) += lr * (G - Q(s,a))
            # Fixed learning rate: Q(s,a) += lr * (G - Q(s,a))
            q_values[action] += lr * (target - old_q)

        self.train_steps += 1
        return float(abs(q_values[action] - old_q))

    def get_policy(self, state: int) -> int:
        """
        获取指定状态的贪婪策略(最大Q值对应的动作) / Get greedy policy for a state (argmax Q)

        Args:
            state: 状态键 / State key

        Returns:
            最优动作 / Best action
        """
        q_values = self.get_q_values(state)
        # 处理平局：在最大值中随机选择 / Handle ties: random choice among max values
        max_indices = np.flatnonzero(q_values == q_values.max())
        return int(np.random.choice(max_indices))

    def get_visit_count(self, state: int, action: int) -> int:
        """
        获取指定状态-动作对的访问次数 / Get visit count for a state-action pair

        Args:
            state: 状态键 / State key
            action: 动作 / Action

        Returns:
            访问次数 / Visit count
        """
        if state not in self.visit_counts:
            return 0
        return int(self.visit_counts[state][action])

    def is_state_visited(self, state: int) -> bool:
        """
        检查状态是否曾被访问 / Check if state has ever been visited

        Args:
            state: 状态键 / State key

        Returns:
            是否访问过 / Whether visited
        """
        return state in self.data

    def num_states(self) -> int:
        """
        返回已访问状态数量 / Return number of visited states
        """
        return len(self.data)

    def to_serializable(self) -> dict:
        """
        转换为可序列化格式(用于保存) / Convert to serializable format (for saving)

        Returns:
            可序列化字典 / Serializable dict
        """
        return {
            "action_size": self.action_size,
            "default_q": self.default_q,
            "data": {int(s): v.tolist() for s, v in self.data.items()},
            "visit_counts": {int(s): v.tolist() for s, v in self.visit_counts.items()},
            "train_steps": self.train_steps,
        }

    @classmethod
    def from_serializable(cls, payload: dict, action_size: Optional[int] = None) -> "SparseQTable":
        """
        从可序列化格式加载 / Load from serializable format

        Args:
            payload: 序列化数据 / Serialized data
            action_size: 动作空间大小(覆盖) / Action space size (override)

        Returns:
            SparseQTable实例 / SparseQTable instance
        """
        act_size = action_size or payload.get("action_size", 4)
        obj = cls(action_size=act_size, default_q=payload.get("default_q", 0.0))

        for state_str, q_list in payload.get("data", {}).items():
            state = int(state_str)
            obj.data[state] = np.array(q_list, dtype=np.float64)

        for state_str, v_list in payload.get("visit_counts", {}).items():
            state = int(state_str)
            obj.visit_counts[state] = np.array(v_list, dtype=np.int32)

        obj.train_steps = payload.get("train_steps", 0)
        return obj

    def get_full_q_table(self, state_size: int) -> np.ndarray:
        """
        获取完整的密集Q表(用于兼容旧接口) / Get full dense Q-table (for backward compatibility)

        Args:
            state_size: 总状态数 / Total number of states

        Returns:
            形状为(state_size, action_size)的密集数组 / Dense array of shape (state_size, action_size)
        """
        q_table = np.full((state_size, self.action_size), self.default_q, dtype=np.float64)
        for state, q_values in self.data.items():
            if 0 <= state < state_size:
                q_table[state] = q_values
        return q_table


class SparsePolicyTable:
    """
    稀疏策略表：从Q表中派生出贪心策略
    Sparse policy table: derives greedy policy from Q-table

    与Q-table不同，这里的策略是确定性的(每个状态一个最佳动作)
    Unlike Q-table, this policy is deterministic (one best action per state)
    """

    def __init__(self, action_size: int = 4):
        """
        初始化稀疏策略表 / Initialize sparse policy table

        Args:
            action_size: 动作空间大小 / Size of action space
        """
        self.action_size = action_size
        # policy dict: state -> best_action
        # 策略字典: 状态 → 最佳动作
        self.data: Dict[int, int] = {}

    def get_action(self, state: int) -> int:
        """
        获取指定状态的策略动作 / Get policy action for a state

        Args:
            state: 状态键 / State key

        Returns:
            动作 / Action
        """
        return self.data.get(state, 0)  # Default: first action

    def update(self, state: int, action: int):
        """
        更新状态的策略 / Update policy for a state

        Args:
            state: 状态键 / State key
            action: 最佳动作 / Best action
        """
        self.data[state] = int(action)

    def contains(self, state: int) -> bool:
        """Check if state has a policy entry."""
        return state in self.data
