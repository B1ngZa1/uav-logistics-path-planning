#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
无人机配送 — Q-learning模型模块
Drone Delivery — Q-learning Model Module

职责 / Responsibility:
    提供稀疏Q表(SparseQTable)，用于表格型Q-learning的状态-动作价值存储。
    Provides sparse Q-table (SparseQTable) for tabular Q-learning state-action
    value storage.

参考 / Reference:
    kaiwu_gorge_walk_v2/agent_q_learning/model/model.py
    agent_sarsa/algorithm/algorithm.py (SparseQTable pattern)
    agent_monte_carlo/model/model.py (SparseQTable pattern)

领域适配 / Domain Adaptation:
    Gorge Walk treasure collection → Drone pickup-delivery:
    - hero_pos → drone_pos (x,y grid)
    - treasure → delivery_order
    - treasure_state (10-bit binary) → order_status_mask (5-bit)
    - endpoint → depot at [30,10]
    - 4-direction (UP/DOWN/LEFT/RIGHT) → same
    - max_step=1000 → max_steps=500
    - treasure_score=100 → delivery_reward=+100
    - end_score=150 → safe_return_bonus=+500
"""

import numpy as np
import os
from typing import Dict, Optional


class SparseQTable:
    """
    稀疏Q表 — 用于表格型Q-learning的内存高效状态-动作价值存储。
    Sparse Q-table — memory-efficient state-action value storage for tabular Q-learning.

    无人机配送状态空间 (Drone delivery state space):
        position(40x40=1600) * order_status(2^5=32) * carrying(2) = 102,400 states
    使用字典仅为实际访问过的状态分配内存，而非预分配整个密集数组。
    Uses a dictionary to allocate memory only for actually-visited states,
    rather than pre-allocating the entire dense array.

    Usage / 用法:
        table = SparseQTable(action_size=4)
        q_vals = table[state_key]       # np.zeros(4) if not visited
        best_action = table.get_max_action(state_key)
        table.update(state_key, 0, 5.0) # set Q(s, a=0) = 5.0
    """

    def __init__(self, action_size: int = 4, default_q: float = 0.0):
        """
        初始化稀疏Q表 / Initialize sparse Q-table.

        Args:
            action_size: 动作空间大小 / Number of discrete actions
            default_q: 未访问状态的默认Q值 / Default Q-value for unvisited states
        """
        self.action_size = action_size
        self.default_q = default_q

        # 核心存储: 状态键 → Q值向量
        # Core storage: state key → Q-value vector
        self.table: Dict[int, np.ndarray] = {}

        # 累计训练步数 (更新次数)
        # Cumulative training steps (update count)
        self.train_steps: int = 0

    # ============================================================
    # 访问与索引 (Access and indexing)
    # ============================================================

    def _get_row(self, state_key: int) -> np.ndarray:
        """
        获取或创建指定状态的Q值行。
        Get or create the Q-value row for a given state.

        惰性初始化: 首次访问时创建零向量。
        Lazy initialization: creates zero vector on first access.

        Args:
            state_key: 整数状态键 / Integer state key

        Returns:
            形状为(action_size,)的Q值数组 / Q-value array of shape (action_size,)
        """
        state_key = int(state_key)
        if state_key not in self.table:
            self.table[state_key] = np.zeros(self.action_size, dtype=np.float32)
        return self.table[state_key]

    def __getitem__(self, key):
        """
        获取Q值，支持多种索引方式。
        Get Q-value(s) with flexible indexing.

        支持 / Supports:
            table[state_key]            → 整行 / full row, np.zeros(4) if not visited
            table[state_key, action]    → 标量 / scalar Q-value
            table[state_key, :]         → 整行 (切片) / full row (slice)

        Args:
            key: 整数状态键 或 (state, action) 元组

        Returns:
            Q值 / Q-value(s)
        """
        if isinstance(key, tuple):
            state_key, action = key
            row = self._get_row(state_key)
            if isinstance(action, slice):
                return row[action]
            return float(row[int(action)])
        return self._get_row(key)

    def __setitem__(self, key, value):
        """
        设置Q值，支持多种索引方式。
        Set Q-value(s) with flexible indexing.

        Args:
            key: (state, action) 或 (state, slice)
            value: 新Q值 / New Q-value
        """
        if isinstance(key, tuple):
            state_key, action = key
            row = self._get_row(state_key)
            if isinstance(action, slice):
                row[action] = value
            else:
                row[int(action)] = float(value)
        else:
            self.table[int(key)] = np.asarray(value, dtype=np.float32)

    def __len__(self) -> int:
        """返回已访问状态数量 / Return number of visited states."""
        return len(self.table)

    # ============================================================
    # 贪婪策略辅助 (Greedy policy helpers)
    # ============================================================

    def get_max_action(self, state_key: int, legal_actions: Optional[list] = None) -> int:
        """
        获取指定状态下Q值最大的动作 (用于贪婪/利用)。
        Get the action with maximum Q-value for a given state (for greedy/exploitation).

        当有多个最大值时随机打破平局，避免系统性偏向第一个动作。
        Randomly breaks ties when multiple actions share the maximum, avoiding
        systematic bias toward the first action.

        Args:
            state_key: 整数状态键 / Integer state key
            legal_actions: 合法动作列表 (可选) / Legal action list (optional)

        Returns:
            最佳动作索引 / Best action index
        """
        q_values = self._get_row(state_key).copy()
        if legal_actions is not None:
            # 掩码非法动作为负无穷 / Mask illegal actions to -inf
            for a in range(self.action_size):
                if a not in legal_actions:
                    q_values[a] = -np.inf
        max_indices = np.flatnonzero(q_values == q_values.max())
        return int(np.random.choice(max_indices))

    def get_max_q(self, state_key: int, legal_actions: Optional[list] = None) -> float:
        """
        获取指定状态下的最大Q值。
        Get the maximum Q-value for a given state.

        Args:
            state_key: 整数状态键 / Integer state key
            legal_actions: 合法动作列表 (可选) / Legal action list (optional)

        Returns:
            最大Q值 / Maximum Q-value
        """
        q_values = self._get_row(state_key).copy()
        if legal_actions is not None:
            for a in range(self.action_size):
                if a not in legal_actions:
                    q_values[a] = -np.inf
        if np.all(np.isneginf(q_values)):
            return float(self.default_q)
        return float(np.max(q_values))

    # ============================================================
    # 增量更新 (Incremental update)
    # ============================================================

    def update(self, state_key: int, action: int, value: float) -> float:
        """
        更新指定状态-动作对的Q值 (直接赋值)。
        Update Q-value for a state-action pair (direct assignment).

        Args:
            state_key: 整数状态键 / Integer state key
            action: 动作索引 / Action index
            value: 新Q值 / New Q-value

        Returns:
            更新变化量的绝对值 / Absolute value of the change
        """
        row = self._get_row(state_key)
        old_val = float(row[int(action)])
        row[int(action)] = float(value)
        self.train_steps += 1
        return float(abs(value - old_val))

    # ============================================================
    # 持久化 (Persistence)
    # ============================================================

    def to_serializable(self) -> dict:
        """
        转换为可序列化字典 (用于保存)。
        Convert to serializable dict (for saving).

        Returns:
            包含 action_size, default_q, table, train_steps 的字典
        """
        return {
            "action_size": self.action_size,
            "default_q": self.default_q,
            "table": {int(k): v.tolist() for k, v in self.table.items()},
            "train_steps": self.train_steps,
        }

    @classmethod
    def from_serializable(cls, payload: dict, action_size: Optional[int] = None,
                          default_q: float = 0.0) -> "SparseQTable":
        """
        从序列化字典恢复Q表 (用于加载)。
        Restore Q-table from a serialized dict (for loading).

        Args:
            payload: 序列化数据字典 / Serialized data dict
            action_size: 覆盖动作空间大小 / Override action size
            default_q: 覆盖默认Q值 / Override default Q-value

        Returns:
            SparseQTable 实例 / SparseQTable instance
        """
        act_size = action_size or int(payload.get("action_size", 4))
        table = cls(action_size=act_size, default_q=default_q)
        table.train_steps = int(payload.get("train_steps", 0))
        for state_str, q_list in payload.get("table", {}).items():
            state_key = int(state_str)
            arr = np.asarray(q_list, dtype=np.float32)
            if arr.shape[0] == act_size:
                table.table[state_key] = arr
        return table

    def save(self, filepath: str) -> None:
        """
        保存Q表到文件。
        Save Q-table to file.

        Args:
            filepath: 文件路径 (推荐 .npy 扩展名) / File path (recommended .npy extension)
        """
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        payload = self.to_serializable()
        np.save(filepath, payload, allow_pickle=True)

    def load(self, filepath: str) -> None:
        """
        从文件加载Q表。
        Load Q-table from file.

        Args:
            filepath: 文件路径 / File path
        """
        payload = np.load(filepath, allow_pickle=True)
        if isinstance(payload, np.ndarray) and payload.shape == ():
            payload = payload.item()
        if isinstance(payload, dict):
            restored = self.from_serializable(payload)
            self.action_size = restored.action_size
            self.default_q = restored.default_q
            self.table = restored.table
            self.train_steps = restored.train_steps
