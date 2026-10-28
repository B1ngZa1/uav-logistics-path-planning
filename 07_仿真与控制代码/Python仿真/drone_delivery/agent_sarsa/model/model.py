#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
无人机配送 — SARSA模型模块
Drone Delivery — SARSA Model Module

职责 / Responsibility:
    管理SARSA算法的稀疏Q表。
    Manages the sparse Q-table for SARSA algorithm.

SARSA的Q表与Q-learning在结构上完全相同（都是Q(s,a)表格），
但SARSA在更新时使用实际执行的下一个动作a'而非max Q。
SARSA's Q-table has the exact same structure as Q-learning (both are Q(s,a) tables),
but SARSA uses the actual next action a' instead of max Q during updates.

参考 / Reference:
    kaiwu_gorge_walk_v2/agent_sarsa/model/model.py



领域适配 / Domain Adaptation:
    峡谷行走(Gorge Walk) — 4方向小型网格
      → 无人机配送(Drone Delivery) — 4方向大型网格 + 订单状态
    Gorge Walk (small grid, 4-direction)
      → Drone Delivery (large grid + order status, same 4-direction)
"""

import numpy as np
from typing import Dict, Optional


class SparseQTable:
    """
    稀疏Q表 — 用于大规模离散状态空间的表格型强化学习。
    Sparse Q-table — for tabular RL with large discrete state spaces.

    无人机配送状态空间 = 40*40*32*2 = 102,400 种状态，
    密集numpy数组(102400×4)使用约3.3MB内存（float32），
    使用稀疏字典仅在实际访问时分配，大幅降低内存使用。
    Drone delivery state space = 40*40*32*2 = 102,400 states,
    dense numpy array (102400×4) uses ~3.3MB (float32),
    sparse dict allocates only on actual visits, drastically reducing memory.

    Attributes:
        action_size: 动作空间大小 / Number of actions
        default_value: 新状态的默认Q值 / Default Q-value for unseen states
        data: dict, state_id → np.ndarray of shape (action_size,) / 状态ID→Q值数组映射
    """

    def __init__(self, action_size: int = 4, default_value: float = 1.0):
        """
        初始化稀疏Q表。
        Initialize sparse Q-table.

        使用1.0进行乐观初始化（encourage exploration / 鼓励探索）。

        Args:
            action_size: 动作数量 / Number of actions
            default_value: 默认Q值 / Default Q-value
        """
        self.action_size = action_size
        self.default_value = default_value
        # 字典: int state_id → np.ndarray of shape (action_size,) / 状态ID到Q值数组的映射
        self.data: Dict[int, np.ndarray] = {}

    def row(self, state: int) -> np.ndarray:
        """
        获取或创建给定状态的Q值行。
        Get or create Q-value row for a given state.

        惰性分配: 仅在状态首次访问时创建新行。
        Lazy allocation: only creates a new row on first visit.

        Args:
            state: 状态键(整数) / State key (integer)

        Returns:
            Q值数组(numpy array), 形状(action_size,) / Q-value array, shape (action_size,)
        """
        state = int(state)
        if state not in self.data:
            # 乐观初始化: 所有Q值初始为default_value (1.0), 鼓励探索
            # Optimistic initialization: all Q-values start at default_value (1.0)
            self.data[state] = np.ones(self.action_size, dtype=np.float32) * self.default_value
        return self.data[state]

    def __getitem__(self, key):
        """
        NumPy风格索引获取Q值。
        NumPy-style indexing for getting Q-values.

        支持 / Supports:
            Q[state]            → 完整行 / full row (shape: action_size,)
            Q[state, action]    → 标量Q值 / scalar Q-value
            Q[state, :]         → 完整行(切片) / full row (slice)

        Args:
            key: 状态索引或(state, action)元组 / State index or (state, action) tuple

        Returns:
            Q值 / Q-value(s)
        """
        if isinstance(key, tuple):
            state, action = key
            row_data = self.row(state)
            if isinstance(action, slice):
                return row_data[action]
            return row_data[int(action)]
        return self.row(key)

    def __setitem__(self, key, value):
        """
        NumPy风格索引设置Q值。
        NumPy-style indexing for setting Q-values.

        Args:
            key: (state, action)元组 / (state, action) tuple
            value: 新的Q值 / New Q-value
        """
        state, action = key
        row_data = self.row(state)
        if isinstance(action, slice):
            row_data[action] = value
        else:
            row_data[int(action)] = value

    def __len__(self) -> int:
        """返回已访问状态的数量 / Return number of visited states."""
        return len(self.data)

    def __contains__(self, state: int) -> bool:
        """检查状态是否已被访问 / Check if state has been visited."""
        return int(state) in self.data

    # ============================================================
    # 序列化 (Serialization) — 用于模型保存/加载
    # ============================================================

    def to_serializable(self) -> dict:
        """
        将Q表转换为可序列化字典。
        Convert Q-table to serializable dictionary.

        用于与numpy save(allow_pickle=True)配合保存。
        For use with numpy save(allow_pickle=True).

        Returns:
            包含 action_size, default_value, data 的字典
            Dictionary with action_size, default_value, and data
        """
        return {
            "action_size": self.action_size,
            "default_value": self.default_value,
            "data": self.data,
        }

    @classmethod
    def from_serializable(cls, payload, action_size: int,
                          default_value: float = 1.0) -> "SparseQTable":
        """
        从序列化字典恢复Q表。
        Restore Q-table from serialized dictionary.

        Args:
            payload: 序列化字典或类字典对象 / Serialized dict or dict-like object
            action_size: 期望的动作空间大小 / Expected action size
            default_value: 默认Q值(仅用于不匹配的旧格式) / Default Q-value

        Returns:
            恢复的SparseQTable实例 / Restored SparseQTable instance

        Raises:
            ValueError: 动作空间大小不匹配时 / When action sizes don't match
        """
        table = cls(action_size=action_size, default_value=default_value)

        if isinstance(payload, dict) and "data" in payload:
            # 新格式: 序列化字典 / New format: serialized dict
            saved_action_size = int(payload.get("action_size", action_size))
            if saved_action_size != int(action_size):
                raise ValueError(
                    f"checkpoint action size {saved_action_size} "
                    f"does not match {action_size}"
                )
            table.action_size = saved_action_size
            table.default_value = float(payload.get("default_value", default_value))
            for key, value in payload["data"].items():
                row_data = np.asarray(value, dtype=np.float32)
                if row_data.shape != (saved_action_size,):
                    raise ValueError(
                        f"invalid Q row shape for state {key}: {row_data.shape}"
                    )
                table.data[int(key)] = row_data
            return table

        # 旧格式: 密集numpy数组 / Legacy format: dense numpy array
        if isinstance(payload, np.ndarray) and payload.ndim == 2:
            if payload.shape[1] != int(action_size):
                raise ValueError(f"invalid dense Q shape: {payload.shape}")
            for state in range(payload.shape[0]):
                row_data = payload[state]
                if not np.all(row_data == default_value):
                    table.data[state] = np.asarray(row_data, dtype=np.float32)
            return table

        raise ValueError("checkpoint does not contain a compatible Q table")

    def num_states(self) -> int:
        """
        返回已访问状态数量 / Return number of visited states.
        提供给监控面板使用 / Provided for monitor panel usage.
        """
        return len(self.data)
