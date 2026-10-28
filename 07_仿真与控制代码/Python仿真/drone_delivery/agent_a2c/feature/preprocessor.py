#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
================================================================================
feature/preprocessor.py -- A2C 特征预处理器 / A2C Feature Preprocessor
================================================================================

职责 / Responsibility:
    将原始环境观测转换为 Agent 所需的 ObsData 和 ActData:
    - 提取50维向量观测 / Extract 50-dim vector observation
    - 基于网格位置计算8方向合法动作掩码 / Compute 8-direction legal action mask
    Convert raw environment observations into ObsData and ActData
    for the A2C agent.

参考 / Reference:
    - kaiwu_infinity_valley agent_a2c 的 preprocessor.py
    - agent_dqn/feature/preprocessor.py (same env, same observation format)

领域适配 / Domain Adaptation:
    - Random2DDeliveryEnv 返回 dict: {"position": (2,), "vector": (50,)}
    - 8方向动作, 网格边界外方向标记为非法 / 8-direction, boundary outsides illegal
    - 方向映射必须与 Random2DDeliveryEnv.ACTION_DELTAS_8 一致
    - Direction mapping must match Random2DDeliveryEnv.ACTION_DELTAS_8 exactly

================================================================================
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from agent_a2c.feature.definition import ObsData, ActData
from agent_a2c.conf.conf import Config as config


# =============================================================================
# 方向向量映射 / Direction Vector Mapping
# =============================================================================

# 8方向对应的 (dx, dy) 偏移量 / (dx, dy) offsets for 8 directions
# 此方向映射与 Random2DDeliveryEnv.ACTION_DELTAS_8 完全一致
# This direction mapping matches Random2DDeliveryEnv.ACTION_DELTAS_8 exactly
#
# env action index: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT,
#                   4=UP_LEFT, 5=UP_RIGHT, 6=DOWN_LEFT, 7=DOWN_RIGHT
ACTION_DIRECTIONS: np.ndarray = np.array([
    [ 0,  1],    # 0: UP 上
    [ 0, -1],    # 1: DOWN 下
    [-1,  0],    # 2: LEFT 左
    [ 1,  0],    # 3: RIGHT 右
    [-1,  1],    # 4: UP_LEFT 左上
    [ 1,  1],    # 5: UP_RIGHT 右上
    [-1, -1],    # 6: DOWN_LEFT 左下
    [ 1, -1],    # 7: DOWN_RIGHT 右下
], dtype=np.int32)


class Preprocessor:
    """
    A2C 特征预处理器 / A2C Feature Preprocessor

    执行两步转换 / Two-step conversion:
      1. observation_process: 原始 env_obs -> ObsData (含 legal_act 合法动作掩码)
      2. action_process:      (act, val, log_prob) -> ActData

    Usage / 用法::

        pp = Preprocessor(grid_size=40)
        obs_data = pp.observation_process(raw_obs)
        act_data = pp.action_process(act=action_idx, val=v, log_prob=lp)
    """

    def __init__(self, grid_size: int = 40) -> None:
        """
        Initialize preprocessor.

        Args / 参数:
            grid_size (int): 网格尺寸 (默认40, 与 Random2DDeliveryEnv 一致)
                             Grid size (default 40, matching Random2DDeliveryEnv)
        """
        self._grid_size: int = grid_size
        self._obs_count: int = 0
        """已处理观测次数 / Processed observation count"""

    # -------------------------------------------------------------------------
    # observation_process -- 观测处理 / Observation Processing
    # -------------------------------------------------------------------------

    def observation_process(
        self, env_obs: Union[Dict, tuple, list, np.ndarray]
    ) -> ObsData:
        """
        将环境原始观测转为结构化 ObsData。
        Convert raw environment observation into structured ObsData.

        支持的输入格式 / Supported input formats:
            - dict:  {"position": np.array([x,y]), "vector": np.array([...])}
            - list(env_obs): [{"position": ..., "vector": ...}]  -- externally wrapped
            - np.ndarray: 直接作为特征向量 / directly as feature vector

        Args / 参数:
            env_obs: 环境返回的观测 / Environment returned observation

        Returns / 返回:
            ObsData: 包含 feature (50,) 和 legal_act (8,) 的结构化数据
                     Structured data with feature and legal_act
        """
        self._obs_count += 1

        # 提取特征向量与位置 / Extract feature vector and position
        position: Optional[np.ndarray] = None

        if isinstance(env_obs, dict):
            # Random2DDeliveryEnv 的标准观测格式
            # Standard Random2DDeliveryEnv observation format
            feature = env_obs.get("vector")
            position = env_obs.get("position")
        elif isinstance(env_obs, (list, tuple)) and len(env_obs) >= 1:
            # 被包装在 list 中的 dict 观测 (框架兼容)
            # Dict observation wrapped in list (framework compatibility)
            inner = env_obs[0]
            if isinstance(inner, dict) and "vector" in inner:
                feature = inner.get("vector")
                position = inner.get("position")
            else:
                feature = inner
        else:
            # 备用: 直接作为特征数组 / Fallback: treat as feature array directly
            feature = env_obs
            position = None

        feature = np.asarray(feature, dtype=np.float32).flatten()

        # 确保特征维度为50 / Ensure feature dimension is state_dim
        if len(feature) != config.STATE_DIM:
            padded = np.zeros(config.STATE_DIM, dtype=np.float32)
            copy_len = min(len(feature), config.STATE_DIM)
            padded[:copy_len] = feature[:copy_len]
            feature = padded

        # 计算合法动作掩码 / Compute legal action mask
        legal_act: np.ndarray = self._compute_legal_actions(position)

        return ObsData(feature=feature, legal_act=legal_act)

    def _compute_legal_actions(
        self, position: Optional[np.ndarray]
    ) -> np.ndarray:
        """
        基于无人机位置计算8方向合法动作掩码。
        Compute 8-direction legal action mask based on drone position.

        位于网格边界时, 越界方向标记为非法。
        At grid boundaries, out-of-bounds directions are marked illegal.

        A2C 使用此掩码在策略分布中屏蔽非法动作 (logit设为 -inf),
        确保采样的动作始终在合法方向内。
        A2C uses this mask to block illegal actions in policy distribution
        (logit set to -inf), ensuring sampled actions stay legal.

        Args / 参数:
            position (Optional[np.ndarray]): 无人机坐标 (x, y) / Drone coordinates

        Returns / 返回:
            np.ndarray: 合法动作掩码 (8,), 1=合法(legal), 0=非法(illegal)
        """
        if position is None:
            return np.ones(config.ACTION_SIZE, dtype=np.float32)

        x, y = float(position[0]), float(position[1])
        g = float(self._grid_size)
        legal = np.ones(config.ACTION_SIZE, dtype=np.float32)

        for i, (dx, dy) in enumerate(ACTION_DIRECTIONS):
            new_x, new_y = x + dx, y + dy
            if new_x < 0 or new_x >= g or new_y < 0 or new_y >= g:
                legal[i] = 0.0

        # 确保至少有一个合法动作 (极端情况保护) / Ensure at least one legal action
        if legal.sum() < 0.5:
            legal[:] = 1.0

        return legal

    # -------------------------------------------------------------------------
    # action_process -- 动作处理 / Action Processing
    # -------------------------------------------------------------------------

    def action_process(
        self,
        act: int,
        val: float = 0.0,
        log_prob: float = 0.0,
    ) -> ActData:
        """
        将动作索引转换为 ActData。
        Convert action index into ActData.

        Args / 参数:
            act (int): 动作索引 (0-7) / Action index
            val (float): 状态价值 V(s) / State value
            log_prob (float): 动作对数概率 / Log probability of action

        Returns / 返回:
            ActData: 结构化动作数据 / Structured action data
        """
        return ActData(act=act, val=val, log_prob=log_prob)

    # -------------------------------------------------------------------------
    # 辅助方法 / Utility Methods
    # -------------------------------------------------------------------------

    @property
    def obs_count(self) -> int:
        """返回已处理观测次数 / Return processed observation count"""
        return self._obs_count

    def reset_count(self) -> None:
        """重置处理计数器 / Reset processing counter"""
        self._obs_count = 0

    @property
    def grid_size(self) -> int:
        """返回网格尺寸 / Return grid size"""
        return self._grid_size

    @staticmethod
    def get_direction(act: int) -> Tuple[int, int]:
        """
        获取动作对应的方向向量 / Get direction vector for an action.

        Args / 参数:
            act (int): 动作索引 / Action index

        Returns / 返回:
            Tuple[int, int]: (dx, dy) 方向向量 / direction vector
        """
        return tuple(ACTION_DIRECTIONS[act].tolist())
