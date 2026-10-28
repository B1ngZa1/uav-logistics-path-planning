# -*- coding: utf-8 -*-
"""
================================================================================
feature/preprocessor.py — 特征预处理器 / Feature Preprocessor
================================================================================
职责 / Responsibility:
    将原始环境观测转换为 Agent 所需的 ObsData 和 ActData,
    包括观测向量提取、合法动作计算。
    Convert raw environment observations into ObsData and ActData
    for the agent, including observation vector extraction and legal action
    computation.

参考 / Reference:
    - KaiWu DRL 框架的 feature/preprocessor.py 模式
    - Gorge Walk DIY agent 的 Preprocessor 设计
领域适配 / Domain Adaptation:
    - 环境返回字典: {"position": (2,), "vector": (50,)}
    - 8方向动作, 边界外方向标记为非法
    - Environment returns dict: {"position": (2,), "vector": (50,)}
    - 8-direction actions, out-of-bounds directions marked as illegal
================================================================================
"""

from typing import List, Optional, Tuple, Union

import numpy as np

from feature.definition import ObsData, ActData
from conf.conf import config


# =============================================================================
# 方向向量映射 / Direction Vector Mapping
# =============================================================================

# 8方向对应的 (dx, dy) 偏移量 / (dx, dy) offsets for 8 directions
# Action index: 0=UP, 1=UP_RIGHT, 2=RIGHT, 3=DOWN_RIGHT,
#               4=DOWN, 5=DOWN_LEFT, 6=LEFT, 7=UP_LEFT
ACTION_DIRECTIONS: np.ndarray = np.array([
    [ 0,  1],   # 0: UP
    [ 1,  1],   # 1: UP_RIGHT
    [ 1,  0],   # 2: RIGHT
    [ 1, -1],   # 3: DOWN_RIGHT
    [ 0, -1],   # 4: DOWN
    [-1, -1],   # 5: DOWN_LEFT
    [-1,  0],   # 6: LEFT
    [-1,  1],   # 7: UP_LEFT
], dtype=np.int32)


class Preprocessor:
    """
    特征预处理器 — 原始观测 → 结构化特征 / Feature Preprocessor — raw obs → structured features

    执行两步转换:
      1. observation_process:  原始env_obs → ObsData (含legal_act)
      2. action_process:       act_idx → ActData

    Two-step conversion:
      1. observation_process:  raw env_obs → ObsData (with legal_act)
      2. action_process:       act_idx → ActData

    Usage / 用法::

        pp = Preprocessor()
        obs_data = pp.observation_process(raw_obs)
        act_data = pp.action_process(selected_action_idx)
    """

    def __init__(self, grid_size: int = 10) -> None:
        """
        Initialize preprocessor.

        Args / 参数:
            grid_size (int): 网格尺寸 / Grid size (default: 10×10)
        """
        self._grid_size: int = grid_size
        self._obs_count: int = 0

    # -------------------------------------------------------------------------
    # observation_process — 观测处理 / Observation Processing
    # -------------------------------------------------------------------------

    def observation_process(
        self, env_obs: Union[dict, tuple, list]
    ) -> ObsData:
        """
        将环境原始观测转为结构化 ObsData。
        Convert raw environment observation into structured ObsData.

        Args / 参数:
            env_obs: 环境返回的观测, 格式为:
                     {"position": np.array([x,y]), "vector": np.array([...])}
                     Environment returned observation dict.

        Returns / 返回:
            ObsData: 包含 feature (50,) 和 legal_act (8,) 的结构化数据
                     Structured data with feature and legal_act
        """
        self._obs_count += 1

        # 提取特征向量 / Extract feature vector
        if isinstance(env_obs, dict):
            feature = env_obs.get("vector")
            position = env_obs.get("position")
        elif isinstance(env_obs, (list, tuple)) and len(env_obs) == 2:
            feature = env_obs[1]
            position = env_obs[0]
        else:
            # 备用: 直接作为特征数组 / Fallback: treat as feature array directly
            feature = env_obs
            position = None

        feature = np.asarray(feature, dtype=np.float32).flatten()

        # 计算合法动作掩码 / Compute legal action mask
        legal_act = self._compute_legal_actions(position)

        return ObsData(feature=feature, legal_act=legal_act)

    def _compute_legal_actions(
        self, position: Optional[np.ndarray]
    ) -> np.ndarray:
        """
        基于无人机位置计算合法动作掩码。
        Compute legal action mask based on drone position.

        位于网格边界时, 越界方向标记为非法。
        At grid boundaries, out-of-bounds directions are marked illegal.

        Args / 参数:
            position (Optional[np.ndarray]): 无人机坐标 (x, y) / Drone coordinates

        Returns / 返回:
            np.ndarray: 合法动作掩码 (8,), 1=合法/legal, 0=非法/illegal
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

        return legal

    # -------------------------------------------------------------------------
    # action_process — 动作处理 / Action Processing
    # -------------------------------------------------------------------------

    def action_process(self, act: int, val: float = 0.0) -> ActData:
        """
        将动作索引转换为 ActData。
        Convert action index into ActData.

        Args / 参数:
            act (int): 动作索引 (0-7) / Action index
            val (float): 对应的Q值 / Corresponding Q-value

        Returns / 返回:
            ActData: 结构化动作数据 / Structured action data
        """
        return ActData(act=act, val=val)

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
