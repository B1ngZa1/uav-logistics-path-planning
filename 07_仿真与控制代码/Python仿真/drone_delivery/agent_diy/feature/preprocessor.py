# -*- coding: utf-8 -*-
"""
================================================================================
feature/preprocessor.py — 特征预处理器 (DIY占位符)
                          Feature Preprocessor (DIY Placeholder)
================================================================================
职责 / Responsibility:
    将原始环境观测转换为 Agent 所需的 ObsData 和 ActData。
    学生需要实现观测向量提取、合法动作计算等逻辑。
    Convert raw environment observations into ObsData and ActData for the agent.
    Students need to implement observation extraction, legal action computation, etc.

参考 / Reference:
    - agent_dqn/feature/preprocessor.py (DQN完整实现参考 / full DQN reference)
    - agent_q_learning/feature/preprocessor.py (表格型RL预处理参考)
    - KaiWu DRL 框架的 feature/preprocessor.py 模式

领域适配 / Domain Adaptation:
    - 环境返回字典: {"position": (2,), "vector": (50,)}
    - 8方向动作, 边界外方向标记为非法
    - Environment returns dict: {"position": (2,), "vector": (50,)}
    - 8-direction actions, out-of-bounds directions marked as illegal

提示 / Tips:
    - observation_process: 从 env_obs 提取 position 和 vector, 构建 ObsData
    - action_process: 将 ActData 转换为环境可执行的动作格式
    - _compute_legal_actions: 根据无人机位置和网格边界计算哪些方向合法
================================================================================
"""

from typing import Optional

import numpy as np

from agent_diy.feature.definition import ObsData


# =============================================================================
# 方向向量映射 / Direction Vector Mapping
# =============================================================================

# 8方向对应的 (dx, dy) 偏移量 / (dx, dy) offsets for 8 directions
# Action index / 动作索引:
#   0: UP (上)          1: UP_RIGHT (右上)
#   2: RIGHT (右)       3: DOWN_RIGHT (右下)
#   4: DOWN (下)        5: DOWN_LEFT (左下)
#   6: LEFT (左)        7: UP_LEFT (左上)
ACTION_DIRECTIONS: np.ndarray = np.array([
    [ 0,  1],   # 0: UP          / 上
    [ 1,  1],   # 1: UP_RIGHT    / 右上
    [ 1,  0],   # 2: RIGHT       / 右
    [ 1, -1],   # 3: DOWN_RIGHT  / 右下
    [ 0, -1],   # 4: DOWN        / 下
    [-1, -1],   # 5: DOWN_LEFT   / 左下
    [-1,  0],   # 6: LEFT        / 左
    [-1,  1],   # 7: UP_LEFT     / 左上
], dtype=np.int32)


class Preprocessor:
    """
    特征预处理器 — 原始观测 → 结构化特征 (DIY占位符)
    Feature Preprocessor — raw obs → structured features (DIY placeholder)

    学生需要实现两步转换:
      1. observation_process:  原始env_obs → ObsData (含legal_act)
      2. action_process:       ActData → 可执行动作

    Students need to implement two-step conversion:
      1. observation_process:  raw env_obs → ObsData (with legal_act)
      2. action_process:       ActData → executable action

    Usage / 用法::

        pp = Preprocessor()
        obs_data = pp.observation_process(raw_obs)
        action = pp.action_process(act_data)
    """

    def __init__(self, grid_size: int = 40) -> None:
        """
        Initialize preprocessor.

        Args / 参数:
            grid_size (int): 网格尺寸 / Grid size (default: 40x40)
        """
        self._grid_size: int = grid_size
        self._obs_count: int = 0
        """已处理观测计数 / Processed observation counter"""

    # -------------------------------------------------------------------------
    # observation_process — 观测处理 / Observation Processing
    # -------------------------------------------------------------------------

    def observation_process(self, env_obs) -> ObsData:
        """
        将环境原始观测转为结构化 ObsData。
        Convert raw environment observation into structured ObsData.

        TODO: 将环境观测转换为 ObsData / Convert env obs to ObsData
        建议步骤 / Suggested steps:
            1. 提取特征向量 (50维) / Extract feature vector (50-dim)
            2. 提取无人机位置 / Extract drone position
            3. 计算合法动作掩码 / Compute legal action mask
            4. 包装为 ObsData 并返回 / Wrap into ObsData and return

        Args / 参数:
            env_obs: 环境返回的观测 (dict格式)
                     Environment returned observation (dict format)
                     示例: {"position": np.array([x,y]), "vector": np.array([...])}

        Returns / 返回:
            ObsData: 包含 feature (50,) 和 legal_act (8,) 的结构化数据
                     Structured data with feature and legal_act
        """
        self._obs_count += 1

        # TODO: 在此实现观测处理逻辑 / Implement observation processing here

        # 步骤参考 / Step reference:
        # 1. 从 env_obs 中提取 feature (50维向量) / Extract feature from env_obs
        #    if isinstance(env_obs, dict):
        #        feature = env_obs.get("vector")
        #        position = env_obs.get("position")
        #    feature = np.asarray(feature, dtype=np.float32).flatten()

        # 2. 基于 position 和 grid_size 计算合法动作 / Compute legal actions
        #    legal_act = np.ones(8, dtype=np.float32)
        #    for i, (dx, dy) in enumerate(ACTION_DIRECTIONS):
        #        nx, ny = position[0] + dx, position[1] + dy
        #        if nx < 0 or nx >= grid_size or ny < 0 or ny >= grid_size:
        #            legal_act[i] = 0.0

        # 3. 返回 ObsData(feature=feature, legal_act=legal_act)

        # 占位符实现 / Placeholder implementation:
        return ObsData(feature=None, legal_act=list(range(4)))

    # -------------------------------------------------------------------------
    # action_process — 动作处理 / Action Processing
    # -------------------------------------------------------------------------

    def action_process(self, act_data):
        """
        将 ActData 转换为可执行的动作。
        Convert ActData to executable action.

        TODO: 将 ActData 转换为可执行动作 / Convert ActData to executable action

        该方法通常很简单: 直接返回 act_data.act 即可,
        但学生可按需添加后处理 (如钳位、映射等)。
        This method is usually simple: just return act_data.act,
        but students may add post-processing (clamping, remapping, etc.) as needed.

        Args / 参数:
            act_data: ActData 对象 (含 act 和 val 字段)
                      ActData object (with act and val fields)

        Returns / 返回:
            int: 可执行的动作索引 (0-7) / Executable action index
        """
        # TODO: 如需额外处理, 在此添加 / Add extra processing here if needed
        return act_data.act if hasattr(act_data, "act") else act_data

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
