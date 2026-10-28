#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
无人机配送 — Q-learning数据定义模块
Drone Delivery — Q-learning Data Definition Module

职责 / Responsibility:
    定义观测、动作、样本数据结构，以及SampleManager轨迹收集器和reward_shaping。
    Defines observation, action, sample data structures, SampleManager trajectory
    collector, and reward_shaping function.

参考 / Reference:
    kaiwu_gorge_walk_v2/agent_q_learning/feature/definition.py

领域适配 / Domain Adaptation:
    Gorge Walk treasure collection → Drone pickup-delivery:
    - hero_pos → drone_pos (x,y grid)
    - treasure → delivery_order
    - treasure_state (10-bit binary) → order_status_mask (5-bit)
    - endpoint → depot at [30,10]
    - max_step=1000 → max_steps=500
    - treasure_score=100 → delivery_reward=+100
    - end_score=150 → safe_return_bonus=+500
"""

import numpy as np
from typing import List, Optional, Dict, Any


# ============================================================
# 简单数据容器类 (Simple data container classes)
# ============================================================

class ObsData:
    """
    观测数据容器 — 封装环境观测处理后的特征。
    Observation data container — encapsulates features after environment
    observation processing.

    Attributes:
        feature: 状态特征 / State feature (integer state key for Q-table lookup)
        legal_actions: 合法动作列表 / List of legal action indices
        raw_obs: 原始环境观测 (可选) / Raw environment observation dict (optional)
    """

    def __init__(self, feature: Optional[int] = None,
                 legal_actions: Optional[List[int]] = None,
                 raw_obs: Optional[Dict[str, Any]] = None):
        self.feature = feature                 # 状态键 / State key
        self.legal_actions = (
            legal_actions if legal_actions is not None else [0, 1, 2, 3]
        )
        self.raw_obs = raw_obs or {}

    def __repr__(self):
        return f"ObsData(feature={self.feature}, legal_actions={self.legal_actions})"


class ActData:
    """
    动作数据容器 — 封装动作选择结果。
    Action data container — encapsulates action selection result.

    Attributes:
        act: 动作索引 / Action index (0=UP, 1=DOWN, 2=LEFT, 3=RIGHT)
        val: 值 (可选, 用于记录相关信息) / Value (optional, for auxiliary info)
    """

    def __init__(self, act: Optional[int] = None, val: Optional[float] = None):
        self.act = act      # 动作索引 / Action index
        self.val = val      # 关联值 (Q值等) / Associated value (Q-value, etc.)

    def __repr__(self):
        return f"ActData(act={self.act})"


class SampleData:
    """
    样本数据容器 — 一条经验转移。
    Sample data container — one experience transition (s, a, r, s', done).

    Q-learning使用此结构记录每步的环境交互结果。
    Q-learning uses this structure to record per-step environment interaction results.

    Attributes:
        obs: 当前状态特征 / Current state feature (int state key)
        _obs: 下一状态特征 / Next state feature (int state key)
        act: 执行的动作 / Action taken (int)
        rew: 获得的奖励 / Reward received (float)
        done: 是否终止 (terminated or truncated) / Whether episode ended
        ret: 累计回报 (Monte Carlo用, Q-learning暂不使用)
             Cumulative return (for Monte Carlo, not used in Q-learning)
    """

    def __init__(self, obs: Optional[int] = None,
                 _obs: Optional[int] = None,
                 act: Optional[int] = None,
                 rew: float = 0.0,
                 done: bool = False,
                 ret: float = 0.0):
        self.obs = obs    # 当前状态键 / Current state key
        self._obs = _obs  # 下一状态键 / Next state key
        self.act = act    # 执行的动作 / Action taken
        self.rew = rew    # 奖励 / Reward
        self.done = done  # 终止标志 / Termination flag
        self.ret = ret    # 累计回报 / Cumulative return

    def __repr__(self):
        return (f"SampleData(obs={self.obs}, act={self.act}, rew={self.rew:.1f}, "
                f"done={self.done})")


# ============================================================
# 奖励塑形 (Reward shaping)
# ============================================================

def reward_shaping(reward: float,
                   env_obs: Optional[Dict[str, Any]] = None,
                   prev_env_obs: Optional[Dict[str, Any]] = None,
                   action: Optional[int] = None) -> float:
    """
    奖励塑形函数 — 无人机配送领域 (当前为直通版本)
    Reward shaping function — Drone Delivery Domain (pass-through for now)

    对原始环境奖励进行透传，保留后续扩展入口。
    Passes through the raw environment reward, preserving hooks for future extensions.

    未来扩展方向 / Future extension directions:
        - 取货小型鼓励 (pickup bonus for reaching merchant)
        - 距离塑形 (distance-based shaping toward targets)
        - 原地不动惩罚 (no-move penalty for wall collisions)
        - 截断惩罚 (truncation/timeout penalty)

    Args:
        reward: 环境原始奖励 / Raw environment reward
        env_obs: 当前环境观测 (可选) / Current environment observation (optional)
        prev_env_obs: 前一环境观测 (可选) / Previous environment observation (optional)
        action: 执行的动作 (可选) / Action taken (optional)

    Returns:
        塑形后的奖励 / Shaped reward (float)
    """
    # 直通: 直接返回原始奖励
    # Pass-through: return raw reward directly
    return float(reward) if reward is not None else 0.0


# ============================================================
# 样本处理 (Sample processing)
# ============================================================

def sample_process(list_game_data: List[Any]) -> List[Dict[str, Any]]:
    """
    将游戏帧数据转换为统一的样本字典格式。
    Convert game frame data to unified sample dictionary format.

    支持 SampleData 对象和原始字典输入。
    Supports SampleData objects and raw dict inputs.

    Args:
        list_game_data: 游戏数据列表 / List of game frame data (SampleData or dict)

    Returns:
        统一格式的样本字典列表 / List of unified sample dicts with keys:
        state, action, reward, next_state, done
    """
    samples = []
    for frame in list_game_data:
        if isinstance(frame, SampleData):
            samples.append({
                "state": frame.obs,
                "action": frame.act,
                "reward": frame.rew,
                "next_state": frame._obs,
                "done": frame.done,
            })
        elif isinstance(frame, dict):
            samples.append({
                "state": frame.get("state"),
                "action": frame.get("action"),
                "reward": frame.get("reward", 0.0),
                "next_state": frame.get("next_state"),
                "done": frame.get("done", False),
            })
        else:
            # 尝试属性访问 / Try attribute access
            samples.append({
                "state": getattr(frame, "state", None),
                "action": getattr(frame, "action", None),
                "reward": getattr(frame, "reward", 0.0),
                "next_state": getattr(frame, "next_state", None),
                "done": getattr(frame, "done", False),
            })
    return samples


# ============================================================
# 轨迹收集器 (Trajectory collector / SampleManager)
# ============================================================

class SampleManager:
    """
    轨迹样本管理器 — 收集一个回合内的所有状态转移。
    Trajectory sample manager — collects all state transitions within one episode.

    用于Q-learning的在线更新: 每一步都产生一条样本，回合结束时一起传入learn。
    Used for Q-learning online updates: each step produces one sample; all samples
    are passed to learn() at episode end.

    Usage / 用法::

        collector = SampleManager()
        for step in episode:
            collector.add(obs=obs_data, _obs=next_obs_data,
                          act=action, rew=reward, done=done)
        collector.process_last_frame()  # 计算最终回报 (MC用)
        train_data = collector.get_game_data()
    """

    def __init__(self):
        # 本回合的所有样本 / All samples for this episode
        self._samples: List[SampleData] = []
        # 上一帧的观测 (用于最后一帧处理) / Previous frame observation
        self._last_obs: Optional[ObsData] = None

    def add(self,
            obs: Optional[ObsData] = None,
            _obs: Optional[ObsData] = None,
            act: Optional[int] = None,
            rew: float = 0.0,
            done: bool = False,
            ret: float = 0.0) -> None:
        """
        添加一条转移记录。
        Add one transition record.

        Args:
            obs: 当前观测数据 / Current observation data (ObsData)
            _obs: 下一观测数据 / Next observation data (ObsData)
            act: 执行的动作 / Action taken (int)
            rew: 获得的奖励 / Reward received (float)
            done: 是否终止 / Whether episode ended (bool)
            ret: 累计回报 (Monte Carlo用) / Cumulative return (for Monte Carlo)
        """
        # 从ObsData中提取状态特征 / Extract state feature from ObsData
        state_feature = obs.feature if isinstance(obs, ObsData) else obs
        next_feature = _obs.feature if isinstance(_obs, ObsData) else _obs

        sample = SampleData(
            obs=state_feature,
            _obs=next_feature,
            act=act,
            rew=rew,
            done=done,
            ret=ret,
        )
        self._samples.append(sample)
        self._last_obs = _obs

    def process_last_frame(self) -> None:
        """
        处理最后一帧 — 为Monte Carlo计算最终折扣回报。
        Process last frame — compute final discounted return for Monte Carlo.

        从最后一帧反向传播累计回报 (Q-learning中此值不使用，但保持兼容)。
        Back-propagates cumulative return from the last frame.
        (Not used by Q-learning but maintained for compatibility.)
        """
        if not self._samples:
            return
        # 反向传播折扣累计回报 / Back-propagate discounted cumulative return
        cumulative = 0.0
        gamma = 0.99
        for sample in reversed(self._samples):
            cumulative = sample.rew + gamma * cumulative
            sample.ret = cumulative

    def get_game_data(self) -> List[SampleData]:
        """
        获取本回合所有样本数据。
        Get all sample data for this episode.

        Returns:
            样本列表 / List of SampleData objects
        """
        return list(self._samples)

    def clear(self) -> None:
        """清空所有记录 / Clear all records."""
        self._samples.clear()
        self._last_obs = None

    def __len__(self) -> int:
        return len(self._samples)

    def __repr__(self) -> str:
        return f"SampleManager(samples={len(self._samples)})"
