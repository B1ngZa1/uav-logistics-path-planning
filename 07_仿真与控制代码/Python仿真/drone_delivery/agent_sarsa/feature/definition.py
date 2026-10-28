#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
无人机配送 — SARSA数据定义模块
Drone Delivery — SARSA Data Definition Module

职责 / Responsibility:
    定义SARSA所需的观测、动作、样本数据结构，奖励塑形函数和轨迹收集器。
    Defines observation, action, sample data structures, reward shaping function,
    and trajectory collector for SARSA.

SARSA特殊要求 / SARSA-Specific Requirements:
    - SampleData必须包含 next_act 字段，因为SARSA是在线策略，需要知道
      在下一状态s'中实际执行的动作a'来更新Q(s,a)
    - SampleData MUST include next_act field because SARSA is on-policy
      and needs to know the actual action a' taken in next state s' to update Q(s,a)

参考 / Reference:
    kaiwu_gorge_walk_v2/agent_sarsa/feature/definition.py



领域适配 / Domain Adaptation:
    峡谷行走(Gorge Walk) → 无人机配送(Drone Delivery)
    Gorge Walk hero_pos    → drone_pos         (无人机网格位置)
    Gorge Walk treasure    → delivery_order     (配送订单)
    Gorge Walk organ_status→ order_status_mask  (订单完成状态)
    Gorge Walk endpoint    → depot              (配送中心)
"""

import numpy as np
from typing import List, Optional, Dict, Any

from common_python.utils.common_func import create_cls


# ============================================================
# 数据类 — 使用 create_cls 模式 (Kaiwu 框架标准)
# Data classes — using create_cls pattern (Kaiwu framework standard)
# ============================================================

# create_cls 创建轻量级数据容器，支持命名参数默认值
# create_cls creates lightweight data containers with named parameter defaults

ObsData = create_cls(
    "ObsData",
    feature=None,         # 状态特征(整数状态键) / State feature (integer state key)
    legal_actions=None,   # 合法动作列表 / List of legal action indices
    raw_obs=None,         # 原始环境观测字典 / Raw environment observation dict
)

ActData = create_cls(
    "ActData",
    act=None,             # 动作索引(0=UP,1=DOWN,2=LEFT,3=RIGHT) / Action index
    sub_act=None,         # 子动作(可选/预留) / Sub-action (optional/reserved)
)

SampleData = create_cls(
    "SampleData",
    obs=None,             # 当前状态特征 s / Current state feature s
    _obs=None,            # 下一状态特征 s' / Next state feature s'
    act=None,             # 执行的动作 a / Action taken a
    rew=0.0,              # 奖励 r / Reward r
    done=False,           # 是否终止 / Whether episode ended
    ret=0.0,              # 累计回报(Monte Carlo用) / Cumulative return (for Monte Carlo)
    next_act=None,        # SARSA关键字段: 下一动作 a'/ SARSA key field: next action a'
)


# ============================================================
# 奖励塑形 (Reward Shaping)
# ============================================================

class _RewardConfig:
    """奖励塑形本地配置常量 / Local config constants for reward shaping."""

    DELIVERY_SCORE = 100.0           # 完成一单配送 / Complete one delivery
    SAFE_RETURN_SCORE = 500.0        # 全部完成返回配送中心 / All done + return to depot
    STEP_PENALTY = -1.0              # 每步惩罚 / Per-step penalty
    COLLISION_PENALTY = -200.0       # 碰撞惩罚 / Collision penalty
    PICKUP_BONUS = 10.0              # 取货小型奖励 / Small pickup bonus
    APPROACH_MERCHANT = 5.0          # 接近商家奖励 / Approach merchant reward
    APPROACH_DELIVERY = 3.0          # 接近配送点奖励 / Approach delivery point reward
    APPROACH_DEPOT = 2.0             # 接近配送中心奖励 / Approach depot reward
    STAY_PENALTY = -3.0              # 撞墙/原地不动惩罚 / Wall collision penalty
    TRUNCATION_PENALTY = -100.0      # 超时截断惩罚 / Timeout truncation penalty
    DELIVERY_REWARD = 100.0          # 配送成功奖励(环境原始值) / Delivery reward (env raw)

    @staticmethod
    def _manhattan(p1, p2):
        """曼哈顿距离 / Manhattan distance"""
        if p1 is None or p2 is None:
            return float("inf")
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


def reward_shaping(env_reward: float,
                   env_obs: Optional[Dict[str, Any]] = None,
                   prev_env_obs: Optional[Dict[str, Any]] = None,
                   action: Optional[int] = None) -> float:
    """
    奖励塑形函数 — 无人机配送SARSA版本。
    Reward shaping function — Drone Delivery (SARSA version).

    对原始环境奖励进行增强，加入辅助信号帮助SARSA更快收敛。
    SARSA比Q-learning对探索中的惩罚更敏感，因此塑形奖励需要更平滑，
    避免在探索阶段产生过高的方差。

    Enhances raw environment reward with auxiliary signals to help SARSA
    converge faster. SARSA is more sensitive to exploration penalties than
    Q-learning, so shaped rewards should be smoother to avoid excessive
    variance during exploration.

    Args:
        env_reward: 环境原始奖励 / Raw environment reward
        env_obs: 当前环境观测 / Current environment observation dict
        prev_env_obs: 前一环境观测 / Previous environment observation dict
        action: 执行的动作 / Action taken

    Returns:
        塑形后的奖励 / Shaped reward (float)
    """
    reward = float(env_reward) if env_reward is not None else 0.0

    # 安全处理None观测 / Safely handle None observations
    if env_obs is None and prev_env_obs is None:
        return reward

    def _safe_get(obs, key, default=None):
        """安全获取观测字段 / Safely get observation field"""
        if obs is None:
            return default
        if isinstance(obs, dict):
            return obs.get(key, default)
        return default

    # ============================================================
    # 1. 取货鼓励 / Pickup Encouragement
    # ============================================================
    if prev_env_obs is not None and env_obs is not None:
        prev_carrying = _safe_get(prev_env_obs, "carrying", 0)
        curr_carrying = _safe_get(env_obs, "carrying", 0)
        if curr_carrying == 1 and prev_carrying == 0:
            reward += _RewardConfig.PICKUP_BONUS

    # ============================================================
    # 2. 距离塑形 — 引导无人机接近目标
    #    Distance Shaping — guide drone toward target
    # ============================================================
    if prev_env_obs is not None and env_obs is not None:
        prev_pos = _safe_get(prev_env_obs, "position")
        curr_pos = _safe_get(env_obs, "position")
        curr_carrying = _safe_get(env_obs, "carrying", 0)
        order_status = _safe_get(env_obs, "order_status")

        if prev_pos is not None and curr_pos is not None and order_status is not None:
            from agent_sarsa.conf.conf import Config

            depot = Config.DEPOT  # [30, 10]

            # 固定商家和配送点(来自环境默认值)
            # Fixed merchants and delivery points (from environment defaults)
            merchants = [
                [16, 17], [22, 15], [14, 12], [25, 18], [18, 24]
            ]
            delivery_points = [
                [20, 25], [28, 22], [10, 20], [32, 12], [16, 17]
            ]

            # 确定当前应前往的目标
            # Determine the current target to go to
            target = None
            if curr_carrying == 0:
                # 空手 → 最近未完成的商家 / Empty → nearest unfinished merchant
                best_dist = float("inf")
                for k in range(Config.NUM_ORDERS):
                    status = _safe_get_index(order_status, k, 0)
                    if status == 0 and k < len(merchants):
                        dist = _RewardConfig._manhattan(curr_pos, merchants[k])
                        if dist < best_dist:
                            best_dist = dist
                            target = merchants[k]
            else:
                # 携带中 → 最近未完成的配送点 / Carrying → nearest unfinished delivery
                best_dist = float("inf")
                for k in range(Config.NUM_ORDERS):
                    status = _safe_get_index(order_status, k, 0)
                    if status == 0 and k < len(delivery_points):
                        dist = _RewardConfig._manhattan(curr_pos, delivery_points[k])
                        if dist < best_dist:
                            best_dist = dist
                            target = delivery_points[k]

            if target is None:
                # 所有完成 → 回配送中心 / All done → return to depot
                target = depot

            if target is not None:
                prev_dist = _RewardConfig._manhattan(prev_pos, target)
                curr_dist = _RewardConfig._manhattan(curr_pos, target)
                if curr_dist < prev_dist:
                    # 接近目标 / Approaching target
                    if curr_carrying == 0:
                        reward += _RewardConfig.APPROACH_MERCHANT
                    else:
                        reward += _RewardConfig.APPROACH_DELIVERY
                elif curr_dist > prev_dist:
                    # 远离目标 / Moving away from target (SARSA: softer penalty)
                    reward -= 0.5

    # ============================================================
    # 3. 原地不动惩罚 / No-move Penalty
    # ============================================================
    if prev_env_obs is not None and env_obs is not None:
        prev_pos = _safe_get(prev_env_obs, "position")
        curr_pos = _safe_get(env_obs, "position")
        if prev_pos is not None and curr_pos is not None:
            terminated = _safe_get(env_obs, "terminated", False)
            if (np.array_equal(prev_pos, curr_pos)
                    and not terminated
                    and action is not None):
                reward += _RewardConfig.STAY_PENALTY

    # ============================================================
    # 4. 截断惩罚 / Truncation Penalty
    # ============================================================
    if env_obs is not None:
        truncated = _safe_get(env_obs, "truncated", False)
        terminated = _safe_get(env_obs, "terminated", False)
        if truncated and not terminated:
            reward += _RewardConfig.TRUNCATION_PENALTY

    # 裁剪到合理范围 / Clip to reasonable range
    return float(np.clip(reward, -500.0, 1000.0))


def _safe_get_index(arr, idx, default=0):
    """安全获取数组/列表索引 / Safely get array/list index"""
    if arr is None:
        return default
    if isinstance(arr, np.ndarray):
        return int(arr[idx]) if idx < len(arr) else default
    if isinstance(arr, (list, tuple)):
        return int(arr[idx]) if idx < len(arr) else default
    return default


# ============================================================
# 轨迹收集器 (Trajectory Collector / SampleManager)
# ============================================================

class SampleManager:
    """
    SARSA轨迹样本管理器 — 收集一个回合内的所有转移。
    SARSA trajectory sample manager — collects all transitions within one episode.

    SARSA版本的特殊处理 / SARSA-specific handling:
        - add() 方法接收 next_act 参数，这是SARSA更新必需的
        - add() method accepts next_act parameter, which is required for SARSA updates
        - process_last_frame() 为Monte Carlo计算最终回报(保留兼容)
        - process_last_frame() computes final returns for Monte Carlo (kept for compatibility)

    用法 / Usage::

        collector = SampleManager()
        for step in episode:
            collector.add(obs=obs_data, _obs=next_obs_data,
                          act=action, rew=reward, done=done,
                          ret=0.0, next_act=next_action)
        collector.process_last_frame()
        train_data = collector.get_game_data()
    """

    def __init__(self):
        # 本回合的所有样本 / All samples for this episode
        self._samples: List[Any] = []  # List[SampleData]
        # 上一帧观测(用于最后一帧处理) / Previous frame observation (for last-frame processing)
        self._last_obs: Optional[Any] = None  # Optional[ObsData]

    def add(self,
            obs: Optional[Any] = None,
            _obs: Optional[Any] = None,
            act: Optional[int] = None,
            rew: float = 0.0,
            done: bool = False,
            ret: float = 0.0,
            next_act: Optional[int] = None) -> None:
        """
        添加一条转移记录（SARSA版本 — 必须包含next_act）。
        Add one transition record (SARSA version — must include next_act).

        Args:
            obs: 当前观测数据(ObsData或整数状态键) / Current observation (ObsData or int state key)
            _obs: 下一观测数据(ObsData或整数状态键) / Next observation (ObsData or int state key)
            act: 执行的动作a / Action taken a
            rew: 获得的奖励r / Reward received r
            done: 是否终止 / Whether episode ended
            ret: 累计回报(Monte Carlo用, 保留兼容) / Cumulative return (for MC, kept for compat)
            next_act: SARSA关键参数: 下一状态中实际执行的动作a'
                      SARSA key parameter: actual action a' taken in next state
        """
        # 从ObsData中提取特征值; 若传入原始整数也行
        # Extract feature from ObsData; raw int also accepted
        state_feature = obs.feature if hasattr(obs, "feature") else obs
        next_feature = _obs.feature if hasattr(_obs, "feature") else _obs

        # 终止状态: next_act 设为 -1 表示没有下一个动作
        # Terminal state: next_act set to -1 means no next action
        if done and next_act is None:
            next_act = -1

        sample = SampleData(
            obs=state_feature,
            _obs=next_feature,
            act=act,
            rew=rew,
            done=done,
            ret=ret,
            next_act=next_act,
        )
        self._samples.append(sample)
        self._last_obs = _obs

    def process_last_frame(self) -> None:
        """
        处理最后一帧 — 反向传播累计回报(Monte Carlo兼容)。
        Process last frame — back-propagate cumulative return (Monte Carlo compat).

        从最后一帧反向计算累计折扣回报 G_t。
        Computes cumulative discounted return G_t backwards from last frame.
        """
        if not self._samples:
            return
        # 从最后一步反向累计回报 / Accumulate return backwards from last step
        cumulative = 0.0
        gamma = 0.99
        for sample in reversed(self._samples):
            cumulative = sample.rew + gamma * cumulative
            sample.ret = cumulative

    def get_game_data(self) -> List[Any]:
        """
        获取本回合所有样本数据。
        Get all sample data for this episode.

        Returns:
            样本列表(List[SampleData]) / List of SampleData objects
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


# ============================================================
# 样本处理 (Sample Processing)
# ============================================================

def sample_process(list_game_data: List[Any]) -> List[Dict[str, Any]]:
    """
    将游戏帧数据转换为统一的样本字典格式。
    Convert game frame data to unified sample dictionary format.

    支持SampleData对象和原始字典格式。
    对SARSA，特别处理next_action字段的提取。
    Supports SampleData objects and raw dict formats.
    For SARSA, specifically handles extraction of the next_action field.

    Args:
        list_game_data: 游戏数据列表 / List of game frame data

    Returns:
        统一格式的样本字典列表 / List of unified sample dicts
    """
    samples = []
    for frame in list_game_data:
        if hasattr(frame, "obs"):  # SampleData-like object / SampleData类对象
            samples.append({
                "state": frame.obs,
                "action": frame.act,
                "reward": frame.rew,
                "next_state": frame._obs,
                "next_action": getattr(frame, "next_act", -1),
                "done": getattr(frame, "done", False),
            })
        elif isinstance(frame, dict):
            samples.append({
                "state": frame.get("state"),
                "action": frame.get("action"),
                "reward": frame.get("reward", 0.0),
                "next_state": frame.get("next_state"),
                "next_action": frame.get("next_action", frame.get("next_act", -1)),
                "done": frame.get("done", False),
            })
        else:
            # 尝试属性访问 / Try attribute access
            samples.append({
                "state": getattr(frame, "state", None),
                "action": getattr(frame, "action", None),
                "reward": getattr(frame, "reward", 0.0),
                "next_state": getattr(frame, "next_state", None),
                "next_action": getattr(frame, "next_action",
                                       getattr(frame, "next_act", -1)),
                "done": getattr(frame, "done", False),
            })
    return samples
