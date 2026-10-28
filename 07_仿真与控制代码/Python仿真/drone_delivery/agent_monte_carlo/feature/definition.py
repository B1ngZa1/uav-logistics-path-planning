#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright (c) 2026 Drone Delivery RL Project. All Rights Reserved.
###########################################################################
"""
Monte Carlo Feature Definition Module for Drone Food Delivery.

蒙特卡洛特征定义模块，用于无人机外卖配送任务。
参考实现：Kaiwu Gorge Walk v2 agent_monte_carlo/feature/definition.py

职责 / Responsibility:
    定义观测(ObsData)、动作(ActData)、样本(SampleData)数据结构，
    以及SampleManager轨迹收集器和reward_shaping奖励塑形函数。
    Defines observation, action, sample data structures, SampleManager
    trajectory collector, and reward_shaping function.

蒙特卡洛特性 / Monte Carlo Characteristics:
    - SampleManager收集整个回合的全部时间步用于回合结束学习
      SampleManager collects ALL timesteps for the entire episode for end-of-episode learning
    - SampleData包含ret字段存储完整累积回报G_t
      SampleData includes ret field to store full cumulative return G_t
    - process_last_frame()从最后一帧反向计算所有G_t
      process_last_frame() computes all G_t via backward pass from last frame

领域适配 / Domain Adaptation:
    Gorge Walk v2 → Drone Delivery:
        hero_pos       → drone_pos         (无人机网格位置)
        treasure       → delivery_order    (配送订单)
        treasure_state → order_status_mask (订单完成状态)
        endpoint       → depot             (配送中心)
        treasure_score=100 → delivery_reward=+100 (配送奖励)
"""


from common_python.utils.common_func import create_cls
import numpy as np
from typing import List, Optional, Dict, Any


# ============================================================================
# 使用 create_cls 模式创建数据类
# Use create_cls pattern to create data classes
# ============================================================================
# create_cls是Kaiwu框架的标准模式，用于创建轻量级的命名属性数据容器。
# create_cls is the standard Kaiwu framework pattern for creating
# lightweight named-attribute data containers.

ObsData = create_cls(
    "ObsData",
    feature=None,          # 状态特征键 (整数) / State feature key (integer)
    legal_actions=None,    # 合法动作列表 / List of legal action indices
    raw_obs=None,          # 原始环境观测字典 / Raw environment observation dict
)

ActData = create_cls(
    "ActData",
    act=None,              # 动作索引 0=UP,1=DOWN,2=LEFT,3=RIGHT / Action index
    sub_act=None,          # 子动作(可选,MC不使用) / Sub-action (optional, not used by MC)
)

SampleData = create_cls(
    "SampleData",
    obs=None,              # 当前状态特征键 / Current state feature key (int)
    _obs=None,             # 下一状态特征键 / Next state feature key (int)
    act=None,              # 执行的动作 / Action taken (int)
    rew=0.0,               # 即时奖励 / Immediate reward (float)
    ret=0.0,               # 累积折扣回报 G_t / Cumulative discounted return G_t (float)
    done=False,            # 回合是否终止 / Whether episode ended (bool)
    next_act=None,         # 下一动作(SARSA用,MC不使用) / Next action (for SARSA, not used by MC)
)


# ============================================================================
# 奖励塑形本地配置
# Local config for reward shaping
# ============================================================================

class _RewardConfig:
    """奖励塑形使用的本地配置常量 / Local config constants for reward shaping."""
    DELIVERY_SCORE = 100.0           # 完成一单配送的基础奖励
    SAFE_RETURN_SCORE = 500.0        # 全部完成并返回配送中心的奖励
    STEP_PENALTY = -1.0              # 每步时间惩罚
    COLLISION_PENALTY = -200.0       # 碰撞惩罚
    PICKUP_BONUS = 10.0              # 取货小型奖励 / Small bonus for picking up an order
    APPROACH_MERCHANT = 5.0          # 接近商家的奖励 / Reward for approaching a merchant
    APPROACH_DELIVERY = 3.0          # 接近配送点的奖励 / Reward for approaching a delivery point
    APPROACH_DEPOT = 2.0             # 接近配送中心的奖励 / Reward for approaching depot
    STAY_PENALTY = -3.0              # 原地不动惩罚 / Penalty for not moving (wall collision)
    TRUNCATION_PENALTY = -100.0      # 超时截断惩罚 / Truncation (timeout) penalty

    @staticmethod
    def _manhattan(pos1, pos2):
        """
        计算曼哈顿距离 / Compute Manhattan distance.

        Args:
            pos1: First position (list/tuple/array of [x, y])
            pos2: Second position (list/tuple/array of [x, y])

        Returns:
            曼哈顿距离 / Manhattan distance
        """
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


# ============================================================================
# 奖励塑形函数
# Reward shaping function
# ============================================================================

def reward_shaping(env_reward: float,
                   env_obs: Optional[Dict[str, Any]] = None,
                   prev_env_obs: Optional[Dict[str, Any]] = None,
                   action: Optional[int] = None) -> float:
    """
    奖励塑形函数 — 无人机配送领域 (蒙特卡洛版本)
    Reward shaping function — Drone Delivery Domain (Monte Carlo version)

    对原始环境奖励进行增强，加入辅助信号帮助MC更快收敛：
    Enhance raw environment reward with auxiliary signals to help MC converge faster.

    MC-specific considerations (蒙特卡洛特有考虑):
        - MC使用完整回报G_t，奖励信号应该更清晰以减少方差
          MC uses full return G_t, reward signals should be clearer to reduce variance
        - 避免过于密集的塑形奖励（会增加G_t方差）
          Avoid overly dense shaping rewards (would increase G_t variance)
        - 优先使用稀疏但有意义的奖励
          Prefer sparse but meaningful rewards

    奖励塑形组件 / Reward shaping components:
        1. Pickup bonus: +10 when drone starts carrying a package
           取货奖励：无人机开始携带包裹时+10
        2. Distance shaping: +5/-3/+2 for approaching merchants/deliveries/depot
           距离塑形：接近商家+5/配送点+3/配送中心+2
        3. No-move penalty: -3 when drone position unchanged (wall collision)
           原地不动惩罚：位置未改变时-3
        4. Truncation penalty: -100 when episode truncated by timeout
           截断惩罚：超时截断时-100

    Args:
        env_reward: 环境原始奖励 / Raw environment reward
        env_obs: 当前环境观测 / Current environment observation dict
        prev_env_obs: 前一环境观测 / Previous environment observation dict
        action: 执行的动作 / Action taken

    Returns:
        塑形后的奖励 / Shaped reward (float)
    """
    reward = float(env_reward) if env_reward is not None else 0.0

    # 安全处理 None 观测 / Safely handle None observations
    if env_obs is None and prev_env_obs is None:
        return reward

    # -----------------------------------------------------------------------
    # 1. 取货鼓励 — 当无人机到达商家位置并开始携带货物时
    #    Pickup encouragement — when drone reaches a merchant and starts carrying
    # -----------------------------------------------------------------------
    if prev_env_obs is not None and env_obs is not None:
        prev_carrying = prev_env_obs.get("carrying", 0) if isinstance(prev_env_obs, dict) else 0
        curr_carrying = env_obs.get("carrying", 0) if isinstance(env_obs, dict) else 0
        if curr_carrying == 1 and prev_carrying == 0:
            reward += _RewardConfig.PICKUP_BONUS

    # -----------------------------------------------------------------------
    # 2. 距离塑形 — 引导无人机接近未完成的目标
    #    Distance shaping — guide drone toward incomplete targets
    # -----------------------------------------------------------------------
    if prev_env_obs is not None and env_obs is not None:
        prev_pos = prev_env_obs.get("position", None) if isinstance(prev_env_obs, dict) else None
        curr_pos = env_obs.get("position", None) if isinstance(env_obs, dict) else None
        curr_carrying = env_obs.get("carrying", 0) if isinstance(env_obs, dict) else 0
        order_status = env_obs.get("order_status", None) if isinstance(env_obs, dict) else None

        if prev_pos is not None and curr_pos is not None and order_status is not None:
            from agent_monte_carlo.conf.conf import Config as MainConfig

            depot = MainConfig.DEPOT

            # 固定商家和配送点 (来自环境默认值)
            # Fixed merchants and delivery points (from environment defaults)
            merchants = [
                [16, 17], [22, 15], [14, 12], [25, 18], [18, 24]
            ]
            delivery_points = [
                [20, 25], [28, 22], [10, 20], [32, 12], [16, 17]
            ]

            # 找到当前应前往的目标 / Find the current target to go to
            target = None
            if curr_carrying == 0:
                # 空手 → 取最近未完成商家
                # Empty → nearest unfinished merchant
                best_dist = float("inf")
                for k, status in enumerate(order_status):
                    if status == 0 and k < len(merchants):
                        dist = _RewardConfig._manhattan(curr_pos, merchants[k])
                        if dist < best_dist:
                            best_dist = dist
                            target = merchants[k]
            else:
                # 携带中 → 取最近未完成配送点
                # Carrying → nearest unfinished delivery point
                best_dist = float("inf")
                for k, status in enumerate(order_status):
                    if status == 0 and k < len(delivery_points):
                        dist = _RewardConfig._manhattan(curr_pos, delivery_points[k])
                        if dist < best_dist:
                            best_dist = dist
                            target = delivery_points[k]

            if target is None:
                # 所有订单完成 → 引导回配送中心
                # All orders complete → guide to depot
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
                    # 远离目标 / Moving away from target
                    reward -= 1.0

    # -----------------------------------------------------------------------
    # 3. 原地不动惩罚 (撞墙 / 无效移动)
    #    No-move penalty (wall collision / invalid move)
    # -----------------------------------------------------------------------
    if prev_env_obs is not None and env_obs is not None:
        prev_pos = prev_env_obs.get("position", None) if isinstance(prev_env_obs, dict) else None
        curr_pos = env_obs.get("position", None) if isinstance(env_obs, dict) else None
        if prev_pos is not None and curr_pos is not None:
            terminated = env_obs.get("terminated", False) if isinstance(env_obs, dict) else False
            if (np.array_equal(prev_pos, curr_pos)
                    and not terminated
                    and action is not None):
                reward += _RewardConfig.STAY_PENALTY

    # -----------------------------------------------------------------------
    # 4. 截断惩罚
    #    Truncation penalty
    # -----------------------------------------------------------------------
    if env_obs is not None:
        truncated = env_obs.get("truncated", False) if isinstance(env_obs, dict) else False
        terminated = env_obs.get("terminated", False) if isinstance(env_obs, dict) else False
        if truncated and not terminated:
            reward += _RewardConfig.TRUNCATION_PENALTY

    # 裁剪到合理范围 / Clip to reasonable range
    return float(np.clip(reward, -500.0, 1000.0))


# ============================================================================
# 样本处理函数
# Sample processing function
# ============================================================================

def sample_process(list_game_data: List[Any]) -> List[Dict[str, Any]]:
    """
    将游戏帧数据转换为统一的样本字典格式。
    Convert game frame data to unified sample dictionary format.

    支持 SampleData 对象和原始字典两种格式。
    Supports both SampleData objects and raw dicts.

    MC uses this to convert the episode trajectory into a list of
    standardized dicts consumed by Algorithm.learn().
    MC使用此函数将回合轨迹转换为Algorithm.learn()使用的标准化字典列表。

    Args:
        list_game_data: 游戏数据列表 / List of game frame data (SampleData or dict)

    Returns:
        统一格式的样本字典列表 / List of unified sample dicts with keys:
            state, action, reward, next_state, done, ret
    """
    samples = []
    for frame in list_game_data:
        if hasattr(frame, "obs") and hasattr(frame, "act"):
            # SampleData object (create_cls pattern)
            # SampleData对象（create_cls模式）
            samples.append({
                "state": getattr(frame, "obs", None),
                "action": getattr(frame, "act", None),
                "reward": float(getattr(frame, "rew", 0.0)),
                "next_state": getattr(frame, "_obs", None),
                "done": bool(getattr(frame, "done", False)),
                "ret": float(getattr(frame, "ret", 0.0)),
            })
        elif isinstance(frame, dict):
            samples.append({
                "state": frame.get("state"),
                "action": frame.get("action"),
                "reward": float(frame.get("reward", 0.0)),
                "next_state": frame.get("next_state"),
                "done": bool(frame.get("done", False)),
                "ret": float(frame.get("ret", 0.0)),
            })
        else:
            # Fallback: try attribute access / 回退：尝试属性访问
            samples.append({
                "state": getattr(frame, "state", None),
                "action": getattr(frame, "action", None),
                "reward": float(getattr(frame, "reward", 0.0)),
                "next_state": getattr(frame, "next_state", None),
                "done": bool(getattr(frame, "done", False)),
                "ret": float(getattr(frame, "ret", 0.0)),
            })
    return samples


# ============================================================================
# 轨迹收集器 — MC版本 (收集整个回合的全部时间步)
# Trajectory collector — MC version (collects ALL timesteps for entire episode)
# ============================================================================

class SampleManager:
    """
    蒙特卡洛轨迹样本管理器 — 收集一个回合内的所有转移。
    Monte Carlo trajectory sample manager — collects all transitions
    within one episode.

    MC关键特性 (MC Key Features):
        - 收集整个回合的全部(s,a,r,s')转移
          Collects ALL (s,a,r,s') transitions for the entire episode
        - process_last_frame()在回合结束时反向计算所有G_t
          process_last_frame() computes all G_t via backward pass at episode end
        - get_game_data()返回完整回合样本列表供agent.learn()使用
          get_game_data() returns the full episode sample list for agent.learn()

    与TD方法SampleManager的区别 (Difference from TD method SampleManager):
        - TD的SampleManager逐帧产生样本给learn()
          TD's SampleManager yields samples frame-by-frame to learn()
        - MC的SampleManager收集全部帧，在回合结束时一次性提交
          MC's SampleManager collects all frames, submits them all at once at episode end

    Usage (用法)::

        collector = SampleManager()
        for step in episode:
            collector.add(obs=state_key, _obs=next_state_key,
                          act=action, rew=reward, done=done)
        # At episode end — compute G_t for all steps
        # 回合结束 — 为所有步骤计算G_t
        collector.process_last_frame()
        # Get full episode data and pass to agent.learn()
        # 获取完整回合数据并传递给agent.learn()
        all_samples = collector.get_game_data()
        agent.learn(all_samples)
    """

    def __init__(self):
        """
        初始化MC样本管理器。
        Initialize MC sample manager.
        """
        # 本回合的所有样本 / All samples for this episode
        self._samples: List = []

        # 上一帧的观测 / Previous frame observation (for last-frame processing)
        self._last_obs = None

        # 回合总步数 / Total steps in the episode
        self.episode_steps: int = 0

        # 回合累计奖励 / Episode cumulative reward
        self.episode_reward: float = 0.0

    def add(self,
            obs=None,
            _obs=None,
            act: Optional[int] = None,
            rew: float = 0.0,
            done: bool = False,
            ret: float = 0.0,
            next_act: Optional[int] = None) -> None:
        """
        添加一条转移记录到本回合。
        Add one transition record to the current episode.

        Args:
            obs: 当前状态特征键 / Current state feature key (int)
            _obs: 下一状态特征键 / Next state feature key (int)
            act: 执行的动作 / Action taken (int)
            rew: 获得的即时奖励 / Immediate reward received (float)
            done: 回合是否终止 / Whether episode ended (bool)
            ret: 累积回报(初始为0, process_last_frame时计算) /
                 Cumulative return (initially 0, computed in process_last_frame)
            next_act: 下一动作(MC不使用) / Next action (not used by MC)
        """
        # 从ObsData提取特征键 / Extract feature key from ObsData
        state_feature = obs.feature if hasattr(obs, "feature") else obs
        next_feature = _obs.feature if hasattr(_obs, "feature") else _obs

        sample = SampleData(
            obs=state_feature,
            _obs=next_feature,
            act=act,
            rew=rew,
            ret=ret,
            done=done,
            next_act=next_act,
        )
        self._samples.append(sample)
        self._last_obs = _obs
        self.episode_steps += 1
        self.episode_reward += rew

    def process_last_frame(self, gamma: float = 0.99) -> None:
        """
        处理最后一帧 — 反向计算所有步骤的累积折扣回报 G_t。
        Process last frame — compute cumulative discounted return G_t
        for all steps via backward pass.

        这是MC方法的核心操作之一。在回合结束时，从最后一步开始
        反向遍历，为每一步计算完整的折扣回报。
        This is one of the core operations of the MC method. At episode end,
        traverse backwards from the last step to compute the full discounted
        return for each step.

        Formula (公式):
            G_T = r_T
            G_{T-1} = r_{T-1} + gamma * G_T
            G_{T-2} = r_{T-2} + gamma * G_{T-1}
            ...

        Args:
            gamma: 折扣因子 / Discount factor (默认0.99)
        """
        if not self._samples:
            return

        # 反向计算累积折扣回报
        # Backward computation of cumulative discounted return
        cumulative = 0.0
        for sample in reversed(self._samples):
            cumulative = sample.rew + gamma * cumulative
            sample.ret = cumulative

    def get_game_data(self) -> List:
        """
        获取本回合所有样本数据(含已计算的G_t)。
        Get all sample data for this episode (with computed G_t).

        Should be called AFTER process_last_frame() to ensure
        G_t values are populated.
        应在process_last_frame()之后调用，以确保G_t值已填充。

        Returns:
            样本数据列表 / List of SampleData objects with ret populated
        """
        return list(self._samples)

    def clear(self) -> None:
        """
        清空所有记录，为下一回合做准备。
        Clear all records to prepare for the next episode.
        """
        self._samples.clear()
        self._last_obs = None
        self.episode_steps = 0
        self.episode_reward = 0.0

    def __len__(self) -> int:
        """返回本回合收集的样本数 / Return number of samples collected this episode."""
        return len(self._samples)

    def __repr__(self) -> str:
        return (f"SampleManager(samples={len(self._samples)}, "
                f"steps={self.episode_steps}, "
                f"total_rew={self.episode_reward:.1f})")
