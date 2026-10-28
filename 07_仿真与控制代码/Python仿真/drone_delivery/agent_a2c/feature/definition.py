#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
================================================================================
feature/definition.py -- A2C 数据定义与样本管理 / A2C Data Definitions & Sample Manager
================================================================================

职责 / Responsibility:
    定义 A2C 训练数据流中的核心数据结构:
    ObsData, ActData, SampleData, SampleManager 及 reward_shaping。
    Define core data structures in the A2C training data flow.

参考 / Reference:
    - kaiwu_infinity_valley agent_a2c 的 definition.py
    - agent_dqn/feature/definition.py (same env, same data structure)
    - agent_monte_carlo/feature/definition.py (episode-based SampleManager)

领域适配 / Domain Adaptation:
    - 为 A2C 增加 log_prob 和 val 字段 (策略梯度所需)
    - Adds log_prob and val fields for A2C (required for policy gradient)
    - SampleData.to_dict() 转换供 Algorithm.learn() 批量处理
    - SampleData.to_dict() conversion for Algorithm.learn() batch processing

================================================================================
"""

from typing import Any, Dict, List, Optional

import numpy as np

from agent_a2c.conf.conf import Config as config


# =============================================================================
# 观测数据 / Observation Data
# =============================================================================

class ObsData:
    """
    A2C 观测数据容器 / A2C Observation Data Container

    存储从环境观测中提取的结构化特征, 包含:
    - 50维向量特征 (传入ActorCritic网络)
    - 8维合法动作掩码 (用于过滤非法方向)

    Attributes / 属性:
        feature (np.ndarray): 50维特征向量 / 50-dim feature vector
        legal_act (np.ndarray): 8维合法动作掩码, 0=非法(illegal), 1=合法(legal)
    """

    __slots__ = ("feature", "legal_act")

    def __init__(
        self,
        feature: np.ndarray,
        legal_act: Optional[np.ndarray] = None,
    ) -> None:
        """
        Initialize A2C observation data.

        Args / 参数:
            feature (np.ndarray): 50维观测向量 / 50-dim observation vector
            legal_act (Optional[np.ndarray]): 合法动作掩码, 默认全合法
                                              Legal action mask, default all legal
        """
        self.feature: np.ndarray = np.asarray(feature, dtype=np.float32)
        if legal_act is None:
            self.legal_act: np.ndarray = np.ones(config.ACTION_SIZE, dtype=np.float32)
        else:
            self.legal_act: np.ndarray = np.asarray(legal_act, dtype=np.float32)

    def __repr__(self) -> str:
        return (
            f"ObsData(feature_shape={self.feature.shape}, "
            f"legal_act_sum={int(self.legal_act.sum())}/{config.ACTION_SIZE})"
        )


# =============================================================================
# 动作数据 / Action Data
# =============================================================================

class ActData:
    """
    A2C 动作数据容器 / A2C Action Data Container

    为 A2C 增加了 log_prob 和 val 字段, 用于策略梯度计算。
    Added log_prob and val fields for A2C policy gradient computation.

    Attributes / 属性:
        act (int): 动作索引 0-7 / Action index 0-7
        val (float): 状态价值 V(s) -- 用于优势计算 / State value for advantage
        log_prob (float): 动作的对数概率 log pi(a|s) -- 用于策略梯度
                           Log probability of the action for policy gradient
    """

    __slots__ = ("act", "val", "log_prob")

    def __init__(
        self,
        act: int,
        val: float = 0.0,
        log_prob: float = 0.0,
    ) -> None:
        """
        Initialize A2C action data.

        Args / 参数:
            act (int): 动作索引 0-7 / Action index 0-7
            val (float): 状态价值 / State value, default 0.0
            log_prob (float): 对数概率 / Log probability, default 0.0
        """
        self.act: int = int(act)
        self.val: float = float(val)
        self.log_prob: float = float(log_prob)

    def __repr__(self) -> str:
        return (
            f"ActData(act={self.act}, val={self.val:.4f}, "
            f"log_prob={self.log_prob:.4f})"
        )


# =============================================================================
# 样本数据 (用于完整回合轨迹收集) / Sample Data (for full episode trajectory)
# =============================================================================

class SampleData:
    """
    A2C 单条转移样本 / A2C Single Transition Sample

    用于完整回合轨迹收集。每个回合的样本在回合结束时批量传给 Algorithm.learn()。
    Used for full episode trajectory collection. Batch-passed to Algorithm.learn()
    at episode end.

    A2C特有字段 / A2C-specific fields:
        - log_prob: 策略梯度计算需要 / required for policy gradient
        - value: 原始价值估计, 用于计算优势 / original value estimate for advantage

    Attributes / 属性:
        state (np.ndarray):  当前观测特征 (50,) / Current observation feature
        action (int):        执行的动作索引 / Executed action index
        reward (float):      即时奖励 / Immediate reward
        log_prob (float):    动作的对数概率 / Log probability of action
        value (float):       状态价值估计 V(s) / State value estimate
        done (bool):         是否终止 / Whether episode terminated
    """

    __slots__ = ("state", "action", "reward", "log_prob", "value", "done")

    def __init__(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        log_prob: float = 0.0,
        value: float = 0.0,
        done: bool = False,
    ) -> None:
        """
        Initialize an A2C transition sample.

        Args / 参数:
            state (np.ndarray): 当前观测 (50,) / Current observation
            action (int): 动作索引 / Action index
            reward (float): 奖励 / Reward
            log_prob (float): 对数概率 / Log probability
            value (float): 状态价值 / State value
            done (bool): 终止标志 / Termination flag
        """
        self.state: np.ndarray = np.asarray(state, dtype=np.float32)
        self.action: int = int(action)
        self.reward: float = float(reward)
        self.log_prob: float = float(log_prob)
        self.value: float = float(value)
        self.done: bool = bool(done)

    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典, 供 Algorithm.learn() 使用。
        Convert to dict for Algorithm.learn() consumption.

        Returns / 返回:
            Dict[str, Any]: 样本字典 / Sample dict with keys:
                state, action, reward, log_prob, value, done
        """
        return {
            "state": self.state,
            "action": self.action,
            "reward": self.reward,
            "log_prob": self.log_prob,
            "value": self.value,
            "done": self.done,
        }

    def __repr__(self) -> str:
        return (
            f"SampleData(act={self.action}, rew={self.reward:.2f}, "
            f"log_prob={self.log_prob:.4f}, done={self.done})"
        )


# =============================================================================
# 奖励塑形函数 / Reward Shaping Function
# =============================================================================

def reward_shaping(
    env_reward: float,
    env_obs: Optional[Any] = None,
) -> float:
    """
    A2C 奖励塑形函数 -- 当前为直通模式。
    A2C Reward shaping function -- currently pass-through mode.

    A2C 使用策略梯度, 对奖励信号的稳定性有一定要求。
    当前保持直通, 可根据需要扩展辅助奖励 (距离塑形, 进度奖励等)。
    A2C uses policy gradients which benefit from stable reward signals.
    Currently pass-through; extensible with auxiliary rewards.

    Args / 参数:
        env_reward (float): 环境原始奖励 / Raw environment reward
        env_obs (Optional[Any]): 环境观测 (保留扩展接口) / Environment obs (reserved)

    Returns / 返回:
        float: 塑形后的奖励 / Shaped reward
    """
    return float(env_reward) if env_reward is not None else 0.0


# =============================================================================
# A2C 样本管理器 -- On-Policy 轨迹收集 / A2C Sample Manager -- On-Policy Collection
# =============================================================================

class SampleManager:
    """
    A2C 轨迹样本管理器 -- 收集完整回合的所有转移。
    A2C Trajectory Sample Manager -- collects all transitions for a full episode.

    A2C 是 on-policy 算法, 必须在回合结束时用完整轨迹进行学习。
    SampleManager 收集一个回合的全部 (s, a, r, log_prob, v) 转移,
    在回合结束时一次性提交给 Algorithm.learn()。

    A2C is on-policy, so it must learn from complete trajectories at episode end.
    SampleManager collects all transitions and submits them in batch.

    Usage / 用法::

        manager = SampleManager()
        for step in episode:
            manager.add(state=obs, action=act, reward=r,
                        log_prob=lp, value=v, done=d)
        # At episode end:
        samples = manager.get_dicts()
        agent.learn(samples)
        manager.clear()
    """

    def __init__(self) -> None:
        """初始化A2C样本管理器 / Initialize A2C sample manager"""
        self._states: List[np.ndarray] = []
        self._actions: List[int] = []
        self._rewards: List[float] = []
        self._log_probs: List[float] = []
        self._values: List[float] = []
        self._dones: List[bool] = []

        self.episode_steps: int = 0
        """回合总步数 / Total steps in the episode"""

        self.episode_reward: float = 0.0
        """回合累计奖励 / Episode cumulative reward"""

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        log_prob: float = 0.0,
        value: float = 0.0,
        done: bool = False,
    ) -> None:
        """
        添加一条转移记录到本回合。
        Add one transition record to the current episode.

        Args / 参数:
            state (np.ndarray): 当前观测 (50,) / Current observation
            action (int): 执行的动作 / Action taken
            reward (float): 即时奖励 / Immediate reward
            log_prob (float): 对数概率 / Log probability of the action
            value (float): 状态价值 / State value estimate
            done (bool): 是否终止 / Whether episode terminated
        """
        self._states.append(np.asarray(state, dtype=np.float32))
        self._actions.append(int(action))
        self._rewards.append(float(reward))
        self._log_probs.append(float(log_prob))
        self._values.append(float(value))
        self._dones.append(bool(done))

        self.episode_steps += 1
        self.episode_reward += float(reward)

    def get_samples(self) -> List[SampleData]:
        """
        获取本回合所有样本 (SampleData对象列表)。
        Get all episode samples as SampleData objects.

        Returns / 返回:
            List[SampleData]: 样本对象列表 / List of SampleData objects
        """
        samples: List[SampleData] = []
        for i in range(len(self._states)):
            samples.append(
                SampleData(
                    state=self._states[i],
                    action=self._actions[i],
                    reward=self._rewards[i],
                    log_prob=self._log_probs[i],
                    value=self._values[i],
                    done=self._dones[i],
                )
            )
        return samples

    def get_dicts(self) -> List[Dict[str, Any]]:
        """
        获取本回合所有样本 (字典格式), 供 Algorithm.learn() 直接使用。
        Get all episode samples as dicts for direct use by Algorithm.learn().

        Returns / 返回:
            List[Dict[str, Any]]: 样本字典列表 / List of sample dicts
        """
        dicts: List[Dict[str, Any]] = []
        for i in range(len(self._states)):
            dicts.append({
                "state": self._states[i],
                "action": self._actions[i],
                "reward": self._rewards[i],
                "log_prob": self._log_probs[i],
                "value": self._values[i],
                "done": self._dones[i],
            })
        return dicts

    def clear(self) -> None:
        """
        清空所有记录, 为下一回合做准备。
        Clear all records to prepare for the next episode.
        """
        self._states.clear()
        self._actions.clear()
        self._rewards.clear()
        self._log_probs.clear()
        self._values.clear()
        self._dones.clear()
        self.episode_steps = 0
        self.episode_reward = 0.0

    def __len__(self) -> int:
        """返回本回合收集的样本数 / Return number of samples collected this episode."""
        return len(self._states)

    def __bool__(self) -> bool:
        """是否有样本 / Whether there are samples"""
        return len(self._states) > 0

    def __repr__(self) -> str:
        return (
            f"SampleManager(samples={len(self._states)}, "
            f"steps={self.episode_steps}, "
            f"total_rew={self.episode_reward:.1f})"
        )
