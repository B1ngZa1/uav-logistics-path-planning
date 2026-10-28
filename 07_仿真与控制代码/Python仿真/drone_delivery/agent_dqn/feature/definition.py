# -*- coding: utf-8 -*-
"""
================================================================================
feature/definition.py — 数据定义与样本管理 / Data Definitions & Sample Management
================================================================================
职责 / Responsibility:
    定义强化学习训练数据流中的核心数据结构:
    ObsData, ActData, SampleData, 以及 SampleManager。
    Define core data structures in the RL training data flow:
    ObsData, ActData, SampleData, and SampleManager.
参考 / Reference:
    - KaiWu DRL 框架的 feature/definition.py 模式
    - Gorge Walk DIY agent 的数据定义方式
领域适配 / Domain Adaptation:
    - ObsData.feature: 50维环境观测向量 / 50-dim environment observation vector
    - ObsData.legal_act: 8维合法动作掩码 / 8-dim legal action mask
    - ActData.act: 0-7 离散动作索引 / discrete action index 0-7
================================================================================
"""

from typing import List, Optional, Tuple
from collections import deque
import numpy as np


# =============================================================================
# 观测数据 / Observation Data
# =============================================================================

class ObsData:
    """
    观测数据容器 / Observation Data Container

    Attributes / 属性:
        feature (np.ndarray): 50维特征向量 / 50-dim feature vector
            [drone_x, drone_y,                          # 无人机位置 (2维)
                order_0_distance, order_0_tw_lower, ...  # 订单0的6个特征
                ...                                      # 订单1~7的6个特征
                order_7_distance, ..., order_7_tw_upper, # (8 * 6 = 48维)
                carrying_status,                          # 携带状态 (1维)
                remaining_steps]                          # 剩余步数 (1维)
        legal_act (np.ndarray): 8维合法动作掩码, 0=非法, 1=合法
                                8-dim legal action mask, 0=illegal, 1=legal
    """

    __slots__ = ("feature", "legal_act")

    def __init__(self, feature: np.ndarray, legal_act: Optional[np.ndarray] = None) -> None:
        """
        Initialize observation data.

        Args / 参数:
            feature (np.ndarray): 50维观测向量 / 50-dim observation vector
            legal_act (Optional[np.ndarray]): 合法动作掩码, 默认全合法
                                              Legal action mask, default all legal
        """
        self.feature: np.ndarray = np.asarray(feature, dtype=np.float32)
        if legal_act is None:
            self.legal_act: np.ndarray = np.ones(8, dtype=np.float32)
        else:
            self.legal_act: np.ndarray = np.asarray(legal_act, dtype=np.float32)

    def __repr__(self) -> str:
        return (f"ObsData(feature_shape={self.feature.shape}, "
                f"legal_act_sum={int(self.legal_act.sum())}/8)")


# =============================================================================
# 动作数据 / Action Data
# =============================================================================

class ActData:
    """
    动作数据容器 / Action Data Container

    Attributes / 属性:
        act (int): 动作索引 (0-7) / Action index (0-7)
        val (float): 动作对应的Q值(推理用) 或0(训练用) / Q-value (inference) or 0 (training)
    """

    __slots__ = ("act", "val")

    def __init__(self, act: int, val: float = 0.0) -> None:
        """
        Initialize action data.

        Args / 参数:
            act (int): 动作索引 0-7 / Action index 0-7
            val (float): Q值 / Q-value, default 0.0
        """
        self.act: int = int(act)
        self.val: float = float(val)

    def __repr__(self) -> str:
        return f"ActData(act={self.act}, val={self.val:.4f})"


# =============================================================================
# 样本数据 (Transition) / Sample Data
# =============================================================================

class SampleData:
    """
    经验回放中的单条转移样本 / A single transition sample for experience replay.

    Attributes / 属性:
        obs (np.ndarray):    当前观测特征 (50,) / Current observation feature
        act (int):           执行的动作索引 / Executed action index
        rew (float):         即时奖励 / Immediate reward
        next_obs (np.ndarray): 下一观测特征 (50,) / Next observation feature
        done (bool):         是否终止 / Whether episode terminated
        legal_act (np.ndarray): 当前观测的合法动作掩码 (8,) / Legal action mask
    """

    __slots__ = ("obs", "act", "rew", "next_obs", "done", "legal_act")

    def __init__(
        self,
        obs: np.ndarray,
        act: int,
        rew: float,
        next_obs: np.ndarray,
        done: bool,
        legal_act: Optional[np.ndarray] = None,
    ) -> None:
        """
        Initialize a transition sample.

        Args / 参数:
            obs (np.ndarray): 当前观测 (50,) / Current observation
            act (int): 动作索引 / Action index
            rew (float): 奖励值 / Reward value
            next_obs (np.ndarray): 下一观测 (50,) / Next observation
            done (bool): 终止标志 / Termination flag
            legal_act (Optional[np.ndarray]): 合法动作掩码 / Legal action mask
        """
        self.obs: np.ndarray = np.asarray(obs, dtype=np.float32)
        self.act: int = int(act)
        self.rew: float = float(rew)
        self.next_obs: np.ndarray = np.asarray(next_obs, dtype=np.float32)
        self.done: bool = bool(done)
        self.legal_act: np.ndarray = (
            np.asarray(legal_act, dtype=np.float32)
            if legal_act is not None
            else np.ones(8, dtype=np.float32)
        )

    def __repr__(self) -> str:
        return (f"SampleData(act={self.act}, rew={self.rew:.2f}, "
                f"done={self.done})")


# =============================================================================
# 奖励塑形 (直通) / Reward Shaping (Pass-Through)
# =============================================================================

def reward_shaping(env_reward: float, obs: Optional[np.ndarray] = None) -> float:
    """
    奖励塑形函数 — 当前为直通模式, 不额外修改奖励。
    Reward shaping function — currently pass-through, no extra modification.

    如果需要添加稀疏奖励辅助信号 (距离惩罚、进度奖励等), 可在此扩展。
    Extend here if auxiliary signals (distance penalty, progress bonus) are needed.

    Args / 参数:
        env_reward (float): 环境原始奖励 / Raw environment reward
        obs (Optional[np.ndarray]): 当前观测 (保留接口) / Current observation (reserved)

    Returns / 返回:
        float: 塑形后的奖励 / Shaped reward
    """
    return env_reward


# =============================================================================
# 样本管理器 / Sample Manager
# =============================================================================

class SampleManager:
    """
    样本管理器 — 负责从 episode 轨迹数据中构造 SampleData。
    SampleManager — Constructs SampleData from episode trajectory data.

    Usage / 用法::

        manager = SampleManager()
        sample = manager.make_sample(obs, act, rew, next_obs, done)
    """

    def __init__(self) -> None:
        """初始化样本管理器 / Initialize sample manager"""
        self._sample_count: int = 0
        """已生成样本计数器 / Generated sample counter"""

    def make_sample(
        self,
        obs: np.ndarray,
        act: int,
        rew: float,
        next_obs: np.ndarray,
        done: bool,
        legal_act: Optional[np.ndarray] = None,
    ) -> SampleData:
        """
        从单步转移构造样本 / Construct a sample from a single transition.

        Args / 参数:
            obs (np.ndarray): 当前观测 / Current observation
            act (int): 动作索引 / Action index
            rew (float): 奖励值 / Reward value
            next_obs (np.ndarray): 下一观测 / Next observation
            done (bool): 终止标志 / Termination flag
            legal_act (Optional[np.ndarray]): 合法动作掩码 / Legal action mask

        Returns / 返回:
            SampleData: 转移样本 / Transition sample
        """
        self._sample_count += 1
        return SampleData(
            obs=obs,
            act=act,
            rew=rew,
            next_obs=next_obs,
            done=done,
            legal_act=legal_act,
        )

    def make_sample_list(
        self,
        transitions: List[Tuple[np.ndarray, int, float, np.ndarray, bool]],
    ) -> List[SampleData]:
        """
        批量构造样本 / Batch construct samples.

        Args / 参数:
            transitions: List of (obs, act, rew, next_obs, done) tuples

        Returns / 返回:
            List[SampleData]: 转移样本列表 / List of transition samples
        """
        samples: List[SampleData] = []
        for obs, act, rew, next_obs, done in transitions:
            samples.append(self.make_sample(obs, act, rew, next_obs, done))
        return samples

    @property
    def sample_count(self) -> int:
        """返回已生成样本总数 / Return total generated sample count"""
        return self._sample_count

    def reset_count(self) -> None:
        """重置样本计数器 / Reset sample counter"""
        self._sample_count = 0
