# -*- coding: utf-8 -*-
"""
================================================================================
feature/definition.py — 数据定义与样本管理 (DIY占位符)
                        Data Definitions & Sample Management (DIY Placeholder)
================================================================================
职责 / Responsibility:
    定义强化学习训练数据流中的核心数据结构:
    ObsData, ActData, SampleData, reward_shaping 以及 SampleManager。
    使用 create_cls 动态创建数据类（与 KaiWu 框架一致）。
    Define core data structures in the RL training data flow:
    ObsData, ActData, SampleData, reward_shaping, and SampleManager.
    Uses create_cls to dynamically create data classes (consistent with KaiWu).

参考 / Reference:
    - agent_dqn/feature/definition.py (DQN完整实现参考 / full DQN reference)
    - agent_q_learning/feature/definition.py (Q-learning参考)
    - KaiWu DRL 框架的 feature/definition.py 模式
    - Gorge Walk DIY agent 的数据定义方式

领域适配 / Domain Adaptation:
    - ObsData.feature: 50维环境观测向量 / 50-dim environment observation vector
    - ObsData.legal_act: 8维合法动作掩码 / 8-dim legal action mask
    - ActData.act: 0-7 离散动作索引 / discrete action index 0-7
================================================================================
"""

from typing import List, Optional
import numpy as np
from common_python.utils.common_func import create_cls


# =============================================================================
# 观测数据 / Observation Data
# =============================================================================

ObsData = create_cls(
    "ObsData",
    feature=None,       # np.ndarray: 50维特征向量 / 50-dim feature vector
    legal_act=None,     # np.ndarray: 8维合法动作掩码, 0=非法, 1=合法
                        #             8-dim legal action mask, 0=illegal, 1=legal
)
"""
观测数据容器 / Observation Data Container

Attributes / 属性:
    feature (np.ndarray): 50维特征向量 / 50-dim feature vector
        [drone_x, drone_y,                         # 无人机位置 (2维)
         order_0_distance, order_0_tw_lower, ...    # 订单0的6个特征
         ...                                         # 订单1~7的6个特征 (8*6=48维)
         carrying_status,                            # 携带状态 (1维)
         remaining_steps]                            # 剩余步数 (1维)
    legal_act (np.ndarray): 8维合法动作掩码 / 8-dim legal action mask

TODO: 学生需理解各字段并正确使用 / Students must understand and use each field correctly
"""


# =============================================================================
# 动作数据 / Action Data
# =============================================================================

ActData = create_cls(
    "ActData",
    act=None,   # int: 动作索引 (0-7) / Action index (0-7)
    val=None,   # float: 动作对应的值 (Q值/概率/对数概率等)
                #        Action value (Q-value / prob / log-prob / etc.)
)
"""
动作数据容器 / Action Data Container

Attributes / 属性:
    act (int): 动作索引 (0-7) / Action index (0-7)
    val (float): 动作值 (Q值/概率/对数概率等)
                 Action value (Q-value / prob / log-prob / etc.)

TODO: 学生需理解各字段并正确使用 / Students must understand and use each field correctly
"""


# =============================================================================
# 样本数据 (Transition) / Sample Data
# =============================================================================

SampleData = create_cls(
    "SampleData",
    obs=None,       # np.ndarray: 当前观测特征 (50,) / Current observation feature
    _obs=None,      # np.ndarray: 下一观测特征 (50,) / Next observation feature
    act=None,       # int: 执行的动作索引 / Executed action index
    rew=None,       # float: 即时奖励 / Immediate reward
    done=None,      # bool: 是否终止 / Whether episode terminated
    ret=None,       # float: 累积回报 (MC用) / Cumulative return (for MC)
)
"""
经验回放中的单条转移样本 / A single transition sample for experience replay.

Attributes / 属性:
    obs (np.ndarray):   当前观测特征 (50,) / Current observation feature
    _obs (np.ndarray):  下一观测特征 (50,) / Next observation feature
    act (int):          执行的动作索引 / Executed action index
    rew (float):        即时奖励 / Immediate reward
    done (bool):        是否终止 / Whether episode terminated
    ret (float):        累积回报 (蒙特卡洛用) / Cumulative return (for Monte Carlo)

TODO: 学生需理解各字段并正确使用 / Students must understand and use each field correctly
"""


# =============================================================================
# 奖励塑形 (直通) / Reward Shaping (Pass-Through)
# =============================================================================

def reward_shaping(env_reward: float, env_obs: Optional[np.ndarray] = None) -> float:
    """
    奖励塑形函数 — 当前为直通模式, 不额外修改奖励。
    Reward shaping function — currently pass-through, no extra modification.

    学生可在此添加自定义奖励塑形逻辑:
    - 距离惩罚: 越远离目标, 奖励越低
    - 进度奖励: 每完成一个子任务给予额外奖励
    - 时间窗口奖励: 在时间窗口内送达给额外奖励
    Students may add custom reward shaping logic here:
    - Distance penalty: farther from target → lower reward
    - Progress bonus: extra reward for completing sub-tasks
    - Time-window bonus: extra reward for on-time delivery

    提示 / Tips:
        - 环境原始奖励较为稀疏, 额外塑形可加速学习
        - Environment raw reward is sparse; extra shaping can accelerate learning
        - 注意不要过度塑形导致策略偏离最优
        - Be careful not to over-shape and cause policy deviation

    Args / 参数:
        env_reward (float): 环境原始奖励 / Raw environment reward
        env_obs (Optional[np.ndarray]): 当前观测 (保留接口) / Current obs (reserved)

    Returns / 返回:
        float: 塑形后的奖励 / Shaped reward
    """
    # TODO: 学生可在此添加自定义奖励塑形 / Students may add custom shaping here
    return env_reward


# =============================================================================
# 样本管理器 / Sample Manager
# =============================================================================

class SampleManager:
    """
    样本管理器 — 负责从 episode 轨迹数据中构造和管理 SampleData。
    SampleManager — Constructs and manages SampleData from episode trajectory.

    学生需要理解此类的使用方式并在训练工作流中正确调用。
    Students need to understand how to use this class and call it correctly
    in the training workflow.

    Usage / 用法::

        manager = SampleManager()
        manager.add(obs=obs, _obs=next_obs, act=act, rew=rew, done=done, ret=0.0)
        # ... 在回合结束时 ...
        manager.process_last_frame()
        train_data = manager.get_game_data()

    TODO: 学生可在此添加自定义样本处理方法 / Students may add custom sample processing
    """

    def __init__(self) -> None:
        """初始化样本管理器 / Initialize sample manager"""
        self.samples: List = []
        """样本列表 / Sample list"""

    def add(
        self,
        obs: np.ndarray,
        _obs: np.ndarray,
        act: int,
        rew: float,
        done: bool,
        ret: float = 0.0,
        **kwargs,
    ) -> None:
        """
        添加一条转移样本 / Add a transition sample.

        Args / 参数:
            obs (np.ndarray): 当前观测 / Current observation
            _obs (np.ndarray): 下一观测 / Next observation
            act (int): 动作索引 / Action index
            rew (float): 奖励值 / Reward value
            done (bool): 终止标志 / Termination flag
            ret (float): 累积回报 (MC用) / Cumulative return (for MC)
            **kwargs: 额外字段 / Extra fields
        """
        self.samples.append(
            SampleData(obs=obs, _obs=_obs, act=act, rew=rew, done=done, ret=ret)
        )

    def process_last_frame(self) -> None:
        """
        处理最后一帧 — 用于 MC 算法计算累积回报。
        Process last frame — used by MC algorithm to compute cumulative returns.

        在当前实现中为直通 (pass-through), 学生可按需扩展。
        Currently pass-through; students may extend as needed.
        """
        pass

    def get_game_data(self) -> List:
        """
        获取收集到的所有样本 / Get all collected samples.

        Returns / 返回:
            List: 样本数据列表 / List of sample data
        """
        return self.samples

    def clear(self) -> None:
        """清空所有样本 / Clear all samples"""
        self.samples = []
