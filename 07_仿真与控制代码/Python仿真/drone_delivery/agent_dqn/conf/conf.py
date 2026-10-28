# -*- coding: utf-8 -*-
"""
================================================================================
conf/conf.py — Dueling DQN 配置模块 / Dueling DQN Configuration Module
================================================================================
职责 / Responsibility:
    集中管理所有超参数与网络结构配置。
    Centralized management of all hyperparameters and network architecture configs.
参考 / Reference:
    - Gorge Walk DIY agent 的 Dueling DQN 配置模式
    - KaiWu DRL 框架的 Config 设计模式 / KaiWu DRL framework Config pattern
领域适配 / Domain Adaptation:
    - 无人机配送任务: 50维向量观测, 8方向离散动作(4 cardinal + 4 diagonal)
    - Drone delivery: 50-dim vector observation, 8-direction discrete actions
================================================================================
"""


class Config:
    """Dueling DQN 全局配置 / Dueling DQN Global Configuration"""

    # =========================================================================
    # 环境参数 / Environment Parameters
    # =========================================================================
    STATE_DIM: int = 50
    """观测向量维度 / Observation vector dimension
    (position:2 + 8 orders * 6 features + carrying:1 + steps:1) = 50"""

    ACTION_SIZE: int = 8
    """动作空间大小 / Action space size
    0: UP, 1: UP_RIGHT, 2: RIGHT, 3: DOWN_RIGHT,
    4: DOWN, 5: DOWN_LEFT, 6: LEFT, 7: UP_LEFT"""

    # =========================================================================
    # 网络结构 / Network Architecture
    # =========================================================================
    HIDDEN_DIM: int = 256
    """特征提取层隐藏单元数 / Hidden units in feature extraction layer"""

    # =========================================================================
    # DQN 超参数 / DQN Hyperparameters
    # =========================================================================
    GAMMA: float = 0.99
    """折扣因子 / Discount factor"""

    LEARNING_RATE: float = 1e-3
    """学习率 / Learning rate"""

    BATCH_SIZE: int = 256
    """训练批大小 / Training batch size"""

    BUFFER_SIZE: int = 100000
    """经验回放缓冲区容量 / Experience replay buffer capacity"""

    TARGET_UPDATE: int = 100
    """目标网络硬更新间隔（步数） / Target network hard update interval (steps)"""

    # =========================================================================
    # Epsilon-贪心探索 / Epsilon-Greedy Exploration
    # =========================================================================
    EPSILON_START: float = 1.0
    """初始探索概率 / Initial exploration probability"""

    EPSILON_MIN: float = 0.05
    """最小探索概率 / Minimum exploration probability"""

    EPSILON_DECAY: float = 0.999
    """探索概率衰减因子（每步） / Exploration probability decay factor (per step)"""

    # =========================================================================
    # 训练控制 / Training Control
    # =========================================================================
    MIN_BUFFER_SIZE_FOR_LEARN: int = 2000
    """开始学习所需的最小缓冲区样本数 / Minimum buffer samples before learning starts"""

    LEARN_INTERVAL: int = 1
    """每隔多少步学习一次 / Learn every N steps"""

    DEVICE: str = "cpu"
    """计算设备 / Computation device (cpu / cuda)"""

    # =========================================================================
    # 模型保存 / Model Saving
    # =========================================================================
    MODEL_SAVE_INTERVAL: int = 500
    """每隔多少 episode 保存一次模型 / Save model every N episodes"""


# 单例配置实例 / Singleton config instance
config = Config()
