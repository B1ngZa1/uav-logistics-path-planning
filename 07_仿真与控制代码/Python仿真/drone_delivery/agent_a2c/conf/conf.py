# -*- coding: utf-8 -*-
"""
================================================================================
conf/conf.py — A2C 配置模块 / A2C Configuration Module
================================================================================
职责 / Responsibility:
    集中管理 Advantage Actor-Critic 算法的所有超参数与网络结构配置。
    Centralized management of all hyperparameters and network architecture
    configs for the Advantage Actor-Critic algorithm.

参考 / Reference:
    - kaiwu_infinity_valley agent_a2c 的 Config 设计模式
    - agent_dqn 的 Config 设计模式
领域适配 / Domain Adaptation:
    - 无人机配送任务: 50维向量观测, 8方向离散动作(4 cardinal + 4 diagonal)
    - Drone delivery: 50-dim vector observation, 8-direction discrete actions
================================================================================
"""


class Config:
    """A2C 全局配置 / A2C Global Configuration"""

    # =========================================================================
    # 环境参数 / Environment Parameters
    # =========================================================================
    STATE_DIM: int = 50
    """观测向量维度 / Observation vector dimension"""

    ACTION_SIZE: int = 8
    """动作空间大小 / Action space size
    0: UP, 1: DOWN, 2: LEFT, 3: RIGHT,
    4: UP_LEFT, 5: UP_RIGHT, 6: DOWN_LEFT, 7: DOWN_RIGHT"""

    # =========================================================================
    # 网络结构 / Network Architecture
    # =========================================================================
    HIDDEN_DIM: int = 256
    """隐藏层单元数 / Hidden layer units"""

    # =========================================================================
    # A2C 超参数 / A2C Hyperparameters
    # =========================================================================
    GAMMA: float = 0.99
    """折扣因子 / Discount factor"""

    LR: float = 3e-4
    """学习率 / Learning rate"""

    ENTROPY_COEF: float = 0.01
    """熵正则化系数 / Entropy regularization coefficient"""

    VALUE_COEF: float = 0.5
    """价值损失权重 / Value loss weight"""

    MAX_GRAD_NORM: float = 0.5
    """梯度裁剪范数 / Gradient clipping norm"""

    # =========================================================================
    # 训练控制 / Training Control
    # =========================================================================
    NUM_EPISODES: int = 5000
    """训练回合数 / Number of training episodes"""

    EVAL_INTERVAL: int = 100
    """评估间隔 (回合数) / Evaluation interval in episodes"""

    MAX_STEPS: int = 800
    """单回合最大步数 / Maximum steps per episode"""

    DEVICE: str = "cpu"
    """计算设备 / Computation device (cpu / cuda)"""

    # =========================================================================
    # 模型保存 / Model Saving
    # =========================================================================
    MODEL_SAVE_INTERVAL: int = 500
    """每隔多少回合保存一次模型 / Save model every N episodes"""


# 单例配置实例 / Singleton config instance
config = Config()
