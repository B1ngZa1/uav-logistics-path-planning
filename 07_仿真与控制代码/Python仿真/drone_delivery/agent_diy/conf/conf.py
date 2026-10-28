# -*- coding: utf-8 -*-
"""
================================================================================
conf/conf.py — 学生DIY配置模块 / Student DIY Configuration Module
================================================================================
职责 / Responsibility:
    集中管理所有超参数与网络结构配置的占位符。
    学生在此文件中设置自己的超参数。
    Centrally manages placeholder hyperparameters and network architecture configs.
    Students set their own hyperparameters in this file.

参考 / Reference:
    - KaiWu DRL 框架的 Config 设计模式 / KaiWu DRL framework Config pattern
    - agent_dqn/conf/conf.py (DQN完整实现参考 / full DQN implementation reference)
    - agent_q_learning/conf/conf.py (Q-learning完整实现参考)

领域适配 / Domain Adaptation:
    - 无人机配送任务: 50维向量观测, 8方向离散动作 (4 cardinal + 4 diagonal)
    - Drone delivery: 50-dim vector observation, 8-direction discrete actions
    - 配送环境: 奖励稀疏, 需要合适的探索策略
    - Delivery env: sparse rewards; appropriate exploration strategy needed
================================================================================
"""


class Config:
    """
    学生DIY全局配置 — 所有值均为占位符, 需学生自行设置。
    Student DIY Global Configuration — all values are placeholders;
    students must set their own values.

    提示 / Tips:
        - 可参考 agent_dqn/conf/conf.py 中的完整DQN配置
        - 可参考 agent_q_learning/conf/conf.py 中的表格型RL配置
        - 完成所有 TODO 项后删除 TODO 注释
        - Refer to agent_dqn/conf/conf.py for full DQN-style config
        - Refer to agent_q_learning/conf/conf.py for tabular RL config
        - Remove TODO comments after completing them
    """

    # =========================================================================
    # 环境参数 / Environment Parameters
    # =========================================================================
    # TODO: 学生在此设置超参数 / Students set hyperparameters here

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
    # TODO: 学生在此设置网络结构参数 / Students set network architecture params here

    HIDDEN_DIM: int = 128
    """隐藏层维度 (占位符, 可按需调整)
    Hidden layer dimension (placeholder, adjust as needed)"""

    # =========================================================================
    # 算法超参数 / Algorithm Hyperparameters
    # =========================================================================
    # TODO: 学生在此设置算法超参数 / Students set algorithm hyperparameters here

    GAMMA: float = 0.99
    """折扣因子 / Discount factor (placeholder)"""

    LEARNING_RATE: float = 1e-3
    """学习率 / Learning rate (placeholder)"""

    BATCH_SIZE: int = 64
    """训练批大小 / Training batch size (placeholder)"""

    BUFFER_SIZE: int = 100000
    """经验回放缓冲区容量 (如需) / Experience replay buffer capacity (if needed)"""

    TARGET_UPDATE: int = 100
    """目标网络更新间隔 (如需) / Target network update interval (if needed)"""

    # =========================================================================
    # 探索策略 / Exploration Strategy
    # =========================================================================
    # TODO: 学生在此设置探索参数 / Students set exploration parameters here

    EPSILON_START: float = 1.0
    """初始探索概率 / Initial exploration probability (placeholder)"""

    EPSILON_MIN: float = 0.01
    """最小探索概率 / Minimum exploration probability (placeholder)"""

    EPSILON_DECAY: float = 0.995
    """探索概率衰减因子 (每步或每回合) / Exploration decay factor (per step or episode)"""

    # =========================================================================
    # 训练控制 / Training Control
    # =========================================================================
    # TODO: 学生在此设置训练控制参数 / Students set training control parameters here

    NUM_EPISODES: int = 1000
    """训练回合总数 / Total training episodes (placeholder)"""

    MAX_STEPS: int = 500
    """每回合最大步数 / Maximum steps per episode (placeholder)"""

    EVAL_INTERVAL: int = 100
    """评估间隔 (回合数) / Evaluation interval in episodes (placeholder)"""

    EVAL_EPISODES: int = 10
    """每次评估的回合数 / Number of evaluation episodes per eval (placeholder)"""

    MODEL_SAVE_INTERVAL: int = 500
    """模型保存间隔 (回合数) / Model save interval in episodes (placeholder)"""

    DEVICE: str = "cpu"
    """计算设备 / Computation device: "cpu" or "cuda" (placeholder)"""

    # =========================================================================
    # 学生可在此添加更多自定义参数
    # Students may add more custom parameters below
    # =========================================================================
    # TODO: 添加你的自定义超参数 / Add your custom hyperparameters here
    pass


# 单例配置实例 / Singleton config instance
# 注意: 所有模块通过 `from conf.conf import config` 引用此实例
# Note: all modules reference this instance via `from conf.conf import config`
config = Config()
