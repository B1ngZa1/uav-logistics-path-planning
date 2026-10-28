#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
无人机配送 — SARSA配置模块
Drone Delivery — SARSA Configuration Module

职责 / Responsibility:
    定义SARSA算法和环境的所有超参数与常量。
    Defines all hyper-parameters and constants for SARSA algorithm and environment.

SARSA特性 / SARSA Characteristics:
    - 在线策略(On-Policy): 更新使用当前策略实际采取的动作a'，而非max Q
      On-Policy: update uses the actual action a' taken by current policy, not max Q
    - 对探索更敏感: SARSA在探索过程中会"意识到"危险动作的代价
      More exploration-sensitive: SARSA "accounts for" the cost of risky actions during exploration

参考 / Reference:
    kaiwu_gorge_walk_v2/agent_sarsa/conf/conf.py
"""


class Config:
    """
    无人机配送SARSA配置
    Drone Delivery SARSA Configuration
    """

    # ============================================================
    # 状态空间与动作空间
    # State space and action space
    # ============================================================
    # 状态编码: position(40x40=1600) * order_status(2^5=32) * carrying(2) = 102400
    # State encoding: position(40x40=1600) * order_status(2^5=32) * carrying(2) = 102400
    GRID_SIZE = 40                    # 网格尺寸 / Grid size
    NUM_ORDERS = 5                    # 订单数量 / Number of orders
    MAX_CARRY = 1                     # 最大同时携带数 / Max simultaneous carrying
    STATE_SIZE = 40 * 40 * (2 ** 5) * 2   # = 102400
    ACTION_SIZE = 4                   # 4方向: 上/下/左/右 (UP/DOWN/LEFT/RIGHT)

    # ============================================================
    # SARSA 超参数 (在线策略时序差分学习)
    # SARSA hyper-parameters (on-policy TD learning)
    # ============================================================
    GAMMA = 0.99                      # 折扣因子 / Discount factor
    LR = 0.1                          # 学习率 / Learning rate (SARSA uses constant LR)
    START_LR = 0.1                    # 初始学习率 (与LR相同) / Initial learning rate (same as LR)

    # ============================================================
    # Epsilon-greedy 探索策略
    # Epsilon-greedy exploration strategy
    # ============================================================
    EPSILON_START = 1.0               # 初始探索率 / Initial exploration rate
    EPSILON_MIN = 0.05                # 最小探索率 / Minimum exploration rate
    EPSILON_DECAY = 0.9995            # 每回合衰减因子 / Per-episode decay factor
    EPISODES = 10000                  # 最大训练回合数 / Maximum training episodes

    # ============================================================
    # 环境配置
    # Environment configuration
    # ============================================================
    MAX_STEPS = 500                   # 每回合最大步数 / Max steps per episode
    DEPOT = [30, 10]                  # 配送中心坐标 / Depot coordinates

    # ============================================================
    # 奖励设计参数
    # Reward design parameters
    # ============================================================
    DELIVERY_SCORE = 100.0            # 完成一单配送的奖励 / Reward for one delivery
    SAFE_RETURN_SCORE = 500.0         # 全部完成并返回配送中心的奖励 / All done + return bonus
    STEP_PENALTY = -1.0               # 每步时间惩罚 / Per-step time penalty
    COLLISION_PENALTY = -200.0        # 碰撞惩罚 / Collision penalty

    # ============================================================
    # 训练与评估
    # Training and evaluation
    # ============================================================
    VALIDATION_INTERVAL = 50          # 每隔多少回合进行一次评估 / Validation interval (episodes)
    VALIDATION_EPISODES = 5           # 每次评估的回合数 / Number of validation episodes
    MODEL_SAVE_INTERVAL = 30          # 模型保存最小间隔(秒) / Min interval between saves (seconds)
    MAX_SAVE_COUNT = 100              # 最大保存次数 / Max save count
    CONVERGENCE_WINDOW = 100          # 收敛检测窗口 / Convergence detection window size
    CONVERGENCE_WIN_RATE = 0.90       # 收敛所需胜率 / Win rate required for convergence

    # ============================================================
    # 观测特征维度
    # Observation feature dimension
    # ============================================================
    OBSERVATION_SHAPE = 55            # 观测特征向量维度 / Observation feature vector dimension
