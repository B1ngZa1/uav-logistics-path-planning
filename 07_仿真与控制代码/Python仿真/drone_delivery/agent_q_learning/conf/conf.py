#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
无人机配送 — Q-learning配置模块
Drone Delivery — Q-learning Configuration Module

职责 / Responsibility:
    定义Q-learning算法和环境的所有超参数与常量。
    Defines all hyper-parameters and constants for Q-learning algorithm and environment.

参考 / Reference:
    kaiwu_gorge_walk_v2/agent_q_learning/conf/conf.py

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


class Config:
    """
    无人机配送Q-learning配置
    Drone Delivery Q-learning Configuration

    State space / 状态空间:
        position(40x40=1600) * order_status(2^5=32) * carrying(2) = 102,400
    """

    # ============================================================
    # 核心 Q-Learning 超参数
    # Core Q-Learning hyper-parameters
    # ============================================================
    GAMMA = 0.99                      # 折扣因子 / Discount factor
    LEARNING_RATE = 0.1               # 学习率 / Learning rate

    # ============================================================
    # Epsilon-greedy 探索策略
    # Epsilon-greedy exploration strategy
    # ============================================================
    EPSILON_START = 1.0               # 初始探索率 / Initial exploration rate
    EPSILON_MIN = 0.05                # 最小探索率 / Minimum exploration rate
    EPSILON_DECAY = 0.9995            # 每回合衰减因子 / Per-episode decay factor

    # ============================================================
    # 状态空间与动作空间
    # State space and action space
    # ============================================================
    GRID_SIZE = 40                    # 网格尺寸 / Grid size
    NUM_ORDERS = 5                    # 订单数量 / Number of orders
    STATE_SIZE = 40 * 40 * (2 ** 5) * 2  # = 102,400
    ACTION_SIZE = 4                   # 4方向: 上/下/左/右 (UP/DOWN/LEFT/RIGHT)

    # ============================================================
    # 环境配置
    # Environment configuration
    # ============================================================
    MAX_STEPS = 500                   # 每回合最大步数 / Max steps per episode
    DEPOT = [30, 10]                  # 配送中心坐标 / Depot coordinates

    # ============================================================
    # 训练配置
    # Training configuration
    # ============================================================
    NUM_EPISODES = 10000              # 最大训练回合数 / Maximum training episodes
    EVAL_INTERVAL = 100               # 评估间隔 (回合) / Evaluation interval (episodes)
    EVAL_EPISODES = 10                # 每次评估的回合数 / Number of evaluation episodes

    # ============================================================
    # 奖励设计参数
    # Reward design parameters
    # ============================================================
    DELIVERY_REWARD = 100.0           # 完成一单配送的奖励 / Reward for one delivery
    SAFE_RETURN_BONUS = 500.0         # 全部完成并返回配送中心的奖励 / All done + return bonus
    STEP_PENALTY = -1.0               # 每步时间惩罚 / Per-step time penalty
    COLLISION_PENALTY = -200.0        # 碰撞惩罚 / Collision penalty

    # ============================================================
    # 模型保存
    # Model saving
    # ============================================================
    MODEL_SAVE_INTERVAL = 300         # 模型保存最小间隔(秒) / Min interval between saves (seconds)
    MAX_SAVE_COUNT = 100              # 最大保存次数 / Max save count

    # ============================================================
    # 观测特征维度 (框架兼容性)
    # Observation feature dimension (framework compatibility)
    # ============================================================
    OBSERVATION_SHAPE = 55            # 观测特征向量维度 / Observation feature vector dimension
