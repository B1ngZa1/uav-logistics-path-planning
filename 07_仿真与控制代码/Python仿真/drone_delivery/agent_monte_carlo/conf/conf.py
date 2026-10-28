#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
无人机配送 — 蒙特卡洛配置模块
Drone Delivery — Monte Carlo Configuration Module

职责 / Responsibility:
    定义Monte Carlo算法和环境的所有超参数与常量。
    Defines all hyper-parameters and constants for Monte Carlo algorithm and environment.

蒙特卡洛特性 / Monte Carlo Characteristics:
    - 回合完整学习: 仅在完整回合结束后进行更新
      Episode-complete learning: Updates only after a full episode finishes
    - 首次访问MC: 只统计每个(state, action)对在回合中的首次出现
      First-visit MC: Only the first visit to each (state, action) pair in an episode counts
    - 完整回报: 使用实际累积折扣回报G_t，无自举(bootstrapping)
      Full return: Uses actual cumulative discounted return G_t, no bootstrapping
    - 高方差: 完整回报方差大，需要更低的learning rate
      High variance: Full returns have high variance, needs lower learning rate

参考 / Reference:
    kaiwu_gorge_walk_v2/agent_monte_carlo/conf/conf.py
"""


class Config:
    """
    无人机配送蒙特卡洛配置
    Drone Delivery Monte Carlo Configuration

    注意 / Note:
        MC算法使用lower learning rate (0.01 vs Q-learning的0.1)，
        因为完整回报G_t的方差远大于TD目标的方差。
        MC uses lower learning rate (0.01 vs Q-learning's 0.1),
        because the variance of full return G_t is much larger than TD targets.
    """

    # ============================================================
    # 状态空间与动作空间
    # State space and action space
    # ============================================================
    # 状态编码: position(40x40=1600) * order_status(2^5=32) * carrying(2) = 102400
    # State encoding: position(40x40=1600) * order_status(2^5=32) * carrying(2) = 102400
    GRID_SIZE = 40                      # 网格尺寸 / Grid size
    NUM_ORDERS = 5                      # 订单数量 / Number of orders
    MAX_CARRY = 1                       # 最大同时携带数 / Max simultaneous carrying
    STATE_SIZE = 40 * 40 * (2 ** 5) * 2  # = 102400
    ACTION_SIZE = 4                     # 4方向: 上/下/左/右 (UP/DOWN/LEFT/RIGHT)

    # 动作映射 / Action mapping
    ACTION_NAMES = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}

    # ============================================================
    # Monte Carlo 超参数 (MC-specific hyper-parameters)
    # ============================================================
    GAMMA = 0.99                        # 折扣因子 / Discount factor
    # MC需要更低的学习率 / MC needs lower learning rate
    # Q-learning用0.1, MC用0.01因为完整回报方差大
    START_LR = 0.01                     # 初始学习率 / Initial learning rate
    LR_MIN = 0.0005                     # 最小学习率 / Minimum learning rate (MC使用更低的min)
    LR_DECAY_STEPS = 150000             # 学习率衰减步数 / LR decay steps

    # ============================================================
    # Epsilon-greedy 探索策略 (较TD方法更慢的衰减)
    # Epsilon-greedy exploration strategy (slower decay than TD methods)
    # ============================================================
    EPSILON_START = 1.0                 # 初始探索率 / Initial exploration rate
    EPSILON_MIN = 0.05                  # 最小探索率 / Minimum exploration rate
    EPSILON_DECAY = 0.999              # 每回合衰减因子(MC用更慢衰减) / Per-episode decay (MC uses slower)
    EPISODES = 10000                    # 最大训练回合数 / Maximum training episodes

    # ============================================================
    # 蒙特卡洛训练控制 (MC training control)
    # ============================================================
    # MC不使用经验回放，不使用目标网络，不使用max算子
    # MC does NOT use experience replay, target network, or max operator
    USE_INCREMENTAL_MEAN = True         # 使用递增均值更新 / Use incremental mean update
    # Q(s,a) += (G - Q(s,a)) / N(s,a)  （递增均值）
    # 或 Q(s,a) += lr * (G - Q(s,a))   （固定学习率）
    # Q(s,a) += (G - Q(s,a)) / N(s,a)  (incremental mean)
    # or Q(s,a) += lr * (G - Q(s,a))   (fixed learning rate)

    # ============================================================
    # 环境配置
    # Environment configuration
    # ============================================================
    MAX_STEPS = 500                     # 每回合最大步数 / Max steps per episode
    DEPOT = [30, 10]                    # 配送中心坐标 / Depot coordinates
    MERCHANTS = [                       # 商家坐标 / Merchant locations
        [16, 17], [22, 15], [14, 12], [25, 18], [18, 24]
    ]
    DELIVERY_POINTS = [                 # 配送点坐标 / Delivery point locations
        [20, 25], [28, 22], [10, 20], [32, 12], [16, 17]
    ]

    # ============================================================
    # 奖励设计参数
    # Reward design parameters
    # ============================================================
    DELIVERY_SCORE = 100.0              # 完成一单配送的奖励 / Reward for one delivery
    SAFE_RETURN_SCORE = 500.0           # 全部完成并返回配送中心的奖励 / All done + return bonus
    STEP_PENALTY = -1.0                 # 每步时间惩罚 / Per-step time penalty
    COLLISION_PENALTY = -200.0          # 碰撞惩罚 / Collision penalty

    # ============================================================
    # 训练与评估
    # Training and evaluation
    # ============================================================
    VALIDATION_INTERVAL = 50            # 每隔多少回合进行一次评估 / Validation interval (episodes)
    VALIDATION_EPISODES = 5             # 每次评估的回合数 / Number of validation episodes
    MODEL_SAVE_INTERVAL = 300           # 模型保存最小间隔(秒) / Min interval between saves (seconds)
    MAX_SAVE_COUNT = 100                # 最大保存次数 / Max save count
    CONVERGENCE_WINDOW = 100            # 收敛检测窗口 / Convergence detection window size
    CONVERGENCE_WIN_RATE = 0.90         # 收敛所需胜率 / Win rate required for convergence

    # ============================================================
    # 观测特征维度
    # Observation feature dimension
    # ============================================================
    OBSERVATION_SHAPE = 55              # 观测特征向量维度 / Observation feature vector dimension
