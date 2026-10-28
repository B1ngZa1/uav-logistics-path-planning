#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
无人机配送 — Q-learning监控构建器模块
Drone Delivery — Q-learning Monitor Builder Module

职责 / Responsibility:
    构建Kaiwu风格的监控配置字典，定义训练过程中需要追踪的指标组。
    Builds Kaiwu-style monitor configuration dict, defining metric groups
    to track during training.

参考 / Reference:
    kaiwu_gorge_walk_v2/agent_q_learning/conf/monitor_builder.py

领域适配 / Domain Adaptation:
    Gorge Walk treasure collection → Drone pickup-delivery:
    - treasure_score=100 → delivery metric tracking
    - end_score=150 → safe return bonus tracking
    - max_step=1000 → max_steps=500
"""

from agent_q_learning.conf.conf import Config


def build_monitor():
    """
    构建监控配置字典。
    Build monitor configuration dictionary.

    定义以下指标组 / Defines the following metric groups:
        - train: 训练指标 (奖励、TD误差、epsilon等)
          Training metrics (reward, TD error, epsilon, etc.)
        - eval: 评估指标 (评估奖励、配送完成数等)
          Evaluation metrics (eval reward, delivery completions, etc.)
        - model: 模型指标 (Q表状态数、训练步数等)
          Model metrics (visited state count, training steps, etc.)

    Returns:
        监控配置字典 / Monitor configuration dict
    """
    monitor = {
        # ============================================================
        # 训练指标组
        # Training metric groups
        # ============================================================
        "train": {
            # 回合奖励 / Episode reward
            "episode_reward": {
                "name": "训练回合奖励 / Training Episode Reward",
                "type": "scalar",
                "display": True,
            },
            # 平均TD误差 / Average TD error
            "td_error": {
                "name": "TD误差 / TD Error",
                "type": "scalar",
                "display": False,
            },
            # 当前探索率 / Current epsilon
            "epsilon": {
                "name": "探索率 / Epsilon",
                "type": "scalar",
                "display": True,
            },
            # 学习率 / Learning rate
            "learning_rate": {
                "name": "学习率 / Learning Rate",
                "type": "scalar",
                "display": False,
            },
            # 回合步数 / Episode steps
            "episode_steps": {
                "name": "回合步数 / Episode Steps",
                "type": "scalar",
                "display": True,
            },
            # 配送数量 / Number of deliveries
            "deliveries": {
                "name": "配送完成数 / Deliveries Completed",
                "type": "scalar",
                "display": True,
            },
            # 是否全部完成 / Whether all orders completed
            "all_delivered": {
                "name": "全部配送完成 / All Orders Delivered",
                "type": "scalar",
                "display": True,
            },
            # Q表已访问状态数 / Visited state count
            "visited_states": {
                "name": "已访问状态数 / Visited States",
                "type": "scalar",
                "display": False,
            },
        },

        # ============================================================
        # 评估指标组
        # Evaluation metric groups
        # ============================================================
        "eval": {
            # 平均评估奖励 / Average evaluation reward
            "eval_avg_reward": {
                "name": "评估平均奖励 / Eval Average Reward",
                "type": "scalar",
                "display": True,
            },
            # 平均评估步数 / Average evaluation steps
            "eval_avg_steps": {
                "name": "评估平均步数 / Eval Average Steps",
                "type": "scalar",
                "display": True,
            },
            # 平均配送完成数 / Average deliveries per eval
            "eval_avg_deliveries": {
                "name": "评估平均配送数 / Eval Avg Deliveries",
                "type": "scalar",
                "display": True,
            },
            # 评估成功率 (全部5单完成) / Eval success rate (all 5 orders completed)
            "eval_success_rate": {
                "name": "评估成功率 / Eval Success Rate",
                "type": "scalar",
                "display": True,
            },
            # 评估最大奖励 / Eval max reward
            "eval_max_reward": {
                "name": "评估最大奖励 / Eval Max Reward",
                "type": "scalar",
                "display": False,
            },
            # 评估最小奖励 / Eval min reward
            "eval_min_reward": {
                "name": "评估最小奖励 / Eval Min Reward",
                "type": "scalar",
                "display": False,
            },
        },

        # ============================================================
        # 模型指标组
        # Model metric groups
        # ============================================================
        "model": {
            # Q表大小 (已访问状态数) / Q-table size (visited states)
            "q_table_size": {
                "name": "Q表大小 / Q-Table Size",
                "type": "scalar",
                "display": False,
            },
            # 累计训练更新次数 / Total training update count
            "train_steps": {
                "name": "训练步数 / Training Steps",
                "type": "scalar",
                "display": False,
            },
        },

        # ============================================================
        # 全局配置
        # Global configuration
        # ============================================================
        "config": {
            "eval_interval": Config.EVAL_INTERVAL,
            "eval_episodes": Config.EVAL_EPISODES,
            "num_episodes": Config.NUM_EPISODES,
            "max_steps": Config.MAX_STEPS,
        },
    }

    return monitor


# ============================================================
# 模块级单例 (Module-level singleton)
# ============================================================

# 监控配置单例，避免重复构建
# Monitor config singleton to avoid repeated construction
MONITOR_CONFIG = build_monitor()
