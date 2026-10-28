#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright (c) 2026 Drone Delivery RL Project. All Rights Reserved.
###########################################################################
"""
Monitor Builder for Monte Carlo Agent in Drone Food Delivery.

蒙特卡洛智能体的监控构建器，用于无人机外卖配送任务。
参考实现：Kaiwu Gorge Walk v2 agent_monte_carlo/conf/monitor_builder.py

职责 / Responsibility:
    定义需要监控的指标及其配置，用于训练过程中的可视化与日志记录。
    Defines metrics to monitor and their configuration for visualization
    and logging during training.

蒙特卡洛特有监控指标 / Monte Carlo-specific monitoring metrics:
    - mean_return: 回合平均累积回报 (不同于TD方法, MC使用完整G_t)
      Episode mean cumulative return (unlike TD methods, MC uses full G_t)
    - num_first_visits: 每回合首次访问的(s,a)对数量
      Number of first-visit (s,a) pairs per episode
    - num_states_visited: 累计已访问状态数量
      Cumulative number of visited states
    - q_value_mean: Q值均值 (MC的Q值收敛速度)
      Mean Q-value (MC Q-value convergence speed)

领域适配 / Domain Adaptation:
    Gorge Walk v2 → Drone Delivery:
        treasure_collected → orders_completed  (完成配送订单数)
        score              → delivery_score    (配送得分)
        step               → episode_steps     (回合步数)
"""


from kaiwudrl.common.monitor.monitor_config_builder import (
    MonitorConfigBuilder,
    Dashboard,
    Timeseries,
    Histogram,
    Scalar,
)
from agent_monte_carlo.conf.conf import Config


def build_monitor():
    """
    Build the monitor configuration for the Monte Carlo drone delivery agent.

    构建蒙特卡洛无人机配送智能体的监控配置。

    Returns:
        MonitorConfigBuilder instance with all metrics configured.
        配置了所有监控指标的MonitorConfigBuilder实例。
    """
    monitor = MonitorConfigBuilder()

    # =========================================================================
    # 训练面板 (Training Dashboard)
    # =========================================================================
    # 主要训练指标面板
    # Primary training metrics dashboard
    monitor.add_dashboard(
        Dashboard()
        .name("monte_carlo_train")
        .label("蒙特卡洛训练指标 / MC Training Metrics")
        .interval(1)  # 每回合更新一次 / Update every episode
        .add_panel(
            Timeseries()
            .title("回合得分 / Episode Score")
            .add_metric("episode_score", "score", "sum")
        )
        .add_panel(
            Timeseries()
            .title("回合步数 / Episode Steps")
            .add_metric("episode_steps", "steps", "last")
        )
        .add_panel(
            Timeseries()
            .title("完成订单数 / Orders Completed")
            .add_metric("orders_done", "count", "last")
        )
        .add_panel(
            Timeseries()
            .title("已访问状态数 / Visited States")
            .add_metric("visited_states", "count", "last")
        )
        .add_panel(
            Timeseries()
            .title("首次访问(s,a)对数 / First-Visit Pairs")
            .add_metric("num_updates", "count", "last")
        )
        .add_panel(
            Timeseries()
            .title("回合累积回报 / Episode Mean Return (G_t)")
            .add_metric("mean_return", "avg_G", "last")
        )
    )

    # =========================================================================
    # 学习指标面板 (Learning Metrics Dashboard)
    # =========================================================================
    # MC学习相关指标
    # MC learning-related metrics
    monitor.add_dashboard(
        Dashboard()
        .name("monte_carlo_learning")
        .label("MC学习指标 / MC Learning Metrics")
        .interval(1)
        .add_panel(
            Timeseries()
            .title("总TD误差 / Total TD Error")
            .add_metric("total_error", "error", "last")
        )
        .add_panel(
            Timeseries()
            .title("Q值均值 / Mean Q-Value")
            .add_metric("mean_q", "q_val", "last")
        )
        .add_panel(
            Timeseries()
            .title("Q值最大值 / Max Q-Value")
            .add_metric("max_q", "q_val", "last")
        )
        .add_panel(
            Timeseries()
            .title("学习率 / Learning Rate")
            .add_metric("learning_rate", "lr", "last")
        )
        .add_panel(
            Timeseries()
            .title("探索率 (epsilon) / Exploration Rate")
            .add_metric("epsilon", "eps", "last")
        )
        .add_panel(
            Timeseries()
            .title("训练步数 / Training Step")
            .add_metric("train_step", "step", "last")
        )
    )

    # =========================================================================
    # Q值分布面板 (Q-Value Distribution Dashboard)
    # =========================================================================
    # 监控Q值分布，检测MC学习的收敛情况
    # Monitor Q-value distribution to detect MC learning convergence
    monitor.add_dashboard(
        Dashboard()
        .name("monte_carlo_q_dist")
        .label("Q值分布 / Q-Value Distribution")
        .interval(10)  # 每10回合更新一次以减少开销
        .add_panel(
            Histogram()
            .title("Q值直方图 / Q-Value Histogram")
            .add_metric("q_histogram", "q_val", "histogram")
        )
    )

    # =========================================================================
    # 评估面板 (Evaluation Dashboard)
    # =========================================================================
    # 评估/验证指标面板
    # Evaluation/validation metrics dashboard
    monitor.add_dashboard(
        Dashboard()
        .name("monte_carlo_eval")
        .label("MC评估指标 / MC Evaluation Metrics")
        .interval(Config.VALIDATION_INTERVAL)  # 每50回合评估一次
        .add_panel(
            Timeseries()
            .title("评估平均得分 / Eval Avg Score")
            .add_metric("eval_avg_score", "eval_score", "avg")
        )
        .add_panel(
            Timeseries()
            .title("评估平均步数 / Eval Avg Steps")
            .add_metric("eval_avg_steps", "eval_steps", "avg")
        )
        .add_panel(
            Timeseries()
            .title("评估完成率 / Eval Completion Rate")
            .add_metric("eval_completion_rate", "eval_cr", "last")
        )
        .add_panel(
            Scalar()
            .title("近100回合胜率 (收敛检测) / Win Rate (Convergence)")
            .add_metric("win_rate_100", "conv", "last")
        )
    )

    return monitor


def build_monitor_for_workflow():
    """
    Build monitor configuration for use within the training workflow.

    为训练工作流构建监控配置。

    这是 workflow/train_workflow.py 中使用的便捷函数。
    Convenience function used in workflow/train_workflow.py.

    Returns:
        MonitorConfigBuilder instance / MonitorConfigBuilder实例
    """
    return build_monitor()
