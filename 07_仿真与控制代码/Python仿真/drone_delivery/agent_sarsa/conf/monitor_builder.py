#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
无人机配送 — SARSA监控面板配置
Drone Delivery — SARSA Monitor Panel Configuration

职责 / Responsibility:
    构建SARSA训练过程的可视化监控面板配置。
    Build the visualization monitor panel configuration for SARSA training.

参考 / Reference:
    kaiwu_gorge_walk_v2/agent_sarsa/conf/monitor_builder.py



领域适配 / Domain Adaptation:
    峡谷行走(Gorge Walk) — 寻路任务
      → 无人机配送(Drone Delivery) — 取送货任务
    Gorge Walk (pathfinding task)
      → Drone Delivery (pickup-delivery task)
"""


def build_monitor():
    """
    构建SARSA无人机配送的监控面板配置。
    Build monitor panel configuration for SARSA drone delivery.

    Returns:
        config_dict: 监控面板配置字典 / Monitor panel configuration dict
    """
    # 使用Kaiwu框架的MonitorConfigBuilder构建监控面板
    # Use Kaiwu framework's MonitorConfigBuilder to build monitor panels
    from kaiwudrl.common.monitor.monitor_config_builder import MonitorConfigBuilder

    monitor = MonitorConfigBuilder()

    config_dict = (
        monitor.title("无人机配送-SARSA")
        # ============================================================
        # 面板组: 算法指标
        # Panel group: Algorithm metrics
        # ============================================================
        .add_group(group_name="算法指标", group_name_en="algorithm")
        # --- 面板: 累积回报 / Panel: Cumulative Return ---
        .add_panel(
            name="累积回报",
            name_en="cumulative_reward",
            description="每回合的总累积回报 / Total cumulative reward per episode",
            type="line",
        )
        .add_metric(metrics_name="累积回报", expr="avg(reward{})")
        .end_panel()
        # --- 面板: TD误差 / Panel: TD Error ---
        .add_panel(
            name="TD误差",
            name_en="td_error",
            description="SARSA时序差分误差 / SARSA temporal difference error",
            type="line",
        )
        .add_metric(metrics_name="TD误差", expr="avg(td_error{})")
        .end_panel()
        # --- 面板: Epsilon探索率 / Panel: Epsilon ---
        .add_panel(
            name="探索率",
            name_en="epsilon",
            description="Epsilon-greedy探索率随回合的变化 / Epsilon decay over episodes",
            type="line",
        )
        .add_metric(metrics_name="探索率", expr="avg(epsilon{})")
        .end_panel()
        # --- 面板: 学习率 / Panel: Learning Rate ---
        .add_panel(
            name="学习率",
            name_en="learning_rate",
            description="当前学习率 / Current learning rate",
            type="line",
        )
        .add_metric(metrics_name="学习率", expr="avg(lr{})")
        .end_panel()
        .end_group()
        # ============================================================
        # 面板组: 任务指标
        # Panel group: Task metrics
        # ============================================================
        .add_group(group_name="任务指标", group_name_en="task")
        # --- 面板: 配送完成数 / Panel: Deliveries ---
        .add_panel(
            name="配送完成数",
            name_en="deliveries",
            description="每回合完成的配送订单数 / Number of completed deliveries per episode",
            type="line",
        )
        .add_metric(metrics_name="配送完成数", expr="avg(deliveries{})")
        .end_panel()
        # --- 面板: 回合步数 / Panel: Episode Steps ---
        .add_panel(
            name="回合步数",
            name_en="episode_steps",
            description="每回合执行的步数(越少效率越高) / Steps per episode (fewer = more efficient)",
            type="line",
        )
        .add_metric(metrics_name="回合步数", expr="avg(episode_steps{})")
        .end_panel()
        # --- 面板: 获胜率 / Panel: Win Rate ---
        .add_panel(
            name="获胜率",
            name_en="win_rate",
            description="最近N回合的获胜率(全部完成并返回配送中心) / Win rate over recent N episodes",
            type="line",
        )
        .add_metric(metrics_name="获胜率", expr="avg(win_rate{})")
        .end_panel()
        .end_group()
        # ============================================================
        # 面板组: 表格统计
        # Panel group: Table statistics
        # ============================================================
        .add_group(group_name="表格统计", group_name_en="table_stats")
        # --- 面板: 已访问状态数 / Panel: Visited States ---
        .add_panel(
            name="已访问状态数",
            name_en="visited_states",
            description="Q表中已访问过的状态数量 / Number of visited states in Q-table",
            type="stat",
            unit="个",
        )
        .add_metric(metrics_name="已访问状态数", expr="avg(visited_states{})")
        .end_panel()
        .end_group()
        .build()
    )
    return config_dict
