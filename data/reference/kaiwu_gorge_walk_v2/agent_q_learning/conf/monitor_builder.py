#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


from kaiwudrl.common.monitor.monitor_config_builder import MonitorConfigBuilder


def build_monitor():
    """
    # This function is used to create monitoring panel configurations for custom indicators.
    # 该函数用于创建自定义指标的监控面板配置。
    """
    monitor = MonitorConfigBuilder()

    config_dict = (
        monitor.title("峡谷漫步V2")
        .add_group(
            group_name="算法指标",
            group_name_en="algorithm",
        )
        .add_panel(
            name="累积回报",
            name_en="reward",
            type="line",
            unit="",
        )
        .add_metric(
            metrics_name="reward",
            expr="avg(reward{})",
        )
        .end_panel()
        .add_panel(
            name="训练损失",
            name_en="loss",
            type="line",
            unit="",
        )
        .add_metric(
            metrics_name="loss",
            expr="avg(loss{})",
        )
        .end_panel()
        .add_panel(
            name="TD误差",
            name_en="td_error",
            type="line",
            unit="",
        )
        .add_metric(
            metrics_name="td_error",
            expr="avg(td_error{})",
        )
        .end_panel()
        .add_panel(
            name="胜率",
            name_en="win_rate",
            type="line",
            unit="%",
        )
        .add_metric(
            metrics_name="win_rate",
            expr="avg(win_rate{})",
        )
        .end_panel()
        .add_panel(
            name="平均Episode步数",
            name_en="episode_steps",
            type="line",
            unit="",
        )
        .add_metric(
            metrics_name="episode_steps",
            expr="avg(episode_steps{})",
        )
        .end_panel()
        .add_panel(
            name="平均宝箱收集数",
            name_en="treasure_count",
            type="stat",
            unit="",
        )
        .add_metric(
            metrics_name="treasure_count",
            expr="avg(treasure_count{})",
        )
        .end_panel()
        .add_panel(
            name="探索率",
            name_en="epsilon",
            type="line",
            unit="",
        )
        .add_metric(
            metrics_name="epsilon",
            expr="avg(epsilon{})",
        )
        .end_panel()
        .end_group()
        .build()
    )
    return config_dict
