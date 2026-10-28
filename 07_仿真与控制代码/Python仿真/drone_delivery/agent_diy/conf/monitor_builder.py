# -*- coding: utf-8 -*-
"""
================================================================================
conf/monitor_builder.py — 监控指标构建器 (DIY占位符)
                         Monitor Builder (DIY Placeholder)
================================================================================
职责 / Responsibility:
    构建 KaiWu 风格的监控配置占位符。
    学生可在此定义自己的监控指标分组, 方便接入监控面板进行实时可视化。
    Build KaiWu-style monitor configuration placeholder.
    Students define their own monitoring metric groups here for real-time
    visualization in the monitoring dashboard.

参考 / Reference:
    - agent_dqn/conf/monitor_builder.py (完整实现参考 / full reference)
    - KaiWu DRL 框架的 MonitorConfig 设计模式

使用方法 / How to Use:
    1. 定义指标名称列表 (如 ["loss", "reward", "epsilon"])
    2. 将指标分组到 _groups 字典中
    3. 通过 monitor_config.all_metrics 获取扁平列表供外部消费

    1. Define metric name lists (e.g. ["loss", "reward", "epsilon"])
    2. Group metrics in the _groups dict
    3. Access flat list via monitor_config.all_metrics for external consumers
================================================================================
"""

from typing import Dict, List, Any


# =============================================================================
# 指标组定义 — 占位符, 学生可自行修改
# Metric Group Definitions — placeholders; students may modify these
# =============================================================================

# TODO: 学生在此定义需要监控的指标 / Students define metrics to monitor here

MODEL_PERFORMANCE_METRICS: List[str] = [
    "loss",
    "reward_mean",
]
"""
模型性能指标组 (占位符) / Model performance metrics (placeholder):
    loss              — 损失值 / Loss value
    reward_mean       — 平均奖励 / Mean reward
"""

TRAINING_PROGRESS_METRICS: List[str] = [
    "epsilon",
    "episode",
    "episode_reward",
    "episode_length",
]
"""
训练进度指标组 (占位符) / Training progress metrics (placeholder):
    epsilon           — 当前探索率 / Current exploration rate
    episode           — 当前回合编号 / Current episode number
    episode_reward    — 回合累积奖励 / Cumulative episode reward
    episode_length    — 回合步数 / Episode length in steps
"""

DRONE_DELIVERY_METRICS: List[str] = [
    "completed",
    "completion_rate",
]
"""
无人机配送业务指标组 (占位符) / Drone delivery business metrics (placeholder):
    completed         — 完成订单数 / Number of completed orders
    completion_rate   — 完成率 / Completion rate
"""


class MonitorConfig:
    """
    KaiWu 风格监控配置 (占位符) / KaiWu-Style Monitor Configuration (placeholder)

    将所有监控指标按类别分组, 提供扁平列表供外部使用者查询。
    Groups all monitor metrics by category and provides a flat list for
    external consumers to query.

    Usage / 用法::

        from conf.monitor_builder import MonitorConfig
        mc = MonitorConfig()
        print(mc.all_metrics)   # 获取全部指标名称 / get all metric names
        print(mc.groups)        # 获取分组字典 / get groups dict

    TODO: 学生可添加更多指标分组 / Students may add more metric groups
    """

    def __init__(self) -> None:
        """构建指标分组 (占位符) / Build metric groups (placeholder)"""
        # TODO: 学生在此添加/修改指标分组 / Students add/modify metric groups here
        self._groups: Dict[str, List[str]] = {
            "model_performance": MODEL_PERFORMANCE_METRICS,
            "training_progress": TRAINING_PROGRESS_METRICS,
            "drone_delivery": DRONE_DELIVERY_METRICS,
        }

    @property
    def groups(self) -> Dict[str, List[str]]:
        """返回指标分组字典 / Return metric groups dict"""
        return self._groups

    @property
    def all_metrics(self) -> List[str]:
        """返回所有指标的扁平列表 / Return flat list of all metric names"""
        result: List[str] = []
        for metric_list in self._groups.values():
            result.extend(metric_list)
        return result

    def get_group(self, group_name: str) -> List[str]:
        """
        获取指定分组的指标列表 / Get metric list for a specific group.

        Args / 参数:
            group_name (str): 分组名称 / Group name

        Returns / 返回:
            List[str]: 该组内的指标名称列表, 若不存在则返回空列表
                       Metric name list within the group; empty if not found
        """
        return self._groups.get(group_name, [])

    def to_dict(self) -> Dict[str, Any]:
        """
        导出为可序列化字典 / Export as serializable dict.

        Returns / 返回:
            Dict[str, Any]: {
                "groups": {...},
                "all_metrics": [...]
            }
        """
        return {
            "groups": dict(self._groups),
            "all_metrics": list(self.all_metrics),
        }


# 单例监控配置 / Singleton monitor config
monitor_config = MonitorConfig()
