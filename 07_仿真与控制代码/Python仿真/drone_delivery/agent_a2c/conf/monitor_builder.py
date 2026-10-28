# -*- coding: utf-8 -*-
"""
================================================================================
conf/monitor_builder.py — A2C 监控指标构建器 / A2C Monitor Builder (KaiWu-Style)
================================================================================
职责 / Responsibility:
    构建 KaiWu 风格的监控配置, 将 A2C 训练过程中的各类指标分组管理,
    方便接入监控面板进行实时可视化。
    Build KaiWu-style monitor configuration for A2C training metrics,
    enabling real-time visualization in the monitoring dashboard.

参考 / Reference:
    - kaiwu_infinity_valley agent_a2c 的 monitor_builder 模式
    - agent_dqn 的 monitor_builder 实现
领域适配 / Domain Adaptation:
    - 关注策略损失、价值损失、熵值和配送业务指标
    - Track policy loss, value loss, entropy, and delivery business metrics
================================================================================
"""

from typing import Dict, List, Any


# =============================================================================
# 指标组定义 / Metric Group Definitions
# =============================================================================

ALGORITHM_METRICS: List[str] = [
    "reward",
    "policy_loss",
    "value_loss",
    "entropy",
    "gradient_norm",
]
"""算法指标组 / Algorithm metrics:
    reward            — 回合累积回报 / Cumulative episode reward
    policy_loss       — 策略损失 (actor loss) / Policy loss (actor loss)
    value_loss        — 价值损失 (critic loss) / Value loss (critic loss)
    entropy           — 策略熵值 / Policy entropy
    gradient_norm     — 梯度范数 / Gradient norm
"""

TRAINING_METRICS: List[str] = [
    "episode",
    "episode_length",
    "learning_rate",
]
"""训练进度指标组 / Training progress metrics:
    episode           — 当前回合编号 / Current episode number
    episode_length    — 回合步数 / Episode length
    learning_rate     — 当前学习率 / Current learning rate
"""

DELIVERY_METRICS: List[str] = [
    "total_orders",
    "completed",
    "on_time",
    "failed",
    "completion_rate",
    "on_time_rate",
]
"""无人机配送业务指标组 / Drone delivery business metrics:
    total_orders      — 回合总订单数 / Total orders per episode
    completed         — 完成订单数 / Completed orders
    on_time           — 准时送达数 / On-time deliveries
    failed            — 失败订单数 / Failed orders
    completion_rate   — 完成率 / Completion rate
    on_time_rate      — 准时率 / On-time rate
"""


class MonitorConfig:
    """
    KaiWu 风格 A2C 监控配置 / KaiWu-Style A2C Monitor Configuration

    将所有监控指标按类别分组, 提供扁平列表供外部使用者查询。
    Groups all monitor metrics by category and provides a flat list for
    external consumers to query.

    Usage / 用法::

        from conf.monitor_builder import monitor_config as mc
        print(mc.all_metrics)   # 获取全部指标名称 / get all metric names
        print(mc.groups)        # 获取分组字典 / get groups dict
    """

    def __init__(self) -> None:
        """构建指标分组 / Build metric groups"""
        self._groups: Dict[str, List[str]] = {
            "algorithm": ALGORITHM_METRICS,
            "training": TRAINING_METRICS,
            "delivery": DELIVERY_METRICS,
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
            group_name (str): 分组名称, 如 "algorithm", "training", "delivery"
                              Group name, e.g. "algorithm"

        Returns / 返回:
            List[str]: 该组内的指标名称列表, 若不存在则返回空列表
                       Metric name list within the group; empty if not found
        """
        return self._groups.get(group_name, [])

    def to_dict(self) -> Dict[str, Any]:
        """
        导出为可序列化字典 / Export as serializable dict.

        Returns / 返回:
            Dict[str, Any]: { "groups": {...}, "all_metrics": [...] }
        """
        return {
            "groups": dict(self._groups),
            "all_metrics": list(self.all_metrics),
        }


# 单例监控配置 / Singleton monitor config
monitor_config = MonitorConfig()
