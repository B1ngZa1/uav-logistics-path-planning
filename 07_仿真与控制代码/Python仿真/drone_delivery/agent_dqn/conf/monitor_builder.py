# -*- coding: utf-8 -*-
"""
================================================================================
conf/monitor_builder.py — 监控指标构建器 / Monitor Builder (KaiWu-Style)
================================================================================
职责 / Responsibility:
    构建 KaiWu 风格的监控配置, 将训练过程中的各类指标分组管理,
    方便接入监控面板进行实时可视化。
    Build KaiWu-style monitor configuration with metric groups for
    real-time visualization in the monitoring dashboard.
参考 / Reference:
    - KaiWu DRL 框架的 MonitorConfig 设计模式
    - Gorge Walk DIY agent 的监控分组方式
领域适配 / Domain Adaptation:
    - 关注模型性能+训练进度+配送业务指标三个维度
    - Track model performance, training progress, and delivery business metrics
================================================================================
"""

from typing import Dict, List, Any


# =============================================================================
# 指标组定义 / Metric Group Definitions
# =============================================================================

MODEL_PERFORMANCE_METRICS: List[str] = [
    "loss",
    "q_value_mean",
    "q_value_max",
    "td_error_mean",
    "gradient_norm",
]
"""模型性能指标组 / Model performance metrics:
    loss              — 单步TD误差的MSE损失 / MSE of TD error per step
    q_value_mean      — Q值均值 / Mean Q-value
    q_value_max       — Q值最大值 / Max Q-value
    td_error_mean     — TD误差均值 / Mean TD error
    gradient_norm     — 梯度范数 / Gradient norm
"""

TRAINING_PROGRESS_METRICS: List[str] = [
    "epsilon",
    "total_steps",
    "episode",
    "episode_reward",
    "episode_length",
    "buffer_size",
]
"""训练进度指标组 / Training progress metrics:
    epsilon           — 当前探索率 / Current exploration rate
    total_steps       — 全局训练步数 / Global training steps
    episode           — 当前回合编号 / Current episode number
    episode_reward    — 回合累积奖励 / Cumulative episode reward
    episode_length    — 回合步数 / Episode length in steps
    buffer_size       — 当前缓冲区样本数 / Current replay buffer size
"""

DRONE_DELIVERY_METRICS: List[str] = [
    "total_orders",
    "completed",
    "on_time",
    "failed",
    "completion_rate",
    "on_time_rate",
    "avg_step_distance",
]
"""无人机配送业务指标组 / Drone delivery business metrics:
    total_orders      — 回合同总订单数 / Total orders per episode
    completed         — 完成订单数 / Number of completed orders
    on_time           — 准时送达数 / Number of on-time deliveries
    failed            — 失败订单数 / Number of failed orders
    completion_rate   — 完成率 = completed / total_orders
    on_time_rate      — 准时率 = on_time / total_orders
    avg_step_distance — 平均每步移动距离 / Average distance per step
"""


class MonitorConfig:
    """
    KaiWu 风格监控配置 / KaiWu-Style Monitor Configuration

    将所有监控指标按类别分组, 提供扁平列表供外部使用者查询。
    Groups all monitor metrics by category and provides a flat list for
    external consumers to query.

    Usage / 用法::

        from conf.monitor_builder import MonitorConfig
        mc = MonitorConfig()
        print(mc.all_metrics)   # 获取全部指标名称 / get all metric names
        print(mc.groups)        # 获取分组字典 / get groups dict
    """

    def __init__(self) -> None:
        """构建指标分组 / Build metric groups"""
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
            group_name (str): 分组名称, 如 "model_performance"
                              Group name, e.g. "model_performance"

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
                "groups": {...},       # 指标分组 / metric groups
                "all_metrics": [...]   # 扁平列表 / flat list
            }
        """
        return {
            "groups": dict(self._groups),
            "all_metrics": list(self.all_metrics),
        }


# 单例监控配置 / Singleton monitor config
monitor_config = MonitorConfig()
