"""多目标优先、返航可行的分层配送策略。

该模块把峡谷漫步增强策略中的“收集目标子集并预留到终点的预算”
映射为无人机的定序层。PPO 只输出连续飞行动作，目标选择由这里给出
可解释的 12 维评分。小规模任务使用全排列精确搜索；目标数较多时可
替换为 Orienteering/VRP 求解器而不改变环境动作协议。
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

from environment.wuhan_drone_env import (
    DroneRuntimeState,
    WuhanDroneDeliveryEnv,
)


@dataclass(frozen=True)
class DeliveryRoutePlan:
    """一次可行的“配送若干目标后返航”计划。"""

    target_ids: Tuple[int, ...]
    estimated_distance_m: float
    estimated_time_s: float
    estimated_energy_wh: float
    feasible: bool


class ScoreFirstDeliveryPlanner:
    """按“目标数优先、总代价次优”选择配送序列。

    比较键采用 ``(目标数, -距离)``，因此不会像单纯加权总分那样因
    少走一段路而放弃一个本来可安全完成的配送目标。
    """

    def __init__(self, max_exact_targets: int = 8):
        if max_exact_targets < 1:
            raise ValueError("max_exact_targets 至少为 1")
        self.max_exact_targets = max_exact_targets

    @staticmethod
    def _route_metrics(
        env: WuhanDroneDeliveryEnv,
        state: DroneRuntimeState,
        target_ids: Sequence[int],
    ) -> Tuple[float, float, float]:
        points = [state.position_m]
        points.extend(
            env._service_position(env.scene.task_by_id(task_id))
            for task_id in target_ids
        )
        points.append(env._service_position(env.scene.depot()))
        distance_m = sum(
            float(np.linalg.norm(end - start))
            for start, end in zip(points, points[1:])
        )
        distance_m *= env.config.return_route_factor
        time_s = distance_m / max(
            env.config.max_speed_xy_mps, 1e-6
        )
        energy_wh = (
            distance_m * env.config.return_energy_wh_per_m
        )
        return distance_m, time_s, energy_wh

    @staticmethod
    def _is_feasible(
        env: WuhanDroneDeliveryEnv,
        state: DroneRuntimeState,
        time_s: float,
        energy_wh: float,
    ) -> bool:
        remaining_time_s = (
            env.config.max_episode_time_s - env.sim_time_s
        )
        reserve_wh = (
            env.config.battery_capacity_wh
            * env.config.reserve_soc
        )
        return (
            time_s + env.config.return_time_margin_s
            < remaining_time_s
            and energy_wh + reserve_wh < state.battery_wh
        )

    def plan(
        self,
        env: WuhanDroneDeliveryEnv,
        drone_id: int = 0,
        excluded_target_ids: Iterable[int] = (),
    ) -> DeliveryRoutePlan:
        state = env.drone_states[drone_id]
        depot_id = env.scene.depot().task_id
        excluded = set(excluded_target_ids)
        pending = sorted(
            task.task_id
            for task in env.scene.tasks
            if task.task_id != depot_id
            and task.task_id not in excluded
            and not env.served_mask[task.task_id - 1]
        )

        if env._mission_phase == "return" or not pending:
            distance_m, time_s, energy_wh = self._route_metrics(
                env, state, ()
            )
            return DeliveryRoutePlan(
                (), distance_m, time_s, energy_wh, True
            )

        # 精确枚举适用于当前每分区 2 个配送点，也覆盖小规模扩展场景。
        # 超过阈值时先按离当前位置的距离截取候选，避免阶乘爆炸。
        if len(pending) > self.max_exact_targets:
            pending = sorted(
                pending,
                key=lambda task_id: float(
                    np.linalg.norm(
                        state.position_m
                        - env._service_position(
                            env.scene.task_by_id(task_id)
                        )
                    )
                ),
            )[: self.max_exact_targets]

        best: Optional[DeliveryRoutePlan] = None
        best_key = None
        for count in range(1, len(pending) + 1):
            for order in permutations(pending, count):
                distance_m, time_s, energy_wh = self._route_metrics(
                    env, state, order
                )
                if not self._is_feasible(
                    env, state, time_s, energy_wh
                ):
                    continue
                key = (count, -distance_m, tuple(-item for item in order))
                if best_key is None or key > best_key:
                    best_key = key
                    best = DeliveryRoutePlan(
                        tuple(order),
                        distance_m,
                        time_s,
                        energy_wh,
                        True,
                    )

        if best is not None:
            return best
        distance_m, time_s, energy_wh = self._route_metrics(
            env, state, ()
        )
        return DeliveryRoutePlan(
            (), distance_m, time_s, energy_wh, False
        )

    @staticmethod
    def target_scores(
        plan: DeliveryRoutePlan, depot_id: int
    ) -> np.ndarray:
        """把路线顺序编码成环境所需的 12 维目标评分。"""
        scores = np.full(12, -1.0, dtype=np.float32)
        if not plan.target_ids:
            scores[depot_id - 1] = 1.0
            return scores
        for rank, target_id in enumerate(plan.target_ids):
            scores[target_id - 1] = max(0.1, 1.0 - 0.1 * rank)
        return scores

    def compose_action(
        self,
        env: WuhanDroneDeliveryEnv,
        physical_actions,
    ) -> np.ndarray:
        """合并 PPO 的 4 维控制量与规划器的 12 维目标评分。"""
        physical = np.asarray(physical_actions, dtype=np.float32)
        if env.num_drones == 1:
            if physical.shape != (4,):
                raise ValueError("单机 physical_actions 形状必须为 (4,)")
            plan = self.plan(env, 0)
            return np.concatenate(
                [
                    np.clip(physical, -1.0, 1.0),
                    self.target_scores(
                        plan, env.scene.depot().task_id
                    ),
                ]
            ).astype(np.float32)

        if physical.shape != (env.num_drones, 4):
            raise ValueError(
                "多机 physical_actions 形状必须为 "
                f"({env.num_drones}, 4)"
            )
        combined = []
        assigned_first_targets = set()
        for drone_id in range(env.num_drones):
            plan = self.plan(
                env, drone_id, assigned_first_targets
            )
            if plan.target_ids:
                assigned_first_targets.add(plan.target_ids[0])
            combined.append(
                np.concatenate(
                    [
                        np.clip(
                            physical[drone_id], -1.0, 1.0
                        ),
                        self.target_scores(
                            plan, env.scene.depot().task_id
                        ),
                    ]
                )
            )
        return np.asarray(combined, dtype=np.float32)
