"""数据驱动的武汉参考场景多无人机配送环境。

接口遵循 Gymnasium：

``reset() -> observation, info``
``step()  -> observation, reward, terminated, truncated, info``

单机动作是 16 维连续向量：三轴加速度、偏航角速度和 12 个任务点评分。
多机动作形状为 ``(num_drones, 16)``，每架无人机仍使用相同的动作语义。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces

    GYMNASIUM_API = True
except ImportError:  # pragma: no cover - 兼容项目现有 gym 依赖
    import gym
    from gym import spaces

    GYMNASIUM_API = False

from .wuhan_scene import DynamicEntity, ScenePackage, TaskPoint


@dataclass(frozen=True)
class WuhanEnvConfig:
    zone_id: str = "A"
    scene_dir: Optional[str] = None
    num_drones: int = 1
    physics_dt_s: float = 0.1
    control_dt_s: float = 0.5
    max_episode_time_s: float = 900.0
    min_altitude_m: float = 10.0
    max_altitude_m: float = 200.0
    max_speed_xy_mps: float = 15.0
    max_speed_z_mps: float = 5.0
    max_accel_xy_mps2: float = 5.0
    max_accel_z_mps2: float = 3.0
    max_yaw_rate_rps: float = 1.2
    drone_radius_m: float = 1.0
    safety_clearance_m: float = 8.0
    battery_capacity_wh: float = 250.0
    reserve_soc: float = 0.15
    map_size: int = 32
    map_extent_m: float = 256.0
    lidar_rays: int = 36
    lidar_range_m: float = 180.0
    target_switch_margin: float = 0.15
    target_replan_interval_s: float = 2.0
    reward_progress: float = 12.0
    reward_delivery: float = 120.0
    reward_all_delivered: float = 300.0
    reward_return: float = 500.0
    penalty_mission_failure: float = 1000.0
    penalty_collision: float = 250.0
    penalty_hard_risk: float = 300.0
    penalty_clearance: float = 4.0
    penalty_energy_per_wh: float = 0.8
    penalty_soft_risk: float = 1.5
    penalty_jerk: float = 0.03
    penalty_time_per_s: float = 0.08
    return_time_margin_s: float = 30.0
    return_route_factor: float = 1.25
    return_energy_wh_per_m: float = 0.015
    # 1000 大于默认时限内全部时间分与返航分之和，使“多送一单”
    # 严格优先于次级的时间优化；这是对峡谷漫步 100 分宝箱的任务化增强。
    score_per_delivery: float = 1000.0
    score_safe_return: float = 150.0
    score_efficiency_per_s: float = 0.2
    seed: int = 42

    @classmethod
    def from_dict(
        cls, values: Optional[Dict[str, object]]
    ) -> "WuhanEnvConfig":
        if not values:
            return cls()
        allowed = cls.__dataclass_fields__.keys()
        return cls(
            **{key: value for key, value in values.items() if key in allowed}
        )

    def validate(self) -> None:
        if self.zone_id.upper() not in {"A", "B", "C", "D"}:
            raise ValueError("zone_id 必须为 A、B、C 或 D")
        if self.num_drones < 1:
            raise ValueError("num_drones 至少为 1")
        ratio = self.control_dt_s / self.physics_dt_s
        if abs(ratio - round(ratio)) > 1e-8 or ratio < 1:
            raise ValueError("control_dt_s 必须是 physics_dt_s 的正整数倍")
        if not 8 <= self.map_size <= 128:
            raise ValueError("map_size 应位于 [8, 128]")
        if not 8 <= self.lidar_rays <= 144:
            raise ValueError("lidar_rays 应位于 [8, 144]")
        if not 0.0 < self.reserve_soc < 1.0:
            raise ValueError("reserve_soc 必须位于 (0, 1)")
        if self.return_route_factor < 1.0:
            raise ValueError("return_route_factor 不能小于 1")
        if self.return_energy_wh_per_m <= 0.0:
            raise ValueError("return_energy_wh_per_m 必须为正数")


@dataclass
class DroneRuntimeState:
    position_m: np.ndarray
    velocity_mps: np.ndarray
    acceleration_mps2: np.ndarray
    previous_acceleration_mps2: np.ndarray
    yaw_rad: float
    yaw_rate_rps: float
    battery_wh: float
    payload_kg: float
    active_target_id: int
    previous_target_distance_m: float
    last_replan_time_s: float
    gps_bias_m: np.ndarray
    comm_quality: float
    collision_kind: Optional[str] = None


class WuhanDroneDeliveryEnv(gym.Env):
    """武汉脱敏参考场景中的单机/多机协同配送环境。"""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 2}

    def __init__(
        self,
        config: Optional[Dict[str, object] | WuhanEnvConfig] = None,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        self.config = (
            config
            if isinstance(config, WuhanEnvConfig)
            else WuhanEnvConfig.from_dict(config)
        )
        self.config.validate()
        self.render_mode = render_mode
        scene_dir = (
            Path(self.config.scene_dir) if self.config.scene_dir else None
        )
        self.scene = ScenePackage.load(self.config.zone_id, scene_dir)
        self.num_drones = self.config.num_drones
        self.physics_substeps = int(
            round(self.config.control_dt_s / self.config.physics_dt_s)
        )
        action_shape = (
            (16,) if self.num_drones == 1 else (self.num_drones, 16)
        )
        self.action_space = spaces.Box(
            -1.0, 1.0, shape=action_shape, dtype=np.float32
        )
        self.observation_space = self._build_observation_space()
        self.np_random = np.random.default_rng(self.config.seed)
        self.drone_states: List[DroneRuntimeState] = []
        self.dynamic_entities: List[DynamicEntity] = []
        self.served_mask = np.zeros(12, dtype=np.int8)
        self.sim_time_s = 0.0
        self.step_count = 0
        self.total_energy_wh = 0.0
        self._all_delivered_awarded = False
        self._termination_reason: Optional[str] = None
        self._last_reward_terms: Dict[str, float] = {}
        self._mission_phase = "delivery"
        self._return_trigger: Optional[str] = None
        self._mission_score = 0.0
        self._score_delta = 0.0

    def _build_observation_space(self):
        n = self.num_drones

        def shape(single_shape: Tuple[int, ...]) -> Tuple[int, ...]:
            return single_shape if n == 1 else (n,) + single_shape

        size = self.config.map_size
        rays = self.config.lidar_rays
        return spaces.Dict(
            {
                "flight": spaces.Box(
                    -1.0, 1.0, shape=shape((18,)), dtype=np.float32
                ),
                "payload_battery": spaces.Box(
                    0.0, 1.0, shape=shape((6,)), dtype=np.float32
                ),
                "delivery_mask": spaces.MultiBinary(12),
                "target": spaces.Box(
                    -1.0, 1.0, shape=shape((4,)), dtype=np.float32
                ),
                "semantic_map": spaces.Box(
                    0.0,
                    1.0,
                    shape=shape((9, size, size)),
                    dtype=np.float32,
                ),
                "building_height": spaces.Box(
                    0.0,
                    1.0,
                    shape=shape((1, size, size)),
                    dtype=np.float32,
                ),
                "lidar": spaces.Box(
                    0.0,
                    1.0,
                    shape=shape((3, rays)),
                    dtype=np.float32,
                ),
                "weather": spaces.Box(
                    0.0, 1.0, shape=shape((8,)), dtype=np.float32
                ),
                "gps": spaces.Box(
                    -1.0, 1.0, shape=shape((6,)), dtype=np.float32
                ),
                "communication": spaces.Box(
                    0.0, 1.0, shape=shape((4,)), dtype=np.float32
                ),
            }
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, object]] = None,
    ):
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
            try:
                super().reset(seed=seed)
            except TypeError:  # old gym
                pass
        _ = options or {}
        self.sim_time_s = 0.0
        self.step_count = 0
        self.total_energy_wh = 0.0
        self._termination_reason = None
        self._last_reward_terms = {}
        self._all_delivered_awarded = False
        self._mission_phase = "delivery"
        self._return_trigger = None
        self._mission_score = 0.0
        self._score_delta = 0.0
        self.served_mask = np.zeros(12, dtype=np.int8)
        self.dynamic_entities = self.scene.create_dynamic_entities()

        depot = self.scene.depot()
        delivery_tasks = [
            task for task in self.scene.tasks if task.task_id != depot.task_id
        ]
        initial_target = (
            delivery_tasks[0].task_id if delivery_tasks else depot.task_id
        )
        self.drone_states = []
        for drone_id in range(self.num_drones):
            offset = np.array(
                [2.5 * drone_id, -2.5 * drone_id, 0.0], dtype=np.float64
            )
            start = depot.position_enu_m.copy() + offset
            start[2] = max(
                self.config.min_altitude_m + 5.0,
                depot.position_enu_m[2] + 5.0 + 1.5 * drone_id,
            )
            start[2] = min(
                start[2], self.config.max_altitude_m - 5.0
            )
            target = self.scene.task_by_id(initial_target)
            self.drone_states.append(
                DroneRuntimeState(
                    position_m=start,
                    velocity_mps=np.zeros(3, dtype=np.float64),
                    acceleration_mps2=np.zeros(3, dtype=np.float64),
                    previous_acceleration_mps2=np.zeros(
                        3, dtype=np.float64
                    ),
                    yaw_rad=0.0,
                    yaw_rate_rps=0.0,
                    battery_wh=self.config.battery_capacity_wh,
                    payload_kg=sum(
                        task.demand_kg for task in delivery_tasks
                    )
                    / self.num_drones,
                    active_target_id=initial_target,
                    previous_target_distance_m=float(
                        np.linalg.norm(
                            start - self._service_position(target)
                        )
                    ),
                    last_replan_time_s=-self.config.target_replan_interval_s,
                    gps_bias_m=np.zeros(3, dtype=np.float64),
                    comm_quality=1.0,
                )
            )
        return self._get_observation(), self._get_info()

    def step(self, action):
        actions = self._validate_actions(action)
        previous = [
            {
                "distance": state.previous_target_distance_m,
                "acceleration": state.acceleration_mps2.copy(),
                "battery": state.battery_wh,
            }
            for state in self.drone_states
        ]
        reward_terms = {
            "progress": 0.0,
            "delivery": 0.0,
            "all_delivered": 0.0,
            "return": 0.0,
            "mission_failure": 0.0,
            "collision": 0.0,
            "hard_risk": 0.0,
            "clearance": 0.0,
            "energy": 0.0,
            "soft_risk": 0.0,
            "jerk": 0.0,
            "time": 0.0,
            "cooperation": 0.0,
        }
        self._score_delta = 0.0
        self._update_mission_phase()
        self._decode_targets(actions)
        terminated = False
        for _ in range(self.physics_substeps):
            self._update_dynamic_entities()
            for drone_id, state in enumerate(self.drone_states):
                self._advance_drone(state, actions[drone_id, :4])
                event = self._safety_event(state, drone_id)
                if event is not None:
                    kind, reason = event
                    state.collision_kind = kind
                    self._termination_reason = reason
                    if kind == "hard_risk":
                        reward_terms[
                            "hard_risk"
                        ] -= self.config.penalty_hard_risk
                    else:
                        reward_terms[
                            "collision"
                        ] -= self.config.penalty_collision
                    terminated = True
            self.sim_time_s += self.config.physics_dt_s
            if terminated:
                break

        for drone_id, state in enumerate(self.drone_states):
            self._reward_drone(
                drone_id, state, previous[drone_id], reward_terms
            )
        reward_terms[
            "time"
        ] -= self.config.penalty_time_per_s * self.config.control_dt_s
        reward_terms["cooperation"] += self._cooperation_reward()
        if not terminated:
            terminated = self._task_terminal(reward_terms)
        truncated = self.sim_time_s >= self.config.max_episode_time_s
        if truncated and not terminated:
            self._termination_reason = "time_limit"
            self._mission_score = 0.0
            reward_terms[
                "mission_failure"
            ] -= self.config.penalty_mission_failure
        elif terminated and self._termination_reason != "mission_complete":
            if self._termination_reason != "safe_return_partial":
                self._mission_score = 0.0

        self.step_count += 1
        reward = float(sum(reward_terms.values()))
        self._last_reward_terms = {
            key: float(value) for key, value in reward_terms.items()
        }
        return (
            self._get_observation(),
            reward,
            bool(terminated),
            bool(truncated),
            self._get_info(),
        )

    def _validate_actions(self, action) -> np.ndarray:
        array = np.asarray(action, dtype=np.float32)
        expected = (
            (1, 16)
            if self.num_drones == 1
            else (self.num_drones, 16)
        )
        if self.num_drones == 1 and array.shape == (16,):
            array = array[None, :]
        if array.shape != expected or not np.isfinite(array).all():
            raise ValueError(
                f"action 必须是有限数组，期望形状 {expected}，实际 {array.shape}"
            )
        return np.clip(array, -1.0, 1.0)

    def _valid_target_mask(
        self, state: Optional[DroneRuntimeState] = None
    ) -> np.ndarray:
        valid = np.zeros(12, dtype=bool)
        depot_id = self.scene.depot().task_id
        if self._mission_phase == "return":
            valid[depot_id - 1] = True
            return valid
        state = state or self.drone_states[0]
        for task in self.scene.tasks:
            if task.task_id != depot_id:
                valid[task.task_id - 1] = (
                    not bool(self.served_mask[task.task_id - 1])
                    and self._target_is_return_feasible(task, state)
                )
        return valid

    def _decode_targets(self, actions: np.ndarray) -> None:
        for drone_id, state in enumerate(self.drone_states):
            valid = self._valid_target_mask(state)
            if not valid.any():
                continue
            scores = actions[drone_id, 4:].astype(np.float64)
            scores[~valid] = -np.inf
            candidate = int(np.argmax(scores)) + 1
            current_score = (
                scores[state.active_target_id - 1]
                if valid[state.active_target_id - 1]
                else -np.inf
            )
            replanning_due = (
                self.sim_time_s - state.last_replan_time_s
                >= self.config.target_replan_interval_s
            )
            if (
                not valid[state.active_target_id - 1]
                or (
                    replanning_due
                    and scores[candidate - 1]
                    >= current_score + self.config.target_switch_margin
                )
            ):
                state.active_target_id = candidate
                state.previous_target_distance_m = self._target_distance(
                    state
                )
                state.last_replan_time_s = self.sim_time_s

    def _wind_vector(
        self, position: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        wind = np.array([1.5, 0.5, 0.0], dtype=np.float64)
        gust = 0.0
        for zone in self.scene.soft_risk_zones:
            exposure = zone.exposure(position)
            if exposure <= 0:
                continue
            if any(token in zone.kind for token in ("风", "阵风", "风廊")):
                angle = math.radians(zone.angle_deg + 90.0)
                wind += exposure * np.array(
                    [
                        math.cos(angle) * 5.0,
                        math.sin(angle) * 5.0,
                        0.5,
                    ]
                )
                gust = max(gust, exposure)
        return wind, float(np.clip(gust, 0.0, 1.0))

    def _advance_drone(
        self, state: DroneRuntimeState, physical_action: np.ndarray
    ) -> None:
        cfg = self.config
        accel = np.array(
            [
                physical_action[0] * cfg.max_accel_xy_mps2,
                physical_action[1] * cfg.max_accel_xy_mps2,
                physical_action[2] * cfg.max_accel_z_mps2,
            ],
            dtype=np.float64,
        )
        yaw_rate = float(physical_action[3] * cfg.max_yaw_rate_rps)
        wind, _ = self._wind_vector(state.position_m)
        effective_accel = accel + 0.08 * wind
        state.previous_acceleration_mps2 = state.acceleration_mps2.copy()
        state.acceleration_mps2 = accel
        state.yaw_rate_rps = yaw_rate
        state.yaw_rad = float(
            (
                state.yaw_rad
                + yaw_rate * cfg.physics_dt_s
                + math.pi
            )
            % (2 * math.pi)
            - math.pi
        )
        state.velocity_mps += effective_accel * cfg.physics_dt_s
        speed_xy = float(np.linalg.norm(state.velocity_mps[:2]))
        if speed_xy > cfg.max_speed_xy_mps:
            state.velocity_mps[:2] *= (
                cfg.max_speed_xy_mps / speed_xy
            )
        state.velocity_mps[2] = float(
            np.clip(
                state.velocity_mps[2],
                -cfg.max_speed_z_mps,
                cfg.max_speed_z_mps,
            )
        )
        state.position_m += state.velocity_mps * cfg.physics_dt_s

        power_w = self._power_w(state)
        energy_wh = power_w * cfg.physics_dt_s / 3600.0
        state.battery_wh = max(
            0.0, state.battery_wh - energy_wh
        )
        self.total_energy_wh += energy_wh

        soft = self.scene.soft_risk_exposure(state.position_m)
        gps_exposure = max(
            [value for kind, value in soft.items() if "GPS" in kind]
            or [0.0]
        )
        comm_exposure = max(
            [value for kind, value in soft.items() if "通信" in kind]
            or [0.0]
        )
        bias_noise = self.np_random.normal(
            0.0, 0.04 + 0.25 * gps_exposure, size=3
        )
        state.gps_bias_m = 0.96 * state.gps_bias_m + bias_noise
        state.comm_quality = float(
            np.clip(1.0 - 0.8 * comm_exposure, 0.0, 1.0)
        )

    def _power_w(self, state: DroneRuntimeState) -> float:
        speed_xy = float(np.linalg.norm(state.velocity_mps[:2]))
        climb = max(float(state.velocity_mps[2]), 0.0)
        accel = float(np.linalg.norm(state.acceleration_mps2))
        return float(
            120.0
            + 12.0 * state.payload_kg
            + 0.08 * speed_xy**3
            + 35.0 * climb
            + 2.5 * accel**2
        )

    def _update_dynamic_entities(self) -> None:
        for entity in self.dynamic_entities:
            entity.update(
                self.config.physics_dt_s, self.scene.extent_m
            )

    def _safety_event(
        self, state: DroneRuntimeState, drone_id: int
    ):
        cfg = self.config
        x, y, z = state.position_m
        if (
            x < 0
            or x > self.scene.extent_m[0]
            or y < 0
            or y > self.scene.extent_m[1]
            or z < cfg.min_altitude_m
            or z > cfg.max_altitude_m
        ):
            return "boundary", "out_of_bounds"
        static = self.scene.static_collision(
            state.position_m, cfg.drone_radius_m
        )
        if static is not None:
            return static[0], f"collision:{static[0]}:{static[1]}"
        risk = self.scene.hard_risk(
            state.position_m, cfg.drone_radius_m
        )
        if risk is not None:
            return "hard_risk", f"hard_risk:{risk.kind}"
        for entity in self.dynamic_entities:
            if (
                entity.clearance(state.position_m)
                <= cfg.drone_radius_m
            ):
                return (
                    "dynamic",
                    f"collision:dynamic:{entity.kind}",
                )
        for other_id, other in enumerate(self.drone_states):
            if (
                other_id != drone_id
                and np.linalg.norm(
                    state.position_m - other.position_m
                )
                < 2 * cfg.drone_radius_m
            ):
                return "uav", f"collision:uav:{other_id}"
        if state.battery_wh <= 0.0:
            return "battery", "battery_depleted"
        return None

    def _target_distance(self, state: DroneRuntimeState) -> float:
        target = self.scene.task_by_id(state.active_target_id)
        return float(
            np.linalg.norm(
                state.position_m - self._service_position(target)
            )
        )

    def _service_position(self, task: TaskPoint) -> np.ndarray:
        result = task.position_enu_m.copy()
        result[2] = max(
            self.config.min_altitude_m,
            task.position_enu_m[2] + 3.0,
        )
        return result

    def _reward_drone(
        self,
        drone_id: int,
        state: DroneRuntimeState,
        previous: Dict[str, object],
        terms: Dict[str, float],
    ) -> None:
        _ = drone_id
        cfg = self.config
        distance = self._target_distance(state)
        progress = (
            float(previous["distance"]) - distance
        ) / max(self.scene.extent_m.max(), 1.0)
        terms["progress"] += cfg.reward_progress * progress

        target = self.scene.task_by_id(state.active_target_id)
        if (
            target.task_id != self.scene.depot().task_id
            and not self.served_mask[target.task_id - 1]
            and distance <= target.landing_radius_m
        ):
            self.served_mask[target.task_id - 1] = 1
            state.payload_kg = max(
                0.0, state.payload_kg - target.demand_kg
            )
            terms["delivery"] += cfg.reward_delivery
            self._score_delta += cfg.score_per_delivery

        clearance = self.scene.nearest_static_clearance(
            state.position_m,
            max_range_m=max(
                cfg.lidar_range_m, cfg.safety_clearance_m
            ),
        )
        if clearance < cfg.safety_clearance_m:
            ratio = 1.0 - clearance / cfg.safety_clearance_m
            terms[
                "clearance"
            ] -= cfg.penalty_clearance * ratio * ratio
        energy_used = (
            float(previous["battery"]) - state.battery_wh
        )
        terms[
            "energy"
        ] -= cfg.penalty_energy_per_wh * max(energy_used, 0.0)
        soft_exposure = sum(
            self.scene.soft_risk_exposure(
                state.position_m
            ).values()
        )
        terms[
            "soft_risk"
        ] -= cfg.penalty_soft_risk * soft_exposure * cfg.control_dt_s
        jerk = (
            np.linalg.norm(
                state.acceleration_mps2
                - np.asarray(previous["acceleration"])
            )
            / cfg.control_dt_s
        )
        terms["jerk"] -= cfg.penalty_jerk * float(jerk)
        state.previous_target_distance_m = distance

    def _local_delivery_done(self) -> bool:
        depot_id = self.scene.depot().task_id
        return all(
            self.served_mask[task.task_id - 1]
            for task in self.scene.tasks
            if task.task_id != depot_id
        )

    def _return_estimates(
        self, state: DroneRuntimeState
    ) -> Tuple[float, float]:
        """估算从当前位置安全返航所需的时间和电量。

        路径系数为直线距离留出绕障余量；能耗估计故意偏保守，使策略在
        电量或时间不再支持下一个配送点时转入返航，而不是赌到耗尽。
        """
        depot = self.scene.depot()
        distance_m = float(
            np.linalg.norm(
                state.position_m - self._service_position(depot)
            )
        )
        route_distance_m = (
            distance_m * self.config.return_route_factor
        )
        return (
            route_distance_m
            / max(self.config.max_speed_xy_mps, 1e-6),
            route_distance_m * self.config.return_energy_wh_per_m,
        )

    def _target_is_return_feasible(
        self, task: TaskPoint, state: DroneRuntimeState
    ) -> bool:
        """仅开放“完成该目标后仍能安全返航”的配送候选。"""
        target_position = self._service_position(task)
        depot_position = self._service_position(self.scene.depot())
        route_distance_m = self.config.return_route_factor * (
            float(np.linalg.norm(state.position_m - target_position))
            + float(np.linalg.norm(target_position - depot_position))
        )
        route_time_s = route_distance_m / max(
            self.config.max_speed_xy_mps, 1e-6
        )
        route_energy_wh = (
            route_distance_m * self.config.return_energy_wh_per_m
        )
        remaining_time_s = (
            self.config.max_episode_time_s - self.sim_time_s
        )
        reserve_wh = (
            self.config.battery_capacity_wh * self.config.reserve_soc
        )
        return (
            route_time_s + self.config.return_time_margin_s
            < remaining_time_s
            and route_energy_wh + reserve_wh < state.battery_wh
        )

    def _has_feasible_delivery_candidate(self) -> bool:
        depot_id = self.scene.depot().task_id
        pending = [
            task
            for task in self.scene.tasks
            if task.task_id != depot_id
            and not self.served_mask[task.task_id - 1]
        ]
        return any(
            self._target_is_return_feasible(task, state)
            for task in pending
            for state in self.drone_states
        )

    def _should_force_return(self) -> Tuple[bool, Optional[str]]:
        if self._local_delivery_done():
            return True, "all_deliveries_completed"
        remaining_time_s = (
            self.config.max_episode_time_s - self.sim_time_s
        )
        reserve_wh = (
            self.config.battery_capacity_wh * self.config.reserve_soc
        )
        for state in self.drone_states:
            return_time_s, return_energy_wh = self._return_estimates(state)
            if (
                remaining_time_s
                <= return_time_s + self.config.return_time_margin_s
            ):
                return True, "time_reserve"
            if state.battery_wh <= return_energy_wh + reserve_wh:
                return True, "energy_reserve"
        if not self._has_feasible_delivery_candidate():
            return True, "no_feasible_delivery"
        return False, None

    def _enter_return_phase(self, trigger: str) -> None:
        if self._mission_phase == "return":
            return
        self._mission_phase = "return"
        self._return_trigger = trigger
        depot_id = self.scene.depot().task_id
        for state in self.drone_states:
            state.active_target_id = depot_id
            state.previous_target_distance_m = self._target_distance(state)
            state.last_replan_time_s = self.sim_time_s

    def _update_mission_phase(self) -> None:
        should_return, trigger = self._should_force_return()
        if should_return:
            self._enter_return_phase(trigger or "safety_reserve")

    def _returned_to_depot(self) -> bool:
        depot = self.scene.depot()
        service_position = self._service_position(depot)
        return all(
            np.linalg.norm(state.position_m - service_position)
            <= depot.landing_radius_m
            for state in self.drone_states
        )

    def _finalize_score(self) -> None:
        delivered_count = int(self.served_mask.sum())
        efficiency_score = max(
            0.0,
            (
                self.config.max_episode_time_s - self.sim_time_s
            )
            * self.config.score_efficiency_per_s,
        )
        self._mission_score = (
            delivered_count * self.config.score_per_delivery
            + self.config.score_safe_return
            + efficiency_score
        )
        self._score_delta = self._mission_score

    def _task_terminal(
        self, terms: Dict[str, float]
    ) -> bool:
        if (
            self._local_delivery_done()
            and not self._all_delivered_awarded
        ):
            terms[
                "all_delivered"
            ] += self.config.reward_all_delivered
            self._all_delivered_awarded = True
        self._update_mission_phase()
        if self._mission_phase == "return" and self._returned_to_depot():
            terms["return"] += self.config.reward_return
            self._termination_reason = (
                "mission_complete"
                if self._local_delivery_done()
                else "safe_return_partial"
            )
            self._finalize_score()
            return True
        return False

    def _cooperation_reward(self) -> float:
        if self.num_drones <= 1:
            return 0.0
        reward = 0.0
        for i in range(self.num_drones):
            for j in range(i + 1, self.num_drones):
                distance = float(
                    np.linalg.norm(
                        self.drone_states[i].position_m
                        - self.drone_states[j].position_m
                    )
                )
                if (
                    2 * self.config.drone_radius_m
                    < distance
                    < 80.0
                ):
                    reward += 0.02
        return reward

    def _get_observation(self) -> Dict[str, np.ndarray]:
        components = [
            self._observation_for_drone(index)
            for index in range(self.num_drones)
        ]
        if self.num_drones == 1:
            result = components[0]
        else:
            result = {
                key: np.stack(
                    [item[key] for item in components], axis=0
                )
                for key in components[0]
                if key != "delivery_mask"
            }
        result["delivery_mask"] = self.served_mask.astype(
            np.int8, copy=True
        )
        return result

    def _observation_for_drone(
        self, drone_id: int
    ) -> Dict[str, np.ndarray]:
        cfg = self.config
        state = self.drone_states[drone_id]
        extent_xyz = np.array(
            [
                self.scene.extent_m[0],
                self.scene.extent_m[1],
                cfg.max_altitude_m,
            ],
            dtype=np.float64,
        )
        pos_norm = np.clip(
            2.0 * state.position_m / extent_xyz - 1.0,
            -1.0,
            1.0,
        )
        vel_norm = np.clip(
            state.velocity_mps
            / np.array(
                [
                    cfg.max_speed_xy_mps,
                    cfg.max_speed_xy_mps,
                    cfg.max_speed_z_mps,
                ]
            ),
            -1.0,
            1.0,
        )
        accel_norm = np.clip(
            state.acceleration_mps2
            / np.array(
                [
                    cfg.max_accel_xy_mps2,
                    cfg.max_accel_xy_mps2,
                    cfg.max_accel_z_mps2,
                ]
            ),
            -1.0,
            1.0,
        )
        clearance = self.scene.nearest_static_clearance(
            state.position_m, cfg.lidar_range_m
        )
        boundary_margin = min(
            state.position_m[0],
            self.scene.extent_m[0] - state.position_m[0],
            state.position_m[1],
            self.scene.extent_m[1] - state.position_m[1],
        )
        flight = np.concatenate(
            [
                pos_norm,
                vel_norm,
                accel_norm,
                [
                    math.sin(state.yaw_rad),
                    math.cos(state.yaw_rad),
                ],
                [
                    np.clip(
                        state.yaw_rate_rps
                        / cfg.max_yaw_rate_rps,
                        -1.0,
                        1.0,
                    )
                ],
                [
                    np.clip(
                        2.0
                        * state.position_m[2]
                        / cfg.max_altitude_m
                        - 1.0,
                        -1.0,
                        1.0,
                    )
                ],
                [
                    np.clip(
                        2.0
                        * (cfg.max_altitude_m - state.position_m[2])
                        / cfg.max_altitude_m
                        - 1.0,
                        -1.0,
                        1.0,
                    )
                ],
                [
                    np.clip(
                        2.0
                        * clearance
                        / cfg.lidar_range_m
                        - 1.0,
                        -1.0,
                        1.0,
                    )
                ],
                [
                    np.clip(
                        2.0
                        * self._target_distance(state)
                        / np.linalg.norm(extent_xyz)
                        - 1.0,
                        -1.0,
                        1.0,
                    )
                ],
                [
                    np.clip(
                        2.0
                        * boundary_margin
                        / min(self.scene.extent_m)
                        - 1.0,
                        -1.0,
                        1.0,
                    )
                ],
                [
                    np.clip(
                        2.0
                        * self.sim_time_s
                        / cfg.max_episode_time_s
                        - 1.0,
                        -1.0,
                        1.0,
                    )
                ],
            ]
        ).astype(np.float32)

        capacity = cfg.battery_capacity_wh
        soc = np.clip(
            state.battery_wh / capacity, 0.0, 1.0
        )
        payload_battery = np.array(
            [
                np.clip(state.payload_kg / 5.0, 0.0, 1.0),
                np.clip(
                    state.payload_kg
                    / max(0.1, state.payload_kg + 1.0),
                    0.0,
                    1.0,
                ),
                soc,
                soc,
                np.clip(self._power_w(state) / 500.0, 0.0, 1.0),
                np.clip(
                    self.served_mask.sum()
                    / max(1, len(self.scene.tasks) - 1),
                    0.0,
                    1.0,
                ),
            ],
            dtype=np.float32,
        )
        target_position = self._service_position(
            self.scene.task_by_id(state.active_target_id)
        )
        relative = target_position - state.position_m
        target = np.concatenate(
            [
                np.clip(relative / extent_xyz, -1.0, 1.0),
                [
                    np.clip(
                        np.linalg.norm(relative)
                        / np.linalg.norm(extent_xyz),
                        0.0,
                        1.0,
                    )
                ],
            ]
        ).astype(np.float32)
        semantic, height = self._local_maps(state)
        lidar = self.scene.ray_distances(
            state.position_m,
            state.yaw_rad,
            pitches_rad=(-0.25, 0.0, 0.25),
            rays=cfg.lidar_rays,
            max_range_m=cfg.lidar_range_m,
        ).astype(np.float32)
        return {
            "flight": flight,
            "payload_battery": payload_battery,
            "delivery_mask": self.served_mask.astype(
                np.int8, copy=True
            ),
            "target": target,
            "semantic_map": semantic,
            "building_height": height,
            "lidar": lidar,
            "weather": self._weather_observation(state),
            "gps": self._gps_observation(state),
            "communication": self._communication_observation(
                state
            ),
        }

    def _local_maps(
        self, state: DroneRuntimeState
    ) -> Tuple[np.ndarray, np.ndarray]:
        cfg = self.config
        size = cfg.map_size
        semantic = np.zeros(
            (9, size, size), dtype=np.float32
        )
        height = np.zeros(
            (1, size, size), dtype=np.float32
        )
        half = cfg.map_extent_m / 2.0
        pixel = cfg.map_extent_m / size
        min_xy = state.position_m[:2] - half
        for collider in self.scene.candidate_prisms(
            state.position_m, cfg.map_extent_m
        ):
            aabb = collider.aabb_enu_m
            x0 = int(
                np.floor((aabb[0] - min_xy[0]) / pixel)
            )
            x1 = int(
                np.ceil((aabb[3] - min_xy[0]) / pixel)
            )
            y0 = int(
                np.floor((aabb[1] - min_xy[1]) / pixel)
            )
            y1 = int(
                np.ceil((aabb[4] - min_xy[1]) / pixel)
            )
            x0, x1 = np.clip([x0, x1], 0, size)
            y0, y1 = np.clip([y0, y1], 0, size)
            if x1 <= x0 or y1 <= y0:
                continue
            channel = (
                0 if collider.kind == "building" else 1
            )
            semantic[
                channel, y0:y1, x0:x1
            ] = 1.0
            height[
                0, y0:y1, x0:x1
            ] = np.maximum(
                height[0, y0:y1, x0:x1],
                np.clip(
                    collider.z_max_m
                    / cfg.max_altitude_m,
                    0.0,
                    1.0,
                ),
            )
        grid_x = (
            min_xy[0]
            + (np.arange(size) + 0.5) * pixel
        )
        grid_y = (
            min_xy[1]
            + (np.arange(size) + 0.5) * pixel
        )
        xx, yy = np.meshgrid(grid_x, grid_y)
        z = state.position_m[2]
        for zone in self.scene.hard_risk_zones:
            if zone.z_min_m <= z <= zone.z_max_m:
                semantic[3] = np.maximum(
                    semantic[3],
                    self._ellipse_mask(xx, yy, zone),
                )
        for zone in self.scene.soft_risk_zones:
            if zone.z_min_m <= z <= zone.z_max_m:
                semantic[4] = np.maximum(
                    semantic[4],
                    self._ellipse_mask(xx, yy, zone)
                    * zone.severity,
                )
        for entity in self.dynamic_entities:
            x = int(
                (entity.position_enu_m[0] - min_xy[0])
                / pixel
            )
            y = int(
                (entity.position_enu_m[1] - min_xy[1])
                / pixel
            )
            if 0 <= x < size and 0 <= y < size:
                semantic[
                    5,
                    max(0, y - 1) : min(size, y + 2),
                    max(0, x - 1) : min(size, x + 2),
                ] = 1.0
        wind, _ = self._wind_vector(state.position_m)
        semantic[6].fill(
            np.clip((wind[0] + 10.0) / 20.0, 0.0, 1.0)
        )
        semantic[7].fill(
            np.clip((wind[1] + 10.0) / 20.0, 0.0, 1.0)
        )
        for task in self.scene.tasks:
            x = int(
                (task.position_enu_m[0] - min_xy[0])
                / pixel
            )
            y = int(
                (task.position_enu_m[1] - min_xy[1])
                / pixel
            )
            if 0 <= x < size and 0 <= y < size:
                semantic[8, y, x] = 1.0
        return semantic, height

    @staticmethod
    def _ellipse_mask(xx: np.ndarray, yy: np.ndarray, zone):
        dx = xx - zone.center_enu_m[0]
        dy = yy - zone.center_enu_m[1]
        angle = math.radians(-zone.angle_deg)
        local_x = math.cos(angle) * dx - math.sin(angle) * dy
        local_y = math.sin(angle) * dx + math.cos(angle) * dy
        radii = np.maximum(
            zone.diameter_enu_m / 2.0, 1e-6
        )
        return (
            (
                (local_x / radii[0]) ** 2
                + (local_y / radii[1]) ** 2
            )
            <= 1.0
        ).astype(np.float32)

    def _weather_observation(
        self, state: DroneRuntimeState
    ) -> np.ndarray:
        wind, gust = self._wind_vector(state.position_m)
        soft = self.scene.soft_risk_exposure(state.position_m)
        rain = max(
            [value for kind, value in soft.items() if "雨" in kind]
            or [0.0]
        )
        visibility = 1.0 - 0.6 * rain
        return np.array(
            [
                np.clip((wind[0] + 10.0) / 20.0, 0.0, 1.0),
                np.clip((wind[1] + 10.0) / 20.0, 0.0, 1.0),
                np.clip((wind[2] + 5.0) / 10.0, 0.0, 1.0),
                gust,
                rain,
                visibility,
                0.5,
                (self.sim_time_s % 600.0) / 600.0,
            ],
            dtype=np.float32,
        )

    def _gps_observation(
        self, state: DroneRuntimeState
    ) -> np.ndarray:
        soft = self.scene.soft_risk_exposure(state.position_m)
        multipath = max(
            [
                value
                for kind, value in soft.items()
                if "GPS" in kind
            ]
            or [0.0]
        )
        bias = np.clip(
            state.gps_bias_m / 20.0, -1.0, 1.0
        )
        quality = np.clip(
            1.0 - multipath, 0.0, 1.0
        )
        return np.array(
            [
                bias[0],
                bias[1],
                bias[2],
                quality,
                1.0 if quality > 0.2 else 0.0,
                multipath,
            ],
            dtype=np.float32,
        )

    def _communication_observation(
        self, state: DroneRuntimeState
    ) -> np.ndarray:
        quality = state.comm_quality
        rssi = np.clip(
            0.2 + 0.8 * quality, 0.0, 1.0
        )
        latency = np.clip(
            0.05 + 0.6 * (1.0 - quality), 0.0, 1.0
        )
        loss = np.clip(
            (1.0 - quality) ** 2, 0.0, 1.0
        )
        return np.array(
            [rssi, quality, latency, loss],
            dtype=np.float32,
        )

    def flatten_observation(
        self,
        observation: Dict[str, np.ndarray],
        include_maps: bool = False,
    ) -> np.ndarray:
        """将字典观测转换为 MLP 基线可用的固定向量。"""
        keys = [
            "flight",
            "payload_battery",
            "delivery_mask",
            "target",
            "lidar",
            "weather",
            "gps",
            "communication",
        ]
        if include_maps:
            keys.extend(
                ["semantic_map", "building_height"]
            )
        values = [
            np.asarray(observation[key], dtype=np.float32).reshape(-1)
            for key in keys
        ]
        return np.concatenate(values).astype(np.float32)

    def _get_info(self) -> Dict[str, object]:
        return {
            "zone_id": self.scene.zone_id,
            "scene_title": self.scene.title,
            "scene_manifest_hash": self.scene.manifest_hash,
            "coordinate_system": self.scene.coordinate_system,
            "sim_time_s": float(self.sim_time_s),
            "step": int(self.step_count),
            "served_task_ids": [
                index + 1
                for index, served in enumerate(self.served_mask)
                if served
            ],
            "local_task_ids": [
                task.task_id for task in self.scene.tasks
            ],
            "active_target_ids": [
                state.active_target_id
                for state in self.drone_states
            ],
            "battery_soc": [
                float(
                    state.battery_wh
                    / self.config.battery_capacity_wh
                )
                for state in self.drone_states
            ],
            "payload_kg": [
                float(state.payload_kg)
                for state in self.drone_states
            ],
            "positions_m": [
                state.position_m.astype(float).tolist()
                for state in self.drone_states
            ],
            "velocities_mps": [
                state.velocity_mps.astype(float).tolist()
                for state in self.drone_states
            ],
            "total_energy_wh": float(self.total_energy_wh),
            "reward_terms": dict(self._last_reward_terms),
            "mission_phase": self._mission_phase,
            "return_trigger": self._return_trigger,
            "score_delta": float(self._score_delta),
            "provisional_delivery_score": float(
                self.served_mask.sum()
                * self.config.score_per_delivery
            ),
            "mission_score": float(self._mission_score),
            "objective_rank": [
                int(self.served_mask.sum()),
                int(
                    self._termination_reason
                    in {"mission_complete", "safe_return_partial"}
                ),
                -float(self.sim_time_s),
                -float(self.total_energy_wh),
            ],
            "termination_reason": self._termination_reason,
            "gymnasium_api": GYMNASIUM_API,
        }

    def render(self):
        if self.render_mode == "rgb_array":
            size = 512
            canvas = np.full(
                (size, size, 3), 245, dtype=np.uint8
            )
            for state in self.drone_states:
                x = int(
                    np.clip(
                        state.position_m[0]
                        / self.scene.extent_m[0]
                        * (size - 1),
                        0,
                        size - 1,
                    )
                )
                y = int(
                    np.clip(
                        state.position_m[1]
                        / self.scene.extent_m[1]
                        * (size - 1),
                        0,
                        size - 1,
                    )
                )
                canvas[
                    max(0, y - 3) : min(size, y + 4),
                    max(0, x - 3) : min(size, x + 4),
                ] = [220, 40, 40]
            return np.flipud(canvas)
        if self.render_mode == "human":
            info = self._get_info()
            print(
                f"[{self.scene.zone_id}] "
                f"t={self.sim_time_s:.1f}s "
                f"served={info['served_task_ids']} "
                f"soc={info['battery_soc']}"
            )
        return None


class LegacyWuhanEnvAdapter:
    """为旧训练脚本提供扁平状态和四元组 step 返回值。"""

    def __init__(
        self,
        env: WuhanDroneDeliveryEnv,
        include_maps: bool = False,
    ):
        self.env = env
        self.include_maps = include_maps
        self.action_space = env.action_space
        sample, _ = env.reset(seed=env.config.seed)
        flattened = env.flatten_observation(
            sample, include_maps=include_maps
        )
        self.observation_space = spaces.Box(
            -np.inf,
            np.inf,
            shape=flattened.shape,
            dtype=np.float32,
        )

    def reset(self):
        observation, _ = self.env.reset()
        return self.env.flatten_observation(
            observation, include_maps=self.include_maps
        )

    def step(self, action):
        observation, reward, terminated, truncated, info = (
            self.env.step(action)
        )
        return (
            self.env.flatten_observation(
                observation, include_maps=self.include_maps
            ),
            reward,
            terminated or truncated,
            info,
        )

    def __getattr__(self, name):
        return getattr(self.env, name)
