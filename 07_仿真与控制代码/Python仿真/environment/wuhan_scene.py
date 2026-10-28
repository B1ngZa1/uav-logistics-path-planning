"""武汉参考配送场景加载、碰撞查询与风险服务。

本模块只读取 ``outputs/环境示意图_3D预览`` 中已经离线生成的场景包。
训练过程中不会访问网络、解析 OSM 或修改场景资产。
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


Array = np.ndarray


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_scene_dir() -> Path:
    """定位包含 ``scene_3d_index.json`` 的场景目录。"""
    output_root = _project_root() / "outputs"
    candidates = sorted(output_root.glob("*/scene_3d_index.json"))
    if not candidates:
        raise FileNotFoundError(
            "未找到 scene_3d_index.json；请先运行 "
            "03_图表脚本/Python脚本/convert_wuhan_reference_scenes_2d_to_3d.py"
        )
    return candidates[0].parent


def _as_array(values: Sequence[float], length: int) -> Array:
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (length,) or not np.isfinite(result).all():
        raise ValueError(f"期望有限的 {length} 维数组，实际为 {result}")
    return result


def point_in_polygon(x: float, y: float, polygon: Array) -> bool:
    """射线法判断二维点是否位于闭合多边形内部。"""
    inside = False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if (yi > y) != (yj > y):
            x_cross = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < x_cross:
                inside = not inside
        j = i
    return inside


def point_segment_distance_2d(point: Array, start: Array, end: Array) -> float:
    segment = end - start
    denom = float(np.dot(segment, segment))
    if denom <= 1e-12:
        return float(np.linalg.norm(point - start))
    t = float(np.clip(np.dot(point - start, segment) / denom, 0.0, 1.0))
    return float(np.linalg.norm(point - (start + t * segment)))


def point_segment_distance_3d(point: Array, start: Array, end: Array) -> float:
    segment = end - start
    denom = float(np.dot(segment, segment))
    if denom <= 1e-12:
        return float(np.linalg.norm(point - start))
    t = float(np.clip(np.dot(point - start, segment) / denom, 0.0, 1.0))
    return float(np.linalg.norm(point - (start + t * segment)))


def polygon_clearance(point_xy: Array, polygon: Array) -> float:
    if point_in_polygon(float(point_xy[0]), float(point_xy[1]), polygon):
        return 0.0
    return min(
        point_segment_distance_2d(point_xy, polygon[i], polygon[(i + 1) % len(polygon)])
        for i in range(len(polygon))
    )


@dataclass(frozen=True)
class PrismCollider:
    collider_id: str
    kind: str
    z_min_m: float
    z_max_m: float
    aabb_enu_m: Array
    footprints: Tuple[Array, ...]

    @classmethod
    def from_payload(cls, payload: Dict[str, object], kind: str) -> "PrismCollider":
        footprints: List[Array] = []
        for item in payload.get("footprints", []):
            polygon = np.asarray(item["exterior_xy_m"], dtype=np.float64)
            if polygon.ndim == 2 and polygon.shape[1] == 2 and len(polygon) >= 3:
                footprints.append(polygon)
        z_min = float(payload.get("z_min_m", 0.0))
        z_max = float(payload.get("z_max_m", z_min))
        raw_aabb = payload.get("aabb_enu_m")
        if raw_aabb is None:
            points = np.concatenate(footprints, axis=0)
            raw_aabb = [
                float(points[:, 0].min()),
                float(points[:, 1].min()),
                z_min,
                float(points[:, 0].max()),
                float(points[:, 1].max()),
                z_max,
            ]
        return cls(
            collider_id=str(payload.get("feature_id", f"{kind}_unknown")),
            kind=kind,
            z_min_m=z_min,
            z_max_m=z_max,
            aabb_enu_m=_as_array(raw_aabb, 6),
            footprints=tuple(footprints),
        )

    def contains(self, position: Array, radius_m: float = 0.0) -> bool:
        x, y, z = position
        aabb = self.aabb_enu_m
        if (
            x < aabb[0] - radius_m
            or x > aabb[3] + radius_m
            or y < aabb[1] - radius_m
            or y > aabb[4] + radius_m
            or z < self.z_min_m - radius_m
            or z > self.z_max_m + radius_m
        ):
            return False
        if not self.footprints:
            return True
        return any(
            polygon_clearance(position[:2], polygon) <= radius_m
            for polygon in self.footprints
        )

    def clearance(self, position: Array) -> float:
        aabb = self.aabb_enu_m
        dx = max(aabb[0] - position[0], 0.0, position[0] - aabb[3])
        dy = max(aabb[1] - position[1], 0.0, position[1] - aabb[4])
        dz = max(self.z_min_m - position[2], 0.0, position[2] - self.z_max_m)
        return float(math.sqrt(dx * dx + dy * dy + dz * dz))


@dataclass(frozen=True)
class CapsuleCollider:
    collider_id: str
    kind: str
    start_enu_m: Array
    end_enu_m: Array
    radius_m: float

    @classmethod
    def from_payload(cls, payload: Dict[str, object]) -> "CapsuleCollider":
        return cls(
            collider_id=str(payload.get("feature_id", "power_line_unknown")),
            kind="power_line",
            start_enu_m=_as_array(payload["start_enu_m"], 3),
            end_enu_m=_as_array(payload["end_enu_m"], 3),
            radius_m=float(payload.get("collision_radius_m", 1.5)),
        )

    def clearance(self, position: Array) -> float:
        return (
            point_segment_distance_3d(position, self.start_enu_m, self.end_enu_m)
            - self.radius_m
        )

    def contains(self, position: Array, radius_m: float = 0.0) -> bool:
        return self.clearance(position) <= radius_m


@dataclass(frozen=True)
class RiskZone:
    kind: str
    hard: bool
    center_enu_m: Array
    diameter_enu_m: Array
    angle_deg: float
    z_min_m: float
    z_max_m: float
    severity: float

    @classmethod
    def from_payload(cls, payload: Dict[str, object]) -> "RiskZone":
        hard = bool(payload.get("hard", False))
        return cls(
            kind=str(payload.get("kind", "unknown")),
            hard=hard,
            center_enu_m=_as_array(payload["center_enu_m"], 2),
            diameter_enu_m=_as_array(payload["diameter_enu_m"], 2),
            angle_deg=float(payload.get("angle_deg", 0.0)),
            z_min_m=float(payload.get("z_min_m", 0.0)),
            z_max_m=float(payload.get("z_max_m", 200.0)),
            severity=1.0 if hard else 0.55,
        )

    def normalized_radius(self, position: Array) -> float:
        delta = position[:2] - self.center_enu_m
        angle = math.radians(-self.angle_deg)
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        local_x = cos_a * delta[0] - sin_a * delta[1]
        local_y = sin_a * delta[0] + cos_a * delta[1]
        radii = np.maximum(self.diameter_enu_m / 2.0, 1e-6)
        return float(
            math.sqrt((local_x / radii[0]) ** 2 + (local_y / radii[1]) ** 2)
        )

    def contains(self, position: Array, margin_m: float = 0.0) -> bool:
        if (
            position[2] < self.z_min_m - margin_m
            or position[2] > self.z_max_m + margin_m
        ):
            return False
        radii = np.maximum(self.diameter_enu_m / 2.0 + margin_m, 1e-6)
        delta = position[:2] - self.center_enu_m
        angle = math.radians(-self.angle_deg)
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        x = cos_a * delta[0] - sin_a * delta[1]
        y = sin_a * delta[0] + cos_a * delta[1]
        return (x / radii[0]) ** 2 + (y / radii[1]) ** 2 <= 1.0

    def exposure(self, position: Array) -> float:
        if not self.contains(position):
            return 0.0
        return float(
            self.severity
            * np.clip(1.0 - 0.35 * self.normalized_radius(position), 0.2, 1.0)
        )


@dataclass
class DynamicEntity:
    entity_id: str
    kind: str
    position_enu_m: Array
    velocity_mps: Array
    half_extent_m: Array

    @classmethod
    def from_payload(cls, payload: Dict[str, object], index: int) -> "DynamicEntity":
        kind = str(payload.get("kind", "dynamic"))
        speed_by_kind = {
            "车辆": 8.0,
            "行人": 1.4,
            "其他无人机": 7.0,
            "飞鸟群": 9.0,
            "船只": 5.0,
            "施工吊臂": 1.0,
        }
        speed = speed_by_kind.get(kind, 3.0)
        heading = math.radians(float(payload.get("direction_deg", 0.0)))
        return cls(
            entity_id=f"dynamic_{index}_{kind}",
            kind=kind,
            position_enu_m=_as_array(payload["position_enu_m"], 3).copy(),
            velocity_mps=np.array(
                [math.cos(heading) * speed, math.sin(heading) * speed, 0.0]
            ),
            half_extent_m=_as_array(
                payload.get("collision_box_size_m", [2, 2, 2]), 3
            )
            / 2.0,
        )

    def update(self, dt: float, extent_m: Array) -> None:
        self.position_enu_m += self.velocity_mps * dt
        for axis in (0, 1):
            low = self.half_extent_m[axis]
            high = extent_m[axis] - self.half_extent_m[axis]
            if self.position_enu_m[axis] < low:
                self.position_enu_m[axis] = low
                self.velocity_mps[axis] = abs(self.velocity_mps[axis])
            elif self.position_enu_m[axis] > high:
                self.position_enu_m[axis] = high
                self.velocity_mps[axis] = -abs(self.velocity_mps[axis])

    def clearance(self, position: Array) -> float:
        delta = np.maximum(
            np.abs(position - self.position_enu_m) - self.half_extent_m, 0.0
        )
        return float(np.linalg.norm(delta))


@dataclass(frozen=True)
class TaskPoint:
    task_id: int
    site_type: str
    position_enu_m: Array
    demand_kg: float
    landing_radius_m: float

    @classmethod
    def from_payload(cls, payload: Dict[str, object]) -> "TaskPoint":
        task_id = int(payload["task_id"])
        site_type = str(payload.get("site_type", "配送点"))
        is_depot = "中心" in site_type or task_id == 1
        return cls(
            task_id=task_id,
            site_type=site_type,
            position_enu_m=_as_array(payload["position_enu_m"], 3),
            demand_kg=0.0 if is_depot else 0.45 + 0.1 * (task_id % 5),
            landing_radius_m=14.0 if is_depot else 12.0,
        )


@dataclass
class ScenePackage:
    zone_id: str
    title: str
    coordinate_system: str
    extent_m: Array
    terrain_z_min_m: float
    terrain_z_max_m: float
    manifest_hash: str
    static_prisms: Tuple[PrismCollider, ...]
    power_lines: Tuple[CapsuleCollider, ...]
    hard_risk_zones: Tuple[RiskZone, ...]
    soft_risk_zones: Tuple[RiskZone, ...]
    tasks: Tuple[TaskPoint, ...]
    dynamic_templates: Tuple[DynamicEntity, ...]

    def __post_init__(self) -> None:
        self._aabbs = np.asarray(
            [collider.aabb_enu_m for collider in self.static_prisms],
            dtype=np.float64,
        )

    @classmethod
    def load(
        cls, zone_id: str = "A", scene_dir: Optional[Path] = None
    ) -> "ScenePackage":
        zone_id = zone_id.upper()
        if zone_id not in {"A", "B", "C", "D"}:
            raise ValueError("zone_id 必须为 A、B、C 或 D")
        scene_dir = Path(scene_dir) if scene_dir is not None else default_scene_dir()
        manifest_path = scene_dir / f"zone_{zone_id}_scene_lod0.json"
        collision_path = scene_dir / f"zone_{zone_id}_collision.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        collision = json.loads(collision_path.read_text(encoding="utf-8"))
        if manifest.get("zone_id") != zone_id or collision.get("zone_id") != zone_id:
            raise ValueError("场景清单与碰撞文件 zone_id 不一致")
        if manifest.get("coordinate_system") != "local_ENU_meters_z_up":
            raise ValueError("只支持 local_ENU_meters_z_up 坐标系")

        prisms: List[PrismCollider] = []
        for item in collision.get("buildings", []):
            collider = PrismCollider.from_payload(item, "building")
            if collider.footprints:
                prisms.append(collider)
        for item in collision.get("bridges", []):
            collider = PrismCollider.from_payload(item, "bridge")
            if collider.footprints:
                prisms.append(collider)

        terrain = manifest.get("terrain", {})
        return cls(
            zone_id=zone_id,
            title=str(manifest.get("title", f"Zone {zone_id}")),
            coordinate_system=str(manifest["coordinate_system"]),
            extent_m=_as_array(manifest["extent_m"], 2),
            terrain_z_min_m=float(terrain.get("elevation_min_m", 0.0)),
            terrain_z_max_m=float(terrain.get("elevation_max_m", 0.0)),
            manifest_hash=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            static_prisms=tuple(prisms),
            power_lines=tuple(
                CapsuleCollider.from_payload(item)
                for item in collision.get("power_lines", [])
            ),
            hard_risk_zones=tuple(
                RiskZone.from_payload(item)
                for item in collision.get("hard_risk_zones", [])
            ),
            soft_risk_zones=tuple(
                RiskZone.from_payload(item)
                for item in manifest.get("risk_volumes", [])
                if not bool(item.get("hard", False))
            ),
            tasks=tuple(
                TaskPoint.from_payload(item)
                for item in manifest.get("task_points", [])
            ),
            dynamic_templates=tuple(
                DynamicEntity.from_payload(item, index)
                for index, item in enumerate(
                    collision.get("dynamic_entities_initial_state", [])
                )
            ),
        )

    def create_dynamic_entities(self) -> List[DynamicEntity]:
        return [
            DynamicEntity(
                entity_id=item.entity_id,
                kind=item.kind,
                position_enu_m=item.position_enu_m.copy(),
                velocity_mps=item.velocity_mps.copy(),
                half_extent_m=item.half_extent_m.copy(),
            )
            for item in self.dynamic_templates
        ]

    def depot(self) -> TaskPoint:
        return min(self.tasks, key=lambda item: item.task_id)

    def task_by_id(self, task_id: int) -> TaskPoint:
        for task in self.tasks:
            if task.task_id == task_id:
                return task
        raise KeyError(f"当前分区没有 task_id={task_id}")

    def candidate_prisms(
        self, position: Array, range_m: float
    ) -> Iterable[PrismCollider]:
        if len(self._aabbs) == 0:
            return ()
        aabbs = self._aabbs
        mask = (
            (position[0] >= aabbs[:, 0] - range_m)
            & (position[0] <= aabbs[:, 3] + range_m)
            & (position[1] >= aabbs[:, 1] - range_m)
            & (position[1] <= aabbs[:, 4] + range_m)
            & (position[2] >= aabbs[:, 2] - range_m)
            & (position[2] <= aabbs[:, 5] + range_m)
        )
        return (self.static_prisms[index] for index in np.flatnonzero(mask))

    def static_collision(
        self, position: Array, radius_m: float
    ) -> Optional[Tuple[str, str]]:
        for collider in self.candidate_prisms(position, radius_m):
            if collider.contains(position, radius_m):
                return collider.kind, collider.collider_id
        for line in self.power_lines:
            if line.contains(position, radius_m):
                return line.kind, line.collider_id
        return None

    def nearest_static_clearance(
        self, position: Array, max_range_m: float = 300.0
    ) -> float:
        clearances = [
            collider.clearance(position)
            for collider in self.candidate_prisms(position, max_range_m)
        ]
        clearances.extend(line.clearance(position) for line in self.power_lines)
        if not clearances:
            return float(max_range_m)
        return float(np.clip(min(clearances), 0.0, max_range_m))

    def hard_risk(
        self, position: Array, radius_m: float = 0.0
    ) -> Optional[RiskZone]:
        return next(
            (
                zone
                for zone in self.hard_risk_zones
                if zone.contains(position, radius_m)
            ),
            None,
        )

    def soft_risk_exposure(self, position: Array) -> Dict[str, float]:
        result: Dict[str, float] = {}
        for zone in self.soft_risk_zones:
            exposure = zone.exposure(position)
            if exposure > 0.0:
                result[zone.kind] = exposure
        return result

    def ray_distances(
        self,
        origin: Array,
        yaw_rad: float,
        pitches_rad: Sequence[float],
        rays: int,
        max_range_m: float,
    ) -> Array:
        """用离散射线步进生成确定性的近似激光观测。"""
        result = np.ones((len(pitches_rad), rays), dtype=np.float32)
        step_m = max(4.0, max_range_m / 40.0)
        distances = np.arange(step_m, max_range_m + step_m, step_m)
        for pitch_index, pitch in enumerate(pitches_rad):
            cos_pitch, sin_pitch = math.cos(pitch), math.sin(pitch)
            for ray_index in range(rays):
                angle = yaw_rad + 2.0 * math.pi * ray_index / rays
                direction = np.array(
                    [
                        math.cos(angle) * cos_pitch,
                        math.sin(angle) * cos_pitch,
                        sin_pitch,
                    ],
                    dtype=np.float64,
                )
                for distance in distances:
                    point = origin + direction * distance
                    if (
                        point[0] < 0
                        or point[0] > self.extent_m[0]
                        or point[1] < 0
                        or point[1] > self.extent_m[1]
                        or point[2] < self.terrain_z_min_m
                    ):
                        result[pitch_index, ray_index] = distance / max_range_m
                        break
                    if self.static_collision(point, 0.5) or self.hard_risk(
                        point, 0.0
                    ):
                        result[pitch_index, ray_index] = distance / max_range_m
                        break
        return np.clip(result, 0.0, 1.0)
