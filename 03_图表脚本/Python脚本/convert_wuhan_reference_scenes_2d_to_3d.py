"""
将已审核的武汉无人机配送 2D 场景转换为可交换的 3D 场景。

输入不是 PNG 像素，而是 plot_wuhan_reference_scenes_2d.py 生成 2D 图时使用的
同一组 OSM / 脱敏概化矢量特征。转换过程保留：

  - 建筑公开轮廓、建筑类型、高度及来源；
  - 水体、绿地、运动场、道路、桥梁、轨道和高压线；
  - 配送点、风险体、动态实体和脱敏校门；
  - 两条仅用于三维审核的分级候选路径（不代表规划结果或现实航线）；
  - 精确建筑棱柱、桥体、高压线和硬风险区碰撞数据。

输出：

  - zone_<ID>_scene_lod<N>.obj / .mtl：通用 3D 模型；
  - zone_<ID>_collision.json：训练/碰撞检测用几何；
  - zone_<ID>_scene_lod<N>.json：场景来源、统计和验收清单；
  - zone_<ID>_scene_lod<N>_preview.png：可选三维审核图；
  - scene_3d_index.json：本次转换索引。

地形策略：

  - 未提供 DEM 时严格使用 z=0 平面，并在清单中标记 flat_no_dem_input；
  - 可在 --terrain-dir 中放置 zone_A_terrain.npz 等文件，包含 x、y、z 数组。
    x/y 为米制局部 ENU 一维坐标，z 为 shape=(len(y), len(x)) 的高程矩阵。

本脚本不抓取或披露敏感设施，不输出真实配送航线。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import triangulate, unary_union

import plot_wuhan_reference_scenes_2d as scene2d


SCHEMA_VERSION = 1
DEFAULT_OUTPUT_DIRNAME = "环境示意图_3D预览"
BUILDING_TYPE_CODES = {
    "低层住宅": "lowrise_residential",
    "多层住宅": "midrise_residential",
    "高层住宅": "highrise_residential",
    "商业综合体": "commercial_complex",
    "办公塔楼": "office_tower",
    "工业厂房": "industrial",
    "公共建筑": "public",
    "景区建筑": "scenic",
}
RISK_KIND_CODES = {
    "GPS多路径": "gps_multipath",
    "通信阴影": "communication_shadow",
    "湖面侧风": "lake_crosswind",
    "江面阵风": "river_gust",
    "河道风廊": "river_wind_corridor",
    "飞鸟活动": "bird_activity",
    "动态交通": "dynamic_traffic",
    "人群密集": "crowd",
    "船舶走廊": "vessel_corridor",
    "桥体硬隔离": "bridge_hard_barrier",
    "轨道硬隔离": "rail_hard_barrier",
    "临时施工": "temporary_construction",
}
THREED_BUILDING_COLORS = {
    "低层住宅": "#D8C6A5",
    "多层住宅": "#C9A27E",
    "高层住宅": "#B86F5B",
    "商业综合体": "#887493",
    "办公塔楼": "#58788A",
    "工业厂房": "#8B969B",
    "公共建筑": "#4F8C88",
    "景区建筑": "#78936E",
}
THREED_ENV_COLORS = {
    "terrain": "#E5E2D8",
    "green": "#A9C99A",
    "forest": "#6F9474",
    "sport": "#8DC9AE",
    "water": "#7BAFCB",
    "road": "#C8C5BD",
    "bridge": "#666E73",
    "rail": "#94677A",
    "power": "#B18A45",
    "roof_equipment": "#566169",
    "path_primary": "#3B8D72",
    "path_secondary": "#C78542",
}
PREVIEW_CAMERAS = {
    "A": {"elev": 52, "azim": -68, "vertical_aspect_m": 430.0},
    "B": {"elev": 48, "azim": -62, "vertical_aspect_m": 500.0},
    "C": {"elev": 47, "azim": -42, "vertical_aspect_m": 460.0},
    "D": {"elev": 53, "azim": -72, "vertical_aspect_m": 420.0},
}


@dataclass(frozen=True)
class Material:
    name: str
    color: Tuple[float, float, float]
    alpha: float = 1.0
    emission: Tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class MeshObject:
    name: str
    material: str
    vertices: List[Tuple[float, float, float]]
    faces: List[Tuple[int, ...]]
    category: str
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class TerrainSurface:
    mode: str
    width_m: float
    height_m: float
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    source_path: Optional[Path] = None

    @classmethod
    def flat(cls, width_m: float, height_m: float) -> "TerrainSurface":
        return cls(
            mode="flat_no_dem_input",
            width_m=width_m,
            height_m=height_m,
            x=np.array([0.0, width_m], dtype=float),
            y=np.array([0.0, height_m], dtype=float),
            z=np.zeros((2, 2), dtype=float),
        )

    @classmethod
    def from_npz(
        cls,
        path: Path,
        width_m: float,
        height_m: float,
    ) -> "TerrainSurface":
        payload = np.load(path)
        missing = {"x", "y", "z"} - set(payload.files)
        if missing:
            raise ValueError(
                "DEM %s 缺少数组: %s" % (path, ", ".join(sorted(missing)))
            )
        x = np.asarray(payload["x"], dtype=float)
        y = np.asarray(payload["y"], dtype=float)
        z = np.asarray(payload["z"], dtype=float)
        if x.ndim != 1 or y.ndim != 1 or z.shape != (len(y), len(x)):
            raise ValueError(
                "DEM 应满足 x/y 为一维，z.shape == (len(y), len(x))"
            )
        if len(x) < 2 or len(y) < 2:
            raise ValueError("DEM 每个方向至少需要 2 个采样点")
        if not np.all(np.diff(x) > 0) or not np.all(np.diff(y) > 0):
            raise ValueError("DEM x/y 必须严格递增")
        if x[0] > 0 or y[0] > 0 or x[-1] < width_m or y[-1] < height_m:
            raise ValueError("DEM 坐标范围未覆盖场景范围")
        if not np.all(np.isfinite(z)):
            raise ValueError("DEM 高程包含 NaN 或无穷值")
        return cls(
            mode="dem_npz",
            width_m=width_m,
            height_m=height_m,
            x=x,
            y=y,
            z=z,
            source_path=path,
        )

    def height_at(self, x_m: float, y_m: float) -> float:
        x_m = float(np.clip(x_m, self.x[0], self.x[-1]))
        y_m = float(np.clip(y_m, self.y[0], self.y[-1]))
        ix = int(np.clip(np.searchsorted(self.x, x_m) - 1, 0, len(self.x) - 2))
        iy = int(np.clip(np.searchsorted(self.y, y_m) - 1, 0, len(self.y) - 2))
        x0, x1 = self.x[ix], self.x[ix + 1]
        y0, y1 = self.y[iy], self.y[iy + 1]
        tx = 0.0 if x1 == x0 else (x_m - x0) / (x1 - x0)
        ty = 0.0 if y1 == y0 else (y_m - y0) / (y1 - y0)
        z00 = self.z[iy, ix]
        z10 = self.z[iy, ix + 1]
        z01 = self.z[iy + 1, ix]
        z11 = self.z[iy + 1, ix + 1]
        return float(
            (1 - tx) * (1 - ty) * z00
            + tx * (1 - ty) * z10
            + (1 - tx) * ty * z01
            + tx * ty * z11
        )

    def summary(self) -> Dict[str, object]:
        digest = None
        if self.source_path is not None:
            digest = hashlib.sha256(self.source_path.read_bytes()).hexdigest()
        return {
            "mode": self.mode,
            "source": str(self.source_path) if self.source_path else None,
            "source_sha256": digest,
            "grid_shape": [int(len(self.y)), int(len(self.x))],
            "elevation_min_m": round(float(np.min(self.z)), 3),
            "elevation_max_m": round(float(np.max(self.z)), 3),
            "max_relief_m": round(float(np.max(self.z) - np.min(self.z)), 3),
        }


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z_.-]+", "_", value)
    return cleaned.strip("_") or "object"


def color_tuple(value: str) -> Tuple[float, float, float]:
    rgb = mcolors.to_rgb(value)
    return float(rgb[0]), float(rgb[1]), float(rgb[2])


def building_material_key(building_type: str) -> str:
    return "building_%s" % BUILDING_TYPE_CODES.get(
        building_type,
        "other",
    )


def risk_material_key(risk_kind: str) -> str:
    return "risk_%s" % RISK_KIND_CODES.get(risk_kind, "other")


def material_library() -> Dict[str, Material]:
    materials: Dict[str, Material] = {
        "terrain": Material("terrain", color_tuple(THREED_ENV_COLORS["terrain"])),
        "green": Material("green", color_tuple(THREED_ENV_COLORS["green"]), 0.88),
        "forest": Material("forest", color_tuple(THREED_ENV_COLORS["forest"]), 0.90),
        "sport": Material("sport", color_tuple(THREED_ENV_COLORS["sport"]), 0.90),
        "water": Material("water", color_tuple(THREED_ENV_COLORS["water"]), 0.72),
        "road": Material("road", color_tuple(THREED_ENV_COLORS["road"])),
        "bridge": Material("bridge", color_tuple(THREED_ENV_COLORS["bridge"])),
        "rail": Material("rail", color_tuple(THREED_ENV_COLORS["rail"])),
        "power": Material("power", color_tuple(THREED_ENV_COLORS["power"])),
        "roof_equipment": Material(
            "roof_equipment",
            color_tuple(THREED_ENV_COLORS["roof_equipment"]),
            0.78,
        ),
        "gate": Material("gate", color_tuple("#263238")),
        "task": Material(
            "task",
            color_tuple(scene2d.TASK_COLOR),
            emission=color_tuple("#5A102C"),
        ),
        "depot": Material(
            "depot",
            color_tuple(scene2d.DEPOT_COLOR),
            emission=color_tuple("#153D19"),
        ),
        "dynamic": Material("dynamic", color_tuple("#455A64")),
        "path_primary": Material(
            "path_primary",
            color_tuple(THREED_ENV_COLORS["path_primary"]),
            emission=color_tuple("#183A30"),
        ),
        "path_secondary": Material(
            "path_secondary",
            color_tuple(THREED_ENV_COLORS["path_secondary"]),
            emission=color_tuple("#4D321B"),
        ),
    }
    for building_type, color in THREED_BUILDING_COLORS.items():
        key = building_material_key(building_type)
        materials[key] = Material(key, color_tuple(color), 0.96)
    for risk_kind, (color, _) in scene2d.RISK_STYLES.items():
        key = risk_material_key(risk_kind)
        materials[key] = Material(key, color_tuple(color), 0.18)
    return materials


def polygon_triangles(polygon: Polygon) -> Iterable[Polygon]:
    if polygon.is_empty or polygon.area <= 1e-8:
        return
    for triangle in triangulate(polygon):
        if triangle.area <= 1e-8:
            continue
        inside_area = triangle.intersection(polygon).area
        if inside_area / triangle.area >= 0.999:
            yield triangle


class ObjSceneBuilder:
    def __init__(self, materials: Dict[str, Material]):
        self.materials = materials
        self.objects: List[MeshObject] = []

    def add_mesh(
        self,
        name: str,
        material: str,
        vertices: Sequence[Sequence[float]],
        faces: Sequence[Sequence[int]],
        category: str,
        metadata: Optional[Dict[str, object]] = None,
    ):
        if not vertices or not faces:
            return
        if material not in self.materials:
            raise KeyError("未定义材质: %s" % material)
        object_vertices = [
            (float(vertex[0]), float(vertex[1]), float(vertex[2]))
            for vertex in vertices
        ]
        object_faces = [tuple(int(index) for index in face) for face in faces]
        self.objects.append(MeshObject(
            name=safe_name(name),
            material=material,
            vertices=object_vertices,
            faces=object_faces,
            category=category,
            metadata=metadata or {},
        ))

    def add_polygon_surface(
        self,
        name: str,
        geometry,
        z_m: float,
        material: str,
        category: str,
        metadata: Optional[Dict[str, object]] = None,
    ):
        part_index = 0
        for polygon in scene2d.iter_polygons(geometry):
            vertices: List[Tuple[float, float, float]] = []
            faces: List[Tuple[int, int, int]] = []
            for triangle in polygon_triangles(polygon):
                coords = list(triangle.exterior.coords)[:3]
                start = len(vertices)
                vertices.extend((x, y, z_m) for x, y in coords)
                faces.append((start, start + 1, start + 2))
            self.add_mesh(
                "%s_p%d" % (name, part_index),
                material,
                vertices,
                faces,
                category,
                metadata,
            )
            part_index += 1

    def add_extruded_polygon(
        self,
        name: str,
        geometry,
        z_min_m: float,
        z_max_m: float,
        material: str,
        category: str,
        metadata: Optional[Dict[str, object]] = None,
    ):
        if z_max_m <= z_min_m:
            raise ValueError("%s 的挤出高度必须为正" % name)
        for part_index, polygon in enumerate(scene2d.iter_polygons(geometry)):
            vertices: List[Tuple[float, float, float]] = []
            faces: List[Tuple[int, ...]] = []

            for triangle in polygon_triangles(polygon):
                coords = list(triangle.exterior.coords)[:3]
                top_start = len(vertices)
                vertices.extend((x, y, z_max_m) for x, y in coords)
                faces.append((top_start, top_start + 1, top_start + 2))
                bottom_start = len(vertices)
                vertices.extend((x, y, z_min_m) for x, y in coords)
                faces.append((bottom_start + 2, bottom_start + 1, bottom_start))

            rings = [polygon.exterior, *polygon.interiors]
            for ring in rings:
                coords = list(ring.coords)
                for index in range(len(coords) - 1):
                    x0, y0 = coords[index]
                    x1, y1 = coords[index + 1]
                    start = len(vertices)
                    vertices.extend([
                        (x0, y0, z_min_m),
                        (x1, y1, z_min_m),
                        (x1, y1, z_max_m),
                        (x0, y0, z_max_m),
                    ])
                    faces.extend([
                        (start, start + 1, start + 2),
                        (start, start + 2, start + 3),
                    ])

            self.add_mesh(
                "%s_p%d" % (name, part_index),
                material,
                vertices,
                faces,
                category,
                metadata,
            )

    def add_box(
        self,
        name: str,
        center: Tuple[float, float, float],
        size: Tuple[float, float, float],
        material: str,
        category: str,
        metadata: Optional[Dict[str, object]] = None,
    ):
        cx, cy, cz = center
        sx, sy, sz = size
        x0, x1 = cx - sx / 2, cx + sx / 2
        y0, y1 = cy - sy / 2, cy + sy / 2
        z0, z1 = cz - sz / 2, cz + sz / 2
        vertices = [
            (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
            (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1),
        ]
        faces = [
            (0, 2, 1), (0, 3, 2),
            (4, 5, 6), (4, 6, 7),
            (0, 1, 5), (0, 5, 4),
            (1, 2, 6), (1, 6, 5),
            (2, 3, 7), (2, 7, 6),
            (3, 0, 4), (3, 4, 7),
        ]
        self.add_mesh(name, material, vertices, faces, category, metadata)

    def add_elliptic_cylinder(
        self,
        name: str,
        center_xy: Tuple[float, float],
        radius_x_m: float,
        radius_y_m: float,
        z_min_m: float,
        z_max_m: float,
        material: str,
        category: str,
        segments: int = 36,
        angle_deg: float = 0.0,
        metadata: Optional[Dict[str, object]] = None,
    ):
        angle = math.radians(angle_deg)
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        vertices: List[Tuple[float, float, float]] = []
        for z_m in (z_min_m, z_max_m):
            for index in range(segments):
                theta = 2 * math.pi * index / segments
                lx = radius_x_m * math.cos(theta)
                ly = radius_y_m * math.sin(theta)
                x = center_xy[0] + lx * cos_a - ly * sin_a
                y = center_xy[1] + lx * sin_a + ly * cos_a
                vertices.append((x, y, z_m))
        vertices.extend([
            (center_xy[0], center_xy[1], z_min_m),
            (center_xy[0], center_xy[1], z_max_m),
        ])
        bottom_center = 2 * segments
        top_center = bottom_center + 1
        faces: List[Tuple[int, ...]] = []
        for index in range(segments):
            nxt = (index + 1) % segments
            faces.extend([
                (index, nxt, segments + nxt),
                (index, segments + nxt, segments + index),
                (bottom_center, nxt, index),
                (top_center, segments + index, segments + nxt),
            ])
        self.add_mesh(name, material, vertices, faces, category, metadata)

    def add_tube_between(
        self,
        name: str,
        start: Sequence[float],
        end: Sequence[float],
        radius_m: float,
        material: str,
        category: str,
        segments: int = 8,
        metadata: Optional[Dict[str, object]] = None,
    ):
        p0 = np.asarray(start, dtype=float)
        p1 = np.asarray(end, dtype=float)
        axis = p1 - p0
        length = float(np.linalg.norm(axis))
        if length <= 1e-6:
            return
        direction = axis / length
        reference = (
            np.array([0.0, 0.0, 1.0])
            if abs(direction[2]) < 0.9
            else np.array([1.0, 0.0, 0.0])
        )
        basis_u = np.cross(direction, reference)
        basis_u /= np.linalg.norm(basis_u)
        basis_v = np.cross(direction, basis_u)

        vertices: List[Tuple[float, float, float]] = []
        for point in (p0, p1):
            for index in range(segments):
                theta = 2 * math.pi * index / segments
                offset = radius_m * (
                    math.cos(theta) * basis_u + math.sin(theta) * basis_v
                )
                vertices.append(tuple(point + offset))
        faces: List[Tuple[int, ...]] = []
        for index in range(segments):
            nxt = (index + 1) % segments
            faces.extend([
                (index, nxt, segments + nxt),
                (index, segments + nxt, segments + index),
            ])
        self.add_mesh(name, material, vertices, faces, category, metadata)

    def validate(self):
        if not self.objects:
            raise ValueError("三维场景没有生成任何网格")
        for obj in self.objects:
            vertex_count = len(obj.vertices)
            if vertex_count < 3:
                raise ValueError("%s 顶点不足" % obj.name)
            for face in obj.faces:
                if len(face) < 3:
                    raise ValueError("%s 存在退化面" % obj.name)
                if min(face) < 0 or max(face) >= vertex_count:
                    raise ValueError("%s 面索引越界" % obj.name)
            array = np.asarray(obj.vertices, dtype=float)
            if not np.all(np.isfinite(array)):
                raise ValueError("%s 顶点含 NaN/无穷值" % obj.name)

    def stats(self) -> Dict[str, object]:
        categories: Dict[str, int] = {}
        for obj in self.objects:
            categories[obj.category] = categories.get(obj.category, 0) + 1
        return {
            "object_count": len(self.objects),
            "vertex_count": sum(len(obj.vertices) for obj in self.objects),
            "triangle_count": sum(
                sum(max(1, len(face) - 2) for face in obj.faces)
                for obj in self.objects
            ),
            "object_counts_by_category": categories,
        }

    def write_obj(self, obj_path: Path) -> Path:
        self.validate()
        obj_path.parent.mkdir(parents=True, exist_ok=True)
        mtl_path = obj_path.with_suffix(".mtl")
        with mtl_path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write("# Wuhan reference scene materials\n")
            for material in self.materials.values():
                handle.write("newmtl %s\n" % material.name)
                handle.write(
                    "Kd %.6f %.6f %.6f\n" % material.color
                )
                handle.write(
                    "Ka %.6f %.6f %.6f\n"
                    % tuple(component * 0.22 for component in material.color)
                )
                handle.write("Ke %.6f %.6f %.6f\n" % material.emission)
                handle.write("d %.6f\n" % material.alpha)
                handle.write("illum 2\n\n")

        vertex_offset = 1
        with obj_path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write("# Wuhan 2D-to-3D research scene\n")
            handle.write("mtllib %s\n" % mtl_path.name)
            for obj in self.objects:
                handle.write("\no %s\n" % obj.name)
                handle.write("usemtl %s\n" % obj.material)
                for x, y, z in obj.vertices:
                    handle.write("v %.6f %.6f %.6f\n" % (x, y, z))
                for face in obj.faces:
                    indices = [str(vertex_offset + index) for index in face]
                    handle.write("f %s\n" % " ".join(indices))
                vertex_offset += len(obj.vertices)
        return mtl_path


def terrain_mesh(
    builder: ObjSceneBuilder,
    terrain: TerrainSurface,
):
    vertices: List[Tuple[float, float, float]] = []
    faces: List[Tuple[int, int, int]] = []
    max_samples = 100
    x_indices = np.unique(np.linspace(
        0, len(terrain.x) - 1, min(len(terrain.x), max_samples), dtype=int
    ))
    y_indices = np.unique(np.linspace(
        0, len(terrain.y) - 1, min(len(terrain.y), max_samples), dtype=int
    ))
    x_values = terrain.x[x_indices]
    y_values = terrain.y[y_indices]
    z_values = terrain.z[np.ix_(y_indices, x_indices)]
    for iy, y_m in enumerate(y_values):
        for ix, x_m in enumerate(x_values):
            vertices.append((float(x_m), float(y_m), float(z_values[iy, ix])))
    nx = len(x_values)
    for iy in range(len(y_values) - 1):
        for ix in range(nx - 1):
            a = iy * nx + ix
            b = a + 1
            c = a + nx
            d = c + 1
            faces.extend([(a, b, d), (a, d, c)])
    builder.add_mesh(
        "terrain",
        "terrain",
        vertices,
        faces,
        "terrain",
        metadata=terrain.summary(),
    )


def lod_geometry(geometry, lod: int):
    if lod == 0:
        return geometry
    if lod == 1:
        simplified = geometry.simplify(2.0, preserve_topology=True)
        return simplified if not simplified.is_empty else geometry
    if lod == 2:
        return geometry.minimum_rotated_rectangle
    raise ValueError("LOD 必须为 0、1 或 2")


def road_width_m(tags: Dict[str, str]) -> float:
    highway = tags.get("highway", "")
    if highway in {"motorway", "trunk", "primary"}:
        return 18.0
    if highway in {"secondary", "tertiary"}:
        return 12.0
    if highway in {"residential", "unclassified"}:
        return 8.0
    if highway == "service":
        return 6.0
    if highway in {"footway", "path", "pedestrian", "cycleway", "steps"}:
        return 2.5
    return 5.0


def geometry_to_footprints(geometry) -> List[Dict[str, object]]:
    result = []
    for polygon in scene2d.iter_polygons(geometry):
        result.append({
            "exterior_xy_m": [
                [round(float(x), 3), round(float(y), 3)]
                for x, y in polygon.exterior.coords
            ],
            "holes_xy_m": [
                [
                    [round(float(x), 3), round(float(y), 3)]
                    for x, y in ring.coords
                ]
                for ring in polygon.interiors
            ],
        })
    return result


def risk_vertical_range(kind: str, hard: bool) -> Tuple[float, float]:
    if hard:
        return 0.0, 180.0
    ranges = {
        "GPS多路径": (8.0, 130.0),
        "通信阴影": (5.0, 120.0),
        "湖面侧风": (10.0, 140.0),
        "江面阵风": (10.0, 150.0),
        "河道风廊": (10.0, 140.0),
        "飞鸟活动": (15.0, 110.0),
        "动态交通": (0.0, 30.0),
        "人群密集": (0.0, 25.0),
        "船舶走廊": (0.0, 35.0),
        "临时施工": (0.0, 90.0),
    }
    return ranges.get(kind, (0.0, 100.0))


def task_altitudes(
    features: Sequence[scene2d.Feature],
    placements: Dict[int, scene2d.TaskPlacement],
    terrain: TerrainSurface,
) -> Dict[int, float]:
    buildings = {
        feature.feature_id: feature
        for feature in features
        if feature.layer == "building"
    }
    altitudes: Dict[int, float] = {}
    for task_id, placement in placements.items():
        target = buildings.get(placement.target_feature_id or "")
        ground_z = terrain.height_at(placement.x_m, placement.y_m)
        if target is None:
            altitudes[task_id] = ground_z + 0.8
        else:
            altitudes[task_id] = ground_z + float(target.height_m or 10.0) + 0.8
    return altitudes


def add_review_paths(
    builder: ObjSceneBuilder,
    zone: scene2d.ZoneSpec,
    features: Sequence[scene2d.Feature],
    placements: Dict[int, scene2d.TaskPlacement],
    task_z: Dict[int, float],
    terrain: TerrainSurface,
) -> List[Dict[str, object]]:
    tasks = list(zone.tasks)
    if len(tasks) < 3:
        return []
    paths = []
    origin = tasks[0]
    for path_index, target in enumerate(tasks[1:], start=1):
        start_placement = placements[origin.point_id]
        end_placement = placements[target.point_id]
        start_xy = np.array([start_placement.x_m, start_placement.y_m])
        end_xy = np.array([end_placement.x_m, end_placement.y_m])
        horizontal_line = LineString([start_xy, end_xy])
        obstacle_top = 0.0
        for feature in features:
            if (
                feature.layer == "building"
                and feature.geometry is not None
                and feature.geometry.intersects(horizontal_line.buffer(30.0))
            ):
                point = feature.geometry.representative_point()
                top = (
                    terrain.height_at(point.x, point.y)
                    + float(feature.height_m or 10.0)
                )
                obstacle_top = max(obstacle_top, top)
        cruise_z = min(190.0, max(70.0, obstacle_top + 25.0))
        if path_index == 2:
            cruise_z = min(195.0, cruise_z + 15.0)

        direction = end_xy - start_xy
        length = float(np.linalg.norm(direction))
        normal = (
            np.array([-direction[1], direction[0]]) / length
            if length > 1e-6
            else np.array([0.0, 0.0])
        )
        lateral_offset = normal * (35.0 if path_index == 2 else 0.0)
        midpoint = (start_xy + end_xy) / 2 + lateral_offset
        points = [
            (start_xy[0], start_xy[1], task_z[origin.point_id] + 2.0),
            (start_xy[0], start_xy[1], cruise_z),
            (midpoint[0], midpoint[1], cruise_z),
            (end_xy[0], end_xy[1], cruise_z),
            (end_xy[0], end_xy[1], task_z[target.point_id] + 2.0),
        ]
        material = "path_primary" if path_index == 1 else "path_secondary"
        grade = "一级审核走廊" if path_index == 1 else "二级审核走廊"
        for segment_index in range(len(points) - 1):
            builder.add_tube_between(
                "review_path_%d_segment_%d" % (path_index, segment_index),
                points[segment_index],
                points[segment_index + 1],
                radius_m=2.8 if path_index == 1 else 2.2,
                material=material,
                category="review_path",
                metadata={
                    "grade": grade,
                    "purpose": "3d_review_only_not_planned_route",
                },
            )
        paths.append({
            "path_id": "review_%d" % path_index,
            "grade": grade,
            "from_task": origin.point_id,
            "to_task": target.point_id,
            "cruise_altitude_m": round(cruise_z, 2),
            "points_enu_m": [
                [round(float(x), 2), round(float(y), 2), round(float(z), 2)]
                for x, y, z in points
            ],
            "purpose": "3d_review_only_not_planned_route",
        })
    return paths


def build_zone_scene(
    zone: scene2d.ZoneSpec,
    features: Sequence[scene2d.Feature],
    placements: Dict[int, scene2d.TaskPlacement],
    terrain: TerrainSurface,
    lod: int,
    include_review_paths: bool,
) -> Tuple[ObjSceneBuilder, Dict[str, object], Dict[str, object]]:
    materials = material_library()
    builder = ObjSceneBuilder(materials)
    terrain_mesh(builder, terrain)

    for feature in features:
        if feature.geometry is None or feature.geometry.is_empty:
            continue
        source = scene2d.feature_geometry_source(feature)
        metadata = {
            "feature_id": feature.feature_id,
            "geometry_source": source,
        }
        if feature.layer == "green":
            material = (
                "forest"
                if feature.tags.get("natural") in {"wood", "forest"}
                else "green"
            )
            builder.add_polygon_surface(
                "green_%s" % feature.feature_id,
                lod_geometry(feature.geometry, min(lod, 1)),
                0.12,
                material,
                "green",
                metadata,
            )
        elif feature.layer == "sport":
            builder.add_polygon_surface(
                "sport_%s" % feature.feature_id,
                lod_geometry(feature.geometry, min(lod, 1)),
                0.22,
                "sport",
                "sport",
                metadata,
            )
        elif feature.layer == "water":
            builder.add_polygon_surface(
                "water_%s" % feature.feature_id,
                lod_geometry(feature.geometry, min(lod, 1)),
                0.18,
                "water",
                "water",
                metadata,
            )
        elif feature.layer == "waterway":
            width = 8.0 if feature.tags.get("waterway") in {"river", "canal"} else 3.0
            surface = feature.geometry.buffer(width / 2, cap_style=2, join_style=2)
            builder.add_polygon_surface(
                "waterway_%s" % feature.feature_id,
                surface,
                0.19,
                "water",
                "waterway",
                metadata,
            )

    bridges = []
    power_collision = []
    for feature in features:
        if feature.geometry is None or feature.geometry.is_empty:
            continue
        if feature.layer == "road":
            if (
                lod == 2
                and feature.tags.get("highway")
                not in {"motorway", "trunk", "primary", "secondary", "tertiary"}
                and feature.tags.get("bridge") in {None, "no"}
            ):
                continue
            width = road_width_m(feature.tags)
            road_polygon = feature.geometry.buffer(
                width / 2,
                cap_style=2,
                join_style=2,
            )
            is_bridge = feature.tags.get("bridge") not in {None, "no"}
            if is_bridge:
                bridge_z = 8.0
                builder.add_extruded_polygon(
                    "bridge_%s" % feature.feature_id,
                    road_polygon,
                    bridge_z,
                    bridge_z + 1.4,
                    "bridge",
                    "bridge",
                    {"feature_id": feature.feature_id, "width_m": width},
                )
                bridges.append({
                    "feature_id": feature.feature_id,
                    "z_min_m": bridge_z,
                    "z_max_m": bridge_z + 1.4,
                    "footprints": geometry_to_footprints(road_polygon),
                })
            else:
                builder.add_polygon_surface(
                    "road_%s" % feature.feature_id,
                    road_polygon,
                    0.28,
                    "road",
                    "road",
                    {"feature_id": feature.feature_id, "width_m": width},
                )
        elif feature.layer == "railway":
            rail_polygon = feature.geometry.buffer(
                3.5, cap_style=2, join_style=2
            )
            builder.add_polygon_surface(
                "rail_%s" % feature.feature_id,
                rail_polygon,
                0.34,
                "rail",
                "railway",
                {"feature_id": feature.feature_id},
            )
        elif feature.layer == "power":
            for line_index, line in enumerate(scene2d.iter_lines(feature.geometry)):
                coords = list(line.coords)
                for segment_index in range(len(coords) - 1):
                    start_xy = coords[segment_index]
                    end_xy = coords[segment_index + 1]
                    start = (start_xy[0], start_xy[1], 35.0)
                    end = (end_xy[0], end_xy[1], 35.0)
                    builder.add_tube_between(
                        "power_%s_%d_%d"
                        % (feature.feature_id, line_index, segment_index),
                        start,
                        end,
                        0.45,
                        "power",
                        "power",
                        metadata={"feature_id": feature.feature_id},
                    )
                    power_collision.append({
                        "feature_id": feature.feature_id,
                        "start_enu_m": [round(v, 3) for v in start],
                        "end_enu_m": [round(v, 3) for v in end],
                        "collision_radius_m": 1.5,
                    })

    target_feature_ids = {
        placement.target_feature_id
        for placement in placements.values()
        if placement.target_feature_id
    }
    building_collisions = []
    building_count = 0
    for feature in features:
        if (
            feature.layer != "building"
            or feature.geometry is None
            or feature.geometry.is_empty
        ):
            continue
        building_count += 1
        geometry = lod_geometry(feature.geometry, lod)
        representative = feature.geometry.representative_point()
        ground_z = terrain.height_at(representative.x, representative.y)
        height_m = float(feature.height_m or 10.0)
        material = building_material_key(
            feature.building_type or "公共建筑"
        )
        builder.add_extruded_polygon(
            "building_%s" % feature.feature_id,
            geometry,
            ground_z,
            ground_z + height_m,
            material,
            "building",
            {
                "feature_id": feature.feature_id,
                "building_type": feature.building_type,
                "height_m": height_m,
                "height_source": feature.height_source,
                "geometry_source": scene2d.feature_geometry_source(feature),
                "lod": lod,
            },
        )
        minx, miny, maxx, maxy = feature.geometry.bounds
        building_collisions.append({
            "feature_id": feature.feature_id,
            "building_type": feature.building_type,
            "height_m": round(height_m, 3),
            "height_source": feature.height_source,
            "geometry_source": scene2d.feature_geometry_source(feature),
            "z_min_m": round(ground_z, 3),
            "z_max_m": round(ground_z + height_m, 3),
            "aabb_enu_m": [
                round(minx, 3), round(miny, 3), round(ground_z, 3),
                round(maxx, 3), round(maxy, 3), round(ground_z + height_m, 3),
            ],
            "footprints": geometry_to_footprints(feature.geometry),
        })
        if (
            lod == 0
            and (
                height_m >= 30.0
                or feature.feature_id in target_feature_ids
            )
        ):
            roof_point = feature.geometry.representative_point()
            equipment_height = 2.5 if height_m < 80 else 4.0
            builder.add_box(
                "roof_equipment_%s" % feature.feature_id,
                (
                    roof_point.x,
                    roof_point.y,
                    ground_z + height_m + equipment_height / 2,
                ),
                (8.0, 5.0, equipment_height),
                "roof_equipment",
                "roof_equipment",
                {"parent_feature_id": feature.feature_id},
            )
            if height_m >= 80.0:
                builder.add_tube_between(
                    "antenna_%s" % feature.feature_id,
                    (
                        roof_point.x,
                        roof_point.y,
                        ground_z + height_m + equipment_height,
                    ),
                    (
                        roof_point.x,
                        roof_point.y,
                        ground_z + height_m + equipment_height + 8.0,
                    ),
                    0.65,
                    "roof_equipment",
                    "roof_equipment",
                    metadata={"parent_feature_id": feature.feature_id},
                )

    risk_records = []
    hard_risk_collisions = []
    for index, risk in enumerate(zone.risks):
        z_min, z_max = risk_vertical_range(risk.kind, risk.hard)
        center = (
            risk.x_ratio * zone.width_m,
            risk.y_ratio * zone.height_m,
        )
        builder.add_elliptic_cylinder(
            "risk_%02d_%s" % (index, risk.kind),
            center,
            risk.width_ratio * zone.width_m / 2,
            risk.height_ratio * zone.height_m / 2,
            z_min,
            z_max,
            risk_material_key(risk.kind),
            "risk_volume",
            segments=42 if lod == 0 else 24,
            angle_deg=risk.angle_deg,
            metadata={"kind": risk.kind, "hard": risk.hard},
        )
        record = {
            "kind": risk.kind,
            "hard": risk.hard,
            "center_enu_m": [round(center[0], 2), round(center[1], 2)],
            "diameter_enu_m": [
                round(risk.width_ratio * zone.width_m, 2),
                round(risk.height_ratio * zone.height_m, 2),
            ],
            "angle_deg": risk.angle_deg,
            "z_min_m": z_min,
            "z_max_m": z_max,
        }
        risk_records.append(record)
        if risk.hard:
            hard_risk_collisions.append(record)

    task_z = task_altitudes(features, placements, terrain)
    task_records = []
    for task in zone.tasks:
        placement = placements[task.point_id]
        z_m = task_z[task.point_id]
        is_depot = task.point_id == 1
        builder.add_elliptic_cylinder(
            "task_P%d" % task.point_id,
            (placement.x_m, placement.y_m),
            8.0 if not is_depot else 11.0,
            8.0 if not is_depot else 11.0,
            z_m,
            z_m + 1.0,
            "depot" if is_depot else "task",
            "task_point",
            segments=24,
            metadata={"task_id": task.point_id, "site_type": task.site_type},
        )
        builder.add_tube_between(
            "task_marker_P%d" % task.point_id,
            (placement.x_m, placement.y_m, z_m + 1.0),
            (placement.x_m, placement.y_m, z_m + 9.0),
            0.7,
            "depot" if is_depot else "task",
            "task_marker",
            metadata={"task_id": task.point_id},
        )
        task_records.append({
            "task_id": task.point_id,
            "site_type": task.site_type,
            "position_enu_m": [
                round(placement.x_m, 3),
                round(placement.y_m, 3),
                round(z_m, 3),
            ],
            "placement_source": placement.placement_source,
            "target_feature_id": placement.target_feature_id,
            "target_geometry_source": placement.target_geometry_source,
            "target_building_type": placement.target_building_type,
        })

    dynamic_records = []
    for index, dynamic in enumerate(zone.dynamics):
        x_m = dynamic.x_ratio * zone.width_m
        y_m = dynamic.y_ratio * zone.height_m
        kind = dynamic.kind
        if "鸟" in kind:
            z_m, size = 70.0, (5.0, 3.0, 2.0)
        elif "无人机" in kind:
            z_m, size = 90.0, (4.0, 4.0, 1.5)
        elif "船" in kind:
            z_m, size = 1.8, (12.0, 4.0, 2.0)
        elif kind in {"车辆", "校车"}:
            z_m, size = 1.5, (8.0, 3.0, 3.0)
        else:
            z_m, size = 1.0, (1.2, 1.2, 2.0)
        builder.add_box(
            "dynamic_%02d" % index,
            (x_m, y_m, terrain.height_at(x_m, y_m) + z_m),
            size,
            "dynamic",
            "dynamic_entity",
            {"kind": kind, "direction_deg": dynamic.direction_deg},
        )
        dynamic_records.append({
            "kind": kind,
            "position_enu_m": [
                round(x_m, 2),
                round(y_m, 2),
                round(terrain.height_at(x_m, y_m) + z_m, 2),
            ],
            "direction_deg": dynamic.direction_deg,
            "collision_box_size_m": list(size),
        })

    gate_records = []
    for feature in features:
        if feature.layer != "gate" or feature.geometry is None:
            continue
        point = feature.geometry
        ground_z = terrain.height_at(point.x, point.y)
        builder.add_box(
            "gate_%s" % feature.tags.get("ref", feature.feature_id),
            (point.x, point.y, ground_z + 2.5),
            (6.0, 2.0, 5.0),
            "gate",
            "gate",
            {"ref": feature.tags.get("ref")},
        )
        gate_records.append({
            "ref": feature.tags.get("ref"),
            "position_enu_m": [
                round(point.x, 2),
                round(point.y, 2),
                round(ground_z, 2),
            ],
        })

    review_paths = (
        add_review_paths(
            builder,
            zone,
            features,
            placements,
            task_z,
            terrain,
        )
        if include_review_paths
        else []
    )

    collision_manifest = {
        "schema_version": SCHEMA_VERSION,
        "coordinate_system": "local_ENU_meters_z_up",
        "zone_id": zone.zone_id,
        "buildings": building_collisions,
        "bridges": bridges,
        "power_lines": power_collision,
        "hard_risk_zones": hard_risk_collisions,
        "dynamic_entities_initial_state": dynamic_records,
        "notes": [
            "建筑碰撞体使用原始 2D 轮廓棱柱，不随可视 LOD 简化。",
            "高压线以线段胶囊半径近似；动态实体需由环境步进器更新。",
        ],
    }
    scene_metadata = {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "converter": Path(__file__).name,
        "zone_id": zone.zone_id,
        "title": zone.title,
        "archetype": zone.archetype,
        "coordinate_system": "local_ENU_meters_z_up",
        "extent_m": [zone.width_m, zone.height_m],
        "lod": lod,
        "lod_definition": {
            "0": "建筑原轮廓 + 屋顶设备/高层天线；近距离审核",
            "1": "建筑轮廓拓扑保持简化；任务走廊审核",
            "2": "建筑最小旋转包围盒；远景与低开销训练",
        },
        "terrain": terrain.summary(),
        "mesh_stats": builder.stats(),
        "building_count": building_count,
        "building_height_range_m": [
            round(min(item["height_m"] for item in building_collisions), 2),
            round(max(item["height_m"] for item in building_collisions), 2),
        ] if building_collisions else [0.0, 0.0],
        "task_points": task_records,
        "risk_volumes": risk_records,
        "dynamic_entities_initial_state": dynamic_records,
        "gates": gate_records,
        "review_paths": review_paths,
        "review_paths_are_real_routes": False,
        "geometry_provenance": scene2d.geometry_source_counts(features),
        "building_inference": scene2d.building_inference_summary(features),
        "water_topology_audit": scene2d.water_topology_audit(features),
        "campus_detail_audit": (
            scene2d.campus_detail_audit(features, placements)
            if zone.zone_id == "B"
            else None
        ),
        "accuracy_class": "research_visualization_not_survey_grade",
        "contains_sensitive_facility_names": False,
    }
    return builder, scene_metadata, collision_manifest


def preview_scene(
    builder: ObjSceneBuilder,
    zone: scene2d.ZoneSpec,
    metadata: Dict[str, object],
    terrain: TerrainSurface,
    output_path: Path,
    dpi: int,
    max_faces: int,
):
    scene2d.apply_style()
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_proj_type("ortho")
    camera = PREVIEW_CAMERAS.get(
        zone.zone_id,
        {"elev": 50, "azim": -65, "vertical_aspect_m": 450.0},
    )
    target_feature_ids = {
        task["target_feature_id"]
        for task in metadata["task_points"]
        if task.get("target_feature_id")
    }

    grid_x = np.linspace(0.0, zone.width_m, 18)
    grid_y = np.linspace(0.0, zone.height_m, 14)
    terrain_x, terrain_y = np.meshgrid(grid_x, grid_y)
    terrain_z = np.vectorize(terrain.height_at)(terrain_x, terrain_y)
    if terrain.mode == "flat_no_dem_input":
        ax.plot_wireframe(
            terrain_x,
            terrain_y,
            terrain_z,
            color="#9E9E9E",
            alpha=0.20,
            linewidth=0.45,
            rstride=1,
            cstride=1,
        )
    else:
        ax.plot_surface(
            terrain_x,
            terrain_y,
            terrain_z,
            color=builder.materials["terrain"].color,
            alpha=0.30,
            linewidth=0,
            antialiased=True,
        )

    all_faces = sum(
        len(obj.faces)
        for obj in builder.objects
        if (
            obj.category != "terrain"
            and not (
                obj.category == "roof_equipment"
                and obj.metadata.get("parent_feature_id")
                not in target_feature_ids
            )
        )
    )
    stride = max(1, math.ceil(all_faces / max_faces))
    face_counter = 0
    for obj in builder.objects:
        if obj.category in {"terrain", "risk_volume"}:
            continue
        if (
            obj.category == "roof_equipment"
            and obj.metadata.get("parent_feature_id")
            not in target_feature_ids
        ):
            continue
        material = builder.materials[obj.material]
        polygons = []
        for face in obj.faces:
            face_counter += 1
            if face_counter % stride != 0 and obj.category not in {
                "task_point", "review_path", "risk_volume"
            }:
                continue
            if len(face) == 3:
                polygons.append([obj.vertices[index] for index in face])
            else:
                anchor = face[0]
                for index in range(1, len(face) - 1):
                    polygons.append([
                        obj.vertices[anchor],
                        obj.vertices[face[index]],
                        obj.vertices[face[index + 1]],
                    ])
        if not polygons:
            continue
        collection = Poly3DCollection(
            polygons,
            facecolor=material.color,
            edgecolor="none",
            alpha=material.alpha,
            linewidth=0.0,
            zorder=2,
        )
        ax.add_collection3d(collection)

    for risk in metadata["risk_volumes"]:
        theta = np.linspace(0.0, 2 * np.pi, 96)
        rx = risk["diameter_enu_m"][0] / 2
        ry = risk["diameter_enu_m"][1] / 2
        angle = math.radians(risk["angle_deg"])
        lx = rx * np.cos(theta)
        ly = ry * np.sin(theta)
        x = (
            risk["center_enu_m"][0]
            + lx * math.cos(angle)
            - ly * math.sin(angle)
        )
        y = (
            risk["center_enu_m"][1]
            + lx * math.sin(angle)
            + ly * math.cos(angle)
        )
        color = scene2d.RISK_STYLES[risk["kind"]][0]
        ax.plot(
            x,
            y,
            np.full_like(theta, risk["z_max_m"]),
            color=color,
            lw=0.75,
            alpha=0.36,
            linestyle=(0, (4, 3)),
        )
        for index in range(0, len(theta) - 1, 24):
            ax.plot(
                [x[index], x[index]],
                [y[index], y[index]],
                [risk["z_min_m"], risk["z_max_m"]],
                color=color,
                lw=0.45,
                alpha=0.20,
                linestyle=(0, (4, 3)),
            )

    task_records = metadata["task_points"]
    for task in task_records:
        x_m, y_m, z_m = task["position_enu_m"]
        ax.scatter(
            [x_m], [y_m], [z_m + 8.0],
            s=58, marker="*", color=scene2d.TASK_COLOR,
            edgecolor="white", linewidth=0.8, depthshade=False,
        )
        ax.text(
            x_m, y_m, z_m + 14.0,
            "P%d" % task["task_id"],
            fontsize=8, fontweight="bold", color="#AD1457",
        )

    max_risk_z = max(
        [record["z_max_m"] for record in metadata["risk_volumes"]] + [150.0]
    )
    max_building_z = max(metadata["building_height_range_m"])
    z_max = max(170.0, min(230.0, max(max_risk_z, max_building_z + 25.0)))
    ax.set_xlim(0, zone.width_m)
    ax.set_ylim(0, zone.height_m)
    ax.set_zlim(0, z_max)
    ax.set_xlabel("局部东向 E (m)", labelpad=10)
    ax.set_ylabel("局部北向 N (m)", labelpad=10)
    ax.set_zlabel("高度 Z (m)", labelpad=7)
    ax.set_title(
        "%s\n2D→3D开发预览｜LOD%d｜非测绘级"
        % (zone.title, metadata["lod"]),
        fontsize=17,
        fontweight="bold",
        pad=20,
    )
    ax.view_init(elev=camera["elev"], azim=camera["azim"])
    ax.set_box_aspect((
        zone.width_m,
        zone.height_m,
        camera["vertical_aspect_m"],
    ))
    ax.grid(True, alpha=0.12)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_facecolor((0.97, 0.97, 0.96, 0.22))
        pane.set_edgecolor((0.72, 0.74, 0.74, 0.35))

    present_building_types = sorted({
        obj.metadata.get("building_type")
        for obj in builder.objects
        if obj.category == "building" and obj.metadata.get("building_type")
    })
    handles = [
        Patch(facecolor=THREED_BUILDING_COLORS[btype], label=btype)
        for btype in present_building_types
    ]
    handles.extend([
        Patch(facecolor=THREED_ENV_COLORS["water"], label="水域"),
        Patch(facecolor=THREED_ENV_COLORS["green"], label="绿地/林地"),
        Patch(facecolor=THREED_ENV_COLORS["sport"], label="运动场"),
        Patch(
            facecolor=THREED_ENV_COLORS["path_primary"],
            label="一级审核走廊",
        ),
        Patch(
            facecolor=THREED_ENV_COLORS["path_secondary"],
            label="二级审核走廊",
        ),
    ])
    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.01, 0.98),
        fontsize=8,
        framealpha=0.90,
    )
    fig.text(
        0.5, 0.018,
        "审核图仅突出任务关联屋顶设备，完整OBJ保留全部细节；"
        "风险体和路径不代表现实限制或配送航线",
        ha="center", fontsize=8, color="#607D8B",
    )
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.06, top=0.91)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def validate_obj_file(path: Path) -> Dict[str, int]:
    vertex_count = 0
    face_count = 0
    max_index = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("v "):
                vertex_count += 1
            elif line.startswith("f "):
                face_count += 1
                indices = [
                    int(token.split("/")[0])
                    for token in line.strip().split()[1:]
                ]
                max_index = max(max_index, *indices)
    if vertex_count == 0 or face_count == 0:
        raise ValueError("OBJ 为空或不含面: %s" % path)
    if max_index > vertex_count:
        raise ValueError(
            "OBJ 面索引 %d 超过顶点数 %d" % (max_index, vertex_count)
        )
    return {
        "vertex_count": vertex_count,
        "face_count": face_count,
        "max_face_vertex_index": max_index,
    }


def terrain_for_zone(
    terrain_dir: Optional[Path],
    zone: scene2d.ZoneSpec,
) -> TerrainSurface:
    if terrain_dir is None:
        return TerrainSurface.flat(zone.width_m, zone.height_m)
    terrain_path = terrain_dir / ("zone_%s_terrain.npz" % zone.zone_id)
    if not terrain_path.exists():
        return TerrainSurface.flat(zone.width_m, zone.height_m)
    return TerrainSurface.from_npz(
        terrain_path,
        zone.width_m,
        zone.height_m,
    )


def parse_args():
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parents[1]
    parser = argparse.ArgumentParser(
        description="将武汉参考场景的2D矢量数据转换为OBJ三维场景"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "outputs" / DEFAULT_OUTPUT_DIRNAME,
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=project_root / "data" / "cache" / "osm_reference_scenes",
    )
    parser.add_argument(
        "--terrain-dir",
        type=Path,
        default=None,
        help="可选DEM目录，文件名为 zone_<ID>_terrain.npz",
    )
    parser.add_argument(
        "--zones",
        default="B",
        help="逗号分隔的分区ID；默认先生成已审核的武汉学院B区",
    )
    parser.add_argument("--lod", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument(
        "--all-lods",
        action="store_true",
        help="为每个分区生成LOD0/1/2",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="额外生成三维PNG审核图",
    )
    parser.add_argument(
        "--preview-lods",
        default=None,
        help=(
            "生成预览的LOD，逗号分隔。默认：单LOD构建时预览所选LOD；"
            "--all-lods时只预览LOD0"
        ),
    )
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--max-preview-faces", type=int, default=50000)
    parser.add_argument(
        "--without-review-paths",
        action="store_true",
        help="不生成两条分级三维审核走廊",
    )
    parser.add_argument(
        "--online",
        action="store_true",
        help="允许缓存缺失时访问Overpass；默认严格离线复用2D缓存",
    )
    parser.add_argument("--refresh-osm", action="store_true")
    parser.add_argument("--seed", type=int, default=scene2d.DEFAULT_SEED)
    return parser.parse_args()


def main():
    args = parse_args()
    requested = {
        item.strip().upper()
        for item in args.zones.split(",")
        if item.strip()
    }
    known = {zone.zone_id for zone in scene2d.ZONE_SPECS}
    unknown = requested - known
    if unknown:
        raise ValueError("未知分区: %s" % ", ".join(sorted(unknown)))
    lods = (0, 1, 2) if args.all_lods else (args.lod,)
    if args.preview_lods is None:
        preview_lods = {0} if args.all_lods else {args.lod}
    else:
        preview_lods = {
            int(item.strip())
            for item in args.preview_lods.split(",")
            if item.strip()
        }
        if not preview_lods.issubset({0, 1, 2}):
            raise ValueError("--preview-lods 只允许 0、1、2")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    index = {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "2d_vector_to_3d_obj",
        "coordinate_system": "local_ENU_meters_z_up",
        "contains_real_delivery_routes": False,
        "contains_sensitive_facility_names": False,
        "zones": [],
    }

    for zone in scene2d.ZONE_SPECS:
        if zone.zone_id not in requested:
            continue
        print("[%s] 载入已审核2D矢量特征..." % zone.zone_id)
        payload, source_meta = scene2d.fetch_osm(
            zone,
            args.cache_dir,
            refresh=args.refresh_osm,
            offline=not args.online,
        )
        features = scene2d.parse_osm(payload, zone, args.seed)
        features = scene2d.augment_reference_morphology(zone, features)
        placements = scene2d.resolve_task_placements(zone, features)
        scene2d.validate_scene(zone, features, placements)
        terrain = terrain_for_zone(args.terrain_dir, zone)

        for lod in lods:
            print("[%s] 构建LOD%d三维场景..." % (zone.zone_id, lod))
            builder, metadata, collision = build_zone_scene(
                zone,
                features,
                placements,
                terrain,
                lod,
                include_review_paths=not args.without_review_paths,
            )
            stem = "zone_%s_scene_lod%d" % (zone.zone_id, lod)
            obj_path = args.output_dir / (stem + ".obj")
            mtl_path = builder.write_obj(obj_path)
            obj_audit = validate_obj_file(obj_path)
            collision_path = args.output_dir / (
                "zone_%s_collision.json" % zone.zone_id
            )
            collision_path.write_text(
                json.dumps(collision, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            metadata.update({
                "source_osm_response_sha256": source_meta.get("response_sha256"),
                "source_osm_query_sha256": source_meta.get("query_sha256"),
                "files": {
                    "obj": obj_path.name,
                    "mtl": mtl_path.name,
                    "collision": collision_path.name,
                },
                "obj_validation": obj_audit,
            })
            scene_path = args.output_dir / (stem + ".json")
            metadata["files"]["scene_manifest"] = scene_path.name
            if args.preview and lod in preview_lods:
                preview_path = args.output_dir / (stem + "_preview.png")
                preview_scene(
                    builder,
                    zone,
                    metadata,
                    terrain,
                    preview_path,
                    args.dpi,
                    args.max_preview_faces,
                )
                metadata["files"]["preview"] = preview_path.name
            scene_path.write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            index["zones"].append({
                "zone_id": zone.zone_id,
                "lod": lod,
                "obj": obj_path.name,
                "mtl": mtl_path.name,
                "collision": collision_path.name,
                "scene_manifest": scene_path.name,
                "preview": metadata["files"].get("preview"),
                "mesh_stats": metadata["mesh_stats"],
                "terrain": metadata["terrain"],
                "validation": metadata["obj_validation"],
            })
            print(
                "[%s] LOD%d完成：%d对象，%d顶点，%d三角面"
                % (
                    zone.zone_id,
                    lod,
                    metadata["mesh_stats"]["object_count"],
                    metadata["mesh_stats"]["vertex_count"],
                    metadata["mesh_stats"]["triangle_count"],
                )
            )

    index_path = args.output_dir / "scene_3d_index.json"
    index_path.write_text(
        json.dumps(index, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("3D转换完成；索引: %s" % index_path)


if __name__ == "__main__":
    main()
