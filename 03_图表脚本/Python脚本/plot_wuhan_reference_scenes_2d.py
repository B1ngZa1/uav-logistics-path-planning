"""
基于四张参考图重新设计武汉无人机配送环境的 2D 审核图。

数据来源：
  - 场景几何：OpenStreetMap / Overpass API
  - 任务点与风险区：脱敏训练设定
  - 武汉学院建筑：校方公开校园图、航拍和建设信息仅用于功能分区与
    相对拓扑校核；输出为自绘概化轮廓，非像素描摹、非测绘级
  - 用户提供的高德截图：仅用于选择场景原型和视觉形态，不作为底图或几何来源

本脚本只输出 2D 图，不生成或修改任何 3D 模型。

输出：
  - zone_A_guanggu_core_2d.png
  - zone_B_lakeside_campus_2d.png
  - zone_C_river_bridgehead_2d.png
  - zone_D_canal_highrise_2d.png
  - scene_2d_manifest.json
"""

import argparse
import hashlib
import json
import math
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, Patch, Polygon as MplPolygon, Rectangle
import numpy as np
import requests
from shapely.affinity import rotate as shapely_rotate
from shapely.affinity import translate as shapely_translate
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiLineString,
    MultiPolygon,
    Point,
    Polygon,
    box,
)
from shapely.ops import polygonize, unary_union
from shapely.validation import make_valid


DEFAULT_SEED = 20260703
EARTH_RADIUS_M = 6371008.8
OVERPASS_ENDPOINTS = (
    "https://overpass.private.coffee/api/interpreter",
    "https://lz4.overpass-api.de/api/interpreter",
    "https://overpass-api.de/api/interpreter",
)
USER_AGENT = "WuhanDroneDeliveryResearch/1.0 (academic 2D scene visualization)"

BUILDING_COLORS = {
    "低层住宅": "#D8C3A5",
    "多层住宅": "#F4A261",
    "高层住宅": "#E76F51",
    "商业综合体": "#9B5DE5",
    "办公塔楼": "#277DA1",
    "工业厂房": "#78909C",
    "公共建筑": "#00A6A6",
    "景区建筑": "#5E9F67",
}

RISK_STYLES = {
    "GPS多路径": ("#8E44AD", "..."),
    "通信阴影": ("#546E7A", "\\\\\\"),
    "湖面侧风": ("#039BE5", "xx"),
    "江面阵风": ("#039BE5", "xx"),
    "河道风廊": ("#00ACC1", "xx"),
    "飞鸟活动": ("#6D4C41", "oo"),
    "动态交通": ("#EF6C00", "++"),
    "人群密集": ("#C2185B", ".."),
    "船舶走廊": ("#1565C0", "//"),
    "桥体硬隔离": ("#D32F2F", "///"),
    "轨道硬隔离": ("#D32F2F", "///"),
    "临时施工": ("#FB8C00", "++"),
}

WATER_COLOR = "#62B4E8"
WATER_EDGE = "#1976B9"
GREEN_COLOR = "#9DDC8A"
FOREST_COLOR = "#4F8E5B"
SPORT_COLOR = "#8AE0BB"
ROAD_CASING = "#AAB7C1"
ROAD_MAIN = "#F9F7F0"
ROAD_SECONDARY = "#F1EFE9"
ROAD_LOCAL = "#ECEBE7"
BRIDGE_COLOR = "#5D6D7E"
RAIL_COLOR = "#C2185B"
POWER_COLOR = "#D99B00"
TASK_COLOR = "#D81B60"
DEPOT_COLOR = "#2E7D32"
TEXT_DARK = "#263238"


@dataclass(frozen=True)
class TaskSpec:
    point_id: int
    x_ratio: float
    y_ratio: float
    site_type: str


@dataclass(frozen=True)
class TaskPlacement:
    point_id: int
    x_m: float
    y_m: float
    placement_source: str
    target_feature_id: Optional[str] = None
    target_geometry_source: Optional[str] = None
    target_building_type: Optional[str] = None
    snap_distance_m: float = 0.0


@dataclass(frozen=True)
class RiskSpec:
    kind: str
    x_ratio: float
    y_ratio: float
    width_ratio: float
    height_ratio: float
    angle_deg: float = 0.0
    hard: bool = False


@dataclass(frozen=True)
class DynamicSpec:
    kind: str
    x_ratio: float
    y_ratio: float
    marker: str
    color: str
    direction_deg: float = 0.0


@dataclass(frozen=True)
class ZoneSpec:
    zone_id: str
    title: str
    archetype: str
    center_lat: float
    center_lon: float
    width_m: float
    height_m: float
    output_name: str
    tasks: Tuple[TaskSpec, ...]
    risks: Tuple[RiskSpec, ...]
    dynamics: Tuple[DynamicSpec, ...]
    scene_hint: str


@dataclass
class Feature:
    feature_id: str
    layer: str
    geometry: object
    tags: Dict[str, str] = field(default_factory=dict)
    building_type: Optional[str] = None
    building_type_source: Optional[str] = None
    height_m: Optional[float] = None
    height_source: Optional[str] = None


@dataclass(frozen=True)
class CampusBuildingSpec:
    name: str
    cluster_id: str
    campus_subtype: str
    center_x_m: float
    center_y_m: float
    width_m: float
    depth_m: float
    angle_deg: float
    shape_family: str
    building_type: str
    height_m: float


ZONE_SPECS = (
    ZoneSpec(
        zone_id="A",
        title="分区 A｜环岛高密核心区",
        archetype="布局参考：光谷广场周边",
        center_lat=30.5082612,
        center_lon=114.3929068,
        width_m=2400.0,
        height_m=1800.0,
        output_name="zone_A_guanggu_core_2d.png",
        tasks=(
            TaskSpec(1, 0.17, 0.18, "配送中心"),
            TaskSpec(2, 0.54, 0.58, "商业裙楼屋顶"),
            TaskSpec(3, 0.83, 0.28, "商业综合体平台"),
        ),
        risks=(
            RiskSpec("GPS多路径", 0.59, 0.58, 0.28, 0.32, -10),
            RiskSpec("动态交通", 0.48, 0.47, 0.45, 0.15, 0),
            RiskSpec("轨道硬隔离", 0.52, 0.35, 0.72, 0.032, 0, True),
            RiskSpec("通信阴影", 0.80, 0.62, 0.18, 0.23, 8),
        ),
        dynamics=(
            DynamicSpec("车辆", 0.45, 0.46, "s", "#455A64", 0),
            DynamicSpec("行人", 0.60, 0.51, "o", "#C2185B", 90),
            DynamicSpec("其他无人机", 0.77, 0.68, "D", "#EC407A", -35),
        ),
        scene_hint="椭圆环岛、放射路网、轨道走廊、商业裙楼与高层塔楼",
    ),
    ZoneSpec(
        zone_id="B",
        title="分区 B｜湖滨校园生态区",
        archetype="布局参考：武汉学院黄家湖校区",
        center_lat=30.4433405,
        center_lon=114.2901048,
        width_m=2400.0,
        height_m=1800.0,
        output_name="zone_B_lakeside_campus_2d.png",
        tasks=(
            TaskSpec(4, 0.76, 0.25, "校园后勤点"),
            TaskSpec(5, 0.52, 0.52, "公共建筑屋顶"),
            TaskSpec(6, 0.41, 0.58, "湖岸维护设施"),
        ),
        risks=(
            RiskSpec("湖面侧风", 0.18, 0.66, 0.30, 0.42, -12),
            RiskSpec("飞鸟活动", 0.30, 0.70, 0.18, 0.18, 0),
            RiskSpec("人群密集", 0.63, 0.43, 0.28, 0.18, -8),
            RiskSpec("通信阴影", 0.78, 0.62, 0.18, 0.20, 0),
        ),
        dynamics=(
            DynamicSpec("飞鸟群", 0.29, 0.72, "v", "#6D4C41", -20),
            DynamicSpec("行人", 0.62, 0.43, "o", "#C2185B", 90),
            DynamicSpec("校车", 0.77, 0.18, "s", "#455A64", 15),
        ),
        scene_hint="不规则湖岸、低中层院落组团、内湖绿地、运动场与校园环路",
    ),
    ZoneSpec(
        zone_id="C",
        title="分区 C｜跨江桥头复合区",
        archetype="布局参考：武汉长江大桥武昌桥头",
        center_lat=30.5513,
        center_lon=114.2927,
        width_m=2800.0,
        height_m=2000.0,
        output_name="zone_C_river_bridgehead_2d.png",
        tasks=(
            TaskSpec(7, 0.74, 0.58, "桥头服务设施"),
            TaskSpec(8, 0.86, 0.29, "老城配送平台"),
            TaskSpec(9, 0.68, 0.18, "滨水公共设施"),
        ),
        risks=(
            RiskSpec("江面阵风", 0.29, 0.61, 0.46, 0.65, -18),
            RiskSpec("桥体硬隔离", 0.46, 0.48, 0.57, 0.040, -17, True),
            RiskSpec("船舶走廊", 0.25, 0.52, 0.42, 0.13, -18),
            RiskSpec("人群密集", 0.78, 0.50, 0.22, 0.18, 0),
            RiskSpec("通信阴影", 0.63, 0.38, 0.15, 0.20, -15),
        ),
        dynamics=(
            DynamicSpec("船只", 0.25, 0.55, ">", "#0D47A1", -18),
            DynamicSpec("车辆", 0.55, 0.45, "s", "#455A64", -17),
            DynamicSpec("飞鸟群", 0.64, 0.74, "v", "#6D4C41", 5),
        ),
        scene_hint="大水面、斜向长桥、桥头匝道、滨水绿带与密集老城街块",
    ),
    ZoneSpec(
        zone_id="D",
        title="分区 D｜窄河水岸商办区",
        archetype="布局参考：楚河汉街周边",
        center_lat=30.5590081,
        center_lon=114.3360216,
        width_m=2400.0,
        height_m=1800.0,
        output_name="zone_D_canal_highrise_2d.png",
        tasks=(
            TaskSpec(10, 0.27, 0.60, "商业裙楼屋顶"),
            TaskSpec(11, 0.49, 0.40, "办公塔楼平台"),
            TaskSpec(12, 0.83, 0.34, "社区公共建筑"),
        ),
        risks=(
            RiskSpec("河道风廊", 0.52, 0.49, 0.70, 0.10, -24),
            RiskSpec("GPS多路径", 0.39, 0.57, 0.31, 0.20, -24),
            RiskSpec("通信阴影", 0.66, 0.46, 0.16, 0.19, -12),
            RiskSpec("动态交通", 0.53, 0.32, 0.56, 0.08, -10),
            RiskSpec("临时施工", 0.82, 0.72, 0.14, 0.16, 0),
        ),
        dynamics=(
            DynamicSpec("车辆", 0.52, 0.29, "s", "#455A64", -10),
            DynamicSpec("行人", 0.43, 0.56, "o", "#C2185B", 65),
            DynamicSpec("施工吊臂", 0.82, 0.72, "P", "#F57C00", 0),
        ),
        scene_hint="窄河道、两岸平行道路、多座短桥、连续商业裙楼与高层街谷",
    ),
)


def apply_style():
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["savefig.facecolor"] = "white"
    plt.rcParams["font.size"] = 10


def bbox_for_zone(zone: ZoneSpec) -> Tuple[float, float, float, float]:
    half_lat = (zone.height_m / 2.0) / 111320.0
    half_lon = (zone.width_m / 2.0) / (
        111320.0 * math.cos(math.radians(zone.center_lat))
    )
    return (
        zone.center_lat - half_lat,
        zone.center_lon - half_lon,
        zone.center_lat + half_lat,
        zone.center_lon + half_lon,
    )


def overpass_query(zone: ZoneSpec) -> str:
    south, west, north, east = bbox_for_zone(zone)
    bbox = "%.7f,%.7f,%.7f,%.7f" % (south, west, north, east)
    # 长江水面以 multipolygon relation 表达，C 区必须读取 member geometry。
    # 其余三区沿用轻量查询，避免大型 landuse relation 使公共端点超时。
    output_clause = (
        "out body geom qt;" if zone.zone_id == "C" else "out tags geom qt;"
    )
    return """
[out:json][timeout:120];
(
  way["building"](%(bbox)s);
  relation["building"](%(bbox)s);
  way["highway"](%(bbox)s);
  way["railway"](%(bbox)s);
  way["bridge"](%(bbox)s);
  way["power"](%(bbox)s);
  way["natural"="water"](%(bbox)s);
  relation["natural"="water"](%(bbox)s);
  way["water"](%(bbox)s);
  relation["water"](%(bbox)s);
  way["waterway"](%(bbox)s);
  relation["waterway"="riverbank"](%(bbox)s);
  way["landuse"](%(bbox)s);
  relation["landuse"](%(bbox)s);
  way["leisure"](%(bbox)s);
  relation["leisure"](%(bbox)s);
  way["natural"="wood"](%(bbox)s);
  relation["natural"="wood"](%(bbox)s);
);
%(output_clause)s
""" % {"bbox": bbox, "output_clause": output_clause}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch_osm(
    zone: ZoneSpec,
    cache_dir: Path,
    refresh: bool = False,
    offline: bool = False,
) -> Tuple[Dict, Dict]:
    query = overpass_query(zone)
    query_hash = hashlib.sha256(query.encode("utf-8")).hexdigest()
    zone_cache = cache_dir / zone.zone_id
    zone_cache.mkdir(parents=True, exist_ok=True)
    json_path = zone_cache / ("%s.json" % query_hash)
    meta_path = zone_cache / ("%s.meta.json" % query_hash)

    if json_path.exists() and not refresh:
        raw = json_path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
        metadata = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        actual_sha256 = sha256_bytes(raw)
        expected_sha256 = metadata.get("response_sha256")
        if expected_sha256 and actual_sha256 != expected_sha256:
            raise ValueError(
                "分区 %s 的 OSM 缓存校验失败：期望 %s，实际 %s"
                % (zone.zone_id, expected_sha256, actual_sha256)
            )
        metadata["response_sha256"] = actual_sha256
        metadata["cache_hit"] = True
        return payload, metadata

    if offline:
        raise FileNotFoundError(
            "离线模式下缺少分区 %s 的 OSM 缓存：%s" % (zone.zone_id, json_path)
        )

    errors = []
    headers = {"User-Agent": USER_AGENT}
    for endpoint in OVERPASS_ENDPOINTS:
        for attempt in range(3):
            try:
                response = requests.post(
                    endpoint,
                    data={"data": query},
                    headers=headers,
                    timeout=180,
                )
                if response.status_code == 429 or response.status_code >= 500:
                    raise RuntimeError("HTTP %s" % response.status_code)
                response.raise_for_status()
                payload = response.json()
                raw = json.dumps(
                    payload, ensure_ascii=False, separators=(",", ":")
                ).encode("utf-8")
                json_path.write_bytes(raw)
                metadata = {
                    "zone_id": zone.zone_id,
                    "query_sha256": query_hash,
                    "response_sha256": sha256_bytes(raw),
                    "endpoint": endpoint,
                    "retrieved_utc": datetime.now(timezone.utc).isoformat(),
                    "attribution": "© OpenStreetMap contributors, ODbL",
                    "cache_hit": False,
                }
                meta_path.write_text(
                    json.dumps(metadata, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                return payload, metadata
            except Exception as exc:
                errors.append("%s attempt %d: %s" % (endpoint, attempt + 1, exc))
                if attempt < 2:
                    time.sleep(2 ** attempt)
    raise RuntimeError("Overpass 获取失败：\n" + "\n".join(errors))


def lonlat_to_local(
    lon: float,
    lat: float,
    zone: ZoneSpec,
) -> Tuple[float, float]:
    lat0 = math.radians(zone.center_lat)
    x = EARTH_RADIUS_M * math.cos(lat0) * math.radians(lon - zone.center_lon)
    y = EARTH_RADIUS_M * math.radians(lat - zone.center_lat)
    return x + zone.width_m / 2.0, y + zone.height_m / 2.0


def coords_from_geometry(
    geometry: Sequence[Dict[str, float]],
    zone: ZoneSpec,
) -> List[Tuple[float, float]]:
    return [
        lonlat_to_local(float(item["lon"]), float(item["lat"]), zone)
        for item in geometry
        if "lon" in item and "lat" in item
    ]


def safe_geometry(geometry):
    if geometry is None or geometry.is_empty:
        return None
    try:
        fixed = make_valid(geometry)
    except Exception:
        fixed = geometry.buffer(0)
    if fixed.is_empty:
        return None
    return fixed


def relation_geometry(element: Dict, zone: ZoneSpec, as_polygon: bool):
    member_geometries = []
    outer_polygons = []
    outer_lines = []
    inner_polygons = []
    inner_lines = []
    for member in element.get("members", []):
        if "geometry" not in member:
            continue
        coords = coords_from_geometry(member["geometry"], zone)
        if len(coords) < 2:
            continue
        if not as_polygon:
            member_geometries.append(LineString(coords))
            continue

        role = member.get("role") or "outer"
        is_inner = role == "inner"
        if coords[0] == coords[-1] and len(coords) >= 4:
            target = inner_polygons if is_inner else outer_polygons
            target.append(Polygon(coords))
        else:
            target = inner_lines if is_inner else outer_lines
            target.append(LineString(coords))

    if not as_polygon:
        return unary_union(member_geometries) if member_geometries else None

    if outer_lines:
        outer_polygons.extend(list(polygonize(unary_union(outer_lines))))
    if inner_lines:
        inner_polygons.extend(list(polygonize(unary_union(inner_lines))))
    if not outer_polygons:
        return None

    outer = unary_union(outer_polygons)
    if inner_polygons:
        outer = outer.difference(unary_union(inner_polygons))
    return outer


def layer_for_tags(tags: Dict[str, str]) -> Optional[str]:
    if "building" in tags:
        return "building"
    if tags.get("natural") == "water" or "water" in tags or tags.get("waterway") == "riverbank":
        return "water"
    if "waterway" in tags:
        return "waterway"
    if tags.get("leisure") in {"pitch", "stadium", "sports_centre", "track"}:
        return "sport"
    if (
        tags.get("leisure") in {"park", "garden"}
        or tags.get("natural") == "wood"
        or tags.get("landuse") in {
            "forest", "grass", "meadow", "recreation_ground",
            "village_green", "cemetery",
        }
    ):
        return "green"
    if "railway" in tags and tags.get("railway") not in {"abandoned", "disused"}:
        return "railway"
    if "highway" in tags:
        return "road"
    if tags.get("power") in {"line", "minor_line"}:
        return "power"
    return None


def element_geometry(element: Dict, zone: ZoneSpec, layer: str):
    polygon_layers = {"building", "water", "green", "sport"}
    as_polygon = layer in polygon_layers
    if element.get("type") == "relation":
        return relation_geometry(element, zone, as_polygon)

    raw_geometry = element.get("geometry", [])
    coords = coords_from_geometry(raw_geometry, zone)
    if as_polygon:
        if len(coords) < 3:
            return None
        if coords[0] != coords[-1]:
            coords.append(coords[0])
        return Polygon(coords)
    if len(coords) < 2:
        return None
    return LineString(coords)


def parse_measure(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", str(value))
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def classify_building(
    tags: Dict[str, str],
    area_m2: float,
    zone: ZoneSpec,
) -> str:
    building = tags.get("building", "yes").lower()
    amenity = tags.get("amenity", "").lower()
    landuse = tags.get("landuse", "").lower()
    office = tags.get("office", "").lower()
    shop = tags.get("shop", "").lower()
    tourism = tags.get("tourism", "").lower()
    name = tags.get("name", "")

    if (
        amenity == "ferry_terminal"
        or tags.get("man_made", "").lower() in {"pier", "quay"}
        or any(token in name for token in ("码头", "渡口", "船坞"))
    ):
        return "公共建筑"
    if amenity == "shelter":
        return "景区建筑"
    if building in {"industrial", "warehouse", "hangar", "factory"} or landuse == "industrial":
        return "工业厂房"
    if building in {"retail", "commercial", "supermarket"} or shop or tags.get("mall") == "yes":
        return "商业综合体"
    if building == "office" or office:
        return "办公塔楼"
    if (
        building in {"school", "university", "college", "hospital", "civic", "public"}
        or amenity in {
            "school", "university", "college", "hospital", "clinic",
            "library", "community_centre", "public_building",
        }
    ):
        return "公共建筑"
    if tourism or building in {"temple", "church", "chapel", "shrine"}:
        return "景区建筑"
    if building in {"apartments", "residential", "dormitory"}:
        levels = parse_measure(tags.get("building:levels"))
        if levels is not None and levels >= 18:
            return "高层住宅"
        return "多层住宅"
    if building in {"house", "detached", "terrace", "semidetached_house", "bungalow"}:
        return "低层住宅"

    if zone.zone_id == "B":
        return "公共建筑" if area_m2 > 850 else "多层住宅"
    if zone.zone_id == "C":
        return "低层住宅" if area_m2 < 650 else "多层住宅"
    if zone.zone_id == "D":
        return "商业综合体" if area_m2 > 1200 else "办公塔楼"
    if area_m2 > 1500:
        return "商业综合体"
    return "多层住宅"


def building_type_source(tags: Dict[str, str]) -> str:
    """标记建筑类型来自 OSM 语义标签还是分区/面积推断。"""
    building = tags.get("building", "").lower()
    semantic_buildings = {
        "industrial", "warehouse", "hangar", "factory",
        "retail", "commercial", "supermarket", "office",
        "school", "university", "college", "hospital", "civic", "public",
        "temple", "church", "chapel", "shrine",
        "apartments", "residential", "dormitory",
        "house", "detached", "terrace", "semidetached_house", "bungalow",
    }
    semantic_keys = ("amenity", "office", "shop", "tourism", "mall")
    if building in semantic_buildings or any(tags.get(key) for key in semantic_keys):
        return "OSM semantic tags"
    if tags.get("landuse", "").lower() == "industrial":
        return "OSM semantic tags"
    return "inferred from zone and footprint"


def resolve_height(
    tags: Dict[str, str],
    building_type: str,
    feature_id: str,
    zone: ZoneSpec,
    seed: int,
) -> Tuple[float, str]:
    explicit = parse_measure(tags.get("height"))
    if explicit is not None and 2.0 <= explicit <= 400.0:
        return explicit, "OSM height"
    levels = parse_measure(tags.get("building:levels"))
    if levels is not None and 1 <= levels <= 120:
        floor_height = {
            "商业综合体": 4.2,
            "办公塔楼": 3.8,
            "工业厂房": 5.0,
            "公共建筑": 3.8,
        }.get(building_type, 3.1)
        return levels * floor_height, "OSM levels"

    priors = {
        "低层住宅": (8, 18),
        "多层住宅": (22, 55),
        "高层住宅": (65, 140),
        "商业综合体": (28, 90),
        "办公塔楼": (55, 165),
        "工业厂房": (12, 35),
        "公共建筑": (16, 48),
        "景区建筑": (8, 24),
    }
    if zone.zone_id == "A" and building_type in {"商业综合体", "办公塔楼", "多层住宅"}:
        priors = dict(priors)
        priors["商业综合体"] = (45, 105)
        priors["办公塔楼"] = (75, 180)
        priors["多层住宅"] = (35, 90)
    elif zone.zone_id == "B":
        priors = dict(priors)
        priors["公共建筑"] = (12, 35)
        priors["多层住宅"] = (18, 40)
    elif zone.zone_id == "C":
        priors = dict(priors)
        priors["低层住宅"] = (7, 18)
        priors["多层住宅"] = (18, 45)
    elif zone.zone_id == "D":
        priors = dict(priors)
        priors["商业综合体"] = (35, 100)
        priors["办公塔楼"] = (70, 180)

    digest = hashlib.sha256(
        ("%s:%s:%s" % (feature_id, zone.zone_id, seed)).encode("utf-8")
    ).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    low, high = priors[building_type]
    return float(rng.uniform(low, high)), "programmatic"


def iter_polygons(geometry) -> Iterable[Polygon]:
    if geometry is None or geometry.is_empty:
        return
    if isinstance(geometry, Polygon):
        yield geometry
    elif isinstance(geometry, MultiPolygon):
        for geom in geometry.geoms:
            yield geom
    elif isinstance(geometry, GeometryCollection):
        for geom in geometry.geoms:
            yield from iter_polygons(geom)


def iter_lines(geometry) -> Iterable[LineString]:
    if geometry is None or geometry.is_empty:
        return
    if isinstance(geometry, LineString):
        yield geometry
    elif isinstance(geometry, MultiLineString):
        for geom in geometry.geoms:
            yield geom
    elif isinstance(geometry, GeometryCollection):
        for geom in geometry.geoms:
            yield from iter_lines(geom)


def parse_osm(payload: Dict, zone: ZoneSpec, seed: int) -> List[Feature]:
    clip_box = box(0, 0, zone.width_m, zone.height_m)
    features = []
    seen = set()
    for element in payload.get("elements", []):
        tags = {str(k): str(v) for k, v in element.get("tags", {}).items()}
        layer = layer_for_tags(tags)
        if layer is None:
            continue
        feature_id = "%s/%s" % (element.get("type"), element.get("id"))
        dedupe_key = (feature_id, layer)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        geometry = safe_geometry(element_geometry(element, zone, layer))
        if geometry is None:
            continue
        try:
            geometry = safe_geometry(geometry.intersection(clip_box))
        except Exception:
            continue
        if geometry is None:
            continue

        feature = Feature(
            feature_id=feature_id,
            layer=layer,
            geometry=geometry,
            tags=tags,
        )
        if layer == "building":
            area = geometry.area
            feature.building_type = classify_building(tags, area, zone)
            feature.building_type_source = building_type_source(tags)
            feature.height_m, feature.height_source = resolve_height(
                tags, feature.building_type, feature_id, zone, seed
            )
        features.append(feature)
    return features


def rotated_rectangle(
    cx: float,
    cy: float,
    width: float,
    height: float,
    angle_deg: float = 0.0,
) -> Polygon:
    points = np.array([
        [cx - width / 2, cy - height / 2],
        [cx + width / 2, cy - height / 2],
        [cx + width / 2, cy + height / 2],
        [cx - width / 2, cy + height / 2],
    ])
    if angle_deg:
        angle = math.radians(angle_deg)
        rotation = np.array([
            [math.cos(angle), -math.sin(angle)],
            [math.sin(angle), math.cos(angle)],
        ])
        points = (points - np.array([cx, cy])) @ rotation.T + np.array([cx, cy])
    return Polygon(points)


def ellipse_polygon(
    cx: float,
    cy: float,
    width: float,
    height: float,
    angle_deg: float = 0.0,
    count: int = 72,
) -> Polygon:
    theta = np.linspace(0, 2 * np.pi, count, endpoint=False)
    points = np.column_stack([
        cx + width / 2 * np.cos(theta),
        cy + height / 2 * np.sin(theta),
    ])
    if angle_deg:
        angle = math.radians(angle_deg)
        rotation = np.array([
            [math.cos(angle), -math.sin(angle)],
            [math.sin(angle), math.cos(angle)],
        ])
        points = (points - np.array([cx, cy])) @ rotation.T + np.array([cx, cy])
    return Polygon(points)


def campus_building_polygon(spec: CampusBuildingSpec) -> Polygon:
    """生成保留院落语义的条形、L/U/H 型校园建筑轮廓。"""
    width = spec.width_m
    depth = spec.depth_m
    wing = max(10.0, min(width, depth) * 0.24)

    if spec.shape_family == "bar":
        local = box(-width / 2, -depth / 2, width / 2, depth / 2)
    elif spec.shape_family == "L":
        local = unary_union([
            box(-width / 2, -depth / 2, width / 2, -depth / 2 + wing),
            box(-width / 2, -depth / 2, -width / 2 + wing, depth / 2),
        ])
    elif spec.shape_family == "U":
        local = unary_union([
            box(-width / 2, -depth / 2, width / 2, -depth / 2 + wing),
            box(-width / 2, -depth / 2, -width / 2 + wing, depth / 2),
            box(width / 2 - wing, -depth / 2, width / 2, depth / 2),
        ])
    elif spec.shape_family == "H":
        local = unary_union([
            box(-width / 2, -depth / 2, -width / 2 + wing, depth / 2),
            box(width / 2 - wing, -depth / 2, width / 2, depth / 2),
            box(-width / 2, -wing / 2, width / 2, wing / 2),
        ])
    else:
        raise ValueError("未知校园建筑形态: %s" % spec.shape_family)

    rotated = shapely_rotate(local, spec.angle_deg, origin=(0.0, 0.0))
    return shapely_translate(
        rotated,
        xoff=spec.center_x_m,
        yoff=spec.center_y_m,
    )


def campus_building_feature(
    spec: CampusBuildingSpec,
    geometry,
) -> Feature:
    return Feature(
        feature_id="campus/B/%s" % spec.name,
        layer="building",
        geometry=geometry,
        tags={
            "building": "yes",
            "source": "official_reference_generalization",
            "geometry_method": "parametric_generalization",
            "accuracy_class": "schematic_not_survey_grade",
            "is_synthetic": "true",
            "reference_image_geometry_digitized": "false",
            "cluster_id": spec.cluster_id,
            "campus_subtype": spec.campus_subtype,
            "shape_family": spec.shape_family,
        },
        building_type=spec.building_type,
        building_type_source="official semantic reference + generalized morphology",
        height_m=spec.height_m,
        height_source="official attribute/generalized prior",
    )


def fit_campus_building(
    spec: CampusBuildingSpec,
    forbidden_geometry,
    scene_boundary: Polygon,
) -> Tuple[Polygon, float]:
    """
    在不改变轮廓形态的前提下做小范围确定性平移。

    参考图只约束相对拓扑；实际落位需尊重 OSM 水体、运动场、道路和
    已接纳建筑，因此允许最多 150 m 的脱敏微调。
    """
    offsets = [
        (float(dx), float(dy))
        for dx in range(-150, 151, 10)
        for dy in range(-150, 151, 10)
    ]
    offsets.sort(
        key=lambda item: (
            math.hypot(item[0], item[1]),
            abs(item[1]),
            abs(item[0]),
            item[1],
            item[0],
        )
    )
    base = campus_building_polygon(spec)
    for dx, dy in offsets:
        candidate = shapely_translate(base, xoff=dx, yoff=dy)
        if not scene_boundary.covers(candidate):
            continue
        if forbidden_geometry is not None and candidate.intersects(forbidden_geometry):
            continue
        return candidate, math.hypot(dx, dy)
    raise ValueError(
        "校园建筑 %s 在 150 m 微调范围内找不到无冲突落位" % spec.name
    )


def morphology_feature(
    feature_id: str,
    layer: str,
    geometry,
    tags: Optional[Dict[str, str]] = None,
    building_type: Optional[str] = None,
    height_m: Optional[float] = None,
) -> Feature:
    feature_tags = tags or {}
    source = feature_tags.get("source", "reference_morphology")
    is_official_generalization = (
        layer == "building"
        and source == "official_reference_generalization"
    )
    return Feature(
        feature_id=feature_id,
        layer=layer,
        geometry=geometry,
        tags=feature_tags,
        building_type=building_type,
        building_type_source=(
            "official semantic reference + generalized morphology"
            if is_official_generalization
            else ("reference morphology" if layer == "building" else None)
        ),
        height_m=height_m,
        height_source=(
            "official attribute/generalized prior"
            if is_official_generalization
            else ("reference morphology" if layer == "building" else None)
        ),
    )


def augment_reference_morphology(
    zone: ZoneSpec,
    features: List[Feature],
) -> List[Feature]:
    """
    在 OSM 覆盖不足时补充参考图明确表达的形态锚点。

    补充对象不复制高德几何；它们是按截图形态语言构造的脱敏训练示意，
    并与 OSM 道路/建筑共同用于首轮 2D 审核。
    """
    augmented = list(features)

    if zone.zone_id == "A":
        green_area = sum(
            feature.geometry.area
            for feature in features
            if feature.layer == "green" and feature.geometry is not None
        )
        if green_area < zone.width_m * zone.height_m * 0.035:
            augmented.append(morphology_feature(
                "morph/A/central_island",
                "green",
                ellipse_polygon(
                    zone.width_m * 0.52,
                    zone.height_m * 0.46,
                    zone.width_m * 0.20,
                    zone.height_m * 0.13,
                    -5,
                ),
            ))

    elif zone.zone_id == "B":
        # 武汉学院 B 区采用“开放底图 + 官方公开资料校核 + 自绘概化轮廓”。
        # OSM 在本范围内仅有极少建筑，但水体、道路和运动场较完整；
        # 因此不伪称补绘建筑来自 OSM，也不对官方效果图做像素描摹。
        augmented = [feature for feature in augmented if feature.layer != "railway"]

        campus_buildings = [
            # B1 北部教学组团：U/L 型院落，不再用同尺寸仓库式长条。
            CampusBuildingSpec("north_u1", "B1_north_teaching", "teaching", 1040, 1430, 112, 70, -5, "U", "公共建筑", 24),
            CampusBuildingSpec("north_l1", "B1_north_teaching", "teaching", 1190, 1430, 96, 62, -5, "L", "公共建筑", 22),
            CampusBuildingSpec("north_u2", "B1_north_teaching", "teaching", 1350, 1420, 118, 70, -5, "U", "公共建筑", 24),
            CampusBuildingSpec("north_bar1", "B1_north_teaching", "teaching", 1090, 1310, 116, 28, -5, "bar", "公共建筑", 20),
            CampusBuildingSpec("north_bar2", "B1_north_teaching", "teaching", 1300, 1300, 108, 28, -5, "bar", "公共建筑", 20),

            # B2 图书馆与湖岸公共建筑。图书馆公开资料为地上 8 层。
            CampusBuildingSpec("library", "B2_lakeside_library", "library", 1090, 1010, 132, 88, -5, "U", "公共建筑", 30),
            CampusBuildingSpec("library_lab", "B2_lakeside_library", "laboratory", 1290, 960, 94, 62, -5, "L", "公共建筑", 22),
            CampusBuildingSpec("library_wing", "B2_lakeside_library", "public_service", 1240, 1080, 86, 26, -5, "bar", "公共建筑", 16),
            CampusBuildingSpec("lakeside_maintenance", "B2_lakeside_library", "lakeside_maintenance", 1160, 990, 24, 12, -5, "bar", "公共建筑", 6),

            # B3 中部教学组团。
            CampusBuildingSpec("central_u1", "B3_central_teaching", "teaching", 1330, 850, 116, 68, -4, "U", "公共建筑", 22),
            CampusBuildingSpec("central_u2", "B3_central_teaching", "teaching", 1500, 850, 116, 68, -4, "U", "公共建筑", 22),
            CampusBuildingSpec("central_l1", "B3_central_teaching", "teaching", 1250, 720, 92, 58, -4, "L", "公共建筑", 20),
            CampusBuildingSpec("central_l2", "B3_central_teaching", "teaching", 1400, 700, 92, 58, -4, "L", "公共建筑", 20),
            CampusBuildingSpec("central_bar1", "B3_central_teaching", "teaching", 1550, 710, 102, 28, -4, "bar", "公共建筑", 20),
            CampusBuildingSpec("central_bar2", "B3_central_teaching", "teaching", 1590, 890, 94, 28, -4, "bar", "公共建筑", 20),

            # B4 东部学生宿舍：三组半围合院落。
            CampusBuildingSpec("dorm_e01", "B4_east_dorms", "student_dormitory", 1970, 900, 72, 24, -4, "bar", "多层住宅", 24),
            CampusBuildingSpec("dorm_e02", "B4_east_dorms", "student_dormitory", 2070, 900, 72, 24, -4, "bar", "多层住宅", 24),
            CampusBuildingSpec("dorm_e03", "B4_east_dorms", "student_dormitory", 2020, 970, 24, 78, -4, "bar", "多层住宅", 24),
            CampusBuildingSpec("dorm_e04", "B4_east_dorms", "student_dormitory", 2140, 1020, 72, 24, -4, "bar", "多层住宅", 24),
            CampusBuildingSpec("dorm_e05", "B4_east_dorms", "student_dormitory", 2240, 1020, 72, 24, -4, "bar", "多层住宅", 24),
            CampusBuildingSpec("dorm_e06", "B4_east_dorms", "student_dormitory", 2190, 1090, 24, 78, -4, "bar", "多层住宅", 24),
            CampusBuildingSpec("dorm_e07", "B4_east_dorms", "student_dormitory", 1960, 1150, 72, 24, -4, "bar", "多层住宅", 24),
            CampusBuildingSpec("dorm_e08", "B4_east_dorms", "student_dormitory", 2060, 1150, 72, 24, -4, "bar", "多层住宅", 24),
            CampusBuildingSpec("dorm_e09", "B4_east_dorms", "student_dormitory", 2010, 1220, 24, 78, -4, "bar", "多层住宅", 24),
            CampusBuildingSpec("dorm_e10", "B4_east_dorms", "student_dormitory", 2140, 1260, 72, 24, -4, "bar", "多层住宅", 24),
            CampusBuildingSpec("dorm_e11", "B4_east_dorms", "student_dormitory", 2240, 1260, 72, 24, -4, "bar", "多层住宅", 24),
            CampusBuildingSpec("dorm_e12", "B4_east_dorms", "student_dormitory", 2190, 1330, 24, 78, -4, "bar", "多层住宅", 24),

            # B5 南部教学/行政。
            CampusBuildingSpec("south_admin", "B5_south_teaching", "administration", 1060, 590, 126, 72, -4, "H", "公共建筑", 22),
            CampusBuildingSpec("south_l1", "B5_south_teaching", "teaching", 1210, 600, 96, 58, -4, "L", "公共建筑", 20),
            CampusBuildingSpec("south_l2", "B5_south_teaching", "teaching", 1350, 620, 96, 58, -4, "L", "公共建筑", 20),
            CampusBuildingSpec("south_bar1", "B5_south_teaching", "teaching", 1500, 620, 96, 28, -4, "bar", "公共建筑", 18),
            CampusBuildingSpec("south_bar2", "B5_south_teaching", "teaching", 1600, 650, 88, 28, -4, "bar", "公共建筑", 18),

            # B6 后勤与餐饮。
            CampusBuildingSpec("service_dining", "B6_service", "service", 1840, 580, 112, 42, -4, "L", "公共建筑", 14),
            CampusBuildingSpec("service_logistics", "B6_service", "service", 1990, 590, 88, 32, -4, "bar", "公共建筑", 12),
            CampusBuildingSpec("service_store", "B6_service", "service", 2100, 620, 70, 28, -4, "bar", "公共建筑", 10),

            # B7 体育馆与场地附属建筑。2026 年公开信息显示体育馆已建成。
            CampusBuildingSpec("gymnasium", "B7_sports", "gymnasium", 1640, 980, 104, 72, -4, "H", "公共建筑", 18),
            CampusBuildingSpec("sports_service", "B7_sports", "sports_service", 1640, 1160, 64, 24, -4, "bar", "公共建筑", 10),

            # B8 东北教师公寓：公开资料为 4 栋、18 层。
            CampusBuildingSpec("faculty_tower_1", "B8_faculty_apartments", "faculty_apartment", 2050, 1440, 46, 30, -4, "bar", "高层住宅", 56),
            CampusBuildingSpec("faculty_tower_2", "B8_faculty_apartments", "faculty_apartment", 2140, 1440, 46, 30, -4, "bar", "高层住宅", 56),
            CampusBuildingSpec("faculty_tower_3", "B8_faculty_apartments", "faculty_apartment", 2230, 1440, 46, 30, -4, "bar", "高层住宅", 56),
            CampusBuildingSpec("faculty_tower_4", "B8_faculty_apartments", "faculty_apartment", 2320, 1440, 46, 30, -4, "bar", "高层住宅", 56),
        ]

        water_geometries = [
            feature.geometry for feature in features
            if feature.layer == "water" and feature.geometry is not None
        ]
        sport_geometries = [
            feature.geometry for feature in features
            if feature.layer == "sport" and feature.geometry is not None
        ]
        existing_buildings = [
            feature.geometry for feature in features
            if feature.layer == "building" and feature.geometry is not None
        ]
        line_obstacles = [
            feature.geometry.buffer(3.0)
            for feature in features
            if feature.layer in {"road", "power"} and feature.geometry is not None
        ]
        forbidden_parts = []
        if water_geometries:
            forbidden_parts.append(unary_union(water_geometries).buffer(12.0))
        if sport_geometries:
            forbidden_parts.append(unary_union(sport_geometries).buffer(5.0))
        if existing_buildings:
            forbidden_parts.append(unary_union(existing_buildings).buffer(3.0))
        forbidden_parts.extend(line_obstacles)
        forbidden = unary_union(forbidden_parts) if forbidden_parts else None
        scene_boundary = box(
            12.0, 12.0, zone.width_m - 12.0, zone.height_m - 12.0
        )

        accepted_campus_buildings = []
        for spec in campus_buildings:
            geometry, shift_m = fit_campus_building(
                spec,
                forbidden,
                scene_boundary,
            )
            feature = campus_building_feature(spec, geometry)
            feature.tags["placement_shift_m"] = "%.1f" % shift_m
            augmented.append(feature)
            accepted_campus_buildings.append(geometry)
            new_obstacle = geometry.buffer(4.0)
            forbidden = (
                new_obstacle
                if forbidden is None
                else unary_union([forbidden, new_obstacle])
            )

        # 校园道路只表达连通骨架，并从建筑、水体和运动场缓冲区中裁开。
        road_cutouts = []
        if water_geometries:
            road_cutouts.append(unary_union(water_geometries).buffer(8.0))
        if sport_geometries:
            road_cutouts.append(unary_union(sport_geometries).buffer(3.0))
        if accepted_campus_buildings:
            road_cutouts.append(
                unary_union(accepted_campus_buildings).buffer(6.0)
            )
        road_cutout = unary_union(road_cutouts) if road_cutouts else None
        campus_road_lines = [
            ("south_axis", [(840, 520), (2150, 520)]),
            ("central_axis", [(900, 760), (1650, 760), (1840, 720), (2320, 720)]),
            ("north_axis", [(940, 1270), (1600, 1270), (1940, 1290), (2320, 1290)]),
            ("west_spine", [(980, 520), (1000, 970), (980, 1470)]),
            ("central_spine", [(1370, 520), (1390, 930), (1420, 1470)]),
            ("east_spine", [(2050, 500), (2050, 900), (2080, 1500)]),
            ("sports_access", [(1500, 760), (1640, 900), (1650, 1210)]),
            ("dorm_court_1", [(1910, 850), (2110, 850), (2110, 1030), (1910, 1030)]),
            ("dorm_court_2", [(2080, 980), (2300, 980), (2300, 1140), (2080, 1140)]),
            ("dorm_court_3", [(1900, 1110), (2110, 1110), (2110, 1280), (1900, 1280)]),
        ]
        for name, coordinates in campus_road_lines:
            geometry = LineString(coordinates)
            if road_cutout is not None:
                geometry = geometry.difference(road_cutout)
            if geometry.is_empty:
                continue
            augmented.append(morphology_feature(
                "campus/B/road_%s" % name,
                "road",
                geometry,
                tags={
                    "highway": "service",
                    "source": "official_reference_generalization",
                    "geometry_method": "schematic_connectivity",
                    "accuracy_class": "schematic_not_survey_grade",
                    "is_synthetic": "true",
                },
            ))

        # 三个校门只保留脱敏编号，用于检查道路图连通性。
        for gate_id, point in (
            ("G1", Point(1370, 505)),
            ("G2", Point(2320, 720)),
            ("G3", Point(2080, 1500)),
        ):
            augmented.append(morphology_feature(
                "campus/B/gate_%s" % gate_id,
                "gate",
                point,
                tags={
                    "ref": gate_id,
                    "source": "official_reference_generalization",
                    "accuracy_class": "schematic_not_survey_grade",
                    "is_synthetic": "true",
                },
            ))

    elif zone.zone_id == "C":
        water_area = sum(
            feature.geometry.area
            for feature in features
            if feature.layer == "water" and feature.geometry is not None
        )
        minimum_water_area = zone.width_m * zone.height_m * 0.15
        if water_area < minimum_water_area:
            raise ValueError(
                "C 区长江水面解析不足（%.0f m² < %.0f m²）。"
                "请刷新含 relation members 的 OSM 缓存；禁止用粗略梯形河面兜底。"
                % (water_area, minimum_water_area)
            )

    elif zone.zone_id == "D":
        river_surfaces = [
            feature for feature in features
            if (
                feature.layer == "water"
                and feature.geometry is not None
                and feature.tags.get("water") in {"river", "canal"}
            )
        ]
        if not river_surfaces:
            raise ValueError(
                "D 区楚河面状水体缺失。请刷新 OSM 数据；"
                "禁止用穿越既有建筑的宽缓冲河面兜底。"
            )

    return augmented


TASK_BUILDING_TYPES = {
    2: {"商业综合体"},
    3: {"商业综合体"},
    4: {"公共建筑"},
    5: {"公共建筑"},
    6: {"公共建筑"},
    7: {"公共建筑", "商业综合体"},
    8: {"低层住宅", "多层住宅", "商业综合体", "公共建筑"},
    9: {"公共建筑", "景区建筑"},
    10: {"商业综合体"},
    11: {"办公塔楼"},
    12: {"公共建筑"},
}

TASK_CAMPUS_SUBTYPES = {
    4: {"service"},
    5: {"library"},
    6: {"lakeside_maintenance"},
}


def feature_geometry_source(feature: Feature) -> str:
    source = feature.tags.get("source")
    if source == "official_reference_generalization":
        return "official_public_reference_generalization"
    if source == "reference_morphology":
        return "reference_morphology"
    return "OpenStreetMap"


def is_authorized_water_structure(feature: Feature) -> bool:
    name = feature.tags.get("name", "")
    return (
        feature.tags.get("amenity", "").lower() == "ferry_terminal"
        or feature.tags.get("amenity", "").lower() == "shelter"
        or feature.tags.get("man_made", "").lower() in {"pier", "quay"}
        or any(token in name for token in ("码头", "渡口", "船坞"))
    )


def water_structure_label(feature: Feature) -> str:
    if feature.tags.get("amenity", "").lower() == "shelter":
        return "水上亭廊"
    return "码头设施"


def risk_geometry(zone: ZoneSpec, risk: RiskSpec) -> Polygon:
    return ellipse_polygon(
        risk.x_ratio * zone.width_m,
        risk.y_ratio * zone.height_m,
        risk.width_ratio * zone.width_m,
        risk.height_ratio * zone.height_m,
        risk.angle_deg,
    )


def resolve_task_placements(
    zone: ZoneSpec,
    features: Sequence[Feature],
) -> Dict[int, TaskPlacement]:
    """
    将设施类任务点吸附到建筑代表点。

    输入坐标仍是脱敏的设计锚点；吸附只发生在局部 ENU 场景内，
    不导出经纬度，也不构成现实配送点或航线。
    """
    buildings = [
        feature for feature in features
        if feature.layer == "building" and feature.geometry is not None
    ]
    water_geometries = [
        feature.geometry for feature in features
        if feature.layer in {"water", "waterway"} and feature.geometry is not None
    ]
    water_union = unary_union(water_geometries) if water_geometries else None
    hard_geometries = [
        risk_geometry(zone, risk) for risk in zone.risks if risk.hard
    ]
    hard_union = unary_union(hard_geometries) if hard_geometries else None

    candidates = []
    for feature in buildings:
        point = feature.geometry.representative_point()
        if water_union is not None and water_union.contains(point):
            continue
        if hard_union is not None and hard_union.contains(point):
            continue
        candidates.append((feature, point))

    placements: Dict[int, TaskPlacement] = {}
    for task in zone.tasks:
        seed_point = Point(
            task.x_ratio * zone.width_m,
            task.y_ratio * zone.height_m,
        )
        if task.point_id == 1:
            placements[task.point_id] = TaskPlacement(
                point_id=task.point_id,
                x_m=seed_point.x,
                y_m=seed_point.y,
                placement_source="脱敏地面锚点",
            )
            continue

        preferred_types = TASK_BUILDING_TYPES.get(task.point_id, set())
        preferred = [
            (feature, point) for feature, point in candidates
            if not preferred_types or feature.building_type in preferred_types
        ]
        preferred_subtypes = TASK_CAMPUS_SUBTYPES.get(task.point_id, set())
        if preferred_subtypes:
            subtype_matches = [
                (feature, point) for feature, point in preferred
                if feature.tags.get("campus_subtype") in preferred_subtypes
            ]
            if subtype_matches:
                preferred = subtype_matches
        ranked_preferred = sorted(
            preferred,
            key=lambda item: seed_point.distance(item[1]),
        )
        ranked_all = sorted(
            candidates,
            key=lambda item: seed_point.distance(item[1]),
        )
        chosen = ranked_preferred[0] if ranked_preferred else None
        if chosen is None or seed_point.distance(chosen[1]) > 350.0:
            chosen = ranked_all[0] if ranked_all else None
        if chosen is None or seed_point.distance(chosen[1]) > 350.0:
            raise ValueError(
                "分区 %s 的任务点 P%d 在 350 m 内找不到可用建筑"
                % (zone.zone_id, task.point_id)
            )

        feature, point = chosen
        placements[task.point_id] = TaskPlacement(
            point_id=task.point_id,
            x_m=float(point.x),
            y_m=float(point.y),
            placement_source="脱敏锚点→建筑代表点",
            target_feature_id=feature.feature_id,
            target_geometry_source=feature_geometry_source(feature),
            target_building_type=feature.building_type,
            snap_distance_m=float(seed_point.distance(point)),
        )
    return placements


def road_style(tags: Dict[str, str]) -> Tuple[str, float]:
    highway = tags.get("highway", "")
    if highway in {"motorway", "trunk", "primary"}:
        return ROAD_MAIN, 4.8
    if highway in {"secondary", "tertiary"}:
        return ROAD_SECONDARY, 3.3
    if highway in {"residential", "unclassified", "service"}:
        return ROAD_LOCAL, 1.8
    if highway in {"footway", "path", "pedestrian", "cycleway", "steps"}:
        return "#D7DCCF", 0.85
    return ROAD_LOCAL, 1.2


def draw_polygon_feature(
    ax,
    geometry,
    facecolor: str,
    edgecolor: str,
    linewidth: float,
    alpha: float,
    zorder: float,
    hatch: Optional[str] = None,
):
    for polygon in iter_polygons(geometry):
        coords = np.asarray(polygon.exterior.coords)
        ax.add_patch(MplPolygon(
            coords,
            closed=True,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
            alpha=alpha,
            hatch=hatch,
            zorder=zorder,
        ))


def draw_base_layers(ax, features: Sequence[Feature]):
    for feature in features:
        if feature.layer == "green":
            draw_polygon_feature(
                ax, feature.geometry, GREEN_COLOR, "#6AA86A", 0.55, 0.72, 1.0
            )
        elif feature.layer == "sport":
            draw_polygon_feature(
                ax, feature.geometry, SPORT_COLOR, "#48A989", 0.75, 0.82, 1.4
            )

    for feature in features:
        if feature.layer == "water":
            draw_polygon_feature(
                ax, feature.geometry, WATER_COLOR, WATER_EDGE, 1.2, 0.92, 2.0
            )
        elif feature.layer == "waterway":
            width = 5.0 if feature.tags.get("waterway") in {"river", "canal"} else 2.2
            for line in iter_lines(feature.geometry):
                x, y = line.xy
                ax.plot(x, y, color=WATER_EDGE, linewidth=width + 1.0, alpha=0.85, zorder=2.0)
                ax.plot(x, y, color=WATER_COLOR, linewidth=width, alpha=0.95, zorder=2.1)

    roads = [f for f in features if f.layer == "road"]
    roads.sort(key=lambda f: road_style(f.tags)[1])
    for feature in roads:
        color, width = road_style(feature.tags)
        is_bridge = feature.tags.get("bridge") not in {None, "no"}
        for line in iter_lines(feature.geometry):
            x, y = line.xy
            casing = BRIDGE_COLOR if is_bridge else ROAD_CASING
            ax.plot(
                x, y, color=casing, linewidth=width + (2.4 if is_bridge else 1.4),
                solid_capstyle="round", alpha=0.92, zorder=4.0 if is_bridge else 3.0,
            )
            ax.plot(
                x, y, color=color, linewidth=width,
                solid_capstyle="round", alpha=1.0, zorder=4.1 if is_bridge else 3.1,
            )

    for feature in features:
        if feature.layer == "railway":
            for line in iter_lines(feature.geometry):
                x, y = line.xy
                ax.plot(
                    x, y, color=RAIL_COLOR, linewidth=1.6,
                    linestyle=(0, (5, 2)), alpha=0.90, zorder=4.5,
                )
        elif feature.layer == "power":
            for line in iter_lines(feature.geometry):
                x, y = line.xy
                ax.plot(
                    x, y, color=POWER_COLOR, linewidth=1.3,
                    linestyle=(0, (6, 3)), alpha=0.90, zorder=4.4,
                )
        elif feature.layer == "gate" and feature.geometry is not None:
            point = feature.geometry
            ax.scatter(
                [point.x], [point.y],
                marker="s", s=34,
                facecolor="#263238", edgecolor="white",
                linewidth=0.8, zorder=7.8,
            )
            ax.annotate(
                feature.tags.get("ref", "G"),
                (point.x, point.y),
                xytext=(5, 4),
                textcoords="offset points",
                fontsize=6.6, fontweight="bold", color="#263238",
                path_effects=[pe.withStroke(linewidth=2.0, foreground="white")],
                zorder=8.0,
            )


def draw_buildings(ax, features: Sequence[Feature]):
    buildings = [f for f in features if f.layer == "building"]
    water_geometries = [
        f.geometry for f in features
        if f.layer == "water" and f.geometry is not None
    ]
    water_union = unary_union(water_geometries) if water_geometries else None
    buildings.sort(key=lambda f: f.height_m or 0)
    for feature in buildings:
        height = float(feature.height_m or 10.0)
        shadow_dx = min(13.0, 2.0 + height * 0.055)
        shadow_dy = -min(11.0, 1.5 + height * 0.045)
        for polygon in iter_polygons(feature.geometry):
            coords = np.asarray(polygon.exterior.coords)
            shadow = coords + np.array([shadow_dx, shadow_dy])
            ax.add_patch(MplPolygon(
                shadow, closed=True,
                facecolor="#455A64", edgecolor="none",
                alpha=min(0.28, 0.10 + height / 900.0),
                zorder=5.0,
            ))
            edge_width = 0.35 if height < 30 else (0.65 if height < 80 else 1.05)
            hatch = "///" if feature.height_source == "programmatic" else None
            is_reference_morphology = feature.height_source in {
                "reference morphology",
                "official attribute/generalized prior",
            }
            ax.add_patch(MplPolygon(
                coords, closed=True,
                facecolor=BUILDING_COLORS[feature.building_type],
                edgecolor="#00695C" if is_reference_morphology else "#37474F",
                linewidth=max(0.9, edge_width) if is_reference_morphology else edge_width,
                linestyle=(0, (2, 1)) if is_reference_morphology else "solid",
                alpha=0.92,
                hatch=hatch,
                zorder=5.2,
            ))
            if (
                is_authorized_water_structure(feature)
                and water_union is not None
                and polygon.intersection(water_union).area > 1.0
            ):
                point = polygon.representative_point()
                ax.text(
                    point.x, point.y,
                    water_structure_label(feature),
                    ha="center", va="center",
                    fontsize=6.2, fontweight="bold", color="#006064",
                    path_effects=[pe.withStroke(linewidth=2.2, foreground="white")],
                    zorder=8.8,
                )


def risk_patch(zone: ZoneSpec, risk: RiskSpec) -> Ellipse:
    color, hatch = RISK_STYLES[risk.kind]
    return Ellipse(
        (
            risk.x_ratio * zone.width_m,
            risk.y_ratio * zone.height_m,
        ),
        risk.width_ratio * zone.width_m,
        risk.height_ratio * zone.height_m,
        angle=risk.angle_deg,
        facecolor=mcolors.to_rgba(color, 0.065 if not risk.hard else 0.13),
        edgecolor=color,
        linewidth=1.3 if not risk.hard else 2.0,
        linestyle=(0, (4, 2)),
        hatch=hatch if risk.hard else None,
        zorder=6.2 if risk.hard else 4.7,
    )


def draw_risks(ax, zone: ZoneSpec):
    for risk in zone.risks:
        patch = risk_patch(zone, risk)
        ax.add_patch(patch)
        color, _ = RISK_STYLES[risk.kind]
        ax.text(
            risk.x_ratio * zone.width_m,
            (risk.y_ratio + risk.height_ratio * 0.36) * zone.height_m,
            risk.kind,
            ha="center", va="center",
            fontsize=7.2, fontweight="bold", color=color,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=1.3),
            zorder=9,
        )


def draw_tasks(
    ax,
    zone: ZoneSpec,
    placements: Dict[int, TaskPlacement],
):
    for task in zone.tasks:
        placement = placements[task.point_id]
        x = placement.x_m
        y = placement.y_m
        is_depot = task.point_id == 1
        ax.scatter(
            [x], [y],
            marker="*" if is_depot else "o",
            s=170 if is_depot else 70,
            facecolor=DEPOT_COLOR if is_depot else "white",
            edgecolor="#1B5E20" if is_depot else TASK_COLOR,
            linewidth=1.7,
            zorder=10,
        )
        ax.annotate(
            "P%d" % task.point_id,
            (x, y),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8.2,
            fontweight="bold",
            color="#1B5E20" if is_depot else "#AD1457",
            path_effects=[pe.withStroke(linewidth=2.4, foreground="white")],
            zorder=11,
        )


def draw_dynamic_objects(ax, zone: ZoneSpec):
    for item in zone.dynamics:
        x = item.x_ratio * zone.width_m
        y = item.y_ratio * zone.height_m
        ax.scatter(
            [x], [y],
            marker=item.marker,
            s=56,
            facecolor=item.color,
            edgecolor="white",
            linewidth=0.8,
            zorder=10,
        )
        length = min(zone.width_m, zone.height_m) * 0.055
        dx = length * math.cos(math.radians(item.direction_deg))
        dy = length * math.sin(math.radians(item.direction_deg))
        ax.annotate(
            "",
            xy=(x + dx, y + dy),
            xytext=(x, y),
            arrowprops=dict(arrowstyle="-|>", color=item.color, lw=1.2),
            zorder=9.5,
        )


def add_cartography(ax, zone: ZoneSpec):
    ax.set_xlim(0, zone.width_m)
    ax.set_ylim(0, zone.height_m)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("局部东向 E (km)", fontweight="bold")
    ax.set_ylabel("局部北向 N (km)", fontweight="bold")
    xticks = np.arange(0, zone.width_m + 1, 500)
    yticks = np.arange(0, zone.height_m + 1, 500)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels(["%.1f" % (v / 1000.0) for v in xticks])
    ax.set_yticklabels(["%.1f" % (v / 1000.0) for v in yticks])
    ax.grid(color="#CFD8DC", linewidth=0.45, alpha=0.45, zorder=-5)

    scale = 500.0
    sx, sy = zone.width_m * 0.07, zone.height_m * 0.065
    ax.plot([sx, sx + scale], [sy, sy], color=TEXT_DARK, linewidth=4.0, zorder=20)
    ax.plot([sx, sx], [sy - 16, sy + 16], color=TEXT_DARK, linewidth=1.2, zorder=20)
    ax.plot([sx + scale, sx + scale], [sy - 16, sy + 16], color=TEXT_DARK, linewidth=1.2, zorder=20)
    ax.text(sx, sy + 35, "0", ha="center", fontsize=7.5, zorder=20)
    ax.text(sx + scale, sy + 35, "500 m", ha="center", fontsize=7.5, zorder=20)

    nx = zone.width_m * 0.93
    ax.annotate(
        "N",
        xy=(nx, zone.height_m * 0.91),
        xytext=(nx, zone.height_m * 0.79),
        ha="center", va="center",
        fontsize=10, fontweight="bold",
        arrowprops=dict(arrowstyle="-|>", color=TEXT_DARK, lw=1.7),
        zorder=20,
    )


def add_legend_panel(ax, zone: ZoneSpec, features: Sequence[Feature]):
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(
        0.03, 0.97, "图层与审核要点",
        fontsize=12.5, fontweight="bold", color=TEXT_DARK, va="top",
    )
    ax.text(
        0.03, 0.915, zone.scene_hint,
        fontsize=8.1, color="#546E7A", va="top", wrap=True,
    )

    present_types = sorted({
        f.building_type for f in features
        if f.layer == "building" and f.building_type
    })
    y = 0.84
    ax.text(0.03, y, "建筑类型", fontsize=9.5, fontweight="bold", color=TEXT_DARK)
    y -= 0.038
    for idx, btype in enumerate(present_types[:8]):
        col = idx % 2
        row = idx // 2
        x = 0.03 + col * 0.48
        yy = y - row * 0.047
        ax.add_patch(Rectangle(
            (x, yy - 0.017), 0.055, 0.032,
            facecolor=BUILDING_COLORS[btype],
            edgecolor="#37474F",
            transform=ax.transAxes,
        ))
        ax.text(x + 0.072, yy, btype, fontsize=7.4, va="center", transform=ax.transAxes)
    y -= max(1, math.ceil(len(present_types[:8]) / 2)) * 0.047 + 0.018

    ax.text(0.03, y, "环境层", fontsize=9.5, fontweight="bold", color=TEXT_DARK)
    y -= 0.040
    environment_handles = []
    if any(f.layer in {"water", "waterway"} for f in features):
        environment_handles.append(
            ("水域/河道", Patch(facecolor=WATER_COLOR, edgecolor=WATER_EDGE))
        )
    if any(f.layer == "green" for f in features):
        environment_handles.append(
            ("绿地/林地", Patch(facecolor=GREEN_COLOR, edgecolor="#6AA86A"))
        )
    if any(f.layer == "sport" for f in features):
        environment_handles.append(
            ("运动场", Patch(facecolor=SPORT_COLOR, edgecolor="#48A989"))
        )
    if any(f.layer == "road" for f in features):
        environment_handles.append(
            ("道路", Line2D([0], [0], color=ROAD_MAIN, lw=5, markeredgecolor=ROAD_CASING))
        )
    if any(
        f.layer == "road" and f.tags.get("bridge") not in {None, "no"}
        for f in features
    ):
        environment_handles.append(
            ("桥梁", Line2D([0], [0], color=BRIDGE_COLOR, lw=5))
        )
    if any(f.layer == "railway" for f in features):
        environment_handles.append(
            ("轨道", Line2D([0], [0], color=RAIL_COLOR, lw=1.8, linestyle="--"))
        )
    if any(f.layer == "gate" for f in features):
        environment_handles.append(
            ("脱敏校门", Line2D(
                [0], [0], marker="s", color="none",
                markerfacecolor="#263238", markeredgecolor="white",
            ))
        )
    for idx, (label, handle) in enumerate(environment_handles):
        col = idx % 2
        row = idx // 2
        x = 0.03 + col * 0.48
        yy = y - row * 0.044
        if isinstance(handle, Patch):
            ax.add_patch(Rectangle(
                (x, yy - 0.014), 0.052, 0.028,
                facecolor=handle.get_facecolor(),
                edgecolor=handle.get_edgecolor(),
                transform=ax.transAxes,
            ))
        else:
            marker = handle.get_marker()
            if marker not in {None, "None", "none", ""}:
                ax.plot(
                    [x + 0.026], [yy],
                    marker=marker, markersize=6.5,
                    markerfacecolor=handle.get_markerfacecolor(),
                    markeredgecolor=handle.get_markeredgecolor(),
                    linestyle="none", transform=ax.transAxes,
                )
            else:
                ax.plot(
                    [x, x + 0.052], [yy, yy],
                    color=handle.get_color(), lw=handle.get_linewidth(),
                    linestyle=handle.get_linestyle(), transform=ax.transAxes,
                )
        ax.text(x + 0.065, yy, label, fontsize=7.2, va="center", transform=ax.transAxes)
    y -= math.ceil(len(environment_handles) / 2) * 0.044 + 0.018

    ax.text(0.03, y, "训练风险层", fontsize=9.5, fontweight="bold", color=TEXT_DARK)
    y -= 0.040
    for idx, risk in enumerate(zone.risks):
        color, hatch = RISK_STYLES[risk.kind]
        col = idx % 2
        row = idx // 2
        x = 0.03 + col * 0.48
        yy = y - row * 0.044
        ax.add_patch(Rectangle(
            (x, yy - 0.014), 0.052, 0.028,
            facecolor=mcolors.to_rgba(color, 0.10),
            edgecolor=color, hatch=hatch if risk.hard else None,
            transform=ax.transAxes,
        ))
        ax.text(x + 0.065, yy, risk.kind, fontsize=7.0, va="center", transform=ax.transAxes)
    y -= math.ceil(len(zone.risks) / 2) * 0.044 + 0.020

    ax.text(0.03, y, "任务点", fontsize=9.5, fontweight="bold", color=TEXT_DARK)
    y -= 0.038
    for idx, task in enumerate(zone.tasks):
        yy = y - idx * 0.041
        marker = "*" if task.point_id == 1 else "o"
        face = DEPOT_COLOR if task.point_id == 1 else "white"
        edge = "#1B5E20" if task.point_id == 1 else TASK_COLOR
        ax.plot(
            [0.06], [yy], marker=marker, markersize=8.5,
            markerfacecolor=face, markeredgecolor=edge,
            linestyle="none", transform=ax.transAxes,
        )
        ax.text(
            0.12, yy,
            "P%d  %s" % (task.point_id, task.site_type),
            fontsize=7.2, va="center", transform=ax.transAxes,
        )

    if zone.zone_id == "B":
        note = (
            "斜线建筑：OSM 轮廓，高度为类型先验\n"
            "青色虚边建筑：官方公开拓扑校核后的自绘概化轮廓\n"
            "非测绘级、未描摹参考图；任务点与校门均脱敏"
        )
    else:
        note = (
            "斜线建筑：OSM 轮廓，高度为类型先验\n"
            "青色虚边建筑：参考形态补绘；类型含推断\n"
            "任务点/风险均脱敏；无规划或现实配送航线"
        )
    ax.text(
        0.03, 0.006, note,
        fontsize=6.5, color="#607D8B", va="bottom",
        transform=ax.transAxes,
    )


def render_zone(
    zone: ZoneSpec,
    features: Sequence[Feature],
    placements: Dict[int, TaskPlacement],
    output_path: Path,
    dpi: int,
):
    fig = plt.figure(figsize=(16, 9.4))
    grid = fig.add_gridspec(1, 2, width_ratios=[4.8, 1.35], wspace=0.035)
    ax = fig.add_subplot(grid[0, 0])
    legend_ax = fig.add_subplot(grid[0, 1])

    ax.set_facecolor("#F7F8F7")
    draw_base_layers(ax, features)
    draw_buildings(ax, features)
    draw_risks(ax, zone)
    draw_dynamic_objects(ax, zone)
    draw_tasks(ax, zone, placements)
    add_cartography(ax, zone)
    add_legend_panel(legend_ax, zone, features)

    fig.suptitle(zone.title, fontsize=19, fontweight="bold", y=0.982)
    subtitle_source = (
        "OSM 水体/道路/运动场 + 官方公开资料校核的概化建筑"
        if zone.zone_id == "B"
        else "OSM 几何 + 参考形态补绘"
    )
    fig.text(
        0.405, 0.948,
        "%s｜纯俯视 2D｜%s + 脱敏任务与风险"
        % (zone.archetype, subtitle_source),
        ha="center", va="center", fontsize=9.3, color="#607D8B",
    )
    fig.text(
        0.5, 0.016,
        "© OpenStreetMap contributors (ODbL)｜仅作科研环境审核，非导航地图；不含高德底图、POI 标注或真实配送路线",
        ha="center", va="bottom", fontsize=7.8, color="#607D8B",
    )
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.075, top=0.915)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def feature_counts(features: Sequence[Feature]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for feature in features:
        counts[feature.layer] = counts.get(feature.layer, 0) + 1
    return counts


def geometry_source_counts(features: Sequence[Feature]) -> Dict[str, object]:
    totals: Dict[str, int] = {}
    by_layer: Dict[str, Dict[str, int]] = {}
    for feature in features:
        source = feature_geometry_source(feature)
        totals[source] = totals.get(source, 0) + 1
        layer_counts = by_layer.setdefault(feature.layer, {})
        layer_counts[source] = layer_counts.get(source, 0) + 1
    return {"total": totals, "by_layer": by_layer}


def building_inference_summary(features: Sequence[Feature]) -> Dict[str, object]:
    buildings = [feature for feature in features if feature.layer == "building"]
    height_counts: Dict[str, int] = {}
    type_counts: Dict[str, int] = {}
    for feature in buildings:
        height_source = feature.height_source or "unknown"
        type_source = feature.building_type_source or "unknown"
        height_counts[height_source] = height_counts.get(height_source, 0) + 1
        type_counts[type_source] = type_counts.get(type_source, 0) + 1
    estimated_height_count = sum(
        count for source, count in height_counts.items()
        if source not in {"OSM height", "OSM levels"}
    )
    inferred_type_count = sum(
        count for source, count in type_counts.items()
        if source != "OSM semantic tags"
    )
    total = len(buildings)
    return {
        "building_count": total,
        "height_source_counts": height_counts,
        "estimated_height_count": estimated_height_count,
        "estimated_height_ratio": round(
            estimated_height_count / total, 6
        ) if total else 0.0,
        "type_source_counts": type_counts,
        "inferred_type_count": inferred_type_count,
        "inferred_type_ratio": round(
            inferred_type_count / total, 6
        ) if total else 0.0,
    }


def campus_detail_audit(
    features: Sequence[Feature],
    placements: Optional[Dict[int, TaskPlacement]] = None,
) -> Dict[str, object]:
    buildings = [
        feature for feature in features
        if (
            feature.layer == "building"
            and feature.geometry is not None
            and feature.tags.get("source") == "official_reference_generalization"
        )
    ]
    cluster_counts: Dict[str, int] = {}
    shape_counts: Dict[str, int] = {}
    subtype_counts: Dict[str, int] = {}
    shifts = []
    for feature in buildings:
        cluster = feature.tags.get("cluster_id", "unknown")
        shape = feature.tags.get("shape_family", "unknown")
        subtype = feature.tags.get("campus_subtype", "unknown")
        cluster_counts[cluster] = cluster_counts.get(cluster, 0) + 1
        shape_counts[shape] = shape_counts.get(shape, 0) + 1
        subtype_counts[subtype] = subtype_counts.get(subtype, 0) + 1
        shifts.append(float(feature.tags.get("placement_shift_m", "0")))

    building_pair_overlaps = []
    for index, feature in enumerate(buildings):
        for other in buildings[index + 1:]:
            overlap_area = feature.geometry.intersection(other.geometry).area
            if overlap_area > 1.0:
                building_pair_overlaps.append({
                    "a": feature.feature_id,
                    "b": other.feature_id,
                    "overlap_area_m2": round(overlap_area, 2),
                })

    water_geometries = [
        feature.geometry for feature in features
        if feature.layer == "water" and feature.geometry is not None
    ]
    sport_geometries = [
        feature.geometry for feature in features
        if feature.layer == "sport" and feature.geometry is not None
    ]
    water_buffer = (
        unary_union(water_geometries).buffer(8.0)
        if water_geometries else None
    )
    sport_union = unary_union(sport_geometries) if sport_geometries else None
    water_clearance_violations = [
        feature.feature_id for feature in buildings
        if (
            water_buffer is not None
            and feature.geometry.intersects(water_buffer)
        )
    ]
    sport_overlap_violations = [
        feature.feature_id for feature in buildings
        if (
            sport_union is not None
            and feature.geometry.intersection(sport_union).area > 1.0
        )
    ]

    generalized_roads = [
        feature for feature in features
        if (
            feature.layer == "road"
            and feature.geometry is not None
            and feature.tags.get("source") == "official_reference_generalization"
        )
    ]
    road_intrusions = []
    for road in generalized_roads:
        for building in buildings:
            intrusion = road.geometry.intersection(
                building.geometry.buffer(5.0)
            ).length
            if intrusion > 0.1:
                road_intrusions.append({
                    "road": road.feature_id,
                    "building": building.feature_id,
                    "length_m": round(intrusion, 2),
                })

    task_subtype_matches = {}
    if placements is not None:
        buildings_by_id = {feature.feature_id: feature for feature in buildings}
        for task_id, expected in TASK_CAMPUS_SUBTYPES.items():
            placement = placements.get(task_id)
            target = (
                buildings_by_id.get(placement.target_feature_id or "")
                if placement is not None
                else None
            )
            actual = target.tags.get("campus_subtype") if target else None
            task_subtype_matches[str(task_id)] = {
                "expected": sorted(expected),
                "actual": actual,
                "match": actual in expected,
            }

    return {
        "geometry_method": "official_public_reference_topology_plus_parametric_generalization",
        "accuracy_class": "schematic_not_survey_grade",
        "reference_image_geometry_digitized": False,
        "generalized_building_count": len(buildings),
        "cluster_counts": cluster_counts,
        "shape_family_counts": shape_counts,
        "campus_subtype_counts": subtype_counts,
        "placement_shift_mean_m": round(float(np.mean(shifts)), 2) if shifts else 0.0,
        "placement_shift_max_m": round(max(shifts), 2) if shifts else 0.0,
        "building_pair_overlap_count": len(building_pair_overlaps),
        "building_pair_overlaps": building_pair_overlaps,
        "water_8m_clearance_violation_count": len(water_clearance_violations),
        "water_8m_clearance_violations": water_clearance_violations,
        "sport_overlap_violation_count": len(sport_overlap_violations),
        "sport_overlap_violations": sport_overlap_violations,
        "generalized_road_5m_intrusion_count": len(road_intrusions),
        "generalized_road_5m_intrusions": road_intrusions,
        "gate_count": sum(1 for feature in features if feature.layer == "gate"),
        "task_subtype_matches": task_subtype_matches,
    }


def water_topology_audit(features: Sequence[Feature]) -> Dict[str, object]:
    water_polygons = [
        feature.geometry for feature in features
        if feature.layer == "water" and feature.geometry is not None
    ]
    water_union = unary_union(water_polygons) if water_polygons else None
    buildings = [
        feature for feature in features
        if feature.layer == "building" and feature.geometry is not None
    ]
    total_building_area = sum(feature.geometry.area for feature in buildings)
    overlapping_buildings = 0
    authorized_water_structures = 0
    unexpected_overlapping_buildings = 0
    materially_overlapping_buildings = 0
    total_building_overlap_area = 0.0
    unexpected_building_overlap_area = 0.0
    nonbridge_road_overlap_length = 0.0
    deep_water_nonbridge_road_overlap_length = 0.0

    if water_union is not None and not water_union.is_empty:
        deep_water = water_union.buffer(-20.0)
        for feature in buildings:
            intersection_area = feature.geometry.intersection(water_union).area
            if intersection_area > 1.0:
                overlapping_buildings += 1
                total_building_overlap_area += intersection_area
                if is_authorized_water_structure(feature):
                    authorized_water_structures += 1
                else:
                    unexpected_overlapping_buildings += 1
                    unexpected_building_overlap_area += intersection_area
                    if intersection_area / max(feature.geometry.area, 1.0) > 0.10:
                        materially_overlapping_buildings += 1
        for feature in features:
            if feature.layer != "road" or feature.geometry is None:
                continue
            if feature.tags.get("bridge") not in {None, "no"}:
                continue
            nonbridge_road_overlap_length += feature.geometry.intersection(
                water_union
            ).length
            if not deep_water.is_empty:
                deep_water_nonbridge_road_overlap_length += (
                    feature.geometry.intersection(deep_water).length
                )

    return {
        "water_area_m2": round(
            water_union.area if water_union is not None else 0.0, 2
        ),
        "overlapping_building_count": overlapping_buildings,
        "authorized_water_structure_count": authorized_water_structures,
        "unexpected_overlapping_building_count": unexpected_overlapping_buildings,
        "materially_overlapping_building_count": materially_overlapping_buildings,
        "total_building_overlap_area_m2": round(total_building_overlap_area, 2),
        "unexpected_building_overlap_area_m2": round(
            unexpected_building_overlap_area, 2
        ),
        "unexpected_building_overlap_ratio": round(
            unexpected_building_overlap_area / total_building_area, 8
        ) if total_building_area else 0.0,
        "nonbridge_road_overlap_length_m": round(
            nonbridge_road_overlap_length, 2
        ),
        "deep_water_nonbridge_road_overlap_length_m": round(
            deep_water_nonbridge_road_overlap_length, 2
        ),
    }


def public_task_placement(
    zone: ZoneSpec,
    task: TaskSpec,
    placement: TaskPlacement,
) -> Dict[str, object]:
    data = {
        "id": task.point_id,
        "site_type": task.site_type,
        "local_enu_m": [round(placement.x_m, 2), round(placement.y_m, 2)],
        "placement_source": placement.placement_source,
        "snap_distance_m": round(placement.snap_distance_m, 2),
    }
    if placement.target_feature_id is not None:
        data.update({
            "target_ref": "%s-T%02d" % (zone.zone_id, task.point_id),
            "target_geometry_source": placement.target_geometry_source,
            "target_building_type": placement.target_building_type,
        })
    return data


def validate_scene(
    zone: ZoneSpec,
    features: Sequence[Feature],
    placements: Dict[int, TaskPlacement],
):
    if len(zone.tasks) != 3:
        raise ValueError("分区 %s 必须恰好包含 3 个任务点" % zone.zone_id)
    if not any(f.layer == "building" for f in features):
        raise ValueError("分区 %s 未解析到建筑" % zone.zone_id)
    if not any(f.layer == "road" for f in features):
        raise ValueError("分区 %s 未解析到道路" % zone.zone_id)
    topology = water_topology_audit(features)
    if topology["unexpected_overlapping_building_count"] > 0:
        raise ValueError(
            "分区 %s 有 %d 栋非水上设施建筑与水体交叠，拒绝渲染"
            % (
                zone.zone_id,
                topology["unexpected_overlapping_building_count"],
            )
        )
    if (
        zone.zone_id in {"B", "D"}
        and topology["deep_water_nonbridge_road_overlap_length_m"] > 100.0
    ):
        raise ValueError(
            "分区 %s 的非桥道路深入水体 %.1f m，拒绝渲染"
            % (
                zone.zone_id,
                topology["deep_water_nonbridge_road_overlap_length_m"],
            )
        )
    if zone.zone_id == "C":
        if topology["unexpected_building_overlap_ratio"] > 0.002:
            raise ValueError(
                "C 区水面与建筑交叠比例 %.4f 超过 0.002，拒绝渲染"
                % topology["unexpected_building_overlap_ratio"]
            )
        if topology["materially_overlapping_building_count"] > 2:
            raise ValueError(
                "C 区有 %d 栋建筑超过 10%% 面积落入水体，拒绝渲染"
                % topology["materially_overlapping_building_count"]
            )
        if topology["deep_water_nonbridge_road_overlap_length_m"] > 500.0:
            raise ValueError(
                "C 区非桥道路进入岸线内 20 m 后仍与水体交叠 %.1f m，拒绝渲染"
                % topology["deep_water_nonbridge_road_overlap_length_m"]
            )
    if zone.zone_id == "B":
        campus_audit = campus_detail_audit(features, placements)
        if campus_audit["generalized_building_count"] < 32:
            raise ValueError("B 区概化校园建筑少于 32 栋，细节不足")
        if campus_audit["building_pair_overlap_count"] > 0:
            raise ValueError("B 区存在概化建筑相互交叠")
        if campus_audit["water_8m_clearance_violation_count"] > 0:
            raise ValueError("B 区概化建筑未满足 8 m 水体退距")
        if campus_audit["sport_overlap_violation_count"] > 0:
            raise ValueError("B 区概化建筑与 OSM 运动场交叠")
        if campus_audit["generalized_road_5m_intrusion_count"] > 0:
            raise ValueError("B 区概化道路侵入建筑 5 m 缓冲区")
        if campus_audit["gate_count"] != 3:
            raise ValueError("B 区应包含 3 个脱敏校门")
        if not all(
            item["match"]
            for item in campus_audit["task_subtype_matches"].values()
        ):
            raise ValueError("B 区任务点未匹配到指定校园设施类型")
    buildings_by_id = {
        feature.feature_id: feature
        for feature in features
        if feature.layer == "building"
    }
    water_geometries = [
        feature.geometry for feature in features
        if feature.layer in {"water", "waterway"} and feature.geometry is not None
    ]
    water_union = unary_union(water_geometries) if water_geometries else None
    hard_geometries = [
        risk_geometry(zone, risk) for risk in zone.risks if risk.hard
    ]
    hard_union = unary_union(hard_geometries) if hard_geometries else None
    for task in zone.tasks:
        if not (0 <= task.x_ratio <= 1 and 0 <= task.y_ratio <= 1):
            raise ValueError("任务点 P%d 超出分区范围" % task.point_id)
        placement = placements.get(task.point_id)
        if placement is None:
            raise ValueError("任务点 P%d 缺少落位结果" % task.point_id)
        point = Point(placement.x_m, placement.y_m)
        if not box(0, 0, zone.width_m, zone.height_m).covers(point):
            raise ValueError("任务点 P%d 的落位超出分区范围" % task.point_id)
        if water_union is not None and water_union.contains(point):
            raise ValueError("任务点 P%d 落入水体" % task.point_id)
        if hard_union is not None and hard_union.contains(point):
            raise ValueError("任务点 P%d 落入硬风险区" % task.point_id)
        if task.point_id != 1:
            target = buildings_by_id.get(placement.target_feature_id or "")
            if target is None or not target.geometry.covers(point):
                raise ValueError("任务点 P%d 未落在目标建筑轮廓内" % task.point_id)


def parse_args():
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parents[1]
    parser = argparse.ArgumentParser(description="生成四张武汉参考场景 2D 审核图")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "outputs" / "环境示意图_2D审核",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=project_root / "data" / "cache" / "osm_reference_scenes",
    )
    parser.add_argument("--dpi", type=int, default=260)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--refresh-osm", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument(
        "--zones",
        default="A,B,C,D",
        help="逗号分隔的分区 ID，默认 A,B,C,D",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    apply_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    requested = {item.strip().upper() for item in args.zones.split(",") if item.strip()}

    manifest = {
        "schema_version": 3,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "mode": "2d-review-only",
        "sources": {
            "base_geometry": "OpenStreetMap / Overpass API",
            "reference_morphology": (
                "用户参考图仅用于场景原型；在 OSM 覆盖不足处以脱敏形态补绘，"
                "不复刻高德底图、POI 或现实配送位置"
            ),
            "wuhan_college_public_references": {
                "method": (
                    "只校核校园功能分区和相对拓扑；使用参数化 L/U/H/条形轮廓"
                    "重新绘制，非像素描摹、非测绘级"
                ),
                "local_source_manifest": (
                    "data/cache/reference_sources/wuhan_college/"
                    "source_manifest.json"
                ),
                "urls": [
                    "https://zs.whxy.edu.cn/info/1097/1575.htm",
                    "https://www.whxy.edu.cn/info/1112/5784.htm",
                    "https://whxy.edu.cn/xyfg/jzfg.htm",
                    "https://whxy.edu.cn/info/1011/3177.htm",
                    "https://www.whxy.edu.cn/info/1013/19384.htm",
                    "https://www.whxy.edu.cn/info/1011/9173.htm",
                ],
            },
            "tasks_and_risks": "脱敏训练设定，仅使用局部 ENU 坐标",
        },
        "attribution": "© OpenStreetMap contributors, ODbL",
        "contains_3d": False,
        "contains_routes": False,
        "zones": [],
    }

    generated = []
    for zone in ZONE_SPECS:
        if zone.zone_id not in requested:
            continue
        print("[%s] 获取开放地图数据..." % zone.zone_id)
        payload, source_meta = fetch_osm(
            zone,
            args.cache_dir,
            refresh=args.refresh_osm,
            offline=args.offline,
        )
        features = parse_osm(payload, zone, args.seed)
        features = augment_reference_morphology(zone, features)
        placements = resolve_task_placements(zone, features)
        validate_scene(zone, features, placements)
        output_path = args.output_dir / zone.output_name
        print("[%s] 渲染 %s 个图层对象..." % (zone.zone_id, len(features)))
        render_zone(zone, features, placements, output_path, args.dpi)
        print("[%s] 已保存: %s" % (zone.zone_id, output_path))
        generated.append(output_path)
        zone_manifest = {
            "zone_id": zone.zone_id,
            "title": zone.title,
            "archetype": zone.archetype,
            "output": zone.output_name,
            "extent_m": [zone.width_m, zone.height_m],
            "feature_counts": feature_counts(features),
            "geometry_source_counts": geometry_source_counts(features),
            "building_inference": building_inference_summary(features),
            "water_topology_audit": water_topology_audit(features),
            "task_points": [
                public_task_placement(zone, task, placements[task.point_id])
                for task in zone.tasks
            ],
            "risk_types": [risk.kind for risk in zone.risks],
            "osm_response_sha256": source_meta.get("response_sha256"),
            "osm_query_sha256": source_meta.get("query_sha256"),
            "osm_endpoint": source_meta.get("endpoint"),
            "osm_retrieved_utc": source_meta.get("retrieved_utc"),
            "cache_hit": source_meta.get("cache_hit", False),
        }
        if zone.zone_id == "B":
            zone_manifest["campus_detail_audit"] = campus_detail_audit(
                features,
                placements,
            )
        manifest["zones"].append(zone_manifest)

    if requested == {"A", "B", "C", "D"} and len(generated) != 4:
        raise RuntimeError("完整审核应生成恰好 4 张图")

    task_ids = [
        task.point_id
        for zone in ZONE_SPECS
        if zone.zone_id in requested
        for task in zone.tasks
    ]
    if len(task_ids) != len(set(task_ids)):
        raise RuntimeError("任务点 ID 不唯一")
    if 1 in task_ids and task_ids.count(1) != 1:
        raise RuntimeError("配送中心 P1 必须唯一")

    manifest_path = args.output_dir / "scene_2d_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("生成完成：%d 张 2D 审核图；未生成 3D。" % len(generated))
    print("清单: %s" % manifest_path)


if __name__ == "__main__":
    main()
