# 武汉参考场景 2D→3D 转换脚本

脚本：`convert_wuhan_reference_scenes_2d_to_3d.py`

转换对象是 2D 脚本使用的矢量特征，不是 PNG 像素。建筑沿原轮廓按高度挤出，
水体、道路、运动场、风险区、任务点和动态实体分别转为三维图层。

## 快速生成武汉学院 B 区

```powershell
python convert_wuhan_reference_scenes_2d_to_3d.py --zones B --lod 0 --preview
```

默认离线读取已经审核的 OSM 缓存，输出到：

```text
outputs/环境示意图_3D预览/
```

主要文件：

- `zone_B_scene_lod0.obj` / `.mtl`：可导入 Blender、Unity、Unreal 的三维模型；
- `zone_B_collision.json`：建筑、桥体、高压线和硬风险区碰撞数据；
- `zone_B_scene_lod0.json`：来源、统计、任务高度及验收结果；
- `zone_B_scene_lod0_preview.png`：三维审核图；
- `scene_3d_index.json`：批量转换索引。

## 批量与 LOD

```powershell
# 四个分区的远景模型
python convert_wuhan_reference_scenes_2d_to_3d.py --zones A,B,C,D --lod 2

# B 区同时生成三个 LOD
python convert_wuhan_reference_scenes_2d_to_3d.py --zones B --all-lods

# 四区全部LOD，并只为LOD0生成审核图
python convert_wuhan_reference_scenes_2d_to_3d.py `
  --zones A,B,C,D --all-lods --preview
```

LOD 定义：

- LOD0：原建筑轮廓、屋顶设备和高层天线；
- LOD1：保持拓扑的简化轮廓；
- LOD2：最小旋转包围盒。

碰撞 JSON 始终保存原始建筑轮廓棱柱，不随可视 LOD 简化。
使用 `--preview-lods 0,1,2` 可以显式指定需要生成审核图的LOD。

## 接入公开 DEM

在指定目录放置 `zone_A_terrain.npz`、`zone_B_terrain.npz` 等文件：

```python
import numpy as np

np.savez(
    "zone_B_terrain.npz",
    x=x_enu_m,                 # 一维、严格递增
    y=y_enu_m,                 # 一维、严格递增
    z=elevation_m,             # shape = (len(y), len(x))
)
```

运行：

```powershell
python convert_wuhan_reference_scenes_2d_to_3d.py `
  --zones B --terrain-dir data/terrain --preview
```

未提供 DEM 时脚本使用零高程平面，并在清单标记
`flat_no_dem_input`，不会把程序化起伏冒充真实地形。

## 审核路径说明

默认生成两条分级审核走廊，用于检查三维高度、建筑遮挡和任务点关系。
它们在清单中标记为 `3d_review_only_not_planned_route`，不是强化学习结果，
也不是现实配送航线。使用 `--without-review-paths` 可关闭。
