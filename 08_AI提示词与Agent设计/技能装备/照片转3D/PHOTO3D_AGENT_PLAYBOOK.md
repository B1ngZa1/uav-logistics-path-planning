# 照片转 3D Agent Playbook

用途：把多视角照片或视频帧转换为可审查的三维视觉资产，并在满足尺度、坐标和安全验收条件后，生成现有无人机规划器可以消费的保守碰撞资产。

## 执行顺序

1. **输入审计**：建立照片清单、EXIF/GPS 摘要、拍摄时间和 SHA-256；剔除严重模糊、重复帧和动态遮挡异常。不得把单张照片当作可量测三维输入。
2. **SfM**：使用固定版本的 COLMAP（或 OpenMVG 对照）完成特征、匹配、建图；保存数据库、相机参数、日志和版本信息。
3. **MVS**：使用 OpenMVS 或 Meshroom/AliceVision 生成稠密点云、网格和纹理；保留孔洞、低置信区域和未观测区域的标记。
4. **米制配准**：用非共线控制点或已知距离求相似变换
   `p_project = s R p_recon + t`；保存拟合点和独立检查点误差。没有尺度与垂直基准时，`ready_for_planner` 必须保持 `false`。
5. **质量门**：检查注册率、重投影误差、点云密度、尺度残差、垂直基准、薄障碍物覆盖和独立检查点；任何一项失败都只能输出视觉层，不能输出“可飞空间”。
6. **规划器适配**：优先将网格转为保守 footprint 棱柱、桥梁/电线转为带高度的体或胶囊，并按
   `d(p,O) >= r_uav + e_recon + e_registration + e_localization + margin`
   膨胀。未知空间按占用处理，不能把无点区域当作自由空间。
7. **离线回放**：在 `scene_contract.example.json` 中填入坐标、尺度、误差和资产路径，先用 `wuhan_scene.py`/toy route 做碰撞回放，再进入轨迹优化或 MARL 训练；不覆盖原始场景资产。

## 停止条件

- 只有视觉网格、NeRF 或 3DGS 而没有尺度、CRS/原点、垂直基准和检查点：停止在视觉层。
- 只有一张图、卫星截图或既有示意图：停止，不声称完成摄影测量重建。
- `scene_contract.example.json` 中 `ready_for_planner=false` 或任一 acceptance 字段为 false：禁止写入规划器的生产碰撞目录。
- 3DGS/NeRF 的外观表示不能直接当作碰撞几何；必须另建网格/体素/棱柱碰撞层。

## 交付物

每次运行至少交付：输入清单、软件锁定清单、日志、稀疏/稠密点云、网格/纹理、相似变换、独立检查点报告、误差包络、碰撞适配 JSON、预览图和一份“未通过项”清单。

工具选择、仓库链接、许可证和本项目接口约束见同目录 `toolchain_matrix.csv`、`github_sources.md`、`README.md` 与 `scene_contract.example.json`。本 playbook 是已装备的执行规范；重型二进制、CUDA 环境和真实照片重建仍须在独立环境中安装和验收，不能以本文件冒充已运行结果。

## 本机可执行性审计（2026-09-20）

本次只读审计结果：

- 项目内可用于摄影测量的照片/视频输入为 1 个 `data/cache/reference_sources/wuhan_college/official_campus_map_2025.jpg`，尺寸 2048×1389；它是单张参考地图，不是多视角照片集，不能运行 SfM/MVS 重建。
- Python 为 3.9.13；已发现 `numpy`、`Pillow`、`opencv-python`、`shapely`、`torch`，未发现 `open3d`、`pycolmap`。
- PATH 未发现 `colmap`、`InterfaceCOLMAP`、`DensifyPointCloud`、`ReconstructMesh`、`meshroom_batch`、`ns-process-data`、`ns-train`、`ns-export`、`instant-ngp`、`ffmpeg`、`cmake`；`nvidia-smi` 可调用只说明驱动查询入口存在，不代表 CUDA 编译链或重建工具可用。
- 因此本轮没有真实运行照片转 3D，没有生成可宣称为摄影测量结果的点云/网格，也没有把现有 2D→3D 示意输出升级为实测场景。

## 后续可执行命令（不在本轮运行）

在独立目录和独立环境中固定版本后，再执行以下最小链路；`$toolRoot`、`$runRoot` 和 release/tag 必须改成实际值，并保存 SHA-256、日志和版本输出。

```powershell
# 0. 先做入口检查；任一缺失都停止，不用“看似成功”的占位输出
Get-Command colmap,InterfaceCOLMAP,DensifyPointCloud,ReconstructMesh -ErrorAction SilentlyContinue
python --version
nvidia-smi

# 1. 输入必须是至少多视角、具有重叠的照片目录；不要把本项目单张地图放入此处
$runRoot = 'D:\photo3d_runs\zone_B_smoke'
colmap feature_extractor --database_path "$runRoot\database.db" --image_path "$runRoot\images"
colmap exhaustive_matcher --database_path "$runRoot\database.db"
colmap mapper --database_path "$runRoot\database.db" --image_path "$runRoot\images" --output_path "$runRoot\sparse"
colmap model_analyzer --path "$runRoot\sparse\0"
colmap image_undistorter --image_path "$runRoot\images" --input_path "$runRoot\sparse\0" --output_path "$runRoot\dense" --output_type COLMAP

# 2. 表面重建；版本不同可能使用不同参数，先执行各程序 -h 并保存输出
InterfaceCOLMAP -i "$runRoot\dense" -o "$runRoot\mvs\scene.mvs" --image-folder "$runRoot\dense\images"
DensifyPointCloud "$runRoot\mvs\scene.mvs"
ReconstructMesh "$runRoot\mvs\scene_dense.mvs"
TextureMesh "$runRoot\mvs\scene_dense.mvs" -m "$runRoot\mvs\scene_dense_mesh.ply"
```

若使用 Nerfstudio，只在已锁定 CUDA/PyTorch 环境运行 `ns-process-data`、`ns-train` 和 `ns-export`；NeRF/3DGS 输出必须经过同一套尺度、坐标、未知空间和碰撞适配门禁，不能跳过 OpenMVS/网格或体素保守化步骤。

## 最小质量门槛

在 `scene_contract.example.json` 中把 `ready_for_planner` 设为 `true` 之前，必须同时满足：

1. 影像有效重叠且相机注册率、重投影误差和稀疏点数达到本次实验预先记录的阈值；失败时只能输出失败报告。
2. 至少 3 个非共线控制点用于相似变换，另留独立检查点；尺度、水平/垂直方向和原点均有记录，独立检查点误差不超过项目事先设定的容差。
3. 稠密点云/网格的覆盖、孔洞、薄障碍物、电线/桥梁和动态物体经过人工或脚本检查；未观测区域标记为 unknown，不得标为空闲。
4. 碰撞适配器已通过静态障碍物、薄体、跨高度线段、起终点和安全膨胀回放；膨胀半径明确包含无人机尺寸、重建误差、配准误差、定位误差和额外安全裕度。
5. 视觉网格、原始重建、碰撞网格/体素、manifest、日志、软件锁定清单和输入 SHA-256 可互相追溯；不得覆盖 `outputs/环境示意图_3D预览` 的既有资产。
