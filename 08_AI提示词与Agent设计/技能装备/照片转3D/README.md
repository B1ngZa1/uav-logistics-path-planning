# 照片转 3D：可装备工具链与无人机环境接入

核验日期：2026-09-17；2026-09-20 复核。状态：调研及现有资产只读核验已开展；未安装、未训练、未做真实照片重建；本地结构化自检已完成。本文命令是下一步操作说明，不是已执行记录。

2026-09-20 复核补充：三个交付文件已存在且可读；`toolchain_matrix.csv` 通过 `Import-Csv` 解析（8 行、14 列、无空工具名），`scene_contract.example.json` 通过 `ConvertFrom-Json` 解析且明确 `ready_for_planner=false`。实际查看了 B 区 LOD0 预览图；图中明确标注“非测绘级”。在线抽查了 COLMAP 官方输出/地理配准文档及 Nerfstudio 官方坐标、几何导出、Splatfacto 文档；结论与下文一致。此次复核仍未安装或运行任何重建工具。

## 1. 建议先装备什么

优先采用 **COLMAP SfM → OpenMVS 稠密重建/网格/纹理 → 米制配准 → 保守碰撞体**。它与本项目已有 OBJ/MTL 显示层、棱柱/胶囊碰撞层最接近。小规模照片先验证几何质量；有航测 GPS/GCP、需要地理参考点云和 DEM 时，另选 ODM。需要逼真视角或视觉强化学习时，再增加 Nerfstudio 的 Nerfacto/Splatfacto。此排序是结合现有代码和本机资源的工程判断，不是统一性能排行榜。

| 优先级 | 装备 | 本项目作用 | 装备条件 |
|---|---|---|---|
| P0 | COLMAP CLI，OpenMVS CLI | 相机定姿、点云、带纹理网格 | 独立工作目录；固定版本；尺度校准 |
| P0 | Open3D Python 库 | 点云清理、网格距离查询、配准辅助 | 不把点云空洞当可飞空间；几何验收 |
| P1 | Nerfstudio | NeRF/3DGS 视觉层对照实验 | 隔离 Python/CUDA 环境；8 GB 显存先小样本 |
| P1 | ODM | 航测照片、GCP、DSM/DTM/地理参考点云 | 确认 CRS/高程基准；核验 Docker 或原生运行环境 |
| P2 | Meshroom/AliceVision | 图形化摄影测量流程、人工检查 | 完整摄影测量包；CUDA 深度估计节点可用 |
| P2 | OpenMVG | SfM 算法对照，输出交给 OpenMVS | 需要做 SfM 研究时再装备 |
| P2 | Instant-NGP、原版 3DGS、torch-ngp | 研究复现、速度/画质对照 | 依赖和许可证分别核验 |
| P3 | 社区 Skill/MCP | 流程指导、模型导入、人工清理辅助 | 代码审计和项目级隔离；不作为测量精度保证 |

完整的输入、输出、系统/GPU、许可证及接入方式见 [toolchain_matrix.csv](toolchain_matrix.csv)；其中 `official_repository` 列直接链接到对应官方仓库。官方仓库、固定提交与证据文档见 [github_sources.md](github_sources.md)。

## 2. 现有项目实况

检查入口：

- `03_图表脚本/Python脚本/README_2D转3D.md`。
- `03_图表脚本/Python脚本/convert_wuhan_reference_scenes_2d_to_3d.py`。
- `outputs/环境示意图_3D预览`：45 个文件，含 12 个 OBJ、12 个 MTL、17 个 JSON、4 张预览图；实际查看了 B 区预览。
- 为确认接入接口，补读 `plot_wuhan_reference_scenes_2d.py` 的坐标转换与 `07_仿真与控制代码/Python仿真/environment/wuhan_scene.py` 的加载/碰撞逻辑。

当前转换器读取 OSM/脱敏概化矢量，而非 PNG 像素，也没有 SfM、MVS 或照片相机标定。OBJ 的材质颜色不能等同于摄影测量纹理。仅有一张示意图、卫星截图或现有预览图时，不能通过这些多视角工具恢复真实遮挡后结构与绝对尺度；单图生成式资产可用于合成训练，但需标为合成几何。

| 区域 | 范围 m | 建筑数 | 估计高度比例 | LOD0 顶点/面 | 地形 |
|---|---:|---:|---:|---:|---|
| A | 2400×1800 | 837 | 89.7252% | 73258 / 37520 | `flat_no_dem_input` |
| B | 2400×1800 | 43 | 100% | 6535 / 3421 | `flat_no_dem_input` |
| C | 2800×2000 | 538 | 87.7323% | 47760 / 20306 | `flat_no_dem_input` |
| D | 2400×1800 | 592 | 65.2027% | 85795 / 39525 | `flat_no_dem_input` |

数值来自各区 `zone_*_scene_lod0.json`，不是实测精度结果。场景标记 `research_visualization_not_survey_grade`；B 区有 41 栋概化建筑。B 区审核路径的 70 m、85 m 是预览示例，不能据此决定现实飞行高度。四区各有自己的局部坐标，不能直接拼成跨区航线。

关键接口事实：

1. 转换器第 1189 行附近的碰撞清单包含 `buildings`、`bridges`、`power_lines`、`hard_risk_zones`、`dynamic_entities_initial_state`；建筑是 footprint＋上下高度，不是三角网格。
2. `wuhan_scene.py` 的 `ScenePackage.load`（约第 337 行起）只消费 scene/collision JSON；直接放入 OBJ/PLY/高斯 PLY 不会改变碰撞结果。
3. `PrismCollider.from_payload` 只读 `exterior_xy_m`，未消费 `holes_xy_m`；中庭等洞口会被保守封闭。`clearance()` 使用 AABB 距离，不是精确网格距离。
4. `ScenePackage` 当前只保存地形最小/最大高度，未加载 DEM 网格作为逐点地形查询。不能只补一份 DEM 元数据就宣称规划器支持真实山地。
5. 原转换器为部分建筑生成的屋顶设备/天线主要是可视层；建筑碰撞上界仍是 `ground_z + height_m`。照片重建若识别出屋顶突出物，需要在新碰撞资产中明确编码。
6. 风险区、电线半径及动态实体包含建模假设；静态照片不能替代风场、临时禁行、实时交通或其他无人机状态。

## 3. 本机资源与范围

实际只读检测：Windows/PowerShell；NVIDIA GeForce RTX 4060 Laptop GPU，8188 MiB 显存，驱动 596.08；物理内存 16849256448 bytes（约 15.69 GiB）。PATH 可找到 Python 3.9 路径、Git、Conda、Docker、WSL 命令；这不证明 Docker 引擎、WSL 发行版或 CUDA 编译环境可用。PATH 未找到 `colmap`、`nvcc`、`ffmpeg`、`cmake`，不排除存在未加入 PATH 的便携程序。

建议首轮用同一静态小建筑/短走廊的 50–150 张照片，最长边 1600–2000 px，先核对注册率、覆盖与尺度，再扩展。张数与分辨率是本项目试验起点，不是官方硬件最低要求或成功保证。当前内存不适合不分块地同时处理四区全量高分辨率影像。

Splatfacto 文档给出的典型显存约 6 GB、big 约 12 GB；原版 3DGS 说明论文评测质量需 24 GB。8 GB 可尝试受限 Splatfacto/降分辨率研究，不承诺复现原版论文全量质量。[S11][S12]

## 4. 三维结果如何接入规划器

### 4.1 统一坐标和尺度

普通单目 SfM 只有相似变换意义下的结构；EXIF GPS、视频中的高度标签或 NeRF 自动归一化都不能直接当作经验证的米制坐标。

使用独立量测的控制点/相机位置求取重建坐标到项目坐标的相似变换：

\[
\mathbf p_{project}=sR\mathbf p_{recon}+\mathbf t,
\quad s>0,\ R^TR=I,\ \det R=1.
\]

`p` 是三维位置，`s` 是米/重建单位，`R` 是旋转，`t` 是米制平移。用加权残差最小化估计，并保留未参与拟合的检查点；至少需要非共线对应点，实际应分布于区域各处并包含高度变化。仅用一个已知长度可约束尺度，不能确定完整朝向和原点。保存检查点误差，不把拟合残差冒充外部精度。

项目 `lonlat_to_local`（2D 脚本第 447–455 行）的实际公式为：

\[
x=R_E\cos\varphi_0(\lambda-\lambda_0)+W/2,\quad
y=R_E(\varphi-\varphi_0)+H/2.
\]

角度以弧度计，`R_E` 取现有脚本常量，`W,H` 是各区宽高。因此中心经纬度映射到 `(W/2,H/2)`；这是球面局部平面近似，并非严格的椭球 ECEF→ENU。兼容现有环境时必须复用该约定或提供显式桥接变换；不能只写“ENU”就省略原点和偏移。真实测量场景宜以一致的测地基准建立局部坐标，再明确转换到项目坐标。

COLMAP 外参是 world→camera：`Xc=Rcw Xw+tcw`，相机中心 `Cw=-Rcwᵀ tcw`，相机轴为右、下、前。Nerfstudio 使用 OpenGL/Blender 相机约定，Y/Z 与前者相反，且数据解析器可能平移、旋转、缩放场景；须保存并逆应用归一化。Nerfstudio 点云 exporter 的 `--save-world-frame True` 只恢复原始数据坐标，仍不自动得到米制项目 ENU。[S03][S10]

高度需区分：相对地面高度 AGL、椭球高、正常高/海拔类高程、项目相对零面。对同一垂直基准，`z_project = z_ground(x,y) + h_AGL`。目前 z=0 平面不代表当地真实海拔。只有 XYZ/OBJ/PLY 而没有 CRS、原点、尺度和高程说明时，禁止认定为真实场景校准成功。

### 4.2 两种接入方式

**方式 A：现有加载器兼容包（优先）。** 在新场景目录中，把已配准网格分区/分高度切片，生成保守 footprint 棱柱；桥面等悬空物体保留实际 `z_min_m>0`，不要全都从地面挤出。树冠、突出物可以放入兼容的静态棱柱列表，另存语义来源；电线保留胶囊近似。保留任务点、风险区、动态状态，输出同名格式的 `zone_B_scene_lod0.json` 和 `zone_B_collision.json`。使用显式 `scene_dir` 做离线对照，不覆盖原资产。近似会损失斜面/拱洞/细杆细节；必须对原网格做碰撞漏检核验。

**方式 B：新增网格或体素后端（后续开发）。** 以 Open3D 的 RaycastingScene 做三角面距离/射线查询，或构建稀疏占据体素及 ESDF，再接入 `ScenePackage` 等碰撞查询接口。现有业务脚本不具备此后端，本次没有实现。Open3D signed distance/occupancy 假设网格封闭且内外明确；开放场景需对未知区单独处理，不可把无点体素等同空闲。[S19]

以下是建议的安全包络定义，并非当前程序已经实现：

\[
d(\mathbf p(t),\mathcal O)\ge r_{uav}+e_{recon}+e_{registration}+e_{localization}+m.
\]

距离、机体包络半径、重建/配准/定位误差界和额外余量的单位都是 m；所有误差应来自验证或保守界，不直接把 RMSE 当最大误差。对边长 `v` 的体素中心近似，可另加入 `sqrt(3)*v/2` 量化界；若几何已膨胀，避免重复计入。未知/未观测区域默认为不可通行，时间变化障碍按 `O(t)` 更新；线段扫掠检测覆盖相邻规划点之间，不能只检查航点。

Gaussian PLY 的位置/尺度/不透明度是渲染参数，不等于表面点云和实体占据概率。NeRF 的密度阈值表面也需额外验证。可视网格与碰撞网格分开保存，PPO/MAPPO 等策略只调用已验证碰撞服务，不能用渲染 PSNR 代替路径安全指标。

建议接口草案见 [scene_contract.example.json](scene_contract.example.json)：未知变换全部为 `null`，`ready_for_planner=false`。这是待实现适配器的配置契约，现有代码不会自动读取或执行其中策略。

## 5. 下一步命令（全部未执行）

### 5.1 工作区与输入

以下为 PowerShell 示例；原始文件使用副本放入新目录。每个原生命令执行后检查 `$LASTEXITCODE`，非零就停止当前管线；复跑使用新的 run 目录。路径中有空格/中文，始终保留引号。若第三方编译器不支持 Unicode 路径，应先解决构建路径问题，不能擅自把任务写到全局目录。

```powershell
$projectRoot = 'C:\University\Files\Training program\无人机项目设计'
$workRoot = Join-Path $projectRoot 'outputs\photo3d_workspace'
$runRoot = Join-Path $workRoot 'runs\pilot_001'
$toolRoot = Join-Path $workRoot 'tools'
New-Item -ItemType Directory -Force -Path $toolRoot,"$runRoot\images","$runRoot\sparse","$runRoot\dense","$runRoot\mvs"
nvidia-smi
Get-Command nvcc,cmake,ffmpeg -ErrorAction SilentlyContinue
```

准备有视差、足够重叠且能覆盖侧面/屋顶的照片，并保存原始 EXIF、相机型号、焦距和控制点量测。视频先抽关键帧，帧号与时间戳另存；抽帧不会创造多视角。已单独取得 FFmpeg 后可用：

```powershell
# 1 fps 仅用于首轮抽帧，需按移动速度/清晰度/重叠调整；-n 防止覆盖。
ffmpeg -n -i "$runRoot\capture.mp4" -vf 'fps=1' -q:v 2 "$runRoot\images\frame_%06d.jpg"
```

### 5.2 COLMAP → OpenMVS（首选）

从 [COLMAP 官方 Releases](https://github.com/colmap/colmap/releases) 选择固定版本 Windows CUDA 包，解压到 `$toolRoot\colmap`；从 [OpenMVS 官方构建入口](https://github.com/cdcseacave/openMVS/wiki/Building) 获取匹配版本或构建制品，放到 `$toolRoot\openmvs`。本次 GitHub Releases API 触发 403 限流，因此未宣称核验最新稳定版、未硬编码猜测的下载资产 URL。安装前记录 release tag、文件 SHA-256 与 CUDA/驱动兼容性。源码提交快照不等于稳定版发布。

需要可审计源码时，先仅下载固定版本：

```powershell
git clone https://github.com/cdcseacave/openMVS.git "$toolRoot\openmvs-src"
git -C "$toolRoot\openmvs-src" checkout --detach e23d25f0a9bd93ec056d4ee7fae2b2aaee80e566
git -C "$toolRoot\openmvs-src" submodule update --init --recursive
# 源码构建须按 S05 配置 Boost/Eigen/OpenCV/CGAL/Ceres 等；不应直接套用缺依赖的 cmake 命令。
```

已准备便携二进制后，按实际解压目录设置路径；先核对 `--help` 与本文参数，源码 main 上的新增选项不保证出现在旧 release 中。[S02][S05]

```powershell
$colmapExe = Join-Path $toolRoot 'colmap\COLMAP.bat'
$mvsBin = Join-Path $toolRoot 'openmvs\bin'
& $colmapExe -h
& "$mvsBin\InterfaceCOLMAP.exe" -h

& $colmapExe feature_extractor --database_path "$runRoot\database.db" --image_path "$runRoot\images"
& $colmapExe exhaustive_matcher --database_path "$runRoot\database.db"
# 视频长序列可改用 sequential_matcher；小规模无序照片先用 exhaustive_matcher。
& $colmapExe mapper --database_path "$runRoot\database.db" --image_path "$runRoot\images" --output_path "$runRoot\sparse"
& $colmapExe model_analyzer --path "$runRoot\sparse\0"
# sparse/0 只是示例；若存在多个子模型，应先比较注册图像及覆盖，不直接选编号最小者。
& $colmapExe image_undistorter --image_path "$runRoot\images" --input_path "$runRoot\sparse\0" --output_path "$runRoot\dense" --output_type COLMAP --max_image_size 2000

Push-Location "$runRoot\mvs"
& "$mvsBin\InterfaceCOLMAP.exe" -i "$runRoot\dense" -o scene.mvs --image-folder "$runRoot\dense\images"
& "$mvsBin\DensifyPointCloud.exe" scene.mvs
& "$mvsBin\ReconstructMesh.exe" scene_dense.mvs
& "$mvsBin\RefineMesh.exe" scene_dense_mesh.mvs
& "$mvsBin\TextureMesh.exe" scene_dense_mesh_refine.mvs --export-type obj
Pop-Location
```

OpenMVS 输出链通常为 `scene_dense.mvs/.ply`、`scene_dense_mesh.mvs/.ply`、refine/textured 成果，实际以日志为准。OBJ 须连同 MTL 和贴图保存。可选只用 COLMAP 稠密分支：

```powershell
& $colmapExe patch_match_stereo --workspace_path "$runRoot\dense" --workspace_format COLMAP --PatchMatchStereo.geom_consistency true
& $colmapExe stereo_fusion --workspace_path "$runRoot\dense" --workspace_format COLMAP --input_type geometric --output_path "$runRoot\dense\fused.ply"
& $colmapExe poisson_mesher --input_path "$runRoot\dense\fused.ply" --output_path "$runRoot\dense\meshed-poisson.ply"
```

当前 COLMAP 文档还列出 `mesh_texturer` 与 `global_mapper`，需要安装版实际支持后再用。GLOMAP 独立仓库已弃用/迁移，优先核验 COLMAP 内置 global mapper。[S02][S20]

### 5.3 OpenMVG 和 Meshroom 替代入口

OpenMVG 是稀疏 SfM 前端，不可替代 OpenMVS 的全部稠密/纹理工作。按 [官方数据管线](https://github.com/openMVG/openMVG/wiki/OpenMVG-on-your-image-dataset) 完成特征、匹配、重建后，用以下官方转换命令接上同一 MVS 管线；`sfm_data.bin` 必须是真实重建输出。[S06][S05]

```powershell
openMVG_main_openMVG2openMVS -i "$runRoot\openmvg\sfm_data.bin" -o "$runRoot\mvs\scene.mvs" -d "$runRoot\openmvg\undistorted_images"
```

Meshroom 从 [官方 Releases](https://github.com/alicevision/Meshroom/releases) 获取包含 AliceVision 摄影测量节点的包，核验 CUDA 节点而非只安装 UI。下列命令对应手册 batch 接口，具体模板名以安装版为准。[S07][S08]

```powershell
& "$toolRoot\meshroom\meshroom_batch.exe" --help
& "$toolRoot\meshroom\meshroom_batch.exe" --input "$runRoot\images" --output "$runRoot\meshroom_output"
```

稠密深度估计依赖 NVIDIA CUDA；CPU 的 Draft/SfM 不能等价于完整稠密重建。保留 `.mg` 管线、节点参数和 MeshroomCache 来源索引以便复跑。

### 5.4 Nerfstudio（视觉实验；独立环境）

官方安装页目前仍给出 Python≥3.8、Torch 2.1.2/cu118 的组合，而核验的 Nerfstudio 源码锁定 `gsplat==1.4.0`；gsplat 独立主线已提出 Torch 2.7+ 要求。不要把两套主线混装。以下是依据安装页与固定 Nerfstudio 提交整理的**候选安装配方，依赖求解和 GPU 运行尚未测试**。Windows CUDA 扩展需要匹配的 C++ Build Tools/CUDA Toolkit，本次未安装。[S09][S10][S13]

```powershell
$nsEnv = Join-Path $workRoot 'envs\nerfstudio'
conda create --prefix "$nsEnv" python=3.10 -y
conda run --prefix "$nsEnv" python -m pip install 'numpy<2' ninja
conda run --prefix "$nsEnv" python -m pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
# 仅环境级 Toolkit；C++ 编译器须事先单独准备，当前命令不会替你安装它。
conda install --prefix "$nsEnv" -c nvidia/label/cuda-11.8.0 cuda-toolkit -y
git clone https://github.com/nerfstudio-project/nerfstudio.git "$toolRoot\nerfstudio-src"
git -C "$toolRoot\nerfstudio-src" checkout --detach 50e0e3c70c775e89333256213363badbf074f29d
conda run --prefix "$nsEnv" python -m pip install "$toolRoot\nerfstudio-src"
conda run --prefix "$nsEnv" python -m pip check
conda run --prefix "$nsEnv" python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

Nerfacto 的高性能 hash encoding 还涉及 tiny-cuda-nn；按安装页在同一环境安装其 torch bindings，并记录所用提交，不能用未锁定的最新依赖宣称精确复现。Splatfacto 使用 gsplat 的 CUDA 扩展；首次编译失败应保留日志后修复编译环境。配置好当前进程可发现的 COLMAP/FFmpeg 后再运行：

```powershell
conda run --prefix "$nsEnv" ns-process-data images --data "$runRoot\images" --output-dir "$runRoot\nsdata"
conda run --prefix "$nsEnv" ns-train nerfacto --data "$runRoot\nsdata" --output-dir "$runRoot\ns_outputs" --pipeline.model.predict-normals True
# 把真实训练输出 config.yml 路径填入，不可沿用任意示例文件。
$nsConfig = 'REPLACE_WITH_REAL_NERFACTO_CONFIG.yml'
conda run --prefix "$nsEnv" ns-export pointcloud --load-config "$nsConfig" --output-dir "$runRoot\ns_pointcloud" --normal-method open3d --save-world-frame True
conda run --prefix "$nsEnv" ns-export poisson --load-config "$nsConfig" --output-dir "$runRoot\ns_mesh"

# 与 Nerfacto 分开训练、分开选择 config，不同时占满 8 GB 显存。
conda run --prefix "$nsEnv" ns-train splatfacto --data "$runRoot\nsdata" --output-dir "$runRoot\splat_outputs"
$splatConfig = 'REPLACE_WITH_REAL_SPLATFACTO_CONFIG.yml'
conda run --prefix "$nsEnv" ns-export gaussian-splat --load-config "$splatConfig" --output-dir "$runRoot\gaussians"
```

Poisson 网格与 `--save-world-frame` 点云可能不在同一坐标空间，必须根据 dataparser 变换逐个恢复并验证。Splatfacto 专门文档指出网格/普通点云导出尚不支持，不能把 NeRF 的 `ns-export poisson/tsdf` 用法机械套给 Splatfacto；以模型专页和安装版能力为准。[S10][S11]

### 5.5 原版 3DGS 与 Instant-NGP（研究对照）

原版 3DGS：先审阅研究用途许可证与 `environment.yml`，在新环境复现，不复用现有项目 Python。`$runRoot\gsdata` 必须包含同一套去畸变 `images/` 与 `sparse/0/`；可从 COLMAP 的 dense/images、dense/sparse 准备副本，保持相机与图像一致。[S12]

```powershell
git clone https://github.com/graphdeco-inria/gaussian-splatting.git "$toolRoot\gaussian-splatting"
git -C "$toolRoot\gaussian-splatting" checkout --detach 54c035f7834b564019656c3e3fcc3646292f727d
git -C "$toolRoot\gaussian-splatting" submodule update --init --recursive
Push-Location "$toolRoot\gaussian-splatting"
conda env create --prefix "$workRoot\envs\graphdeco" --file environment.yml
conda run --prefix "$workRoot\envs\graphdeco" python train.py -s "$runRoot\gsdata" -m "$runRoot\graphdeco_output" -r 4 --data_device cpu
Pop-Location
```

`-r 4`、CPU 存图仅是降显存措施，不能保证 8 GB 成功，也不能称达到论文评测设置。导出的高斯 `point_cloud.ply` 不是可直接供规划器使用的碰撞点云。

Instant-NGP：在已准备 CMake≥3.21、兼容 NVIDIA CUDA 和 C++ 编译器的开发终端中构建。当前许可证 3.3 限定非商业研究/评估。先按仓库准备隔离 Python 环境及 `requirements.txt`，再使用以下步骤。[S14]

```powershell
git clone https://github.com/NVlabs/instant-ngp.git "$toolRoot\instant-ngp"
git -C "$toolRoot\instant-ngp" checkout --detach abe236ee00cf90cfca6e36e65c00435d5b21f50a
git -C "$toolRoot\instant-ngp" submodule update --init --recursive
cmake -S "$toolRoot\instant-ngp" -B "$toolRoot\instant-ngp\build" -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build "$toolRoot\instant-ngp\build" --config RelWithDebInfo -j 4
New-Item -ItemType Directory -Force -Path "$runRoot\colmap_text"
& $colmapExe model_converter --input_path "$runRoot\sparse\0" --output_path "$runRoot\colmap_text" --output_type TXT
# 以下 python 必须来自为 instant-ngp 准备的环境，且能 import pyngp。
python "$toolRoot\instant-ngp\scripts\colmap2nerf.py" --images "$runRoot\images" --text "$runRoot\colmap_text" --out "$runRoot\transforms.json" --keep_colmap_coords
python "$toolRoot\instant-ngp\scripts\run.py" --scene "$runRoot\transforms.json" --n_steps 10000 --save_snapshot "$runRoot\model.ingp" --save_mesh "$runRoot\ngp_mesh.ply"
```

保留 COLMAP 坐标不等于关闭全部内部尺度/AABB 处理；导出网格仍需检查控制点、轴向、单位。Marching Cubes 密度等值面可能有漂浮物/空洞。torch-ngp 是作者独立 PyTorch 替代实现（MIT 主许可），不是 NVIDIA 官方绑定；其 CUDA 扩展及依赖需另审计，不作为本轮默认安装项。[S15]

### 5.6 ODM 与 Open3D

ODM 可选官方 Windows 安装包，或已有 Docker Linux 容器环境。容器首轮可使用 CPU，避免未经核实的 GPU 透传假设。下列 Docker 命令中的镜像标签必须从官方发布记录选择，再锁定 digest；不是 `latest` 复现实验。[S16][S17]

```powershell
$odmImage = 'opendronemap/odm:REPLACE_WITH_VERIFIED_TAG'
docker version
docker pull "$odmImage"
docker image inspect "$odmImage" --format '{{index .RepoDigests 0}}'
# $runRoot\images 是航测照片目录；固定 digest 后替换 odmImage。
docker run --rm --mount "type=bind,source=$runRoot,target=/datasets/pilot" "$odmImage" --project-path /datasets pilot --dsm --dtm
```

有 GCP 才按官方格式另加 GCP 文件与 `--gcp` 参数；不能拿低精度相机 GPS 当独立检查点。地理参考 LAZ/GeoTIFF 与纹理 OBJ 可能采用不同原点/偏移，需要保存转换文件。DTM 是地面模型，DSM 是表面模型，二者均不能完整表示桥下或悬空多层结构。

Open3D 是后处理库，不独立完成照片 SfM。隔离环境安装范例：[S18][S19]

```powershell
$geoEnv = Join-Path $workRoot 'envs\geometry'
conda create --prefix "$geoEnv" python=3.10 -y
conda run --prefix "$geoEnv" python -m pip install open3d==0.19.0
conda run --prefix "$geoEnv" python -m pip check
conda run --prefix "$geoEnv" python -c "import open3d as o3d; print(o3d.__version__)"
```

库安装成功后，仍需另写项目级适配器；本轮没有添加这类业务逻辑。

## 6. Skill / MCP 装备结论

| 候选 | 来源属性 | 实际能力 | 结论 |
|---|---|---|---|
| NVIDIA/nurec-skills | NVIDIA 官方技能仓库 | 提供 NeRF/场景转换/训练/导出流程提示；依赖 NGC、Linux、CUDA 容器环境 | 可作为项目级技能参考；不是本机已安装技能，也不替代摄影测量质量验收 [S18] |
| ahujasid/mcp-for-blender | 社区 MCP；非 Blender 官方 | Blender 导入、清理、导出及 Python 执行 | 只作为已有重建资产的后处理候选；不是 SfM/MVS 引擎，接入前需审查任意 Python 执行权限 [S17] |
| 本机可用的 CLI | COLMAP/OpenMVS/OpenMVG/Meshroom/AliceVision 官方/作者 CLI | 可执行、可记录版本/参数/日志、可复跑 | 本项目首选；当前 PATH 未发现这些程序，本文不宣称已安装 [S01][S03][S04][S05] |

本次检索没有确认 COLMAP、OpenMVS、Nerfstudio 维护者官方发布且可直接装备的专用 MCP。搜索没有命中不等于不存在。CLI 原生执行最容易记录输入、版本、日志和退出码；MCP 只做调度外壳。

若后续选择 MCP，应先下载固定提交审阅。Blender MCP 需要 Blender 插件，并有任意 Python 执行和遥测相关设置；仍要逐项审查工具权限。本文不提供会写全局配置、安装系统包或自动启动服务的注册命令。[S17]

## 7. 验收、风险与未完成项

第一轮真实重建必须保存：输入照片/视频及控制点 SHA-256、采集条件、相机内参、软件提交/制品哈希、Python 包锁、CUDA/驱动、完整命令与退出码、相机注册结果、稀疏/稠密/网格/贴图、配准矩阵、留出检查点误差、未知空间掩码、碰撞包版本。视觉指标与几何指标分别报告。

| 验收项 | 本次状态 / 后续判据 |
|---|---|
| 既有资产结构 | 原脚本 `validate_obj_file` 检查 12 OBJ；结果与 12 份清单相同；17 JSON 解析通过；不是水密性或实测精度证明 |
| 新清单完整性 | CSV/JSON 解析、必需列与来源 ID 对应关系须通过最终检查 |
| 原文件保护 | 对 4 个已读入口与 45 个既有预览资产比对 SHA-256 |
| 环境可运行 | 尚未安装/编译/训练；PATH 探测不等于依赖验收 |
| 几何校准 | 待取得多视角原片、控制点及目标精度；没有填写虚构变换或精度 |
| 碰撞接入 | 待实现独立适配器、保守近似/未知区/薄结构与跨高度线段测试 |
| 真实照片重建门禁 M1 | `未执行`：需取得多视角原片、控制点和目标精度后，单独执行小样本重建与几何验收；本次未宣称工具链已通过 |

主要风险是弱纹理/玻璃/水面、树叶及人车运动、滚动快门、细电线漏重建、背面未观测、跨分区坐标错位、归一化尺度遗失和数据/贴图许可。许可证以锁定版本和依赖为准：OpenMVS/ODM 的 AGPL 要审查部署分发方式；原版 3DGS 和 Instant-NGP 的研究用途限制不能被上层 wrapper 消除。软件许可、照片权利、预训练权重许可分别核验。[S04][S12][S14][S16]

`math-modeling` 使用说明：本资料沿用其证据、量纲、假设和质检要求。用户明确指定本任务交付 README/CSV/来源清单，因此以用户要求覆盖该 Skill 默认“两份建模报告”文件名约束；本轮不进入编程/论文阶段，不额外生成九幅图、论文或执行论文双引擎检索。官方项目文档是此次工具链事实的主要证据。

实际执行的关键只读命令：`Get-Content`/`rg`，`nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv`（退出码 0），对版本锚点中的 7 个仓库执行 `git ls-remote`（提交/标签均匹配），`python -B` 读取并解析原有业务脚本与交付 JSON/CSV。一次 CSV 列数记录错误已修正为 14 列；未执行本文任何安装、抽帧、构建或训练命令。
