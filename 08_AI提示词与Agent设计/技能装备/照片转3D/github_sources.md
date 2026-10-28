# 照片转 3D：来源与版本核查记录

核查日期：2026-09-17（Asia/Shanghai）；2026-09-20 复核。本文仅完成资料核查与方案设计，未安装或运行下述重建工具。

## 2026-09-20 复核记录

- 本地验收：`README.md`、`toolchain_matrix.csv`、`scene_contract.example.json` 均可读；CSV 为 8 行、14 列，JSON 可解析；已记录的既有业务入口 SHA-256 与本清单中的记录一致。
- 预览验收：实际查看 `outputs/环境示意图_3D预览/zone_B_scene_lod0_preview.png`；它是彩色建筑挤出、平面底图和审核走廊的开发预览，标题明确为“非测绘级”。
- 在线抽查：[COLMAP 官方输出格式](https://colmap.github.io/format.html)与[地理配准说明](https://github.com/colmap/colmap/blob/main/doc/faq.rst)支持本文关于输出文件和相似变换配准的表述；[Nerfstudio 坐标约定](https://github.com/nerfstudio-project/nerfstudio/blob/main/docs/quickstart/data_conventions.md)、[几何导出](https://docs.nerf.studio/quickstart/export_geometry.html)和[Splatfacto](https://docs.nerf.studio/nerfology/methods/splat.html)支持本文关于相机轴向、`ns-export` 以及 Gaussian splat 不直接等于碰撞网格的表述。
- 本次复核没有安装、编译、抽帧、训练或修改现有业务脚本；在线页面用于事实抽查，不替代锁定版本、依赖和本地运行验收。

“官方”指算法作者或项目维护方的一手仓库/文档；社区工具的作者仓库是一手来源，但不代表 Blender、COLMAP 等上游厂商认可。搜索摘要仅用于发现，结论优先依据正文、源码和许可证。README 与 CSV 的 `Sxx` 对应本文件。

## 一手来源

| 编号 | 项目及来源级别 | 可追溯链接 | 支撑的结论与版本注意点 |
|---|---|---|---|
| S01 | COLMAP；官方 | [仓库](https://github.com/colmap/colmap)、[4.2.0 发布](https://github.com/colmap/colmap/releases/tag/4.2.0)、[4.2.0 CLI 文档源码](https://github.com/colmap/colmap/blob/4.2.0/doc/cli.rst)、[安装](https://colmap.github.io/install.html)、[许可证](https://github.com/colmap/colmap/blob/4.2.0/COPYING.txt) | SfM/MVS、Windows 发行包、CPU/GPU 路径；BSD-3-Clause 主体许可。4.2.0 已有 `mesh_texturer`，不能再笼统声称 COLMAP 不支持纹理。滚动网站显示 4.3.0.dev0，不能把开发版功能全部套用到旧二进制。 |
| S02 | COLMAP；官方坐标与配准说明 | [输出格式](https://colmap.github.io/format.html)、[地理配准 FAQ](https://colmap.github.io/faq.html#geo-registration)、[4.2.0 model.cc](https://github.com/colmap/colmap/blob/4.2.0/src/colmap/exe/model.cc) | 外参是 world-to-camera；相机中心为 `-R^T t`；`model_aligner` 使用相机中心对应关系。核对了 `ref_is_gps`、`alignment_type`、`alignment_max_error`、`transform_path`。 |
| S03 | OpenMVG；官方 | [仓库](https://github.com/openMVG/openMVG)、[构建](https://github.com/openMVG/openMVG/blob/develop/BUILD.md)、[SfM 文档](https://openmvg.readthedocs.io/en/latest/software/SfM/SfM/)、[许可证](https://github.com/openMVG/openMVG/blob/develop/LICENSE) | SfM/相机标定和稀疏点云；MPL-2.0；支持跨平台 C++ 构建。导出 OpenMVS 的实际命令另见 S04。 |
| S04 | OpenMVS；官方作者仓库及 Wiki | [仓库](https://github.com/cdcseacave/openMVS)、[Usage](https://github.com/cdcseacave/openMVS/wiki/Usage)、[Building](https://github.com/cdcseacave/openMVS/wiki/Building)、[v2.4.0 发布](https://github.com/cdcseacave/openMVS/releases/tag/v2.4.0)、[许可说明](https://github.com/cdcseacave/openMVS/blob/v2.4.0/COPYRIGHT.md)、[许可证](https://github.com/cdcseacave/openMVS/blob/v2.4.0/LICENSE)、[TextureMesh 参数源码](https://github.com/cdcseacave/openMVS/blob/v2.4.0/apps/TextureMesh/TextureMesh.cpp) | 导入 COLMAP/OpenMVG；稠密化、网格、优化、纹理；CUDA 可选；AGPL-3.0。`--export-type obj` 已核对。Wiki 随主线变化，原生 SfM 等新功能不作为 v2.4.0 基线。 |
| S05 | Meshroom；官方 | [仓库](https://github.com/alicevision/Meshroom)、[2025.1.0 发布](https://github.com/alicevision/Meshroom/releases/tag/v2025.1.0)、[许可](https://github.com/alicevision/Meshroom/blob/develop/COPYING.md)、[安装](https://github.com/alicevision/Meshroom/blob/develop/INSTALL.md) | 图形节点工作流与批处理；MPL-2.0。2025.1.0 发布说明写明预构建包使用 CUDA 12、compute capability ≥5.0；具体节点和插件仍需与发布包匹配。 |
| S06 | AliceVision；官方 | [仓库](https://github.com/alicevision/AliceVision)、[构建选项](https://github.com/alicevision/AliceVision/blob/develop/INSTALL.md)、[许可](https://github.com/alicevision/AliceVision/blob/develop/COPYING.md) | Meshroom 的摄影测量后端；MPL-2.0；CUDA 构建与插件选项。典型密集深度管线依赖 NVIDIA CUDA，CPU draft 不等价于完整 MVS。 |
| S07 | Nerfstudio；官方 | [仓库](https://github.com/nerfstudio-project/nerfstudio)、[安装](https://docs.nerf.studio/quickstart/installation.html)、[自有数据](https://docs.nerf.studio/quickstart/custom_dataset.html)、[v1.1.5](https://github.com/nerfstudio-project/nerfstudio/tree/v1.1.5)、[许可](https://github.com/nerfstudio-project/nerfstudio/blob/v1.1.5/LICENSE)、[COLMAP 调用代码](https://github.com/nerfstudio-project/nerfstudio/blob/v1.1.5/nerfstudio/process_data/colmap_utils.py)、[复用位姿代码](https://github.com/nerfstudio-project/nerfstudio/blob/v1.1.5/nerfstudio/process_data/images_to_nerfstudio_dataset.py) | Apache-2.0；官方安装路径包含 CUDA/PyTorch/tiny-cuda-nn。包装器的旧 GPU 参数与 COLMAP 4.2 CLI 存在名称差异；建议 `--skip-colmap` 复用已生成的二进制位姿。 |
| S08 | Nerfstudio；官方几何导出 | [Export geometry](https://docs.nerf.studio/quickstart/export_geometry.html)、[导出源码](https://github.com/nerfstudio-project/nerfstudio/blob/v1.1.5/nerfstudio/scripts/exporter.py) | NeRF 的点云、TSDF、Poisson 与纹理导出；Poisson 示例要求训练 normals。通用文档对“all models”的描述不能覆盖 Splatfacto 专页的限制。 |
| S09 | Nerfstudio；官方坐标契约 | [Data conventions](https://docs.nerf.studio/quickstart/data_conventions.html)、[dataparser 源码](https://github.com/nerfstudio-project/nerfstudio/blob/v1.1.5/nerfstudio/data/dataparsers/nerfstudio_dataparser.py) | OpenGL/Blender 相机约定；`applied_transform`、`applied_scale` 和 dataparser 自动定向、平移、缩放需要追踪，导出结果不能直接假定为米制 ENU。 |
| S10 | Splatfacto/gsplat；官方 | [Splatfacto 专页](https://docs.nerf.studio/nerfology/methods/splat.html)、[文档源码](https://github.com/nerfstudio-project/nerfstudio/blob/main/docs/nerfology/methods/splat.md)、[gsplat 仓库](https://github.com/nerfstudio-project/gsplat)、[许可](https://github.com/nerfstudio-project/gsplat/blob/main/LICENSE)、[Windows 构建](https://github.com/nerfstudio-project/gsplat/blob/main/docs/INSTALL_WIN.md) | Splatfacto 默认约 6 GB、big 约 12 GB 是文档示例，非所有场景的上限；Gaussian PLY 导出；专页未支持通用 mesh/pointcloud 导出。gsplat 为 Apache-2.0 独立实现，CUDA 扩展可能首次运行才编译。 |
| S11 | 原始 3D Gaussian Splatting；论文作者官方实现 | [仓库](https://github.com/graphdeco-inria/gaussian-splatting)、[许可证正文](https://github.com/graphdeco-inria/gaussian-splatting/blob/main/LICENSE.md)、[环境](https://github.com/graphdeco-inria/gaussian-splatting/blob/main/environment.yml) | CUDA compute capability ≥7.0；论文质量训练标注 24 GB VRAM，小场景可以调低；研究/评估用非商用许可，商业使用须另行许可。不是 MIT。 |
| S12 | Instant-NGP；NVIDIA 官方研究仓库 | [仓库](https://github.com/NVlabs/instant-ngp)、[数据准备与归一化](https://github.com/NVlabs/instant-ngp/blob/master/docs/nerf_dataset_tips.md)、[转换脚本](https://github.com/NVlabs/instant-ngp/blob/master/scripts/colmap2nerf.py)、[许可证](https://github.com/NVlabs/instant-ngp/blob/master/LICENSE.txt) | Windows/Linux、NVIDIA CUDA、CMake ≥3.21；NeRF/快照/等值面。NVIDIA 自定义研究评估许可；`scale`、`offset`、`aabb_scale` 影响坐标。`--run_colmap` 路线在 Windows 可能自动下载依赖，本方案避开该选项。 |
| S13 | 2D Gaussian Splatting；论文作者官方实现 | [仓库](https://github.com/hbb1/2d-gaussian-splatting)、[许可证](https://github.com/hbb1/2d-gaussian-splatting/blob/main/LICENSE.md) | 面向几何的 surfel/深度与 TSDF 网格；提供 `train.py`、`render.py`；沿用非商用 Gaussian-Splatting 许可。适合研究对照，不保证生成闭合碰撞体。 |
| S14 | VGGT；Meta 官方研究仓库 | [仓库](https://github.com/facebookresearch/vggt)、[许可证](https://github.com/facebookresearch/vggt/blob/main/LICENSE.txt)、[商业许可权重入口](https://huggingface.co/facebook/VGGT-1B-Commercial) | 多图预测相机、深度、点图并可导出 COLMAP；README 2025-07-29 更新：代码与新商业 checkpoint 的条款有别于原始非商用权重。不能统一写成 CC-BY-NC 或 MIT，也不能把学得尺度当作测量标尺。 |
| S15 | OpenDroneMap；官方 | [ODM 仓库](https://github.com/OpenDroneMap/ODM)、[输出](https://docs.opendronemap.org/outputs/)、[GCP](https://docs.opendronemap.org/gcp/)、[许可](https://github.com/OpenDroneMap/ODM/blob/master/LICENSE) | 航拍影像→地理点云、纹理模型、正射、DSM/DTM；Docker 与原生 Windows 路径；AGPL-3.0。GCP/CRS 支持使其更适合航空摄影测量；树冠影像不能自动提供可靠林下地面。 |
| S16 | Open3D；官方 | [仓库](https://github.com/isl-org/Open3D)、[许可](https://github.com/isl-org/Open3D/blob/main/LICENSE)、[距离/占据查询](https://www.open3d.org/docs/release/tutorial/geometry/distance_queries.html)、[安装](https://www.open3d.org/docs/release/getting_started.html) | MIT；点云处理、网格、TSDF、RaycastingScene；本方案先用 CPU。signed distance/occupancy 的内外判断需要水密且内外定义清楚的网格，不能盲目用于开口城市表面。 |
| S17 | MCP for Blender；第三方集成的作者一手仓库，非 Blender 官方 | [现仓库](https://github.com/ahujasid/mcp-for-blender)、[旧址重定向](https://github.com/ahujasid/blender-mcp)、[固定提交的包入口](https://github.com/ahujasid/mcp-for-blender/blob/6f992ffbca3cb715d111fc640b737b808632273c/pyproject.toml)、[许可](https://github.com/ahujasid/mcp-for-blender/blob/main/LICENSE) | MIT；Python ≥3.10、Blender ≥3.0；包/命令现为 `mcp-for-blender`，版本文件 2.0.0。可执行 Blender Python、编辑/导入场景；本身不完成 SfM。默认遥测及外部生成服务需显式处理。 |
| S18 | NuRec skills；NVIDIA 官方技能仓库 | [仓库](https://github.com/NVIDIA/nurec-skills)、[路由 skill](https://github.com/NVIDIA/nurec-skills/blob/main/skills/nurec-index/SKILL.md)、[NRE skill](https://github.com/NVIDIA/nurec-skills/blob/main/skills/nre/SKILL.md)、[Apache 许可](https://github.com/NVIDIA/nurec-skills/blob/main/LICENSE)、[CC-BY 许可](https://github.com/NVIDIA/nurec-skills/blob/main/LICENSE-CC-BY-4.0) | 真实可读取的技能包：ncore 数据转换、NRE 训练/导出/渲染。文档为 CC-BY-4.0 AND Apache-2.0，代码文件 Apache-2.0，上游容器/权重独立许可。NRE skill 标明 Linux、CUDA 12.8、≥24 GB 显存（建议 48+）、NGC 账号与容器工具链。 |
| S19 | GeoAI point-cloud-lidar；社区技能的作者一手仓库 | [仓库](https://github.com/muend/geoai-skills)、[skill 正文](https://github.com/muend/geoai-skills/blob/main/skills/point-cloud-lidar/SKILL.md)、[许可](https://github.com/muend/geoai-skills/blob/main/LICENSE) | MIT；点云质量、CRS、垂直基准和元数据工作流。可辅助审核摄影测量点云；不是照片重建引擎，也不能将 SfM 云视为 LiDAR。 |
| S20 | FFmpeg；官方文档 | [fps 滤镜](https://ffmpeg.org/ffmpeg-filters.html#fps-1)、[项目许可说明](https://ffmpeg.org/legal.html) | 视频抽帧；2 fps 是本项目试验起点，不是官方重建质量标准。具体发行包许可证受构建选项影响。 |

## 已解析的版本锚点

以下哈希通过 `git ls-remote` 只读取得。HEAD 锚点用于复核本次观察，不代表稳定版或通过本机集成测试。git clone 后仍须记录依赖锁、子模块、模型权重摘要和实际二进制版本。

| 仓库 | 引用 | 实际 SHA |
|---|---|---|
| colmap/colmap | refs/tags/4.2.0 | `be5e29168d4aff238409d60424812df66aac919f` |
| cdcseacave/openMVS | refs/tags/v2.4.0 | `58117204c86bbb11a0b25b26a8987676cf11274d` |
| cdcseacave/openMVS | HEAD | `e23d25f0a9bd93ec056d4ee7fae2b2aaee80e566` |
| nerfstudio-project/nerfstudio | refs/tags/v1.1.5 | `6b60855003011b2ca23c2fe3f8e2ca6314c69924` |
| nerfstudio-project/nerfstudio | HEAD | `50e0e3c70c775e89333256213363badbf074f29d` |
| NVlabs/instant-ngp | HEAD | `abe236ee00cf90cfca6e36e65c00435d5b21f50a` |
| graphdeco-inria/gaussian-splatting | refs/heads/main | `54c035f7834b564019656c3e3fcc3646292f727d` |
| NVIDIA/nurec-skills | HEAD | `5b9d287eb7d271147751a119aa490f41edc90363` |
| ahujasid/mcp-for-blender | HEAD | `6f992ffbca3cb715d111fc640b737b808632273c` |
| muend/geoai-skills | HEAD | `096e5d4e6825a128e376b017783ee4c8c7323f9b` |

## 搜索覆盖与未采纳结果

实际检索包括 `COLMAP official github installation license`、`OpenMVS OpenMVG official github`、`Meshroom AliceVision requirements`、`site:github.com photogrammetry MCP COLMAP`、`site:github.com photogrammetry "SKILL.md"`、`site:github.com nerfstudio mcp server`、`blender-mcp Hyper3D`。没有在本轮检索中验证到由 COLMAP/OpenMVG/OpenMVS/Nerfstudio 官方维护、可直接推荐的专用 MCP 服务；这不是“此类服务不存在”的断言。

- 二手教程、Reddit 安装帖和 `github-wiki-see.page` 镜像只作发现线索，未作为许可证、硬件门槛或命令结论的依据；优先改用官方 Wiki 原址。
- [newo-ether/blender-mcp](https://github.com/newo-ether/blender-mcp)、[soozs1/Blender-MCP](https://github.com/soozs1/Blender-MCP) 是社区变体；未做代码安全审核，不用其宣传代替上游证据。
- [角色照片转 3D skill](https://github.com/laxmimerit/Claude-Skills/blob/main/character-image-to-3d/SKILL.md) 面向角色建模，与米制城市空域碰撞重建不匹配，未推荐。
- GitHub REST API 在本机返回 403 rate limit；改用官方网页、raw 源码和 `git ls-remote` 核查，未杜撰 API 查询结果。尝试的 COLMAP legacy/4.2 文档地址未成功打开，已以 tag 内文档源码替代。Meshroom develop 旧 batch 脚本路径返回 404，批命令参考明确标注版本的官方手册。

## 本项目实物证据

相对路径均以项目根目录为基准，以下是本机检查结果，不是网络来源。

| 文件 | 证据 |
|---|---|
| `03_图表脚本/Python脚本/README_2D转3D.md` | 明确“矢量特征不是 PNG 像素”；LOD 与 DEM 输入契约。SHA256 `7DB79324788F8A7E66A6D9AE11A06EFCDD3CC7F16B52734FA03BBD7B50ACEC65`。 |
| `03_图表脚本/Python脚本/convert_wuhan_reference_scenes_2d_to_3d.py` | `build_zone_scene`、`collision_manifest`、`scene_metadata` 与 CLI；独立碰撞体、零 DEM 回退、研究示意准确度标记。SHA256 `90303A39A7B8566EA3B3F5D35C8380215AB5E4D804DADE4D598AEB08C075B264`。 |
| `03_图表脚本/Python脚本/plot_wuhan_reference_scenes_2d.py` | `lonlat_to_local`（447 行起）、`EARTH_RADIUS_M=6371008.8`、B 区中心与尺寸；球面近似投影后平移，不是严格椭球 ENU。 |
| `outputs/环境示意图_3D预览/scene_3d_index.json` | A/B/C/D 各 LOD0/1/2 共 12 条，全部 `flat_no_dem_input`；现有产物生成时间与本次核查时间不同。 |
| `outputs/环境示意图_3D预览/zone_B_scene_lod0.json` | 2400×1800 m；43 建筑；6–56 m 建筑高度；181 对象、6535 顶点、3421 三角面；研究示意非测绘；审核路线不是真实路线。 |
| `outputs/环境示意图_3D预览/zone_B_scene_lod0_preview.png` | 已打开人工视觉检查：彩色建筑挤出、平面底图、两级审核走廊；标题明确非测绘级。 |
| `07_仿真与控制代码/Python仿真/environment/wuhan_scene.py` | `ScenePackage.load`（337 行起）读取 manifest 与 collision JSON，不读 OBJ/PLY；`PrismCollider` 只读取外环，`static_collision` 查询棱柱与胶囊。SHA256 `7613307B614A847333290F53EEE95F8551A86496E268706CA5DC9EE358BFD9B0`。 |

当前主机只读检查：Windows/PowerShell；RTX 4060 Laptop GPU，8188 MiB 显存，驱动 596.08；物理内存约 15.7 GiB；PATH 可见 Git、Python 3.9 路径、Conda、WSL。PATH 未找到 COLMAP、FFmpeg、CMake，仅说明当前会话无法按这些名字调用。WSL 列出 Ubuntu、podman-machine-default、docker-desktop；未验证这些环境中的 CUDA、依赖及服务运行状态。
