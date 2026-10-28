"""
模块01 — 环境安装与配置 / Module 01 — Environment Setup & Configuration

职责 / Responsibility:
    系统依赖安装、项目目录结构详解、四区场景数据配置和配置文件层级说明。
    System dependency installation, project directory structure details,
    four-zone scene data configuration, and configuration file hierarchy.

参考 / Reference:
    CLAUDE.md (编码规范、四区环境数据)
    config.py (WUHAN_ENV_CONFIG, WUHAN_SCENARIO_CONFIGS)
    outputs/环境示意图_2D审核/scene_2d_manifest.json (四区元数据)
    outputs/环境示意图_3D预览/scene_3d_index.json (3D场景索引)

领域适配 / Domain Adaptation:
    峡谷漫步v2 → 无人机配送:
    - 64x64固定地图 → 40x40网格 + 四区3D城市空域
    - 单场景配置 → 多分区(Wuhan A/B/C/D)场景切换
    - 2D离散网格 → 2D/3D混合仿真环境
"""

---

## 1. 环境安装与配置 / Environment Setup & Configuration

> **版本 / Version**: v2.0  
> **日期 / Date**: 2026-07-11  
> **前置模块 / Prerequisites**: 模块00（概述与快速开始）  
> **预计耗时 / Estimated Time**: 30-60分钟

---

### 1.1 系统要求与依赖安装 / System Requirements & Dependencies

#### 硬件最低要求 / Minimum Hardware Requirements

| 资源 | 最低要求 | 推荐配置 |
|---|---|---|
| CPU | 4核 | 8核+ |
| 内存 / RAM | 8 GB | 16 GB+ |
| 磁盘 / Disk | 2 GB 可用空间 | 5 GB+ (含模型检查点) |
| GPU | 不要求（L1/L2纯CPU可训） | NVIDIA GPU (L3/L4加速) |

#### 必需依赖表 / Required Dependencies

| 包名 | 最低版本 | 用途 / Purpose |
|---|---|---|
| `numpy` | >= 1.21 | 数值计算、数组操作 |
| `torch` | >= 1.12 | 神经网络、深度RL算法（DQN/A2C/PPO） |
| `gymnasium` | >= 0.26 | 环境接口标准（gym.Env兼容） |
| `toml` | >= 0.10 | TOML配置文件解析 |

#### 可选依赖表 / Optional Dependencies

| 包名 | 用途 / Purpose | 何时安装 |
|---|---|---|
| `tensorboard` | 训练曲线可视化 | 需要监控训练过程时 |
| `matplotlib` | 绘制奖励收敛曲线 | 需要生成论文图表时 |
| `pandas` | 实验数据整理 | 需要批量处理实验结果时 |
| `pytest` | 运行单元测试 | 需要验证代码正确性时 |
| `pillow` | 图像处理（可选） | 需要处理场景截图时 |

#### 版本兼容矩阵 / Version Compatibility Matrix

| Python | PyTorch | Gymnasium | NumPy | 状态 |
|---|---|---|---|---|
| 3.8 | 1.12.x | 0.26.x | 1.21.x | 兼容（最低） |
| 3.10 | 2.0.x | 0.29.x | 1.24.x | 推荐 |
| 3.11 | 2.1.x | 0.29.x | 1.26.x | 兼容 |
| 3.12 | 2.2.x | 1.0.x | 1.26.x | 兼容（最新） |

#### 安装步骤 / Installation Steps

```powershell
# 1. 创建虚拟环境 / Create virtual environment
cd "C:\University\Files\Training program\无人机项目设计\07_仿真与控制代码\Python仿真"
python -m venv venv
.\venv\Scripts\activate

# 2. 安装核心依赖 / Install core dependencies
pip install numpy torch gymnasium toml

# 3. 安装可选依赖 / Install optional dependencies
pip install tensorboard matplotlib pandas pytest

# 4. 验证安装 / Verify installation
python -c "import numpy; import torch; import gymnasium; import toml; print('All dependencies OK')"
```

> **GPU支持 / GPU Support**: 如需GPU加速，先安装CUDA版本的PyTorch：  
> `pip install torch --index-url https://download.pytorch.org/whl/cu118`

---

### 1.2 项目目录结构详解 / Project Directory Structure

#### 两套代码体系A/B区别 / Two Code Systems: A vs B

| 维度 | 体系A（现有成熟代码） | 体系B（Kaiwu式重构） |
|---|---|---|
| **根目录** | `07_仿真与控制代码/Python仿真/` | `07_仿真与控制代码/Python仿真/drone_delivery/` |
| **环境** | `environment/simple_delivery_env.py` | 由Agent内部conf/train_env_conf.toml配置 |
| **算法** | `algorithm/ppo_agent.py`, `classical_rl.py` | `agent_<algo>/algorithm/algorithm.py` |
| **配置** | `config.py`（全局Python dict） | `conf/*.toml` + `agent_<algo>/conf/conf.py` |
| **策略** | 不修改，作为终点参考 | 新功能在此开发 |
| **算法覆盖** | PPO、CTDE、经典RL（完整） | Q-learning、SARSA、MC、DQN、A2C、DIY |

#### drone_delivery/ 的6子模块标准结构 / 6-Submodule Standard Structure

每个Agent目录遵循腾讯开悟（Kaiwu）的标准结构：

```text
agent_<algo>/
├── __init__.py                — 包初始化
├── agent.py                   — Agent类（标准接口）
├── algorithm/
│   ├── __init__.py
│   └── algorithm.py           — Algorithm类（核心训练逻辑）
├── conf/
│   ├── __init__.py
│   ├── conf.py                — Config类（纯类变量，超参数）
│   ├── monitor_builder.py     — 监控面板配置
│   └── train_env_conf.toml    — 环境配置（TOML格式）
├── feature/
│   ├── __init__.py
│   ├── definition.py          — ObsData/ActData/SampleData + reward_shaping
│   └── preprocessor.py        — Preprocessor类（特征预处理）
├── model/
│   ├── __init__.py
│   └── model.py               — 神经网络/Q表模型
└── workflow/
    ├── __init__.py
    └── train_workflow.py      — workflow()函数 + EpisodeRunner类
```

#### 已完成的6个Agent目录 / 6 Completed Agent Directories

| Agent目录 | 算法 | 适用层级 | 状态 |
|---|---|---|---|
| `agent_q_learning/` | Q-learning | L1 | 已完成 |
| `agent_sarsa/` | SARSA | L1 | 已完成 |
| `agent_monte_carlo/` | Monte Carlo | L1 | 已完成 |
| `agent_dqn/` | DQN (Dueling) | L2 | 已完成 |
| `agent_a2c/` | A2C | L2/L3 | 已完成 |
| `agent_diy/` | DIY（空白模板） | 自定义 | 已完成（占位） |

#### outputs/ 图片资产索引 / outputs/ Image Asset Index

| 图片路径 | 内容 / Content |
|---|---|
| `outputs/环境示意图/wuhan_environment_overview.png` | 四区总览图 |
| `outputs/环境示意图/wuhan_environment_3d.png` | 3D环境展示 |
| `outputs/环境示意图/wuhan_building_height_heatmap.png` | 建筑高度热力图 |
| `outputs/环境示意图/wuhan_obstacle_legend.png` | 障碍物图例 |
| `outputs/环境示意图/wuhan_zone_top_view.png` | 四区分区图 |
| `outputs/环境示意图_2D审核/zone_A_guanggu_core_2d.png` | Zone A 2D审核图 |
| `outputs/环境示意图_2D审核/zone_B_lakeside_campus_2d.png` | Zone B 2D审核图 |
| `outputs/环境示意图_2D审核/zone_C_river_bridgehead_2d.png` | Zone C 2D审核图 |
| `outputs/环境示意图_2D审核/zone_D_canal_highrise_2d.png` | Zone D 2D审核图 |
| `outputs/环境示意图_3D预览/zone_A_scene_lod0_preview.png` | Zone A 3D预览 |
| `outputs/环境示意图_3D预览/zone_B_scene_lod0_preview.png` | Zone B 3D预览 |
| `outputs/环境示意图_3D预览/zone_C_scene_lod0_preview.png` | Zone C 3D预览 |
| `outputs/环境示意图_3D预览/zone_D_scene_lod0_preview.png` | Zone D 3D预览 |

---

### 1.3 四区场景数据配置 / Four-Zone Scene Data Configuration

#### scene_2d_manifest.json 说明 / scene_2d_manifest.json Description

2D审核清单记录了每个分区的几何统计、任务点坐标和来源声明。关键字段：

```json
{
  "schema_version": 3,
  "generated_utc": "2026-07-04T14:18:09.913797+00:00",
  "seed": 20260703,
  "zones": [
    {
      "zone_id": "A",
      "title": "分区 A | 环岛高密核心区",
      "extent_m": [2400.0, 1800.0],
      "feature_counts": {
        "building": 837,
        "road": 499,
        "railway": 14,
        "water": 4
      }
    }
  ]
}
```

#### scene_3d_index.json 说明 / scene_3d_index.json Description

3D场景索引记录了每个分区的OBJ/MTL文件、碰撞体JSON和风险区定义。

#### Zone B 作为主训练场景的原因 / Why Zone B is the Primary Training Scene

1. **建筑稀疏**：仅43栋，碰撞风险低，适合L1/L2/L3初期训练
2. **地形平坦**：武汉学院区域地势起伏小，简化3D物理建模
3. **电力线明确**：高压线数据完整，可训练避障能力
4. **参考地图可得**：校园公开参考资料充足，便于验证

#### 四区参数对比表 / Four-Zone Parameter Comparison

| 参数 / Parameter | Zone A | Zone B | Zone C | Zone D |
|---|---|---|---|---|
| 参考区域 | 光谷广场 | 武汉学院 | 长江大桥 | 楚河汉界 |
| 建筑数 | 837栋 | 43栋 | 538栋 | 592栋 |
| 范围(m) | 2400x1800 | 较小 | 中等 | 中等 |
| 任务点 | 1,2,3 | 4,5,6 | 7,8,9 | 10,11,12 |
| 基地坐标 | [408,324,0.8] | [1790,589,14.8] | [2139,1189,11.2] | [714,1047,6.8] |
| 适用层级 | L4多机 | L1/L2/L3 | L4扩展 | L4扩展 |
| 难度 | 高 | 低 | 中高 | 中高 |
| 特征 | 商圈、立交、密集高层 | 湖区、电力线、稀疏建筑 | 桥梁、江面、滨水区 | 窄河道、商业住宅混合 |

#### 任务点坐标表 / Task Point Coordinates

| 全局ID | 分区 | 名称 | 类型 | ENU坐标(m) |
|---|---|---|---|---|
| 1 | A | 配送中心 | 基地 | [408, 324, 0.8] |
| 2 | A | 商业裙楼屋顶 | 配送 | [1149, 1073, 15.8] |
| 3 | A | 商业综合体平台 | 配送 | [2013, 491, 9.2] |
| 4 | B | 校园后勤点 | 基地 | [1790, 589, 14.8] |
| 5 | B | 公共建筑屋顶 | 配送 | [945, 1011, 30.8] |
| 6 | B | 湖岸维护设施 | 配送 | [1160, 990, 6.8] |
| 7 | C | 桥头服务设施 | 基地 | [2139, 1189, 11.2] |
| 8 | C | 老城配送平台 | 配送 | [2363, 438, 11.2] |
| 9 | C | 滨水公共设施 | 配送 | [1929, 463, 12.4] |
| 10 | D | 商业裙楼屋顶 | 基地 | [714, 1047, 6.8] |
| 11 | D | 办公塔楼平台 | 配送 | [1249, 642, 138.7] |
| 12 | D | 社区公共建筑 | 配送 | [2028, 689, 47.7] |

*来源 / Source: `outputs/环境示意图_2D审核/scene_2d_manifest.json`*

---

### 1.4 配置文件层级 / Configuration File Hierarchy

#### 三层配置架构 / Three-Layer Configuration Architecture

```text
Layer 3 (最高优先级): agent_<algo>/conf/train_env_conf.toml  — 单算法环境配置
Layer 2 (中间层):     agent_<algo>/conf/conf.py               — 单算法超参数
Layer 1 (最低优先级): config.py                                — 全局默认配置
```

#### 配置优先级规则 / Configuration Priority Rules

```python
# 优先级：算法TOML > 算法conf.py > 全局config.py
# Priority: algo TOML > algo conf.py > global config.py

# 示例：学习率配置
# 1. 全局默认 / Global default (config.py)
PPO_CONFIG = {"learning_rate": 3e-4}

# 2. 算法配置类 / Algorithm config class (agent_<algo>/conf/conf.py)
class Config:
    LEARNING_RATE = 0.1  # 覆盖全局值

# 3. TOML配置（如扩展）/ TOML config (if extended)
# [algo_conf]
# learning_rate = 0.05  # 最高优先级
```

#### TOML vs Python Dict 使用场景 / TOML vs Python Dict Usage

| 场景 | 推荐格式 | 原因 |
|---|---|---|
| 环境参数（坐标、步数） | TOML | 无需改代码即可调参 |
| 算法超参数 | conf.py (Config类) | 代码级类型检查 |
| 全局默认值 | config.py (dict) | 统一管理、向后兼容 |
| 多实验对比 | TOML | 不同文件对应不同实验配置 |

#### TOML配置文件读取示例 / TOML Config File Reading Example

```python
import toml

# Read TOML configuration file
def read_toml_config(filepath):
    """
    读取TOML配置 / Read TOML configuration

    Args:
        filepath: 配置文件路径 / Config file path

    Returns:
        配置字典 / Configuration dictionary
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        config = toml.load(f)
    return config

# Example: Load Q-learning environment config
usr_conf = read_toml_config("agent_q_learning/conf/train_env_conf.toml")
env_conf = usr_conf.get("env_conf", {})
print(f"Depot: {env_conf['depot']}")
print(f"Max steps: {env_conf['max_steps']}")
```

#### 全局配置字典概览 / Global Config Dict Overview

`config.py` 中定义了6个全局配置字典：

| 配置字典 | 键数 | 用途 / Purpose |
|---|---|---|
| `ENV_CONFIG` | 12 | 3D环境基础参数（空域大小、步长、安全距离） |
| `UAV_CONFIG` | 5 | 无人机物理参数（速度、加速度、电池） |
| `PPO_CONFIG` | 12 | PPO算法超参数（学习率、GAE、裁剪） |
| `NETWORK_CONFIG` | 8 | 网络结构配置（隐藏层、激活函数、初始化） |
| `REWARD_CONFIG` | 9 | 奖励函数权重（8分量+边界惩罚） |
| `TRAIN_CONFIG` | 5 | 训练流程参数（回合数、间隔、种子） |
| `WUHAN_ENV_CONFIG` | 25 | 武汉场景专用配置（物理、计分、返航） |
| `WUHAN_SCENARIO_CONFIGS` | 6 | 六个场景变体（WA/WB/WC/WD/WA_MULTI/WB_MULTI） |

---

*模块01结束 / End of Module 01*  
*下一模块 / Next: 模块02 — 四层级问题框架 / Module 02 — Four-Level Problem Framework*
