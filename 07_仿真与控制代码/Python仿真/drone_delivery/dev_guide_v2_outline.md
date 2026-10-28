# 开发指南v2框架大纲 / Development Guide v2 Framework Outline

> **版本 / Version**: v2.0-draft
> **日期 / Date**: 2026-07-11
> **设计者 / Designer**: DocumentAgent
> **预计总行数 / Estimated Lines**: 4300-5000行（12模块 + 附录）
> **预计总字数 / Estimated Words**: ~130,000字（含代码）

---

## 设计理念 / Design Philosophy

v2采用**模块化子文档架构**，将v1.0的单一731行文档扩展为12个模块的完整开发指南。主文档`dev_guide_v2.md`作为导航枢纽和内容容器，通过占位符标记每个模块的位置。

**核心改进点**：
1. API文档全面采用Kaiwu四段式（函数签名 / Parameters / Returns / 不变量）
2. 每个算法独立成节，包含原理+公式+代码+监控指标
3. 补充v1.0完全缺失的6子模块开发流程详解
4. 新增实验设计、科研绘图、代码规范三个独立模块
5. 从PPO指南迁移L3/L4环境说明、奖励函数设计、课程学习

---

## 参考文档索引 / Reference Document Index

| 代号 | 文件路径 | 行数 | 用途 |
|---|---|---:|---|
| [KW] | `data/reference/kaiwu_gorge_walk_v2/dev_guide_optimized.md` | 1859 | Kaiwu风格基准 |
| [DG] | `drone_delivery/dev_guide.md` (v1.0) | 731 | 现有指南，待完善 |
| [PPO] | `outputs/武汉光谷无人机配送_PPO开发指南.md` (v3.0) | 3415 | L3/L4技术内容 |
| [RF] | `reward_function_design.md` | 1723 | 奖励函数理论设计 |
| [CFG] | `config.py` | 123 | 已有配置参数 |
| [ALG] | `06_模型与算法/算法文档/` | ~3000 | 18篇算法设计文档 |
| [KV] | `data/reference/kaiwu_infinity_valley/` | — | 高级算法参考代码 |
| [KA] | `14_大赛准备/Kaiwu结构分析.md` | 733 | 结构分析报告 |
| [PG] | `outputs/无人机配送_渐进式开发指南.md` | 1471 | 渐进式开发参考 |

---

## 模块 0：概述与快速开始 / 00_overview.md

> **预计行数**: 225行 | **难度**: 低 | **优先级**: P2

### 内容摘要

项目背景与研究目标说明（无人机替代外卖骑手的调度优化问题），一句话策略（先规划可返航目标子集→按数量优先排序→PPO执行连续控制→每到达重规划→安全返航结算），30分钟快速上手指南（安装→冒烟测试→完整训练→查看结果），约30个核心术语的中英双语定义表，关键源码入口表（9个核心文件+职责+状态），项目目录结构树，3条推荐阅读路线。

### 子节设计

```
## 0.1 项目背景与研究目标 / Background & Goals
    - 外卖骑手工作模式类比
    - 无人机替代骑手的核心挑战（5项）
    - 项目最终交付物清单
    - 国家级大创立项定位

## 0.2 快速开始：30分钟跑通L1 / Quick Start: Run L1 in 30 Minutes
    - 步骤1: 安装依赖
    - 步骤2: 确认目录结构
    - 步骤3: 运行冒烟测试
    - 步骤4: 查看训练日志
    - 步骤5: 加载模型评估

## 0.3 术语表 / Glossary
    - RL基础术语（episode, reward, policy等）
    - 无人机领域术语（depot, waypoint, no-fly zone等）
    - 项目特有术语（CTDE, Kaiwu结构, 6子模块等）

## 0.4 关键源码入口 / Key Source Entry Points
    - 两套代码体系A/B说明
    - 9个核心文件入口表

## 0.5 项目目录结构 / Project Directory Structure

## 0.6 文档使用指南 / How to Use This Guide
    - 3条推荐阅读路线
    - 模块依赖关系图
    - 状态标记说明
```

### 内容来源映射

| 子节 | 主要来源 | 迁移方式 |
|---|---|---|
| 0.1 | [DG] §1.1-1.2 + CLAUDE.md 项目概述 | 扩写 |
| 0.2 | [KW] 注意事项 + [DG] §7.1 + CLAUDE.md 训练命令 | 组合编写 |
| 0.3 | [KW] 项目简介 + [DG] §1.1 + [ALG] 索引 | 全新编写 |
| 0.4 | [PPO] §0.5 关键源码入口 | 迁移+扩展 |
| 0.5 | [DG] §5.2 + CLAUDE.md 目录结构 | 迁移 |
| 0.6 | [KW] 文档结构 + [KA] §E1 | 全新编写 |

---

## 模块 1：环境安装与配置 / 01_environment_setup.md

> **预计行数**: 275行 | **难度**: 低 | **优先级**: P3

### 内容摘要

Python环境依赖清单（torch, numpy, gymnasium等），四区场景数据配置说明（scene_3d_index.json入口、LOD0/collision JSON），三层TOML配置文件体系（全局→Agent级→Python dict），场景数据加载流程（ScenePackage类），环境验证方法。

### 子节设计

```
## 1.1 系统要求与依赖安装 / System Requirements & Dependencies
    1.1.1 必需依赖 / Required
    1.1.2 可选依赖 / Optional
    1.1.3 版本兼容矩阵 / Version Compatibility Matrix

## 1.2 项目目录结构 / Project Directory Structure
    - 两套代码体系A/B的区别
    - drone_delivery/ 的6子模块标准结构
    - outputs/ 图片资产索引

## 1.3 四区场景数据配置 / Four-Zone Scene Data Configuration
    - scene_2d_manifest.json 读取方式
    - scene_3d_index.json 读取方式
    - Zone B 作为主训练场景的原因

## 1.4 配置文件层级 / Configuration File Hierarchy
    - 三层配置架构
    - 配置优先级规则
    - TOML vs Python dict 使用场景
```

### 内容来源映射

| 子节 | 主要来源 | 迁移方式 |
|---|---|---|
| 1.1 | [KW] 环境配置 + 渐进指南 §0.4 | 迁移 |
| 1.2 | [DG] §5.2 + CLAUDE.md | 扩写 |
| 1.3 | [DG] §1.2 + [PPO] §场景一览 | 组合 |
| 1.4 | [KW] §B4 + [CFG] config.py | 扩写 |

---

## 模块 2：四层级问题框架 / 02_four_levels.md

> **预计行数**: 375行 | **难度**: 中 | **优先级**: P1

### 内容摘要

L1-L4四个层级的完整环境设计说明。每层从Kaiwu峡谷漫步的领域映射出发，说明新增约束的必要性。L1覆盖Simple2DDeliveryEnv完整接口（40x40网格, 5固定订单, Discrete(4)）；L2覆盖Random2DDeliveryEnv（随机订单, 时间窗口, 多单携带, 8方向）；L3覆盖WuhanDroneDeliveryEnv简化版（3D状态166维, Box(3), 禁飞区, 碰撞检测, 能耗）；L4覆盖完整版（多机CTDE, Box(4), 风场, 通信限制）。包含层级递进Mermaid图和四层全量对比表。

### 子节设计

```
## 2.1 L1: 2D固定商家 / Level 1: 2D Fixed Merchants
    - 问题定义与数学模型（状态空间、动作空间、约束、目标函数）
    - Simple2DDeliveryEnv 环境接口
    - 状态编码公式
    - 地图布局与障碍物

## 2.2 L2: 2D随机商家 / Level 2: 2D Random Merchants
    - 新增约束说明
    - Random2DDeliveryEnv 接口
    - 50维向量化观测

## 2.3 L3: 3D固定商家 / Level 3: 3D Fixed Merchants
    - 2D→3D维度跃升
    - 3D状态空间166维分解
    - Box(3) 连续动作空间
    - 禁飞区、碰撞检测、能耗模型

## 2.4 L4: 3D随机+全约束 / Level 4: 3D Random + Full Constraints
    - 多机协同CTDE
    - Box(4) 连续动作空间
    - 风场、通信、动态禁飞

## 2.5 层级对比与升级路径 / Level Comparison & Upgrade Path
    - 四层级统一对比表
    - 渐进升级策略
    - 课程学习C1-C7映射
    - 12个全局任务点坐标表

## 2.6 环境示意图索引 / Environment Visualization Index
    - outputs/ 下13张图片索引
```

### 内容来源映射

| 子节 | 主要来源 | 迁移方式 |
|---|---|---|
| 2.1 | [DG] §2.1 + 渐进指南 §1.1-1.2 | 扩写 |
| 2.2 | [DG] §2.2 + 渐进指南 §2.1 | 扩写 |
| 2.3 | [PPO] 第3-7章 + [DG] §2.3 | 迁移+扩写 |
| 2.4 | [PPO] 第20章 + [DG] §1.2层级四 | 迁移 |
| 2.5 | [DG] §1.5 + CLAUDE.md + [PPO] §18 | 组合 |
| 2.6 | [DG] §2.4 环境示意图 | 迁移 |

---

## 模块 3：算法体系设计 / 03_algorithms.md

> **预计行数**: 550行 | **难度**: 中 | **优先级**: P0

### 内容摘要

六个算法的独立介绍（Q-learning/SARSA/Monte Carlo/DQN/A2C/PPO），每个算法包含原理说明、核心公式LaTeX、关键特性、代码示例、监控指标。六算法全量对比表和算法选择决策树。课程学习C1-C7七阶段完整设计。

### 子节设计

```
## 3.1 算法选择决策树 / Algorithm Selection Decision Tree
    - Mermaid决策流程图
    - 表格型 vs 深度算法适用场景

## 3.2 Q-Learning（L1表格型）
    - 算法原理（贝尔曼最优方程）
    - 核心公式 LaTeX
    - Double Q-Learning
    - 稀疏Q表, ε-greedy, 学习率调度
    - 代码示例, 监控指标

## 3.3 SARSA（L1表格型）
    - 与Q-learning的核心差异（on-policy vs off-policy）
    - 安全边际优势
    - 代码差异点

## 3.4 Monte Carlo（L1表格型）
    - 首次访问MC
    - 完整回报G_t, 偏差-方差权衡
    - 增量均值更新

## 3.5 DQN（L2深度）
    - Dueling Network架构
    - 经验回放 + 目标网络
    - Huber loss, 超参数推荐

## 3.6 A2C（L2/L3深度）
    - Actor-Critic架构
    - 优势函数 + GAE(λ)
    - 熵正则化

## 3.7 PPO（L3/L4深度）
    - tanh-squashed Gaussian策略
    - Clipped Surrogate目标
    - KL散度早停
    - MLP 166→256→256→128→16
    - CNN-MLP混合架构（可选）

## 3.8 算法对比总表 / Algorithm Comparison Summary
    - 6算法统一对比表（策略类型/动作空间/网络/超参数/优势/劣势/适用层级）
    - 雷达图描述（样本效率/稳定性/可扩展性/实现难度/探索能力）

## 3.9 课程学习设计（C1-C7）/ Curriculum Learning Design
    - 七阶段课程表
    - 阶段转换逻辑
    - 通过条件
```

### 内容来源映射

| 子节 | 主要来源 | 迁移方式 |
|---|---|---|
| 3.1 | [DG] §1.5 + [KW] 注意事项 + [KA] §C2 G17 | 组合编写 |
| 3.2 | [DG] §4.1 + [KW] §4 Q-Learning + [ALG] 算法伪代码 | 大幅扩写 |
| 3.3 | [DG] §4.2 + [KW] §4 SARSA | 扩写 |
| 3.4 | [DG] §4.3 + [KW] §4 Monte Carlo | 扩写 |
| 3.5 | [DG] §4.4 + [KW] §B3 + [ALG] 算法伪代码 | 扩写 |
| 3.6 | [DG] §4.5 + [KV] agent_a2c/ | 扩写 |
| 3.7 | [PPO] 第10-15章 + [KV] agent_ppo/ | 大量迁移 |
| 3.8 | [DG] §1.5 + [KA] §C2 G17 | 扩写 |
| 3.9 | [PPO] §18课程学习 + [RF] §6 | 迁移 |

---

## 模块 4：六子模块实现详解 / 04_six_modules.md

> **预计行数**: 650行 | **难度**: 高 | **优先级**: P0

### 内容摘要

Kaiwu标准Agent目录的6个子模块逐一详解。这是v1.0完全缺失的核心内容。每个子模块说明职责边界、关键类设计、核心函数列表、代码示例。数据流图（env.reset→observation_process→predict→action_process→env.step→reward_shaping→sample_process→learn）。

### 子节设计

```
## 4.1 子模块总览与数据流 / Submodule Overview & Data Flow
    - 6子模块目录树
    - 模块间依赖关系Mermaid图
    - 数据流向图Mermaid
    - 建议开发顺序（feature→model→algorithm→agent→workflow→conf）

## 4.2 agent.py: Agent类实现 / Agent Class Implementation
    - 类继承关系
    - 9个标准方法签名
    - __init__ 设计模式
    - 检查点版本管理
    - 日志安全访问模式
    - 四段式API文档

## 4.3 algorithm/: 算法逻辑模块 / Algorithm Logic Module
    - 三层架构（ReplayBuffer→Algorithm→SparseQTable/Model）
    - ReplayBuffer 实现
    - SparseQTable 惰性初始化
    - Algorithm类通用接口
    - 指标字典规范格式

## 4.4 conf/: 配置管理模块 / Configuration Management Module
    - Config类设计（纯类变量）
    - train_env_conf.toml
    - monitor_builder.py
    - 超参数命名规范

## 4.5 feature/: 特征工程模块 / Feature Engineering Module
    - definition.py（接口层：reward_shaping, sample_process, 10+辅助函数）
    - preprocessor.py（纯计算层：状态编码, 合法动作掩码, 路径规划）
    - Held-Karp精确TSP
    - 各层级特征维度对照表

## 4.6 model/: 模型定义模块 / Model Definition Module
    - 表格型模型（Q表/Numpy存储）
    - MLP模型模板（Kaiming init）
    - Actor-Critic双网络
    - PPO tanh-squashed Gaussian
    - save_model/load_model 工具函数

## 4.7 workflow/: 训练工作流模块 / Training Workflow Module
    - workflow() 完整训练循环
    - Episode + Step 循环设计
    - Epsilon衰减策略
    - 验证机制（每50ep×5次评估）
    - 收敛检测条件
    - 模型保存限速（最小30s间隔，最多100次）
    - 监控数据上报
    - 完整workflow代码示例（~100行可运行代码）
```

### 内容来源映射

| 子节 | 主要来源 | 迁移方式 |
|---|---|---|
| 4.1 | [KW] 第6章总述 + [DG] §5.3 + [KA] §B1 | 组合 |
| 4.2 | [KW] §3智能体详述 + [KA] §B2 | 大量迁移 |
| 4.3 | [KW] §B3 + [DG] §4 | 迁移+扩写 |
| 4.4 | [KW] §B4 + [CFG] config.py | 迁移 |
| 4.5 | [KW] §B5 + [RF] + [ALG] 状态空间设计 | 大量迁移 |
| 4.6 | [KW] §B6 + [PPO] 网络结构 + [KV] | 迁移 |
| 4.7 | [KW] §B7 + [KA] §B7 + [DG] §5.1 | 大量迁移 |

---

## 模块 5：数据协议规范 / 05_data_protocol.md

> **预计行数**: 225行 | **难度**: 中 | **优先级**: P1

### 内容摘要

三种数据类（ObsData/ActData/SampleData）的完整定义和字段说明。create_cls工厂函数的实现原理。terminated与truncated的语义区分和正确使用（自然终止不自举 vs 时间截断需自举）。数据流全景图（env→obs→predict→act→step→sample→learn）。

### 子节设计

```
## 5.1 create_cls 动态数据类 / Dynamic Data Classes
    - Kaiwu机制原理
    - 与dataclass/namedtuple对比
## 5.2 ObsData 规范 / ObsData Specification
    - feature字段（L1=int, L2=50-dim, L3=166-dim, L4=多机扩展）
    - legal_act字段（布尔掩码编码）
    - observation_process四段式API
## 5.3 ActData 规范 / ActData Specification
    - act字段（离散int vs 连续ndarray）
    - val字段（Critic状态价值）
    - action_process四段式API
## 5.4 SampleData 规范 / SampleData Specification
    - 6字段说明（obs, _obs, act, rew, done, ret）
    - ret计算（MC: G_t, TD: r+γQ, GAE: λ-return）
    - sample_process四段式API
## 5.5 terminated vs truncated 正确使用 / Correct Usage
    - 语义区分
    - sample_process中不同处理
    - 常见错误示例
## 5.6 数据流全景图 / Full Data Flow Diagram
    - Mermaid数据流图
    - 各层级完整数据流示例
```

### 内容来源映射

| 子节 | 主要来源 | 迁移方式 |
|---|---|---|
| 5.1 | [KA] §C2 G5 + [KW] L779-800 | 迁移 |
| 5.2 | [DG] §3.2 + [KA] §C2 G6 | 扩写 |
| 5.3 | [DG] §3.4 + [KA] §C2 G7 | 扩写 |
| 5.4 | [DG] §3.4 + [KA] §C2 G9 | 扩写 |
| 5.5 | CLAUDE.md 常见陷阱 + [KA] §C1 | 组合编写 |
| 5.6 | [KW] 第6章数据流 | 新编写 |

---

## 模块 6：配置文件格式详解 / 06_configuration.md

> **预计行数**: 275行 | **难度**: 中 | **优先级**: P2

### 内容摘要

TOML配置体系完整说明，Python Config类设计模式，算法路由配置（新增算法注册步骤），四层级推荐超参数表（学习率/折扣因子/ε衰减/batch_size等），环境配置速查表。奖励参数8分量详解和裁剪策略。

### 子节设计

```
## 6.1 TOML配置格式规范 / TOML Configuration Format
    - train_env_conf.toml 结构
    - 双语注释规范
    - 各层级TOML示例
## 6.2 Python配置类设计 / Python Config Class Design
    - Config类模板（纯类变量）
    - config.py全局配置字典
    - 超参数优先级
## 6.3 算法路由配置 / Algorithm Routing Configuration
    - algo_conf_drone_delivery.toml
    - 新增算法注册步骤
    - app_conf_drone_delivery.toml
## 6.4 各层级推荐超参数表 / Recommended Hyperparameters
    - L1/L2/L3/L4完整超参数推荐
    - 参数来源标注
## 6.5 环境配置速查表 / Environment Config Quick Reference
    - 四层级环境参数统一对比
    - WUHAN_ENV_CONFIG 说明
```

### 内容来源映射

| 子节 | 主要来源 | 迁移方式 |
|---|---|---|
| 6.1 | [KW] §B4 TOML + [DG] §2.5 | 迁移+扩写 |
| 6.2 | [CFG] config.py全部 + [KW] §B4 | 迁移 |
| 6.3 | [DG] §5.2 + [KW] §5代码包 | 扩写 |
| 6.4 | [CFG] + [DG] 附录C + [RF] §7 | 组合 |
| 6.5 | CLAUDE.md 四层级表 + [CFG] | 组合 |

---

## 模块 7：训练与评估流程 / 07_training.md

> **预计行数**: 325行 | **难度**: 低 | **优先级**: P1

### 内容摘要

冒烟测试、完整训练、对比实验的完整命令参考。评估模式（exploit确定性 vs predict含探索）和核心指标定义。检查点管理（保存限速、版本管理、断点续训）。监控面板与日志。8个常见故障排查。

### 子节设计

```
## 7.1 训练脚本使用说明 / Training Script Usage
    - train_test.py 冒烟测试
    - run_wuhan_training.py 完整训练
    - run_level1_experiments.py 对比实验
    - 断点续训方法
## 7.2 评估模式与指标 / Evaluation Mode & Metrics
    - exploit() vs predict() 区别
    - 核心指标定义（avg_deliveries, delivery_rate, avg_steps等）
    - 评估频率建议
## 7.3 模型保存与加载 / Model Save & Load
    - 保存格式（.npy / .pkl / .pth）
    - 保存限速策略
    - 断点续训checkpoint加载
## 7.4 监控面板与日志 / Monitoring & Logging
    - monitor_builder.py 配置
    - TensorBoard接入
    - 日志等级规范
## 7.5 常见故障排查 / Troubleshooting
    - 8个问题诊断+解决方案
```

### 内容来源映射

| 子节 | 主要来源 | 迁移方式 |
|---|---|---|
| 7.1 | [DG] §7.1 + CLAUDE.md + [KA] §D7 | 组合 |
| 7.2 | [DG] §2.5 + [KW] §评估模式 | 扩写 |
| 7.3 | [DG] §7.2-7.3 + [KA] §B2 + §C2 G13 | 扩写 |
| 7.4 | [KW] §8 + [KA] §C2 G16 | 迁移 |
| 7.5 | [DG] §8 + CLAUDE.md 陷阱 + [RF] §9.2 | 扩写 |

---

## 模块 8：实验设计指南 / 08_experiment.md

> **预计行数**: 325行 | **难度**: 中 | **优先级**: P2

### 内容摘要

对比实验设计（6算法×4层级，统一控制变量，N>=5重复），消融实验设计（奖励分量消融、网络结构消融、探索策略消融），超参数敏感度分析（网格搜索/随机搜索），奖励函数调参指南（4层分层标定、归一化方法、RewardMonitor），课程学习实验（7阶段配置、CurriculumManager使用）。

### 子节设计

```
## 8.1 对比实验设计 / Comparative Experiment Design
    - 统一控制变量清单
    - 评估指标定义
    - 统计检验方法（t-test/Wilcoxon）
    - 实验结果报告模板
## 8.2 消融实验设计 / Ablation Study Design
    - 奖励分量消融
    - 网络结构消融
    - 探索策略消融
    - 消融表格模板
## 8.3 超参数敏感度分析 / Hyperparameter Sensitivity Analysis
    - 搜索空间定义
    - 网格搜索/随机搜索/Bayesian
    - 可视化方案（热力图/平行坐标）
## 8.4 奖励函数调参指南 / Reward Function Tuning Guide
    - 4层分层标定策略
    - 归一化方法
    - RewardMonitor使用
    - 常见奖励问题诊断表
## 8.5 课程学习实验 / Curriculum Learning Experiments
    - C1-C7完整配置
    - CurriculumManager使用
    - 与固定难度对比实验
```

### 内容来源映射

| 子节 | 主要来源 | 迁移方式 |
|---|---|---|
| 8.1 | [ALG] 技术路线说明 + CLAUDE.md | 新编写 |
| 8.2 | [RF] §7参数标定 + [ALG] PPO设计 | 新编写 |
| 8.3 | [RF] §7.2敏感度分析 | 迁移 |
| 8.4 | [RF] §4-7 + §9 | 大量迁移 |
| 8.5 | [PPO] §18 + [RF] §6 + §8.3 | 迁移 |

---

## 模块 9：科研绘图指南 / 09_visualization.md

> **预计行数**: 275行 | **难度**: 低 | **优先级**: P3

### 内容摘要

训练收敛曲线绘制方法（平滑窗口、置信带、多算法对比），算法对比图表（柱状图、箱线图、雷达图、热力图），环境示意图使用（已有13张图片索引），训练过程可视化（Q表增长、ε衰减、loss曲线），图表导出规范（期刊PDF/SVG、答辩PPT、论文单/双栏）。

### 子节设计

```
## 9.1 学习曲线绘制 / Learning Curve Plotting
    - 标准绘制方法（EMA平滑+置信带）
    - matplotlib/seaborn代码模板
    - SCI/Nature标准规范
## 9.2 算法对比图表 / Algorithm Comparison Charts
    - 柱状图、箱线图、雷达图、热力图
    - 统计显著性标注
    - 配色方案（色盲友好）
## 9.3 环境示意图使用 / Environment Visualization Usage
    - outputs/ 13张图片索引
    - 各图使用场景
    - 引用格式
## 9.4 训练过程可视化 / Training Process Visualization
    - Q表增长曲线、ε衰减曲线
    - 奖励分量贡献堆叠面积图
    - 关键事件标注
## 9.5 图表导出规范 / Chart Export Standards
    - 期刊投稿格式
    - 答辩PPT格式
    - 文件命名规范
```

### 内容来源映射

| 子节 | 主要来源 | 迁移方式 |
|---|---|---|
| 9.1 | 项目绘图脚本 + 科研通用规范 | 新编写 |
| 9.2 | [ALG] 算法设计完成总结 + 通用规范 | 新编写 |
| 9.3 | [DG] §2.4 + outputs/ 索引 | 迁移 |
| 9.4 | [KW] §8监控 + [RF] §9.1 | 迁移 |
| 9.5 | 科研通用规范 | 新编写 |

---

## 模块 10：代码规范与最佳实践 / 10_coding_standards.md

> **预计行数**: 225行 | **难度**: 低 | **优先级**: P3

### 内容摘要

双语注释规范（英文在上中文在下），命名规范（Agent目录/类名/函数/数据类/配置类/TOML键），文件头三栏格式（职责/参考/领域适配），API文档四段式规范，terminated vs truncated严格区分，业务Score与训练Reward分离原则。

### 子节设计

```
## 10.1 双语注释规范 / Bilingual Comment Convention
    - 代码注释格式
    - 文件头三栏
    - docstring规范
    - TOML注释格式
## 10.2 命名规范与目录结构 / Naming Conventions
    - 目录/类/函数/数据类/配置/TOML键
    - 新增Agent目录创建清单
## 10.3 API文档四段式规范 / Four-Part API Documentation
    - 函数签名 + Parameters + Returns + 不变量
    - 表格格式规范
## 10.4 terminated vs truncated 严格区分 / Handling
    - Gymnasium标准定义
    - sample_process处理逻辑
    - 常见错误+修正
## 10.5 业务Score与训练Reward分离 / Separation
    - 字典序验收 vs 密集训练
    - 验收评分公式
    - 调试方法
```

### 内容来源映射

| 子节 | 主要来源 | 迁移方式 |
|---|---|---|
| 10.1 | CLAUDE.md 编码规范 + [KA] §A2 A7 | 迁移 |
| 10.2 | CLAUDE.md 命名规范 + [KA] §B1 | 迁移 |
| 10.3 | [KA] §A4 API四段式 + [KW] 全文 | 迁移 |
| 10.4 | CLAUDE.md 陷阱 + [DG] §2.1-2.2 | 组合 |
| 10.5 | CLAUDE.md 陷阱 + [DG] §1.4 + [RF] | 组合 |

---

## 模块 11：API参考文档 / 11_api_reference.md

> **预计行数**: 500行 | **难度**: 高 | **优先级**: P0

### 内容摘要

所有公共API的四段式文档（函数签名/Parameters/Returns/不变量）。覆盖环境API（Simple2DDeliveryEnv + Random2DDeliveryEnv的reset/step/get_state_key/close/render）、Agent接口（9个标准方法）、特征处理API（observation_process/action_process/reward_shaping/sample_process + preprocessor辅助函数）、算法核心API（Algorithm/ReplayBuffer/SparseQTable）、工作流API（workflow/CurriculumManager/SampleManager/RewardMonitor）、配置与工具API。每个API包含使用示例。附录（环境数据来源、领域映射表、配置速查、算法文档索引、版本历史）。

### 子节设计

```
## 11.1 环境API / Environment API
    - Simple2DDeliveryEnv（__init__, reset, step, get_state_key, close, render）
    - Random2DDeliveryEnv（差异字段）
    - WuhanDroneDeliveryEnv（概述+差异）
    - 每个API四段式
## 11.2 Agent接口API / Agent Interface API
    - 9个标准方法完整四段式
    - reset/predict/exploit/learn/save_model/load_model
    - observation_process/action_process/reward_shaping
## 11.3 特征处理API / Feature Processing API
    - observation_process/action_process/reward_shaping/sample_process
    - preprocessor.py辅助函数
    - 计算复杂度标注
## 11.4 算法核心API / Algorithm Core API
    - Algorithm类接口
    - ReplayBuffer push/sample/__len__
    - SparseQTable get/set/get_max_action
    - 指标字典格式
## 11.5 工作流API / Workflow API
    - workflow() 主函数
    - SampleManager/CurriculumManager/RewardMonitor
## 11.6 配置与工具API / Configuration & Utility API
    - Config类变量列表
    - create_cls/reward_calculator/reward_normalizer
## 附录A-E
    - A: 环境数据来源
    - B: 领域映射表（14项完整版）
    - C: 配置参数速查
    - D: 算法文档索引（18篇）
    - E: 版本历史
```

### 内容来源映射

| 子节 | 主要来源 | 迁移方式 |
|---|---|---|
| 11.1 | [DG] §6 + [KW] §环境详述 + [KA] §C2 G1 | 大幅扩写 |
| 11.2 | [KW] §3 + [KA] §B2 + [KA] §C2 G2-G4 | 大量迁移 |
| 11.3 | [KW] §6 + [KA] §B5 + [KA] §C2 G6-G9 | 迁移 |
| 11.4 | [KW] §B3 + [DG] §4 + [KA] §C2 G11 | 迁移 |
| 11.5 | [KW] §6 + [KA] §B7 + [RF] §8.1-8.3 | 迁移 |
| 11.6 | [CFG] + [RF] §8 + [KW] §B4 | 组合 |
| 附录 | [DG] 附录A-D + CLAUDE.md | 迁移 |

---

## 行数预估汇总 / Line Count Summary

| 模块 | 文件名 | 预计行数 | 占比 | 难度 | 优先级 | 主要内容来源 |
|---|---|---:|---:|---|---|---|
| 0 | 00_overview.md | 225 | 5.2% | 低 | P2 | [DG]迁移 + [PPO]迁移 |
| 1 | 01_environment_setup.md | 275 | 6.4% | 低 | P3 | [KW]迁移 + [CFG] |
| 2 | 02_four_levels.md | 375 | 8.7% | 中 | P1 | [DG]扩写 + [PPO]迁移 |
| 3 | 03_algorithms.md | 550 | 12.8% | 中 | P0 | [DG]扩写 + [PPO]迁移 |
| 4 | 04_six_modules.md | 650 | 15.1% | 高 | P0 | [KA]迁移（全新内容） |
| 5 | 05_data_protocol.md | 225 | 5.2% | 中 | P1 | [DG]扩写 + [KA]迁移 |
| 6 | 06_configuration.md | 275 | 6.4% | 中 | P2 | [CFG]迁移 + [PPO]迁移 |
| 7 | 07_training.md | 325 | 7.6% | 低 | P1 | [DG]扩写 + [PPO]迁移 |
| 8 | 08_experiment.md | 325 | 7.6% | 中 | P2 | [RF]迁移（全新框架） |
| 9 | 09_visualization.md | 275 | 6.4% | 低 | P3 | 全新编写 |
| 10 | 10_coding_standards.md | 225 | 5.2% | 低 | P3 | CLAUDE.md迁移 |
| 11 | 11_api_reference.md | 500 | 11.6% | 高 | P0 | [DG]扩写 + [KA]迁移 |
| 附录 | (在11中) | 75 | 1.7% | 低 | — | [DG]迁移 |
| **合计** | | **~4300** | **100%** | | | |

---

## 优先级与工期排序 / Priority & Effort Order

| 优先级 | 模块 | 理由 | 预计工期 |
|---|---|---|---|
| P0 | 4 六子模块实现详解 | 核心开发指导，缺此不可编码 | 2天 |
| P0 | 3 算法体系设计 | 六算法是项目核心差异 | 1.5天 |
| P0 | 11 API参考文档 | 可直接复用已有内容 | 1天 |
| P1 | 2 四层级问题框架 | 项目框架定义 | 1天 |
| P1 | 5 数据协议规范 | 开发必备知识 | 0.5天 |
| P1 | 7 训练与评估流程 | 可直接复用 | 0.5天 |
| P2 | 0 概述与快速开始 | 入口文档 | 0.5天 |
| P2 | 6 配置文件格式 | 大量可从config.py迁移 | 0.5天 |
| P2 | 8 实验设计指南 | 策划书支撑 | 1天 |
| P3 | 1 环境安装与配置 | 相对独立 | 0.5天 |
| P3 | 9 科研绘图指南 | 辅助文档 | 0.5天 |
| P3 | 10 代码规范 | 可从CLAUDE.md迁移 | 0.5天 |

**总预计工期：~10天（1人全职）**

---

## 与v1.0的内容对应关系 / v1.0 Content Migration Map

| v1.0章节 | v1.0行号 | v2模块 | v2文件 | 迁移方式 |
|---|---|---|---|---|
| §1 项目介绍 | L12-175 | 模块0 + 模块2 | 00 + 02 | 拆分+扩写 |
| §2 环境说明 | L177-308 | 模块2 + 模块1 | 02 + 01 | 拆分+扩写 |
| §3 Agent说明 | L310-406 | 模块4 + 模块5 + 模块10 | 04 + 05 + 10 | 拆分+大幅扩写 |
| §4 算法介绍 | L410-474 | 模块3 | 03 | 大幅扩写 |
| §5 框架概述 | L477-530 | 模块4 + 模块7 | 04 + 07 | 拆分 |
| §6 环境API | L533-600 | 模块11 | 11 | 大幅扩写 |
| §7 训练评估保存 | L604-639 | 模块7 | 07 | 扩写 |
| §8 常见故障 | L643-670 | 模块7 + 模块10 | 07 + 10 | 拆分+扩写 |
| 附录A-D | L673-730 | 模块11附录 | 11 | 迁移 |

---

## 写作规范 / Writing Conventions

1. **全文双语标题**：`## 5. 框架概述 / Framework Overview`
2. **代码块**：使用 ` ```python ``` ` 标准Fenced代码块
3. **表格**：管道符双语表头 `| 中文 / English |`，左对齐文字，右对齐数字
4. **API文档**：四段式（函数签名 / Parameters / Returns / 不变量）
5. **图片引用**：使用项目根目录相对路径，引用`outputs/`下已有图片
6. **警告提示**：`> **注意 / Note**: ...` 格式
7. **代码注释**：英文在上，中文在下（`# English comment` + `# 中文注释`）
8. **状态标记**：`已实现` / `生产增强` / `历史基线`
9. **占位符**：`<!-- PLACEHOLDER_MODULE_X_Y -->` 标记待编写位置

---

*本大纲基于对9份参考文档（共约12,000行）的深度分析，覆盖Kaiwu风格基准、现有v1.0指南、PPO开发指南、奖励函数设计、配置文件、算法文档库、渐进式开发指南和Kaiwu结构分析报告。*
*Outline based on deep analysis of 9 reference documents (~12,000 lines total). Generated by DocumentAgent on 2026-07-11.*
