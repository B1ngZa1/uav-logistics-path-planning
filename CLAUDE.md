# CLAUDE.md — 无人机配送项目 Claude Code 指南

## 项目概述

本项目是大学生创新训练计划项目，研究**基于强化学习的无人机外卖配送调度系统**，目标是替代外卖骑手的生态位。

核心任务是开发指南的设计，任务目标是策划书的撰写。

---

## 当前工作目录

**项目根目录**：`c:\University\Files\Training program\无人机项目设计\`

所有文件路径均以此为基准。以下是根目录下的完整结构：

```text
项目根目录/
├── 00_项目管理/           — 项目计划、进度跟踪
├── 01_参考文献/           — 40+ 篇论文、知识库
├── 02_数据保存/           — 模型检查点、训练日志
├── 03_图表脚本/           — 绘图脚本、Mermaid图
├── 04_脚本图片输出/       — 训练曲线图、环境视图、3D渲染
├── 05_正文版本保存/       — 论文草稿
├── 06_模型与算法/         — 算法文档（PPO、CTDE、状态空间、动作空间等18篇）
├── 07_仿真与控制代码/
│   ├── Python仿真/        — 核心代码库（详见"两套代码体系"）
│   └── 文档说明/
├── 08_AI提示词与Agent设计/
├── 09_软件原型/
├── 10_答辩与展示/
├── 11_国家级立项申报/
├── 12_论文发表准备/
├── 13_国家级项目过程管理/
├── 14_大赛准备/
├── data/
│   ├── cache/osm_reference_scenes/   — OSM原始几何数据（四区）
│   ├── cache/reference_sources/      — 武汉学院参考地图
│   └── reference/
│       ├── kaiwu_gorge_walk_v2/      — Kaiwu峡谷漫步v2完整代码（L1参考）
│       └── kaiwu_infinity_valley/    — Kaiwu无限之谷完整代码（L2/L3参考）
├── outputs/
│   ├── 环境示意图/         — 5张总览图（3D、热力图、障碍物图例等）
│   ├── 环境示意图_2D审核/  — 4张分区2D图 + scene_2d_manifest.json
│   ├── 环境示意图_3D预览/  — 四区3D场景文件（OBJ/MTL/碰撞JSON/预览PNG）
│   ├── 武汉光谷无人机配送_PPO开发指南.md — 3400行PPO高级开发指南（v3.0）
│   └── 无人机配送_渐进式开发指南.md     — 渐进式开发指南（v1.0）
├── reward_function_design.md          — 奖励函数理论设计文档
├── README.md                          — 项目说明
└── CLAUDE.md                          — 本文件
```

代码运行入口在 `07_仿真与控制代码/Python仿真/`，但策划书撰写、文档编辑、图表生成、论文引用等工作的基准目录是项目根目录。

---

## 两套代码体系

### 体系A：已有的成熟代码（现有代码库）

位置：`07_仿真与控制代码/Python仿真/` 下的 `environment/`、`algorithm/`、`config.py`、`run_wuhan_training.py` 等。

特点：

- 完整PPO实现（`07_仿真与控制代码/Python仿真/algorithm/ppo_agent.py`，16维连续动作、tanh-squashed Gaussian）
- 四区武汉3D场景数据（`outputs/环境示意图_3D预览/`，OSM建筑+碰撞体+风险区）
- CTDE多机协同框架（`07_仿真与控制代码/Python仿真/algorithm/ctde_framework.py`）
- 分层规划器（`07_仿真与控制代码/Python仿真/algorithm/delivery_strategy.py`，ScoreFirstDeliveryPlanner）
- 2D简化环境（`07_仿真与控制代码/Python仿真/environment/simple_delivery_env.py`、`random_delivery_env.py`）
- 经典RL算法（`07_仿真与控制代码/Python仿真/algorithm/classical_rl.py`，Q-learning/SARSA/MC/A*）
- 3400+行PPO开发指南（`outputs/武汉光谷无人机配送_PPO开发指南.md`，v3.0）

**不要修改这套代码**，它已是"终点状态"参考。

### 体系B：无限之谷式重构（新建代码）

位置：`07_仿真与控制代码/Python仿真/drone_delivery/`

特点：

- 完全对齐腾讯开悟（Kaiwu）的 `agent_<algo>/` 目录结构
- 每个Agent包含6子模块：`agent.py`、`algorithm/`、`conf/`、`feature/`、`model/`、`workflow/`
- 双语（中文+英文）注释、开悟式API文档格式
- 已完成6个Agent目录：q_learning、sarsa、monte_carlo、dqn、a2c、diy

**新功能开发应在`07_仿真与控制代码/Python仿真/drone_delivery/`下进行。**

---

## 四层级问题框架

项目采用渐进式设计，从简单到复杂：

| 层级 | 名称 | 动作空间 | 关键约束 | 算法 |
|---|---|---|---|---|
| L1 | 2D固定商家 | Discrete(4) | 地图边界+障碍物 | Q-learning、SARSA、MC |
| L2 | 2D随机商家 | Discrete(8) | +时间窗口+多单携带 | DQN、A2C |
| L3 | 3D固定商家 | Box(3) 连续 | +禁飞区+能耗+建筑碰撞 | A2C、PPO |
| L4 | 3D随机+全约束 | Box(4) 连续 | +多机+风+通信+动态禁飞 | PPO+CTDE |

每个层级新增一组约束条件，逐步构建完整的城市配送系统。

---

## 编码规范

### 语言与注释

- **所有代码必须双语注释**：中文在前，英文在后
  ```python
  # 特征处理 / Feature processing
  # ε-greedy探索 / ε-greedy exploration
  ```
- **文件头必须包含三栏**：
  ```python
  """
  职责 / Responsibility: [一句话说明]
  参考 / Reference: [Kaiwu参考源路径]
  领域适配 / Domain Adaptation: [从峡谷→无人机的关键映射]
  """
  ```
- **每个类/函数必须有双语docstring**
- **文档标题用双语**：`## 5. 框架概述 / Framework Overview`

### 命名规范（对齐Kaiwu）

| 元素 | 规范 | 示例 |
|---|---|---|
| Agent目录 | `agent_<algo>` (小写+下划线) | `agent_q_learning` |
| 类名 | PascalCase | `Agent`, `Algorithm`, `Config`, `SampleManager` |
| 函数 | snake_case | `reward_shaping()`, `observation_process()` |
| 数据类 | PascalCase + Data后缀 | `ObsData`, `ActData`, `SampleData` |
| 配置类 | 固定 `Config` | `Config.GAMMA`, `Config.LEARNING_RATE` |
| TOML键 | snake_case | `max_steps`, `grid_size` |

### Agent标准接口

每个Agent必须实现以下方法（Kaiwu标准）：

```python
class Agent:
    def reset(self, env_obs): ...                     # 回合开始
    def predict(self, list_obs_data): ...              # 训练时动作选择（随机）
    def exploit(self, env_obs): ...                    # 评估时动作选择（确定性）
    def learn(self, list_sample_data): ...             # 模型训练
    def save_model(self, path, id): ...                # 保存模型
    def load_model(self, path, id): ...                # 加载模型
    def observation_process(self, env_obs): ...        # 环境观测 → ObsData
    def action_process(self, act_data): ...            # ActData → 动作
    def reward_shaping(self, reward): ...              # 奖励塑形
```

### 数据协议（Kaiwu模式）

```python
ObsData = create_cls("ObsData", feature=None, legal_act=None)
ActData = create_cls("ActData", act=None, val=None)
SampleData = create_cls("SampleData", obs=None, _obs=None, act=None, rew=None, done=None, ret=None)
```

---

## 领域映射表（峡谷→无人机配送）

| 峡谷漫步v2 | → | 无人机配送 |
|---|---|---|
| hero_pos | → | drone_pos |
| treasure | → | delivery_order |
| treasure_state (10-bit) | → | order_status_mask (5-bit) |
| endpoint/start | → | depot |
| obstacle | → | building/no_fly_zone |
| monster | → | dynamic_risk |
| max_step=1000 | → | max_steps=500 |
| treasure_score=100 | → | delivery_reward=+100 |
| end_score=150 | → | safe_return_bonus=+500 |
| BFS距离 | → | A*距离（曼哈顿） |
| 4方向移动 | → | 保持不变（L1） |
| 8方向移动 | → | 保持不变（L2） |
| legal_act | → | 合法动作掩码 |

---

## 环境接口

### Simple2DDeliveryEnv（L1用）

- `reset(seed)` → `({"position":(2,), "order_status":(5,), "carrying":int}, info)`
- `step(action:0-3)` → `(obs, reward, terminated, truncated, info)`
- `get_state_key()` → int（唯一状态编码）
- 动作：0=上UP, 1=下DOWN, 2=左LEFT, 3=右RIGHT
- 地图：40×40网格，5个固定订单，500步上限

### Random2DDeliveryEnv（L2用）

- `reset(seed)` → `({"position":(2,), "vector":(50,)}, info)`
- `step(action:0-7)` → `(obs, reward, terminated, truncated, info)`
- 8方向动作（4基本+4对角）
- 4-8个随机订单/回合，带时间窗口，最多同时携带2单

---

## 开发指南写作规范

`drone_delivery/dev_guide.md` 是核心交付物。写作要求：

1. **全文双语**：标题 `## 5. 框架概述 / Framework Overview`，正文中文为主，英文术语不翻译
2. **代码引用**：所有函数签名用 ` ```python ``` ` 代码块
3. **表格规范**：左对齐文字，右对齐数字（`---:`），代码引用用反引号
4. **图片引用**：直接引用 `outputs/` 下的现有图片，使用相对路径
5. **状态标记**：已实现 / 生产增强 / 历史基线，明确区分
6. **章节编号**：全局连续编号（0-31），H2为章节，H3/H4为子节
7. **API文档格式**：每个函数用"函数签名 / Parameters / Returns / 不变量"四段式

---

## 四区环境数据

| 分区 | 参考区域 | 建筑数 | 任务点 | 适用层级 |
|---|---|---|---|---|
| A | 光谷广场 | 837栋 | 1(基地), 2, 3 | L4多机 |
| B | 武汉学院 | 43栋 | 4(基地), 5, 6 | L1/L2/L3 |
| C | 长江大桥附近 | 538栋 | 7(基地), 8, 9 | L4扩展 |
| D | 楚河汉界 | 592栋 | 10(基地), 11, 12 | L4扩展 |

任务点坐标见 `outputs/环境示意图_2D审核/scene_2d_manifest.json`。

---

## 关键参考文件

| 文件 | 用途 | 何时引用 |
|---|---|---|
| `outputs/武汉光谷无人机配送_PPO开发指南.md` | 3400行PPO高级指南 | 写L3/L4时参考其结构和公式 |
| `data/reference/kaiwu_gorge_walk_v2/` | Kaiwu峡谷漫步v2完整代码 | 仿写L1算法时参考 |
| `data/reference/kaiwu_infinity_valley/` | Kaiwu无限之谷完整代码 | 仿写L2/L3算法时参考 |
| `data/reference/kaiwu_gorge_walk_v2/dev_guide_optimized.md` | 峡谷漫步v2开发指南（1859行） | 写`dev_guide.md`时仿照其风格 |
| `outputs/环境示意图_3D预览/scene_3d_index.json` | 四区3D场景索引 | 查找建筑数、任务点、风险区 |
| `outputs/环境示意图_2D审核/scene_2d_manifest.json` | 四区2D审核清单 | 查找坐标、分类、审计数据 |
| `07_仿真与控制代码/Python仿真/config.py` | 已有配置参数 | 写`drone_delivery/conf/`时参考 |
| `reward_function_design.md` | 奖励函数理论设计文档 | 写奖励函数时参考 |

---

## 用户分工

- **算法设计**：用户负责（Claude Code协助）
- **问题设计**：用户负责（Claude Code协助）
- **环境设计**：用户负责（Claude Code协助）
- **财务部分**（市场规模、前景预测）：用户自己或找人做，Claude Code不涉及

---

## 工作流建议

### 新增算法Agent时

1. 在 `07_仿真与控制代码/Python仿真/drone_delivery/` 下创建 `agent_<algo>/` 目录，包含6子模块
2. 参考已有Agent（如`agent_q_learning/`）的文件结构和代码风格
3. 确保所有文件双语注释，所有函数/类有docstring
4. 在 `07_仿真与控制代码/Python仿真/drone_delivery/conf/algo_conf_drone_delivery.toml` 中注册算法路由
5. 在 `07_仿真与控制代码/Python仿真/drone_delivery/train_test.py` 中添加冒烟测试函数

### 修改开发指南时

1. 在 `07_仿真与控制代码/Python仿真/drone_delivery/dev_guide.md` 中进行
2. 保持全文双语风格，参考峡谷漫步v2指南（`data/reference/kaiwu_gorge_walk_v2/dev_guide_optimized.md`）的写作方式
3. 引用现有 `outputs/` 图片，不要重复生成
4. 参数表格保持规范（左对齐文字、右对齐数字）

### 环境相关工作时

1. 新增2D网格环境：在 `07_仿真与控制代码/Python仿真/drone_delivery/conf/map_data/` 下创建JSON
2. 新增3D场景：利用 `outputs/环境示意图_3D预览/` 已有资产
3. 参考现有 `outputs/环境示意图_2D审核/` 的坐标和分类数据

---

## 训练与评估

### 运行冒烟测试

```powershell
cd "c:\University\Files\Training program\无人机项目设计\07_仿真与控制代码\Python仿真"
python drone_delivery/train_test.py --algo q_learning --episodes 5
python drone_delivery/train_test.py --algo all --episodes 5
```

### 运行完整训练

```powershell
python run_wuhan_training.py --zone B --episodes 1000
```

### 运行对比实验

```powershell
python run_level1_experiments.py --episodes 2000 --eval-interval 100
```

---

## 常见陷阱

1. **不要混淆两套代码**：`07_仿真与控制代码/Python仿真/drone_delivery/` 是新体系，`07_仿真与控制代码/Python仿真/environment/`、`07_仿真与控制代码/Python仿真/algorithm/` 是旧体系
2. **不要省略双语注释**：这是Kaiwu风格的核心要求
3. **不要把terminated和truncated混用**：自然终止不自举，时间截断需要从final observation自举
4. **不要在奖励函数中使用累积奖励代替一次性奖励**：配送奖励只在状态从未完成→完成时发放一次
5. **不要让奖励分量的量级差异过大**：配送+100 vs 碰撞-200是合理的，配送+100 vs 时间-0.01则会淹没信号
6. **业务Score与训练Reward必须分离**：验收用字典序（目标数严格优先），训练用密集Reward

---

## 项目目标

最终输出：

1. 一份渐进式开发指南（`07_仿真与控制代码/Python仿真/drone_delivery/dev_guide.md`），与Kaiwu峡谷漫步v2指南风格一致
2. 策划书的技术方案部分（基于四层级框架和六算法对比）
3. 学习过程图表（奖励收敛曲线、算法对比柱状图、雷达图）
4. 环境示意图引用（利用已有的`outputs/`资源）

---

*本文件由 Claude Code 自动生成，基于项目完整代码库和参考文档的分析。*
*Generated by Claude Code based on full project analysis. Last updated: 2026-07-09*
