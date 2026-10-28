# Kaiwu参考文档结构分析
# Kaiwu Reference Document Structure Analysis

> **任务 / Task**: T3.1 分析参考文档
> **分析日期 / Date**: 2026-07-11
> **分析对象 / Targets**:
> - `data/reference/kaiwu_gorge_walk_v2/dev_guide_optimized.md` (1859行，风格基准)
> - `data/reference/kaiwu_gorge_walk_v2/agent_q_learning/` (代码结构基准)
> - `data/reference/kaiwu_infinity_valley/` (高级算法参考)
> - `drone_delivery/dev_guide.md` v1.0 (731行，待完善文档)
> - `outputs/无人机配送_渐进式开发指南.md` v1.0 (1471行，技术内容参考)
> - `outputs/武汉光谷无人机配送_PPO开发指南.md` v3.0 (3415行，L3/L4参考)

---

## A. 写作风格分析 / Writing Style Analysis

### A1. 标题层级规范 / Heading Hierarchy

Kaiwu dev_guide_optimized.md 使用 **四级标题体系**：

```markdown
# 一级标题 (文档标题，不编号)
## 二级标题 (主要章节，无编号前缀)
### 三级标题 (子模块/核心函数)
#### 四级标题 (细分功能点)
```

**章节结构顺序**（dev_guide_optimized.md 共 8 大章节）：
1. `项目简介` — 任务介绍 + 场景介绍 + 规则介绍 + 注意事项
2. `环境详述` — 环境配置 + 环境信息说明 + 核心函数介绍
3. `智能体详述` — 智能体综述 + 核心接口（predict/exploit/save_model/load_model）
4. `算法介绍` — Dynamic Programming + Monte Carlo + SARSA + Q-Learning + 监控信息 + 模型保存限制 + 模型评估模式
5. `综述` — 训练流程简介 + 代码包简介 + 目录结构
6. `智能体开发` — 特征处理 + 奖励设计 + 样本处理 + 算法开发 + 模型开发 + 工作流开发
7. `分布式计算框架` — 总体架构 + 组件介绍 + 服务介绍 + 工作流
8. `监控与日志` — 监控面板 + 日志查看

**无人机 dev_guide.md v1.0 使用 8 章 + 4 附录**：
1. `项目介绍` — 背景 + 场景 + 任务
2. `环境说明` — 四层级 + 监控指标
3. `Agent说明` — 通用架构 + 特征处理 + 奖励 + 数据协议
4. `算法介绍` — 6算法对比
5. `框架概述` — CTDE架构
6. `环境API参考` — 2个Env类接口
7. `训练/评估/保存` — 命令 + 保存
8. `常见故障` — 4个排障项

### A2. 代码块风格 / Code Block Style

Kaiwu 文档代码块特征：

**双语注释模式**（英文在上，中文在下，用 `#` 注释）：
```python
# Get the environment and agent
# 获取环境和智能体
env, agent = envs[0], agents[0]

# Execute several epochs
# 执行若干次epoch
epoch_num = 1000

# Training loop
# 训练循环
for epoch in range(epoch_num):
    # After each episode, the trajectory data is converted into training samples for training.
    # 在每一个episode结束之后，将轨迹数据转换成训练样本进行训练
    for g_data in run_episodes(episode_num_every_epoch, env, agent, logger, monitor):
```

**TOML配置双语注释**（英文在上，中文在下）：
```toml
[env_conf]
# 起点坐标
# 数组，包含两个整数元素[x, z]，每个坐标取值范围为0~63。
# 起点坐标和终点坐标不能相同。
start = [29, 9]

# 宝箱ID数组。每个ID必须是整数，取值范围：0~9。
# 配置固定宝箱的ID。
treasure_id = []
```

**代码块不使用语言标识符**（Kaiwu原始文档使用无标识的缩进代码块）。
无人机文档应使用 ` ```python ``` ` 标准Fenced代码块。

### A3. 表格规范 / Table Convention

Kaiwu文档使用两种表格格式：

**格式1：带表头分隔线的对齐表格**（用于API文档）：
```markdown
  **参数名**       **介绍**
  ---------------- ------------------------------------------
  **第一个参数**   字符串类型，类的名称
  **其余参数**     类的属性，默认值为None，由开发者自行定义
```

**格式2：环境配置参数表**：
```markdown
  **数据名**             **数据类型**   **数据描述**                                                            **默认值**
  ---------------------- -------------- ----------------------------------------------------------------------- ------------
  **start**              [int,int]      起点位置坐标                                                            [29,9]
  **treasure_random**    bool           是否生成随机宝箱                                                        false
```

**无人机文档已采用的格式**：
```markdown
| 参数 / Parameter | 值 / Value |
|---|---|
| 地图 / Map | 40x40 网格 |
```

**差异**：Kaiwu用加粗字段名 + 双空格对齐；无人机用管道符 + 中文/英文并列。无人机风格更易读，建议保留。

### A4. API文档四段式 / API Doc Pattern

Kaiwu dev_guide_optimized.md 对每个核心函数采用严格的 **四段式** 格式：

```
### functionName

[一段描述，说明功能和用途]

def functionName(params):
    return result

### Parameters

  **参数名**   **介绍**
  ------------ --------------------------------------------------------
  **param1**   类型描述

### Return (或 Returns)

  **参数名**    **介绍**
  ------------- ------------------------------------------------------------------
  **result**    类型描述
```

**示例**（Kaiwu原文中的 observation_process）：
```markdown
### observation_process

将环境返回的观测数据转换成ObsData类型数据。
很多情况下，特征工程包含了大量的数值处理、数据转换和领域知识，
我们建议将大量的特征处理代码在<智能体文件夹>/feature/preprocessor.py文件中实现，
然后由observation_process进行调用。

def observation_process(self, obs, state=None):
    return ObsData(feature=feature, legal_act=legal_actions)

### Parameters

  **参数名**   **介绍**
  ------------ --------------------------------------------------------
  **obs**      Observation类型，env.reset和env.step返回的环境观测数据
  **state**    EnvInfo类型，env.reset和env.step返回的环境状态数据

### Return

  **参数名**    **介绍**
  ------------- ------------------------------------------------------------------
  **ObsData**   开发者定义的ObsData类型的数据，将作为agent.predict()函数的输入。
```

**无人机文档当前做法**：函数签名用代码块，但缺少 Parameters/Return 表格。应补充。

### A5. 图片引用模式 / Image Reference Pattern

Kaiwu原始文档（Word导出格式）：
```markdown
![峡谷漫步地图场景示意图](04_项目内容及任务分工/dev_guide_optimized_media/media/image1.png "图片 16"){width="6.645669291338582in" height="5.770690069991251in"}
```

无人机文档已采用的标准Markdown格式：
```markdown
![武汉学院2D地图](outputs/环境示意图_2D审核/zone_B_lakeside_campus_2d.png)
![四区环境总览](outputs/环境示意图/wuhan_environment_overview.png)
```

**可用图片清单**（已存在于 `outputs/` 下）：
| 目录 | 文件 | 用途 |
|---|---|---|
| `outputs/环境示意图/` | 5张PNG | 总览图（3D、热力图、障碍物图例） |
| `outputs/环境示意图_2D审核/` | 4张PNG + JSON | 分区2D图 |
| `outputs/环境示意图_3D预览/` | 4组(OBJ+MTL+PNG) | 四区3D场景预览 |

### A6. 警告和注意事项标记 / Warning/Note Markers

Kaiwu文档使用 `**注意：**` 标记：

```markdown
**注意**：奖励是用于引导强化学习训练的信号，与环境的得分（score）概念不同。
得分用于衡量模型表现，而奖励用于算法学习。

**注意：当前实验提供的四个算法仅支持单机训练。若需要使用分布式训练，
请在agent_diy文件夹下按照开发框架文档自行完成算法开发。**
```

无人机文档应统一使用：
```markdown
> **注意 / Note**: ...
> **警告 / Warning**: ...
> **提示 / Tip**: ...
```

### A7. 双语风格对比 / Bilingual Style Comparison

| 特征 | Kaiwu原文 | 无人机v1.0 | 建议 |
|---|---|---|---|
| 标题 | 仅中文 | 双语（中/英） | 保留双语 |
| 正文 | 仅中文 | 中文为主+英文术语 | 保留 |
| 表头 | 仅中文（加粗） | 双语（`中 / En`） | 保留双语 |
| 代码注释 | 英文在上+中文在下 | 同上 | 保持一致 |
| 函数名 | 英文 | 英文 | 保持一致 |

---

## B. 六子模块结构分析 / 6-Submodule Structure Analysis

### B1. 总览 / Overview

Kaiwu标准Agent目录结构（以 `agent_q_learning` 为基准）：

```
agent_q_learning/               # Agent根目录
├── __init__.py                 # 包标识
├── agent.py                    # [核心] Agent类，277行
│   ├── class Agent(BaseAgent)  #   继承kaiwudrl.interface.agent.BaseAgent
│   ├── ObsData = create_cls()  #   动态数据类定义
│   ├── ActData = create_cls()  #   动态数据类定义
│   ├── CHECKPOINT_VERSION      #   检查点版本号
│   └── 9个标准方法             #   reset/predict/exploit/learn/save_model/load_model
│                                #   observation_process/action_process/reward_shaping
├── algorithm/
│   ├── __init__.py
│   └── algorithm.py            # [核心] 算法逻辑，343行
│       ├── ReplayBuffer        #   经验回放缓冲（deque）
│       ├── SparseQTable        #   稀疏Q表（惰性初始化）
│       └── Algorithm           #   Double Q-Learning + LR调度
├── conf/
│   ├── __init__.py
│   ├── conf.py                 # Config类，46行，所有超参数常量
│   ├── monitor_builder.py      # 监控面板配置
│   └── train_env_conf.toml     # 环境TOML配置
├── feature/
│   ├── __init__.py
│   ├── definition.py           # [核心] 特征定义+奖励塑形，199行
│   │   ├── reward_shaping()    #   7分量奖励函数
│   │   ├── sample_process()    #   帧→样本转换
│   │   └── _helper函数(10+)    #   观测解析辅助函数
│   └── preprocessor.py         # [核心] 特征工程+路径规划，564行
│       ├── q_learning_feature()#   状态编码 (1024*pos + treasure_binary)
│       ├── optimal_route()     #   Held-Karp精确路径规划（DP）
│       ├── legal_actions_from_obs()  # 合法动作掩码
│       └── shortest_path_action()    # BFS最短路径
├── model/
│   ├── __init__.py
│   └── model.py                # 神经网络模型，93行
│       └── Model(nn.Module)    #   Linear(55,128)→ReLU→Linear(128,64)→ReLU→Linear(64,4)
└── workflow/
    ├── __init__.py
    └── train_workflow.py       # [核心] 训练循环，288行
        └── workflow()           #   完整训练流程：episode循环 + 验证 + 收敛检测 + 模型保存
```

### B2. agent.py 详解 / Agent Module Deep Dive

**核心职责**：连接环境与算法，实现所有标准接口。

**关键设计模式**：

1. **动态数据类**（Kaiwu特有模式）：
```python
from common_python.utils.common_func import create_cls

ObsData = create_cls("ObsData", feature=None, legal_actions=None, raw_obs=None)
ActData = create_cls("ActData", act=None)
```

2. **检查点版本管理**：
```python
CHECKPOINT_VERSION = 2
STATE_ENCODING = "official_position_treasure_mask_v1"
```

3. **标准接口实现**（全部9个方法）：
```python
class Agent(BaseAgent):
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None): ...
    def reset_episode(self): ...              # 重置回合状态
    def predict(self, list_obs_data): ...     # 训练时（含探索）
    def exploit(self, env_obs): ...           # 评估时（确定性）
    def learn(self, list_sample_data): ...    # 训练步骤
    def save_model(self, path, id): ...       # 保存检查点
    def load_model(self, path, id): ...       # 加载检查点
    def observation_process(self, env_obs): ... # 观测→ObsData
    def action_process(self, act_data): ...   # ActData→动作
    def reward_shaping(self, env_reward, env_obs, prev_env_obs, action): ...  # 奖励塑形
```

4. **日志安全访问**（不依赖logger必须存在）：
```python
def _log(self, level, message):
    if self.logger is None:
        return
    log_method = getattr(self.logger, level, None) or getattr(self.logger, "info", None)
    if log_method is not None:
        log_method(message)
```

### B3. algorithm/ 详解 / Algorithm Module Deep Dive

**核心职责**：实现算法核心逻辑，独立于Agent和环境。

**三层架构**：

```
ReplayBuffer (数据层)
    ↓ 采样
Algorithm (算法层)
    ↓ 调用
SparseQTable / Model (参数层)
```

**ReplayBuffer**：
```python
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)  # 循环缓冲
    def push(self, state, action, reward, next_state, done): ...
    def sample(self, batch_size): ...
    def __len__(self): ...
```

**Algorithm 类关键特性**：
- Double Q-Learning（两个Q表交替更新，减少过估计）
- 经验回放（batch_size=128）
- 学习率调度（指数衰减，从0.15到0.003）
- 目标网络同步（硬同步每300步 或 软Polyak更新tau=0.005）
- 返回指标字典：`{"loss": ..., "td_error": ..., "lr": ...}`

### B4. conf/ 详解 / Config Module Deep Dive

**核心职责**：集中管理所有超参数，不散布在代码中。

**Config类设计**（46行，纯类变量）：
```python
class Config:
    STATE_SIZE = 64 * 64 * 1024
    ACTION_SIZE = 4
    LEARNING_RATE = 0.15
    GAMMA = 0.96
    EPSILON_START = 0.35
    EPSILON_MIN = 0.02
    EPSILON_DECAY = 0.985
    EPISODES = 10000
    REPLAY_CAPACITY = 200000
    BATCH_SIZE = 128
    TARGET_UPDATE_FREQ = 300
    LR_MIN = 0.003
    LR_DECAY_STEPS = 150000
    OBSERVATION_SHAPE = 55
    TARGET_SOFT_TAU = 0.005
    USE_SOFT_UPDATE = False
    USE_DNN = False
```

**train_env_conf.toml 设计**（环境专用配置）：
```toml
[env_conf]
# 起点坐标
start = [29, 9]
# 终点坐标
end = [11, 55]
# 宝箱ID数组
treasure_id = []
# 随机宝箱数量
treasure_count = 0
# 是否随机宝箱
treasure_random = false
# 最大步数
max_step = 2000
```

### B5. feature/ 详解 / Feature Module Deep Dive

**两个文件分工明确**：

**definition.py**（与环境交互的接口层）：
- `reward_shaping()` — 7分量奖励塑形函数
- `sample_process()` — 轨迹帧→训练样本转换
- 辅助函数族：`_observation()`, `_game_info()`, `_hero_pos()`, `_organs()`, `_total_score()`, `_target_dist()`, `_nearest_treasure_dist()`, `_active_treasure_count()`, `_treasure_count()`, `_treasure_score()`, `_exploration_bonus()`

**preprocessor.py**（纯计算层，不依赖环境对象）：
- `q_learning_feature(env_obs)` — 状态编码
- `legal_actions_from_obs(env_obs)` — 合法动作掩码
- `optimal_route(env_obs)` — Held-Karp精确路径规划（TSP，O(2^10 * 10^2)）
- `shortest_path_action(env_obs)` — BFS最短路径动作
- `heuristic_action_scores(env_obs)` — 启发式动作评分
- 常量：MAP_SIZE=64, TREASURE_COUNT=10, ACTION_DELTAS

### B6. model/ 详解 / Model Module Deep Dive

**最简模块**（93行），仅在USE_DNN=True时使用：

```python
class Model(nn.Module):
    def __init__(self, state_shape, action_shape=0, softmax=False):
        super().__init__()
        self.fc1 = nn.Linear(state_shape, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_shape)
        # Kaiming uniform init

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

def save_model(model, path, id):
    torch.save(model.state_dict(), f"{path}/model.ckpt-{id}.pkl")

def load_model(model, path, id):
    model.load_state_dict(torch.load(f"{path}/model.ckpt-{id}.pkl"))
```

### B7. workflow/ 详解 / Workflow Module Deep Dive

**核心职责**：定义完整训练循环，是最复杂的模块（288行）。

**关键设计要素**：

1. **配置读取**：从TOML文件读取环境配置
2. **Episode循环**：`for episode in range(EPISODES)`
3. **Step循环**：`while not done:`
4. **数据流**：env.reset → observation_process → predict → action_process → env.step → reward_shaping → sample_process → learn
5. **Epsilon衰减**：每episode结束后衰减（不在step内）
6. **验证机制**：每50个episode运行5次纯评估
7. **收敛检测**：`recent_win_rate >= 0.90 and avg_treasures >= expected_treasures and episode >= 200`
8. **模型保存限速**：最小30秒间隔，最多100次
9. **监控上报**：`monitor.put_data({os.getpid(): monitor_data})`
10. **异常处理**：`handle_disaster_recovery` 错误恢复

### B8. infinity_valley 高级算法结构 / Infinity Valley Advanced Structure

infinity_valley 目录包含更高级的Agent（a2c, ppo, policy_gradient），但子模块结构完全一致：
```
agent_ppo/
├── agent.py          # 复杂Agent（Actor-Critic架构）
├── algorithm/        # PPO算法（GAE、clipped surrogate）
├── conf/             # 配置（更大GAMMA、更多训练参数）
├── feature/          # 更高维特征处理
├── model/            # Actor + Critic 双网络
└── workflow/          # 多episode批处理训练
```

---

## C. 内容差距分析 / Content Gap Analysis

### C1. dev_guide.md v1.0 vs dev_guide_optimized.md 对比

| 维度 | dev_guide_optimized.md (基准) | dev_guide.md v1.0 (当前) | 差距 |
|------|------|------|------|
| **总行数** | 1859行 | 731行 | 差1128行 |
| **章节完整度** | 8章+附录 | 8章+4附录 | 章节数相近，但每章深度不足 |
| **项目介绍** | 任务+场景+规则+注意事项+算法选择+训练模式 | 背景+场景(4级)+任务+12个任务点 | 基本完整，缺少"规则介绍"（总积分公式） |
| **环境详述** | 配置表+env接口(step/reset)+4个算法监控信息 | 四层级+监控指标(7行) | 缺少env.step/reset完整API文档 |
| **智能体详述** | predict/exploit/save_model/load_model 4个完整API | 通用架构+简要接口 | 缺少完整API四段式文档 |
| **算法介绍** | 4算法详述+监控+模型保存限制+评估模式 | 6算法对比(仅表格) | 缺少每个算法的独立介绍段落 |
| **智能体开发** | 6个子模块分别详述(create_cls/observation/action/reward/sample/learn/model/workflow) | 4节(特征/奖励/数据协议/算法对比) | **最大差距**：缺少完整的开发流程指导 |
| **分布式框架** | 完整架构+组件+服务+工作流 | CTDE框架概述(3行) | 需要扩写分布式部分(但可简化) |
| **监控与日志** | 监控面板+日志查看+告警 | 无 | 完全缺失 |
| **图片引用** | 6张图 | 6张图 | 数量接近 |
| **代码示例** | 30+段完整代码 | 15段代码 | 缺少workflow完整示例 |

### C2. 缺失章节清单 / Missing Sections Checklist

| 编号 | 缺失内容 | 优先级 | 来源参考 |
|------|---------|--------|---------|
| G1 | `env.step()` / `env.reset()` 完整API文档（四段式） | 高 | dev_guide_optimized.md L700-716 |
| G2 | `agent.predict()` 完整API文档（装饰器+Parameters+Return） | 高 | dev_guide_optimized.md L1190-1210 |
| G3 | `agent.exploit()` 完整API文档 | 高 | dev_guide_optimized.md L1212-1230 |
| G4 | `agent.save_model()` / `load_model()` 完整API文档 | 高 | dev_guide_optimized.md L1254-1282 |
| G5 | `create_cls` 动态数据类详解 | 中 | dev_guide_optimized.md L779-800 |
| G6 | `observation_process` 四段式API文档 | 高 | dev_guide_optimized.md L810-832 |
| G7 | `action_process` 四段式API文档 | 中 | dev_guide_optimized.md L840-858 |
| G8 | `reward_shaping` 四段式API文档 + 设计说明 | 高 | dev_guide_optimized.md L860-882 |
| G9 | `sample_process` 四段式API文档 | 中 | dev_guide_optimized.md L895-914 |
| G10 | `SampleData2NumpyData` / `NumpyData2SampleData` 分布式支持 | 低 | dev_guide_optimized.md L920-955 |
| G11 | `learn()` 四段式API文档 | 高 | dev_guide_optimized.md L970-987 |
| G12 | `workflow()` 训练工作流完整示例代码 | 高 | dev_guide_optimized.md L1015-1098 |
| G13 | 模型保存限制策略 | 中 | dev_guide_optimized.md L489-504 |
| G14 | 模型评估模式说明 | 中 | dev_guide_optimized.md L506-531 |
| G15 | 分布式计算框架架构图（简化版） | 低 | dev_guide_optimized.md L1284-1314 |
| G16 | 监控面板说明 + 算法指标 | 中 | dev_guide_optimized.md L480-487 |
| G17 | 每个算法的独立介绍段落 | 高 | dev_guide_optimized.md L453-478 |
| G18 | 总积分规则公式 | 中 | dev_guide_optimized.md L30-39 |
| G19 | 训练流程总览图 | 中 | dev_guide_optimized.md L543-566 |
| G20 | 超时判定和失败处理规则 | 中 | dev_guide_optimized.md L39 |

### C3. 已有但需完善的内容 / Content Needing Improvement

| 编号 | 内容 | 当前状态 | 改进方向 |
|------|------|---------|---------|
| U1 | Agent标准接口（3.1节） | 仅列签名 | 补充每个方法的Parameters/Return表格 |
| U2 | 数据协议（3.4节） | 有定义+字段说明 | 补充使用场景示例 |
| U3 | 算法介绍（第4章） | 仅对比表格 | 每个算法补充1-2段原理+代码示例 |
| U4 | 奖励塑形（3.3节） | 仅直通函数 | 需要7分量奖励设计说明（来自reward_function_design.md） |
| U5 | 训练命令（7.1节） | 简要命令 | 补充完整workflow代码示例 |
| U6 | 常见故障（第8章） | 4个排障项 | 补充到6-8个，增加具体错误日志示例 |

---

## D. PPO开发指南可复用内容提取 / PPO Guide Reusable Content

### D1. 概述

`outputs/武汉光谷无人机配送_PPO开发指南.md` v3.0（3415行）是技术内容最丰富的文档，包含大量可以直接迁移至新dev_guide.md的内容。以下按主题提取关键内容段落。

### D2. L3/L4环境设计（可复用章节）

**来源**：PPO开发指南 第3-7章

**可复用内容清单**：

| 内容 | 行数范围 | 复用目标章节 |
|------|---------|------------|
| 武汉四区场景数据（837/43/538/592栋建筑） | 第3章 | 新dev_guide 第2章 |
| 3D状态空间设计（166维向量分解） | 第4章 | 新dev_guide 第6章 |
| Box(16)连续动作空间定义 | 第5章 | 新dev_guide 第6章 |
| 禁飞区分类（永久/动态/临时） | 第3章 | 新dev_guide 第2章 |
| 建筑碰撞检测模型（AABB+球体） | 第3章 | 新dev_guide 第2章 |
| 12个任务点坐标（4区×3点） | 第3章 | 新dev_guide 第2章 |

**3D状态空间166维分解**（直接引用）：
```
drone_state (12维): pos_xyz(3) + vel_xyz(3) + att_rpy(3) + battery(1) + payload(1) + speed(1)
order_state (15维×5单): status(5) + merchant_pos(3×5) + delivery_pos(3×5) + time_remaining(5) + weight(5) + urgency(5)
environment (19维): wind(3) + nfz_flags(4) + nearest_building_dist(1) + altitude_agl(1) + comm_quality(1) + gps_quality(1) + time_of_day(1) + weather(1) + congestion(1) + visibility(1) + airspace_class(1) + nearby_drone_count(1) + collision_warning(1) + emergency_flag(1)
action_history (4维): last_action(4)
context (16维): mission_type(1) + zone_id(4) + priority_level(1) + total_orders(1) + completed_orders(1) + active_orders(1) + time_elapsed(1) + max_episode_time(1) + current_score(1) + carrying_count(1) + returning(1) + emergency_landing(1) + low_battery(1)
```

### D3. 奖励函数设计（可复用章节）

**来源**：PPO开发指南 第8章 + `reward_function_design.md`

**可直接复用的8分量奖励函数**：

| 分量 | 公式 | 量级 | 触发条件 |
|------|------|------|---------|
| 步惩罚 | -0.1/step | -50/episode | 每步 |
| 取货奖励 | +50/pickup | 0-250 | 从未取→已取 |
| 配送奖励 | +100/delivery | 0-500 | 从未完成→已完成 |
| 安全返航 | +500 | 一次 | terminated且all delivered |
| 碰撞惩罚 | -200/collision | 0--1000 | 碰撞发生 |
| 禁飞区惩罚 | -100/violation | 0--500 | 进入禁飞区 |
| 电量警告 | -50 (低电量) | 一次 | battery < 20% |
| 时间超时 | -100 | 一次 | truncated |
| 距离塑形 | +2/-1 | 持续 | 接近/远离目标 |
| 电量效率 | +0.01×(prev-curr) | 持续 | 电量消耗与进度比 |

**奖励分量裁剪**：`np.clip(total, -500, +1000)`

### D4. PPO算法实现（可复用章节）

**来源**：PPO开发指南 第10-15章

**可直接复用的技术内容**：

1. **tanh-squashed Gaussian 策略**（代码+公式）
2. **GAE(λ) 优势估计**（公式+实现）
3. **Clipped Surrogate 目标**（公式+代码）
4. **KL 散度早停**（自适应KL系数）
5. **Advantage 标准化**（逐batch标准化）
6. **MLP Actor-Critic 网络结构**（166→256→256→128→16）
7. **CNN-MLP 混合架构**（可选，用于3D体素输入）

### D5. 课程学习（可复用章节）

**来源**：PPO开发指南 第18章

**7阶段课程学习（C1-C7）**：

| 阶段 | 名称 | 目标 | 约束 | 通过条件 |
|------|------|------|------|---------|
| C1 | 单点直飞 | 1单1点 | 无障碍 | delivery_rate>80%, 500ep |
| C2 | 多点取送 | 1单多点 | +建筑 | delivery_rate>70%, 1000ep |
| C3 | 时间窗口 | 多单 | +时间窗 | delivery_rate>60%, 1500ep |
| C4 | 能耗约束 | 多单 | +电量 | success_rate>60%, 2000ep |
| C5 | 禁飞区 | 多单 | +禁飞 | violation_rate<5%, 2500ep |
| C6 | 风场 | 多单 | +风 | delivery_rate>50%, 3000ep |
| C7 | 全约束 | 完整 | 全部 | score>400, 5000ep |

**阶段转换逻辑**：连续N个episode满足通过条件 → 自动升级

### D6. CTDE框架（可复用章节）

**来源**：PPO开发指南 第20章 + `06_模型与算法/算法文档/`

**可复用内容**：
- 集中训练（Critic共享全局信息）
- 分散执行（Actor仅用局部观测）
- 通信协议（消息传递格式）
- 冲突检测（时空冲突矩阵）

### D7. 训练与评估命令（可复用章节）

**来源**：PPO开发指南 第22章

**可直接复用的命令集**：
```bash
# 冒烟测试
python drone_delivery/train_test.py --algo q_learning --episodes 5
python drone_delivery/train_test.py --algo all --episodes 5

# 完整训练
python run_wuhan_training.py --zone B --episodes 1000

# 对比实验
python run_level1_experiments.py --episodes 2000 --eval-interval 100

# 评估
python run_evaluation.py --model outputs/models/q_learning_best.npy --episodes 100 --render
```

---

## E. 重写建议 / Rewrite Recommendations

### E1. 新dev_guide.md 建议结构

基于上述分析，建议新版开发指南采用以下结构（预计2500-3000行）：

```
# 无人机外卖配送开发指南 / Drone Food Delivery Development Guide

## 0. 文档导读 / How to Use This Guide
## 1. 项目介绍 / Project Introduction
    1.1 任务介绍 / Task Description (含积分规则公式)
    1.2 场景介绍 / Scene Description (四层级概述)
    1.3 注意事项 / Important Notes (算法选择+训练模式)

## 2. 环境详述 / Environment Details
    2.1 L1环境：Simple2DDeliveryEnv
    2.2 L2环境：Random2DDeliveryEnv
    2.3 L3环境：WuhanDrone3DEnv (从PPO指南迁移)
    2.4 L4环境：WuhanMultiDroneEnv (从PPO指南迁移)
    2.5 环境配置 (TOML说明)

## 3. Agent说明 / Agent Details
    3.1 通用架构 / Common Architecture (6子模块树)
    3.2 Agent标准接口 / Standard Interface
    3.3 数据协议 / Data Protocol (ObsData/ActData/SampleData)

## 4. 算法介绍 / Algorithm Introduction
    4.1 Q-Learning (原理+代码+监控指标)
    4.2 SARSA
    4.3 Monte Carlo
    4.4 DQN
    4.5 A2C
    4.6 PPO
    4.7 算法对比表

## 5. 框架概述 / Framework Overview
    5.1 CTDE架构
    5.2 训练流程图
    5.3 代码包目录结构

## 6. 智能体开发 / Agent Development (核心章，最长)
    6.1 特征处理 / Feature Processing
        6.1.1 create_cls 数据结构定义
        6.1.2 observation_process 四段式API
        6.1.3 action_process 四段式API
        6.1.4 preprocessor.py 设计说明
    6.2 奖励设计 / Reward Design
        6.2.1 reward_shaping 四段式API
        6.2.2 7分量奖励函数详解 (从PPO指南+reward_function_design.md迁移)
    6.3 样本处理 / Sample Processing
        6.3.1 sample_process 四段式API
    6.4 算法开发 / Algorithm Development
        6.4.1 learn() 四段式API
        6.4.2 Algorithm类设计模式
    6.5 模型开发 / Model Development
        6.5.1 nn.Module模型模板
        6.5.2 save_model / load_model 四段式API
    6.6 工作流开发 / Workflow Development
        6.6.1 workflow() 四段式API
        6.6.2 完整训练循环代码示例

## 7. 课程学习 / Curriculum Learning
    7.1 C1-C7 阶段设计 (从PPO指南迁移)
    7.2 阶段转换逻辑

## 8. 训练、评估与保存 / Training, Evaluation & Checkpointing
    8.1 命令参考
    8.2 模型保存限制策略
    8.3 评估模式说明

## 9. 监控与日志 / Monitoring & Logging
    9.1 监控面板
    9.2 日志查看
    9.3 常见故障排查 (扩展到8项)

## 附录 / Appendix
    A. 环境数据来源
    B. 领域映射表
    C. 配置参数速查
    D. 版本历史
```

### E2. 优先级排序 / Priority Order for Rewrite

| 优先级 | 内容 | 预计新增行数 | 难度 |
|--------|------|-------------|------|
| P0 | 补全所有API四段式文档（G1-G11） | +600行 | 低 |
| P1 | 每个算法独立介绍段落（G17） | +400行 | 中 |
| P2 | workflow完整代码示例（G12） | +200行 | 中 |
| P3 | 奖励函数7分量详解 | +300行 | 低（从现有文档迁移） |
| P4 | L3/L4环境说明 | +300行 | 低（从PPO指南迁移） |
| P5 | 课程学习章节 | +200行 | 低（从PPO指南迁移） |
| P6 | 监控与日志章节 | +150行 | 中 |
| P7 | 常见故障扩展 | +100行 | 低 |

**预计总新增：约 2250 行，新版指南总行数约 2980 行**

---

*本分析基于对 7 个参考文档/目录的深度阅读，总计覆盖约 7500 行代码和文档。*
*Analysis based on deep reading of 7 reference files/directories, covering approximately 7500 lines total.*
