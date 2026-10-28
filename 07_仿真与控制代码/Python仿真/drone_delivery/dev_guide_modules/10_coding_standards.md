"""
模块10 — 代码规范与最佳实践 / Module 10 — Coding Standards & Best Practices

职责 / Responsibility:
    双语注释规范、命名规范、API文档四段式规范、terminated/truncated区分、
    业务Score与训练Reward分离原则。
    Bilingual comment convention, naming conventions, four-part API documentation,
    terminated/truncated distinction, and Score/Reward separation principle.

参考 / Reference:
    CLAUDE.md (编码规范、命名规范、Agent标准接口)
    drone_delivery/agent_q_learning/conf/conf.py (Config类示例)
    drone_delivery/agent_q_learning/agent.py (Agent标准接口)
    data/reference/kaiwu_gorge_walk_v2/dev_guide_optimized.md (Kaiwu风格)

领域适配 / Domain Adaptation:
    峡谷漫步v2 → 无人机配送:
    - 10-bit宝箱状态 → 5-bit订单状态掩码
    - treasure_done → delivery_completed
    - hero_dead (碰撞) → drone_collision
    - end_reached → depot_reached (安全返航)
"""

---

## 10. 代码规范与最佳实践 / Coding Standards & Best Practices

> **版本 / Version**: v2.0  
> **日期 / Date**: 2026-07-11  
> **前置模块 / Prerequisites**: 模块06（配置文件格式）

---

### 10.1 双语注释规范 / Bilingual Comment Convention

#### 代码注释格式 / Code Comment Format

**核心规则**：中文在上，英文在下（或同一行用`/`分隔，中文在前）。

```python
# 特征处理 / Feature processing
# epsilon-greedy探索 / epsilon-greedy exploration
```

单行内联注释：

```python
state_key = position_x * grid_size + position_y  # 状态编码 / State encoding
```

多行逻辑块注释：

```python
# ============================================================
# 核心 Q-Learning 超参数
# Core Q-Learning hyper-parameters
# ============================================================
```

#### 文件头三栏格式 / Three-Field File Header

每个Python文件必须包含三栏docstring头：

```python
"""
职责 / Responsibility: [一句话说明本文件做什么]
参考 / Reference: [Kaiwu参考源路径，如 kaiwu_gorge_walk_v2/agent_q_learning/...]
领域适配 / Domain Adaptation: [从峡谷→无人机的关键映射]
"""
```

**完整示例 / Complete Example**:

```python
"""
Q-learning算法实现模块
Q-learning algorithm implementation module

职责 / Responsibility:
    实现Q-learning核心训练逻辑，包括Q表更新、epsilon-greedy探索和TD误差计算。
    Implements Q-learning core training logic: Q-table update, epsilon-greedy
    exploration, and TD error computation.

参考 / Reference:
    kaiwu_gorge_walk_v2/agent_q_learning/algorithm/algorithm.py

领域适配 / Domain Adaptation:
    Gorge Walk treasure collection → Drone pickup-delivery:
    - hero_pos → drone_pos (x,y grid)
    - treasure_state (10-bit) → order_status_mask (5-bit)
    - endpoint → depot at [30,10]
    - max_step=1000 → max_steps=500
    - treasure_score=100 → delivery_reward=+100
    - end_score=150 → safe_return_bonus=+500
"""
```

#### docstring规范 / docstring Convention

每个类和函数必须有双语docstring，采用以下格式：

```python
class QLearningAgent:
    """
    Q-learning智能体 / Q-learning Agent

    使用字典存储稀疏Q表，支持epsilon-greedy探索策略。
    Uses dictionary-based sparse Q-table with epsilon-greedy exploration.

    Attributes:
        q_table: Q值表 / Q-value table
        epsilon: 当前探索率 / Current exploration rate
        gamma: 折扣因子 / Discount factor
    """

    def __init__(self, env, learning_rate=0.1, gamma=0.99, epsilon=1.0):
        """
        初始化Q-learning智能体 / Initialize Q-learning agent

        Args:
            env: 环境实例 / Environment instance
            learning_rate: 学习率，范围[0.001, 1.0] / Learning rate, range [0.001, 1.0]
            gamma: 折扣因子，范围[0.9, 0.999] / Discount factor, range [0.9, 0.999]
            epsilon: 初始探索率，范围[0.0, 1.0] / Initial exploration rate

        Returns:
            None

        不变量 / Invariants:
            - 0 < learning_rate <= 1.0
            - 0.9 <= gamma <= 0.999
            - 0.0 <= epsilon <= 1.0
        """
        self.env = env
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = {}
```

#### TOML注释格式 / TOML Comment Format

```toml
# 中文说明 — 详细描述
# English description — detailed explanation
# 类型 / Type: <类型>, 范围 / Range: <范围>
key = value
```

示例：

```toml
# 最大步数 — 每个回合的最大交互步数
# Maximum steps per episode — maximum interactions per episode
# 整型 / Type: int, 范围 / Range: 1~2000
max_steps = 500
```

---

### 10.2 命名规范与目录结构 / Naming Conventions

#### 命名规则总表 / Naming Rules Summary

| 元素 | 规范 | 示例 | 反例 |
|---|---|---|---|
| Agent目录 | `agent_<algo>` (小写+下划线) | `agent_q_learning` | `Q_Learning_Agent/` |
| 类名 | PascalCase | `Agent`, `Algorithm`, `Config` | `agent_class` |
| 函数名 | snake_case | `reward_shaping()`, `observation_process()` | `RewardShaping()` |
| 变量名 | snake_case | `state_dim`, `action_space` | `StateDim` |
| 常量名 | UPPER_CASE | `MAX_STEPS`, `LEARNING_RATE` | `max_steps` |
| 数据类 | PascalCase + Data后缀 | `ObsData`, `ActData`, `SampleData` | `obs_data_class` |
| 配置类 | 固定名称 `Config` | `Config.GAMMA` | `QLearningConfig` |
| TOML键 | snake_case | `max_steps`, `grid_size` | `maxSteps` |
| TOML节名 | snake_case | `[env_conf]`, `[algo_conf]` | `[envConf]` |
| Python文件 | snake_case.py | `train_workflow.py`, `algorithm.py` | `TrainWorkflow.py` |

#### 命名示例代码 / Naming Example Code

```python
# 类名：PascalCase / Class name: PascalCase
class QLearningAgent:
    pass

class ActorCriticNetwork:
    pass

# 函数名：snake_case / Function name: snake_case
def reward_shaping(reward):
    """奖励塑形 / Reward shaping"""
    pass

def observation_process(env_obs):
    """观测处理 / Observation processing"""
    pass

# 变量名：snake_case / Variable name: snake_case
state_dim = 11
action_dim = 4
learning_rate = 3e-4

# 常量：UPPER_CASE / Constant: UPPER_CASE
MAX_STEPS = 1000
LEARNING_RATE = 3e-4
GAMMA = 0.99

# 数据类：PascalCase + Data / Data class: PascalCase + Data
ObsData = create_cls("ObsData", feature=None, legal_act=None)
ActData = create_cls("ActData", act=None, val=None)
SampleData = create_cls("SampleData",
    obs=None,     # 当前观测 / Current observation
    _obs=None,    # 下一观测 / Next observation
    act=None,     # 执行的动作 / Executed action
    rew=None,     # 奖励 / Reward
    done=None,    # 是否终止 / Whether terminated
    ret=None,     # 回报 / Return
)
```

#### 新增Agent目录创建清单（6步）/ New Agent Directory Checklist (6 Steps)

| 步骤 | 文件 | 内容 |
|---|---|---|
| 1 | `agent_<algo>/__init__.py` | 包初始化，可为空或导入Agent类 |
| 2 | `agent_<algo>/agent.py` | Agent类，实现9个标准接口方法 |
| 3 | `agent_<algo>/algorithm/algorithm.py` | Algorithm类，核心训练逻辑 |
| 4 | `agent_<algo>/conf/conf.py` | Config类，纯类变量超参数 |
| 5 | `agent_<algo>/conf/train_env_conf.toml` | TOML环境配置，双语注释 |
| 6 | `agent_<algo>/feature/definition.py` | ObsData/ActData/SampleData + reward_shaping |

可选文件（建议包含）：

| 文件 | 内容 |
|---|---|
| `conf/monitor_builder.py` | 监控面板配置 |
| `feature/preprocessor.py` | Preprocessor类（特征预处理） |
| `model/model.py` | 神经网络/Q表模型 |
| `workflow/train_workflow.py` | workflow()函数 + EpisodeRunner类 |

---

### 10.3 API文档四段式规范 / Four-Part API Documentation

#### 函数签名格式 / Function Signature Format

每个公开函数必须包含四段式文档：**函数签名、Parameters、Returns、不变量**。

```python
def predict(self, list_obs_data):
    """
    训练时动作选择（随机探索） / Select actions during training (with exploration)

    Parameters:
        list_obs_data (list[ObsData]): 观测数据列表，每个元素对应一架无人机。
            ObsData.feature: 状态特征（state_key或向量）
            ObsData.legal_act: 合法动作掩码，list[bool]，长度=动作空间大小
        / List of observation data, one per drone.
            ObsData.feature: state feature (state_key or vector)
            ObsData.legal_act: legal action mask, list[bool], len=action_space_size

    Returns:
        list[ActData]: 动作数据列表。
            ActData.act: 选择的动作值，int，范围[0, action_space_size)
            ActData.val: 状态价值（仅Actor-Critic），float
        / List of action data.
            ActData.act: selected action, int, range [0, action_space_size)
            ActData.val: state value (Actor-Critic only), float

    不变量 / Invariants:
        - len(list_obs_data) == len(return_value)
        - 0 <= act_data.act < action_space_size
        - 如果legal_act不全为False，则选择的动作必须合法
        - If legal_act is not all False, selected action must be legal
    """
```

#### Parameters表格规范 / Parameters Table Convention

对于复杂参数，使用表格形式：

```python
def learn(self, list_sample_data):
    """
    模型训练 / Model training

    Parameters:
        list_sample_data (list[SampleData]): 训练样本列表
            每个SampleData包含以下字段：
            Each SampleData contains:

            | 字段 | 类型 | 说明 |
            |---|---|---|
            | obs | ObsData | 当前观测 |
            | _obs | ObsData | 下一观测 |
            | act | int | 执行的动作 |
            | rew | float | 获得的奖励 |
            | done | bool | 是否终止（非截断） |
            | ret | float | 计算的回报G_t |

    Returns:
        dict: 训练统计信息 / Training statistics
            - "loss": float, 损失值
            - "q_table_size": int, Q表大小（表格型算法）
    """
```

#### 不变量标注 / Invariant Annotation

不变量是函数执行前后必须始终成立的条件：

```python
def reset(self, env_obs):
    """
    回合开始重置 / Reset at episode start

    Parameters:
        env_obs (dict): 环境初始观测
            - "position": np.ndarray, shape=(2,), dtype=int32, 范围[0, 39]
            - "order_status": np.ndarray, shape=(5,), dtype=int8, 范围{0, 1}

    Returns:
        ObsData: 初始观测数据

    不变量 / Invariants:
        - 调用后 self.epsilon >= Config.EPSILON_MIN
        - 调用后 Q表不清空（累积学习）
        - 返回的ObsData.feature >= 0
    """
```

---

### 10.4 terminated vs truncated 严格区分 / Correct Handling

#### Gymnasium标准定义 / Gymnasium Standard Definitions

| 术语 | 英文 | 含义 | 是否自举 |
|---|---|---|---|
| terminated | terminated | 环境自然终止（碰撞、完成任务） | 不自举（G_t = r_t） |
| truncated | truncated | 时间截断（达到max_steps） | 需要从final observation自举 |

#### sample_process中的处理逻辑 / Processing Logic in sample_process

```python
def sample_process(self, sample):
    """
    样本后处理（处理terminated/truncated） / Sample post-processing

    terminated=True:  回报G_t直接使用r_t（环境已真正结束）
    truncated=True:   回报G_t需要从final observation自举

    不变量 / Invariants:
        - terminated和truncated不能同时为True
        - terminated=True时，done=True，ret=r_t（不加gamma*V(s')）
        - truncated=True时，done=True，ret=r_t + gamma*V(s')（自举）
    """
    if sample.done and not sample.truncated:
        # 真正终止：G_t = r_t / True termination: G_t = r_t
        sample.ret = sample.rew
    elif sample.done and sample.truncated:
        # 时间截断：G_t = r_t + gamma * V(s') / Truncation: G_t = r_t + gamma * V(s')
        sample.ret = sample.rew + self.gamma * self.get_value(sample._obs)
```

#### 常见错误与修正 / Common Errors and Corrections

**错误1：混淆terminated和truncated**

```python
# 错误 / Wrong: 不区分terminated和truncated
if done:
    target = reward  # 截断时丢失了未来价值

# 正确 / Correct: 区分处理
if terminated:
    target = reward
elif truncated:
    target = reward + gamma * V(next_obs)  # 自举
```

**错误2：在terminated时自举**

```python
# 错误 / Wrong: terminated时不应自举
if terminated:
    target = reward + gamma * max_q(next_state)  # 错误！

# 正确 / Correct: terminated时不自举
if terminated:
    target = reward  # 环境已真正结束，无未来价值
```

**错误3：在truncated时不自举**

```python
# 错误 / Wrong: 截断时不自举会低估价值
if truncated:
    target = reward  # 丢失了剩余价值

# 正确 / Correct: 截断时需要自举
if truncated:
    target = reward + gamma * V(next_obs)  # 估计剩余价值
```

---

### 10.5 业务Score与训练Reward分离 / Separation Principle

#### 核心原则 / Core Principle

**训练用密集Reward，验收用业务Score，两者不可混用。**

| 维度 | 训练Reward | 业务Score |
|---|---|---|
| 用途 | 梯度更新、策略优化 | 验收排名、论文对比 |
| 密度 | 密集（每步有信号） | 稀疏（回合结束计算） |
| 结构 | 加权求和（8分量） | 字典序（目标数优先） |
| 设计目标 | 可训练性、收敛速度 | 公平性、可解释性 |

#### 验收评分公式 / Evaluation Scoring Formula

```text
mission_score = 1000 * completed_count + 150 + 0.2 * remaining_time
仅在安全返航时有效 / Valid only upon safe return
```

字典序排序规则：

```python
# 先比较完成订单数（严格优先）
# Compare completed order count first (strictly priority)
if completed_a != completed_b:
    return completed_a > completed_b

# 订单数相同时，比较剩余时间
# When order count equal, compare remaining time
return remaining_time_a > remaining_time_b
```

#### 训练Reward分量 / Training Reward Components

```python
REWARD_CONFIG = {
    "distance_weight": -0.1,        # 距离引导（每步连续信号）
    "time_penalty": -1.0,           # 时间惩罚（防止拖延）
    "collision_weight": -10.0,      # 碰撞惩罚（安全约束）
    "energy_weight": -0.05,         # 能耗惩罚（效率优化）
    "delivery_reward_single": 100,  # 单次配送完成（事件触发）
    "delivery_reward_all": 500,     # 全部完成+返航（事件触发）
    "cooperation_weight": 1.0,      # 多机协作奖励
    "safety_weight": -5.0,          # 安全净空惩罚
    "boundary_weight": -20.0,       # 越界惩罚
}
```

#### 调试方法 / Debugging Methods

**方法1：分别记录Score和Reward**

```python
# 在info字典中分别记录 / Record separately in info dict
info = {
    "business_score": mission_score,    # 验收用 / For evaluation
    "training_reward": episode_reward,  # 训练用 / For training
    "completed_deliveries": completed_count,
    "remaining_time": remaining_steps,
}
```

**方法2：验证Score的字典序特性**

```python
# 完成3单、剩余50步 应该优于 完成2单、剩余400步
# 3 deliveries, 50 steps remaining should beat 2 deliveries, 400 steps
score_a = 1000 * 3 + 150 + 0.2 * 50   # = 3160
score_b = 1000 * 2 + 150 + 0.2 * 400  # = 2230
assert score_a > score_b  # 字典序保证 / Lexicographic ordering
```

**方法3：监控Reward各分量**

```python
# 训练时记录各分量比例 / Record component ratios during training
reward_components = {
    "distance": -0.1 * total_distance,
    "time": -1.0 * steps_taken,
    "collision": -10.0 * collision_count,
    "delivery": 100 * deliveries_completed,
    "safe_return": 500 if safe_return else 0,
}
# 检查各分量量级是否合理（不应相差100倍以上）
# Check magnitude balance (should not differ by >100x)
```

---

*模块10结束 / End of Module 10*  
*下一模块 / Next: 模块11 — API参考文档 / Module 11 — API Reference*
