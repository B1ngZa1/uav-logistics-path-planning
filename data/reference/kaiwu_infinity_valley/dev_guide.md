# 1 无限之谷开发指南

## 1.1 项目简介

### 任务介绍

**目标**：训练一个智能体，使其控制环境中的英雄，通过不断探索地图来学习移动策略，避免英雄被怪物捕获，同时可以尝试收集宝箱以获得更多得分。如果英雄被怪物追上或达到最大任务时长，则任务结束。

### 场景介绍

#### 地图

#### 元素介绍

无限之谷地图中包含英雄、点位、道路、障碍物、宝箱和怪物。

| 元素 | 说明 |
|------|------|
| 英雄 | 环境中存在英雄单位，智能体可以控制英雄在地图中进行移动。英雄每执行一次动作指令代表一步。 |
| 点位 | 环境中一共存在12个点位，其中包括1个英雄出生点位、1个怪物出生点位、10个宝箱点位。宝箱点位支持用户配置，配置方式可参考环境详述-环境配置。 |
| 道路 | 英雄可以正常移动的区域。在道路中英雄有8个移动方向，智能体在控制英雄执行一个动作时，会朝该动作表示方向持续移动3帧（1步），然后再等待下一次动作指令。 |
| 障碍物 | 障碍物会阻挡英雄前进，当英雄在移动过程中遇到障碍物时，将紧贴障碍物边缘移动。有关英雄移动的详细介绍，请查看环境详述-动作空间。 |
| 宝箱 | 智能体可以通过控制英雄拾取宝箱增加积分。每一局任务最多可配置10个宝箱，每个宝箱为100积分。 |
| 怪物 | 环境中存在怪物单位，怪物会追逐英雄，每一局任务最多生成2个怪物。任务开始时，第一个怪物会从怪物出生点出发，追逐英雄。当任务进行一定时间（支持用户配置，默认300步）后，第二个怪物会从怪物出生点出发，追逐英雄。 |

在创建训练任务和评估任务时，上述元素的配置方式有所不同。具体请查看环境详述-环境配置部分。

### 计分规则

任务得分与步数、宝箱收集数量有关，任务结束时会按下方规则计算任务得分：

**任务得分 = 步数得分 + 宝箱得分**

- **步数得分**：在躲避怪物的过程中，智能体每控制英雄走一步，即可得分。步数得分的系数为1.5。可通过该公式进行计算：步数得分=步数x1.5。
- **宝箱得分**：每获得一个宝箱，即可增加100分。可通过该公式进行计算：宝箱得分=宝箱个数x100。

**注意**：如果英雄在最大步数内被怪物追逐到，则任务结束。

---

## 1.2 环境详述

### 环境配置

在智能体和环境的交互中，首先会调用env.reset方法，该方法接受一个usr_conf参数，这个参数通过读取agent_算法名/conf/train_env_conf.toml文件的内容来实现定制化的环境配置。因此，用户可以通过修改train_env_conf.toml文件中的内容来调整环境配置。

```python
# usr_conf为代码包中的train_env_conf.toml配置文件，此处以agent_ppo为例
usr_conf = read_usr_conf("agent_ppo/conf/train_env_conf.toml", logger)
env_obs = env.reset(usr_conf=usr_conf)
```

无限之谷环境配置包含的属性如下：

| 数据名 | 数据类型 | 数据描述 |
|--------|----------|----------|
| treasure_random | bool | 是否生成随机宝箱，设置为true表示随机宝箱，设置为false表示固定宝箱。默认值为false。 |
| treasure_count | int | 生成随机宝箱时的宝箱数量，仅在treasure_random=true时生效；当treasure_count=-1时，表示随机值，即从有效的范围内进行不重复的抽取。取值范围为-1、0~10，默认值为0。 |
| treasure_id | list of int | 生成固定宝箱时的宝箱编号，仅在treasure_random=false时生效，id不可重复。取值范围为1~10，默认值为空。 |
| monster_speed | int | 怪物的移动速度（英雄移动速度为3600mm/s）。取值范围为-1、0、1000~3600，-1表示随机速度，默认值为1000。 |
| monster_interval | int | 第二个怪物出现间隔，即预测多少步后出现第二个怪物。取值范围为-1、0~500，-1表示随机间隔，默认值为300。 |
| max_step | int | 单局最大步数。取值范围为1~2000，默认值为2000。 |

> **补充说明：**
> 1. train_env_conf.toml文件中的配置仅在训练时生效，请按上表数据描述进行配置。若配置错误，训练任务会变为"失败"状态，此时可以通过查看env模块的错误日志进行排查。
> 2. 若需调整模型评估任务时的配置，用户需要通过腾讯开悟平台创建评估任务并完成环境配置，详细参数见智能体详述-模型评估模式。

**train_env_conf的默认配置：**

```toml
[env_conf]

# 宝箱是否随机, 布尔值，false - 固定宝箱，true - 随机宝箱
# 若开启固定，则使用treasure_id生成固定宝箱。
# 若开启随机，则使用treasure_count随机生成宝箱。
treasure_random = false

# 整型，取值范围-1、0~10，-1表示随机。
# 宝箱数量
treasure_count = 0

# 宝箱id，数组，id取值范围：1~10
# 配置固定宝箱的id。
treasure_id = []

# 整型，取值范围-1、0、1000~3600，-1表示随机
# 怪物移动速度（智能体移速为3600mm/s）
monster_speed = 1000

# 整型，取值范围-1、0~500，-1表示随机
# 第二个怪物出现间隔，即预测多少步后，出现第二个怪物。
monster_interval = 300

# 整型，取值范围1~2000, 默认2000
# 最大步数
max_step = 2000
```

英雄起点、怪物起点以及宝箱id对应的位置如下图所示：

---

### 环境数据

在调用env.reset与env.step接口时，会返回环境当前的状态。其中env.reset接口返回obs数据，而env.step接口会额外返回reward数据。下面将对这些环境状态的各项数据及其含义进行详细的介绍，帮助用户更好的理解每个返回值的作用。

```python
# usr_conf为用户传入的环境配置
env_obs = env.reset(usr_conf=usr_conf)

# current_action为用户传入的动作
env_reward, env_obs = env.step(current_action)
```

#### 环境返回信息

**env_obs**

| 数据名 | 数据类型 | 数据描述 |
|--------|----------|----------|
| env_id | str | 环境ID |
| frame_no | int | 当前环境实例运行时的帧号 |
| observation | Observation | 环境实例的观测信息 |
| extra_info | ExtraInfo | 环境实例的额外信息（调试用） |
| terminated | bool | 是否任务结束（被怪物捕获） |
| truncated | bool | 是否任务中断（超时） |

**env_reward**

| 数据名 | 数据类型 | 数据描述 |
|--------|----------|----------|
| env_id | str | 环境ID |
| frame_no | int | 当前帧号 |
| reward | int | 当前步的即时奖励 |

下面会对这些数据进行简单介绍，完整的观测数据结构可以参考数据协议。

#### 观测信息（observation）

observation是环境实例针对智能体返回的原始信息，包括局部观测信息frame_state、场景信息game_info、当前的合法动作legal_act以及环境的局部地图信息map_info。具体描述如下：

| 数据名 | 数据类型 | 数据描述 |
|--------|----------|----------|
| step_no | int | 当前步数 |
| frame_state | FrameState | 局部环境数据 |
| game_info | GameInfo | 场景信息 |
| map_info | MapInfo | 地图信息 |
| legal_act | list of int | 合法动作 |

frame_state和map_info仅包含局部视野范围内的状态数据。

#### frame_state 帧状态

frame_state 包含局部视野内的帧状态信息：

| 数据名 | 数据类型 | 数据描述 |
|--------|----------|----------|
| heroes | list | 英雄列表 |
| monsters | list | 怪物列表 |
| organs | list | 机关列表（宝箱等） |

#### game_info 场景信息

observation 包含的 game_info 是当前场景状态的统计信息，包括得分、宝箱收集数量等。分数的计算详见计分规则。

**注意**：得分用于衡量模型在环境中的表现，也作为衡量强化学习训练产出模型的评价指标，与强化学习里的奖励reward 要区别开。

#### 合法动作

在环境中的某一时刻，不是所有的动作都可以被执行，因此我们需要向模型输入legal action，将网络结果进行掩码（masking）从而避免不符合规则的动作被输出。

在无限之谷中，合法动作用于标识8个移动方向的可用性。

#### 观测视野范围

智能体观测到的地图范围是有限的，我们在观测空间中提供的视野域范围是以英雄为中心，分别向"上、下、左、右"四个方向拓宽 50 格数的一个正方形观察域（size=101x101）。

局部地图信息map_info是以英雄为中心，观察域大小的二维数组(size=101x101)，其中0表示不可通行，1表示可以通行。

> **补充说明：**
> - 局部地图信息维度较大，更适合先用CNN进行图片的特征提取。
> - 无限之谷地图长宽各128000（毫米），坐标范围为[-64000,64000]，地图被栅格化为256x256个格子，每个网格大小为500x500（毫米）。

#### 额外信息（extra_info）

环境会提供部分额外信息，训练时作为提供给智能体的观测信息的补充，评估时无法获取。

| 数据名 | 数据类型 | 数据描述 |
|--------|----------|----------|
| result_code | int | 错误码，0表示正常 |
| result_message | string | 错误信息描述 |
| frame_state | FrameState | 环境全局数据 |
| game_info | GameInfo | 其他场景数据 |

---

### 动作空间

无限之谷的动作空间为移动，取值范围为0~7。

移动使用的是8维离散化的动作空间，如下图所示，将360度等分为8份，每45度角一个动作方向，以x轴正方向为起点，逆时针旋转。

对应关系如下：

```cpp
// 方向角，以x轴正方向为起始边
enum Direction {
    Angle_0 = 0;
    Angle_45 = 1;
    Angle_90 = 2;
    Angle_135 = 3;
    Angle_180 = 4;
    Angle_225 = 5;
    Angle_270 = 6;
    Angle_315 = 7;
}
```

#### 执行逻辑

接下来介绍智能体移动的执行逻辑。在一次决策中，首先智能体给出预测的动作指令，然后：

- 移动方向上无障碍物：正常移动。
- 移动方向上有障碍物：移动被障碍物阻挡，会沿障碍物边缘移动。

---

### 时间信息

帧（frame）和步（step）存在一定映射关系。

- **帧**是场景的一个时间单位，表示场景的一个完整更新周期。在每一帧中，场景的所有元素（如宝箱等）都会根据当前的状态和输入进行更新。
- **步**是强化学习环境中的一个时间单位，表示智能体（agent）在环境中执行一个动作并接收反馈的过程。在每一步中，智能体选择一个动作，环境根据该动作更新状态，并返回新的状态、奖励和终止信号。

在本环境中，1个step由3个frame组成。这意味着每个动作对应一个步，在每一步中，智能体将在三个连续的帧中执行同一个动作。环境将在每一步结束后更新状态并返回反馈，场景只有在完成三帧后，环境状态才会返回一次状态的更新。

- **步更新**：在每一步中，智能体选择一个动作，环境更新状态并返回。
- **帧更新**：在一步中，场景进行三次帧更新，更新所有场景中对象的状态并渲染新的画面。

帧（frame），步（step），现实时间秒（s）和现实时间毫秒（ms）的关系如下：

- 1 frame 约等于 66 ms
- 1 step 等于 3 frame
- 1 s 等于 1000 ms

**注意** ：由于运行环境的差异，每一帧的时间会在66毫秒上下浮动。

---

### 环境监控信息

环境上报了得分等指标，用户可以通过腾讯开悟平台/客户端的监控功能查看。

指标说明如下：

| 指标名称 | 说明 |
|----------|------|
| score | 该面板包含两个指标：total_score：任务结束时的总得分。treasure_score：任务结束时收集到的宝箱奖励。 |
| steps | 该面板包含两个指标：max_steps：任务设置的最大步数。finished_steps：任务结束时所用的步数。 |
| treasure_cnt | 该面板包含两个指标：total_treasures：任务设置的宝箱个数。collected_treasures：任务结束时收集到的宝箱个数。 |
| treasure_random | 是否生成随机宝箱。若为0则表示宝箱位置固定，若为1则表示宝箱位置随机。 |
| monster_speed | 怪物的移动速度。 |
| monster_interval | 第二个怪物出现间隔，单位步。 |

---

## 1.3 智能体详述

我们在代码包中提供了智能体的简单实现，本文将对该部分内容进行讲解，包括观测处理及优化指南等。

### 观测处理

环境返回的observation信息包含了针对智能体的局部观测信息，可以在observation_process函数中对这些局部观测信息进行处理。

很多情况下，观测信息体量较大且步骤繁多，我们推荐用户使用预处理器preprocessor对环境返回的observation信息进行预处理：

```python
def observation_process(self, obs, extra_info):
    feature, legal_action, reward = self.preprocessor.process([obs, extra_info], self.last_action)
```

预处理后的信息参考以下这些数据的描述：

| 数据名 | 数据类型 | 数据维度 | 数据描述 |
|--------|----------|----------|----------|
| hero_info | dict | 3 | 英雄的坐标信息 |
| treasure_infos | dict | 12(最大维度) | 宝箱的坐标、相对位置信息 |
| monster_infos | dict | 2(最大维度) | 怪物的坐标、相对位置信息 |
| obstacle_map | int32 | 51x51=2601 | 障碍物的局部视野信息(local) |
| memory_map | float32 | 51x51=2601 | 地图历史记忆信息(local) |
| treasure_map | int32 | 51x51=2601 | 宝箱的局部视野信息(local) |
| legal_actions | list | 8 | 合法动作 |
| legal_actions_feature | list | 8x4 | 合法动作特征 |
| recent_position_map | dict | 自定义 | 最近通过位置信息 |

### 特征处理

#### 位置信息

在特征预处理步骤中，提供了norm(pos)函数对坐标进行归一化，输入Position类型，输出值为元组；提供了convert_pos_to_grid_pos(x, z)函数对坐标进行网格化，输入(x, z)原始坐标，输出值为(x, z)网格化坐标：

```python
# 归一化示例代码
def norm(pos):
    norm_pos = (
        (pos.x + 64250) / 128000,
        (pos.z + 64250) / 128000,
    )
    return norm_pos

# 网格化示例代码
def convert_pos_to_grid_pos(x, z):

    x = (x + 64250) // 500
    z = (z + 64250) // 500

    # This step is necessary in order to be aligned with the order of json files
    # 这一步是为了与json文件的顺序保持一致
    x, z = z, x

    return x, z
```

#### 相对位置信息

RelativePosition表征的是英雄在任意位置时，物件的相对位置信息，比如方向、距离。其中：

- RelativeDirection是一个八维的离散化的方向信息。
- l2_distance是通过将地图网格化后计算出的从英雄当前格子到目标格子的欧式距离。
- grid_distance是通过将地图网格化后计算出的从英雄当前格子到目标格子的网格路径最短距离。

| 数据名 | 数据类型 | 数据描述 |
|--------|----------|----------|
| direction | RelativeDirection | 相对方位(离散化) |
| l2_distance | float | L2距离 |
| grid_distance | int | 网格化后的最短路径距离 |

> **特别说明：**
> - 在计算两点的grid_distance时，如果有一点的位置在障碍物里(无法走到的点位)，grid_distance统一设置为-1。
> - grid_distance表征的是英雄当前位置与某个物件(宝箱)的最短路径距离，该路径距离指的是网格路径。具体做法是先得到英雄和物件(宝箱)的网格坐标，然后通过bfs搜索算法按照L1距离的方式来计算出两个网格相距的最短路径。比如[1, 4]和[5, 10]的最短网格距离为| 1 - 5 | + | 4 - 10 | = 4 + 6 = 10。

### 奖励处理

代码包仅提供了简单的智能体实现，用户可以仔细阅读环境详述和数据协议，根据自己对环境的理解，进行特征工程、奖励开发等工作，不断提升智能体的能力。

---

### 算法介绍

我们在无限之谷代码包中提供了3个核心算法 Policy Gradient，A2C，PPO，同时，我们还提供了一个diy模板算法文件夹，用户可在该文件夹中自定义算法实现。

#### Policy Gradient 算法

Policy Gradient算法，也称为策略梯度方法，是一类用于优化策略函数的强化学习算法。与基于值函数的方法（如Q-learning）不同，策略梯度方法直接对策略进行参数化，并通过优化这些参数来最大化累积回报期望。策略梯度算法的核心思想是通过策略梯度来优化策略函数，具体来说，算法通过采样真实轨迹，计算回报，然后利用梯度上升法来更新策略模型。

#### A2C 算法

A2C（Advantage Actor-Critic）结合了策略梯度方法和价值函数估计，旨在提高学习效率和性能。A2C算法的核心思想是通过并行化环境交互和引入优势函数来优化策略和价值网络。A2C算法利用多个并行环境同时收集经验，然后统一更新网络参数，这样可以提高样本效率并减少训练时间。此外，A2C通过计算优势函数（Advantage Function）来评估动作相对于平均水平的优势，从而减少策略梯度的方差，提高训练稳定性。

#### PPO 算法

PPO（Proximal Policy Optimization）是一种策略优化算法，通过限制策略更新的幅度来提高训练稳定性。PPO使用裁剪的目标函数，确保新策略与旧策略之间的差异不会过大，从而避免训练过程中的剧烈波动。

---

### 算法监控信息

算法上报了reward等指标，用户可以通过腾讯开悟平台/客户端的监控功能查看。

**针对Policy Gradient的指标说明如下：**

| 指标名称 | 说明 |
|----------|------|
| reward | 累积回报，反映智能体的能力，正常训练情况下指标应震荡向上 |
| policy_loss | policy_loss的标量值，反应训练的进程，正常情况下指标应该是逐渐接近于0 |

**针对A2C的指标说明如下：**

| 指标名称 | 说明 |
|----------|------|
| reward | 累积回报，反映智能体的能力，正常训练情况下指标应震荡向上 |
| policy_loss | policy_loss的标量值，反应训练的进程，正常情况下指标应该是逐渐接近于0 |
| value_loss | value_loss的标量值，反应训练的进程，正常情况下指标应该是逐渐接近于0 |

**针对PPO的指标说明如下：**

| 指标名称 | 说明 |
|----------|------|
| reward | 累积回报，反映智能体的能力，正常训练情况下指标应震荡向上 |
| value_loss | 价值函数的预测误差，反映价值函数的准确性 |
| policy_loss | 策略函数的优化目标，反映策略的选择质量，正常情况下指标应在0附近震荡 |
| entropy_loss | 策略的探索性，反映策略的多样性 |

---

### 模型保存限制策略

为了避免用户保存模型的频率过于频繁，开悟平台对模型保存会有安全限制，不同的任务会有不同的限制，限制规则详情如下：

- 保存模型的频率限制: 2次/分钟
- 单个任务保存模型的次数限制：1000次

---

### 模型评估模式

用户可以在腾讯开悟平台上创建模型评估任务。创建任务时，可以对该任务的环境进行配置，配置信息如下：

```toml
# 环境配置
[env_conf]

# 宝箱是否随机
treasure_random = true

# 宝箱数量，-1表示随机
treasure_count = -1

# 宝箱id
treasure_id = []

# 怪物移动速度，-1表示随机
monster_speed = -1

# 怪物出现间隔，单位step，-1表示随机
monster_interval = -1

# 最大步数
max_step = 1000
```

另外，训练模式时，用户一般使用agent.predict方法进行决策；而在评估模式时，平台会调用agent.exploit方法进行决策，一般情况下，模型在训练和评估时的决策会因算法不同和用户设计不同，而有不同的行为，这部分由用户定义和实现。

---

## 1.4 数据协议

为了方便同学们调用原始数据和特征数据，下面提供了完整的数据结构供大家查阅。

---

### env_obs

```json
{
    "env_obs": {
        "env_id": str,             # 环境ID
        "frame_no": int,           # 环境帧号
        "observation": {           # 观测信息
            "step_no": int,
            "frame_state": dict,
            "game_info": dict,
            "map_info": list,
            "legal_act": list
        },
        "extra_info": {            # 额外信息
            "frame_state": dict,
            "game_info": dict,
            "result_code": int,
            "result_message": str
        },
        "terminated": bool,        # 是否任务结束（被怪物捕获,则为True）
        "truncated": bool          # 是否任务中断（超过最大步数,则为True）
    }
}
```

### observation

```json
{
    "step_no": int,                    # 当前步数（每步对应3帧），reset时为0
    "frame_state": {                   # 局部视野内的帧状态
        "heroes": [                    # 英雄列表
            {
                "pos": {"x": float, "z": float},  # 英雄位置
                "score": int,                      # 宝箱分数
                "step_score": int,                 # 步数分数
                "treasure_count": int              # 已收集宝箱数量
            }
        ],
        "monsters": [                  # 怪物列表（局部视野内）
            {
                "monster_id": int,
                "pos": {"x": float, "z": float},
                "start_pos": {"x": float, "z": float},
                "monster_speed": float,
                "monster_interval": int
            }
        ],
        "organs": [                    # 机关列表（局部视野内）
            {
                "sub_type": int,       # 类型：1=宝箱
                "config_id": int,      # 配置ID
                "status": int,         # 状态：1=可获取，0=已收集
                "pos": {"x": float, "z": float}
            }
        ]
    },
    "game_info": {                       # 场景信息
        "total_score": int,              # 总分数
        "step_no": int,                  # 步数
        "step_score": int,               # 步数得分
        "pos": {"x": float, "z": float}, # 英雄位置
        "treasure_collected_count": int, # 已收集宝箱数量
        "treasure_score": int,           # 宝箱得分
        "treasure_id": [int, ...],       # 已生成宝箱ID列表
        "treasure_count": int,           # 宝箱总数
        "monster_interval": int,         # 怪物生成间隔
        "first_monster_speed": float,    # 第一只怪物速度
        "second_monster_speed": float    # 第二只怪物速度

    },
    "map_info": [                      # 局部地图信息（以英雄为中心）
        {"values": [int, int, ...]},   # 第一行
        {"values": [int, int, ...]},   # 第二行
        ...                            # 大小: 101 × 101
    ],
    "legal_act": [int, int, ...]       # 合法动作（8维，对应8个方向）
}
```

**map_info 元素值含义:**

- 类型: list[list[int]]
- 说明: 局部地图信息（以英雄为中心的局部视野）
- 格式: 二维数组，每个元素表示地图格子类型
  - 0: 障碍物
  - 1: 可通行
  - 2: 宝箱
  - 3: 怪物

### extra_info

```json
{
    "frame_state": {
        "frame_no": int,         # 当前帧号
        "heroes": [...],         # 完整英雄信息
        "monsters": [...],       # 完整怪物信息
        "organs": [...]          # 完整物件信息
    },
    "game_info": {
        "total_score": float,           # 总得分
        "step_no": int,                 # 当前步数
        "step_score": float,            # 步数得分
        "pos": {"x": int, "z": int},    # 英雄当前位置
        "treasure_collected_count": int, # 已收集宝箱数量
        "treasure_score": int,          # 宝箱累计得分
        "treasure_count": int,          # 地图上宝箱总数
        "treasure_id": [int, ...],      # 宝箱ID列表
        "monster_interval": int,        # 第二个怪物的出现间隔
        "first_monster_speed": int,     # 第一个怪物的移动速度
        "second_monster_speed": int     # 第二个怪物的移动速度
    },
    "result_code": int,                 # 错误码,0表示正常
    "result_message": str               # 错误信息描述
}
```

**result_code常见错误码说明：**

| 错误码 | 说明 |
|--------|------|
| -100 | reset 配置创建错误 |
| -101 | 配置验证失败 |
| -103 | reset 执行异常 |
| -200 | step 执行异常 |
| -201 | 环境已结束但未重置 |

---

### env_reward

```json
{
    "env_reward": {
        "env_id": str,
        "frame_no": int,
        "reward": int          # 当前步的奖励
    }
}
```

---

# 2 腾讯开悟强化学习框架

## 2.1 综述

欢迎来到腾讯开悟！

腾讯开悟强化学习开发框架是基于强化学习系统系列技术标准打造的标准化开发套件。该框架为开发者提供了标准化的编程接口和丰富的工具集，支持开发者高效完成智能体开发、环境交互，以及模型的训练及预测流程。

### 训练流程简介

本开发框架的完整训练流程如下图所示：

如图，完整训练流程包含以下关键环节：

| 环节 | 介绍 |
|------|------|
| 智能体-环境循环交互 | - 智能体将环境提供的观测和奖励处理为符合预测函数输入要求的数据；- 调用预测函数，生成动作指令；- 将智能体输出的动作指令处理为符合环境输入要求的数据；- 环境执行动作后完成状态转移，并反馈新的观测数据和奖励数据； |
| 样本处理 | - 每个环境有不同的开始与结束逻辑，智能体与环境从开始到结束的完整交互过程，称为episode；- 智能体与环境每一次交互产生的结构化数据，称为样本；一个episode产生的样本序列称为轨迹；- 对轨迹数据进行处理，转换为规范化训练样本(SampleData)； |
| 模型迭代优化 | - 基于训练样本，通过算法持续更新模型参数，实现策略优化； |
| 智能体模型更新 | - 智能体加载最新模型，与环境继续循环交互； |

该流程通过强化学习分布式计算框架提供的训练工作流实现。基于此，开发框架主要包含三大核心模块：

- **强化学习环境系统**：提供标准的强化学习环境接口。开发者可以通过标准接口，实现智能体与环境的交互。
- **强化学习智能体开发套件**：提供标准的强化学习智能体接口，以及算法库、模型组件库等工具函数库。开发者可以通过工具函数库快速完成智能体的构建。
- **强化学习分布式计算框架**：提供标准接口，支持开发者按需实现训练工作流，运行单机或分布式的训练及评估任务。

### 代码包简介

开发者可以通过腾讯开悟平台所提供的强化学习项目使用开发框架。一个强化学习项目的代码目录如下：

```
📦 根目录
├── 📂 agent
│   ├── 📂 algorithm
│       └── 📄 __init__.py
│       └── 📄 algorithm.py
│   ├── 📂 conf
│       └── 📄 __init__.py
│       └── 📄 conf.py
│       └── 📄 train_env_conf.toml
│   ├── 📂 feature
│       └── 📄 __init__.py
│       └── 📄 definition.py
│       └── 📄 preprocessor.py
│   ├── 📂 model
│       └── 📄 __init__.py
│       └── 📄 model.py
│   ├── 📂 workflow
│       └── 📄 __init__.py
│       └── 📄 train_workflow.py
│   ├── 📄 __init__.py
│   └── 📄 agent.py
├── 📂 conf
│   ├── 📄 __init__.py
│   ├── 📄 configure_app.toml
├── 📂 log
└── 📄 train_test.py
```

**代码目录介绍：**

| 目录名 | 介绍 |
|--------|------|
| agent/ | 智能体子目录，智能体相关内容均集中于该目录，是开发者核心工作目录。 |
| conf/ | 配置文件目录，包含运行训练任务相关的配置，例如训练样本批处理大小batch_size等。 |
| log/ | 日志目录，存放运行代码测试脚本时生成的日志文件。 |
| train_test.py | 代码正确性测试脚本，该脚本会使用当前代码包完成一步训练。建议开发者在启动训练任务前，确保代码已通过该脚本检测。 |

**agent 目录详情：**

| 目录/文件名 | 介绍 |
|-------------|------|
| algorithm/ | 算法相关，开发者在该目录下完成算法实现，包含loss计算、模型优化等，详情见算法开发 |
| feature/ | 特征相关，开发者在该目录下完成数据结构定义和数据处理方法，以及样本处理和奖励计算，详情见数据处理与奖励设计 |
| model/ | 模型相关，开发者在该目录下完成模型实现。 详情见模型开发 |
| workflow/ | 工作流目录，开发者在该目录下完成训练工作流的开发。 详情见工作流开发 |
| agent.py | 智能体核心代码文件，开发者在该文件中完成预测、训练等核心函数的实现。 详情见智能体开发 |

标准代码包中都存在一个agent_diy子文件夹，该文件夹是预定义的智能体模板，可供开发者进行智能体的开发。

**conf 目录详情：**

| 文件名 | 介绍 |
|--------|------|
| configure_app.toml | 训练任务相关的配置，包括样本大小、样本池大小等。 |

---

通过对训练流程和代码包的介绍，相信开发者能够对腾讯开悟开发框架建立了初步认知。

接下来，我们将详细介绍每个模块的功能及使用方式。

---

## 2.2 环境

在综述中提到，强化学习训练流程离不开智能体与环境的持续交互，本文将详细介绍强化学习环境系统的功能及标准接口函数。

### 概述

强化学习环境是基于输入动作，输出观测、奖励等反馈的功能模块，用于表达强化学习算法所求解的问题场景。

开发框架通过场景适配模块，对仿真器进行封装，将其特化的接口、协议转换为强化学习环境统一的接口和协议，供智能体调用。

强化学习环境系统主要提供如下功能：

1. 接收配置信息，用于指定自身初始化方式，比如环境中各种元素的初始状态。
2. 输出观测、奖励信息，可用于智能体预测、训练。
3. 输出观测、奖励之外的其他信息，供强化学习系统相关组件使用以实现特定功能。其他信息可包括可视化数据、日志数据等，实现的功能包括环境可视化、运行状况监测等。
4. 接收动作指令，完成状态转移并产生新的观测和奖励。

---

### 环境使用

开发框架通过场景适配模块，将问题场景进行标准化封装，为开发者提供统一的交互接口与通信协议。由于环境之间存在差异，接口中所涉及的观测、奖励等信息的具体数据结构也有所不同，开发者需查阅所使用环境的官方数据协议文档以获取准确信息。

开发者可以在训练工作流的workflow中获取到对应环境的实例，通过标准接口实现智能体与环境的交互。

### 核心函数介绍

#### reset(usr_conf)

reset会将环境重置为环境配置文件中指定的状态，并且返回初始观测。

```python
# usr_conf为开发者传入的环境配置
obs, state = env.reset(usr_conf = usr_conf)
```

**Parameters**

| 参数名 | 介绍 |
|--------|------|
| usr_conf | dict类型，环境配置文件 |

**Returns**

| 参数名 | 介绍 |
|--------|------|
| obs | dict类型，环境观测信息 |
| state | dict类型，环境全局信息 |

---

#### step(act, stop_game = false)

环境会执行传入的act动作指令，完成一次状态转移，并返回新的观测和奖励等信息。

```python
frame_no, _obs, score, terminated, truncated, _state = env.step(act, stop_game = false)
```

**Parameters**

| 参数名 | 介绍 |
|--------|------|
| act | dict类型，环境执行的动作 |
| stop_game | bool类型，是否结束当前对局 |

**Returns**

| 参数名 | 介绍 |
|--------|------|
| frame_no | int类型，当前环境实例运行时的帧号 |
| _obs | dict字典类型，当前帧的观测信息 |
| score | int类型，当前帧的奖励信息 |
| terminated | bool类型，当前环境实例是否结束 |
| truncated | bool类型，当前环境实例是否异常或中断 |
| _state | dict字典类型，当前帧的全部状态信息 |

---

## 2.3 智能体

### 2.3.1 综述

智能体是强化学习系统中的核心模块，在开发框架综述中提到，完整训练流程包括：

| 环节 | 介绍 |
|------|------|
| 智能体-环境循环交互 | - 智能体将环境提供的观测和奖励处理为符合预测函数输入要求的数据；- 调用预测函数，生成动作指令；- 将智能体输出的动作指令处理为符合环境输入要求的数据；- 环境执行动作后完成状态转移，并反馈新的观测数据和奖励数据； |
| 样本处理 | - 每个环境有不同的开始与结束逻辑，智能体与环境从开始到结束的完整交互过程，称为episode；- 智能体与环境每一次交互产生的结构化数据，称为样本；一个episode产生的样本序列称为轨迹；- 对轨迹数据进行处理，转换为规范化训练样本(SampleData)； |
| 模型迭代优化 | - 基于训练样本，通过算法持续更新模型参数，实现策略优化； |
| 智能体模型更新 | - 智能体加载最新模型，与环境继续循环交互； |

基于上述训练流程，我们将智能体的开发分为四个部分：

1. **数据处理及奖励设计**：介绍基于环境观测数据进行特征处理、样本处理和奖励设计的方法。
2. **模型开发**：介绍模型开发接口及开发方法。
3. **算法开发**：介绍包括算法开发接口及开发方法。
4. **工作流开发**：介绍开发者开发自定义训练工作流的方法。

接下来，将通过独立的章节对强化学习智能体开发套件中每个模块的功能及接口函数进行介绍。

---

### 2.3.2 特征处理

环境返回的数据通常无法直接作为智能体预测和训练的输入，开发者需要完成特征处理、样本处理和奖励设计，确保数据结构与类型符合智能体的接口规范。

#### 特征处理

在特征处理时，开发者需要完成四个关键的开发工作，分别是定义数据结构、观测处理、动作处理。

#### 定义数据结构

**开发目录**：`<智能体文件夹>/feature/definition.py`

首先，开发者需要定义智能体可以使用的数据结构（类）。

开发框架已经预先定义好了三种数据类型：ObsData, ActData, SampleData。

- ObsData和ActData分别表示智能体预测的输入和输出，将会由agent.predict()使用；
- SampleData为训练样本的数据类型，训练样本将会被agent.learn()使用，进行模型训练。

**核心函数介绍**

**create_cls**

用于动态创建数据结构（类）。ObsData, ActData, SampleData是训练流程必需的三类，但每一个类的数据结构包含哪些属性完全由开发者自定义，属性名称和属性数量没有限制。

```python
ObsData = create_cls("ObsData",
    feature=None,
)
ActData = create_cls("ActData",
    action=None,
    prob=None,
)
SampleData = create_cls("SampleData",
    npdata=None
)
```

**Parameters**

| 参数名 | 介绍 |
|--------|------|
| 第一个参数 | 字符串类型，类的名称 |
| 其余参数 | 类的属性，默认值为None，由开发者自行定义 |

---

#### 观测处理

**开发目录**：`<智能体文件夹>/agent.py`

由于环境的reset和step接口返回的数据属于原始观测数据，无法直接作为智能体预测时的输入，开发者需要将这部分数据进行特征化。

**核心函数介绍**

**observation_process**

将环境返回的观测数据转换成ObsData类型数据。

很多情况下，特征工程包含了大量的数值处理、数据转换和领域知识，我们建议将大量的特征处理代码在`<智能体文件夹>/feature/preprocessor.py`文件中实现，然后由observation_process进行调用。

```python
def observation_process(self, obs, state=None):
    return ObsData(feature=feature, legal_act=legal_actions)
```

**Parameters**

| 参数名 | 介绍 |
|--------|------|
| obs | Observation类型，env.reset和env.step返回的环境观测数据 |
| state | EnvInfo类型，env.reset和env.step返回的环境状态数据 |

**Return**

| 参数名 | 介绍 |
|--------|------|
| ObsData | 开发者定义的ObsData类型的数据，将作为agent.predict()函数的输入。 |

---

#### 动作处理

**开发目录**：`<智能体文件夹>/agent.py`

由于环境的step接口的输入须要满足环境的特定数据协议，开发者需要将智能体预测的输出转换为符合环境step接口输入要求的数据。

**核心函数介绍**

**action_process**

将智能体预测输出的ActData类型数据转换成环境可以接收的动作数据。

```python
def action_process(self, act_data):
    return act_data.act
```

**Parameters**

| 参数名 | 介绍 |
|--------|------|
| act_data | 开发者定义的ActData类型的数据 |

**Return**

| — | 环境能处理的动作数据类型，作为env.step()的输入 |

---

#### 奖励设计

**开发目录**：`<智能体名称>/feature/definition.py`

这里的奖励特指强化学习中的Reward，注意要与环境反馈的Score进行区分。Score通常用于衡量智能体在环境中的实际表现。开发者在设计Reward时，有非常大的灵活性，不仅可以基于环境返回的观测信息，还可以加入开发者对问题的理解、经验或者知识。

**核心函数介绍**

**reward_shaping**

开发框架预设的奖励设计函数接口，开发者可以通过该函数实现复杂的奖励计算，在训练工作流中调用。

```python
def reward_shaping(obs, _obs, state, _state):
    return reward
```

**Parameters**

| — | 参数个数和类型不限制，可以是环境信息、智能体信息、开发者的经验和知识等。 |

**Return**

| — | 数值类型，计算出的reward值 |

---

#### 样本处理

**开发目录**：`<智能体文件夹>/feature/definition.py`

由于环境与智能体交互过程中产生的轨迹数据无法直接作为智能体训练时的输入，开发者需要将轨迹数据转换为训练样本数据。

**核心函数介绍**

**sample_process**

将环境与智能体交互过程中产生的轨迹数据转换成开发者定义的SampleData类型数据。

```python
@attached
def sample_process(self, list_game_data):
    return [SampleData(**i.__dict__) for i in list_game_data]
```

**Parameters**

| 参数名 | 介绍 |
|--------|------|
| list_game_data | list(Frame)类型， 使用开发者自定义的Frame作为输入，因为样本一般进行批处理，所以传入列表 |

**Return**

| 参数名 | 介绍 |
|--------|------|
| list(SampleData)类型 | SampleData类型的数据组成的列表 |

---

为了支持分布式训练，样本数据需要进行网络传输，由于SampleData无法直接进行网络传输，需要先转换成Numpy的Array，待传输到对端之后再由np.Array转换成SampleData。

因此，开发者需要实现两个转换函数 SampleData2NumpyData和 NumpyData2SampleData，这两个函数互为反函数。

**注意**：由于这两个函数会被分布式计算框架调用，因此这两个函数的实现都必须包含一个装饰器`@attached`

#### SampleData2NumpyData

将SampleData转换为NumpyData。

```python
@attached
def SampleData2NumpyData(g_data):
    return g_data.npdata
```

**Parameters**

| 参数名 | 介绍 |
|--------|------|
| g_data | SampleData 类型 |

**Return**

| — | Numpy.array类型 |

---

#### NumpyData2SampleData

将NumpyData转换为SampleData。

```python
@attached
def NumpyData2SampleData(s_data):
    return SampleData(npdata=s_data)
```

**Parameters**

| 参数名 | 介绍 |
|--------|------|
| s_data | Numpy.array 类型 |

**Return type**

| — | SampleData类型 |

---

### 2.3.3 算法开发

**开发目录**：`<智能体名称>/algorithm/algorithm.py`

在完成特征处理和奖励设计后，开发者还需要实现强化学习算法，以通过特定优化方法更新模型参数。

以下为实现强化学习算法的核心函数介绍，有关函数的更多细节可以查阅分布式计算框架。

#### 核心函数介绍

**learn**

实现强化学习优化算法的核心方法，该函数输入为训练样本数据，开发者需基于不同的算法完成相关实现，包括优化方法、损失计算等。

```python
def learn(self, list_sample_data):
    """
    Implementing the core method of the algorithm
    实现算法的核心方法
    """
    loss = 0                         # 基于不同算法实现loss计算 Calculate loss
    loss.backward()                  # 计算梯度 Calculate gradient
    self.optimizer.step()            # 通过梯度下降等方法更新模型 Update weights
```

**Parameters**

| 参数名 | 介绍 |
|--------|------|
| list_sample_data | list类型，训练样本(SampleData)列表 |

---

### 2.3.4 模型开发

**开发目录**：`<智能体名称>/model/model.py`

一个强化学习模型是基于特征作为输入数据，输出策略的神经网络模型。

开发者需要在model.py文件中，实现神经网络模型。开发框架要求，模型类需继承 torch.nn.Module 类，即符合Pytorch模型的实现规范。

```python
class Model(nn.Module):
    def __init__(self, state_shape, action_shape=0, softmax=False):
        super().__init__()
```

---

### 2.3.5 工作流开发

#### 训练工作流

在完成智能体开发后，需要进一步实现由分布式计算框架提供的训练工作流接口，使智能体和环境持续交互，收集训练样本，迭代模型参数，最终完成策略的优化。

**核心函数介绍**

**workflow**

通过该函数实现强化学习训练工作流，调用智能体和环境提供的接口，完成环境交互、样本收集和模型更新。

```python
@attached
def workflow(envs, agents, logger=None, monitor=None):
```

**Parameters**

| 参数名 | 介绍 |
|--------|------|
| envs | list类型，环境列表，返回当前正在运行的环境。 |
| agents | list类型，智能体列表，通过调用开发者实现的 `<智能体名称>/agent.py` 实例化 Agent, 并作为输入传入 workflow。 |
| logger | Logger类型，框架提供的日志组件，接口与 python 的 logging 库一致。 |
| monitor | Monitor类型，框架提供的监控组件。 |

---

接下来，我们将通过一个训练工作流关键步骤的代码示例（具体实现由开发者完成），说明如何通过训练工作流实现完整训练流程。

```python
@attached
def workflow(envs, agents, logger=None, monitor=None):
    # Get the environment and agent
    # 获取环境和智能体
    env, agent = envs[0], agents[0]

    # Execute several epochs
    # 执行若干次epoch
    epoch_num = 1000

    # Each epoch executes several episodes
    # 每个epoch执行若干个episode
    episode_num_every_epoch = 1000

    # Training loop
    # 训练循环
    for epoch in range(epoch_num):
        # After each episode, the trajectory data is converted into training samples for training.
        # 在每一个episode结束之后，将轨迹数据转换成训练样本进行训练
        for g_data in run_episodes(episode_num_every_epoch, env, agent, logger, monitor):
            # Agent training. If single-machine training, the model is trained directly; if distributed training, samples are sent to the sample-pool.
            # agent进行训练。如果是单机训练，则直接对模型进行训练；如果是分布式训练，则将训练样本发送到样本池。
            agent.learn(g_data)
            # Ensure that the next training sample collected is new
            # 清空g_data，确保下一次搜集的训练样本是新的
            g_data.clear()

        # Save the model at intervals
        # 依据时间间隔保存模型
        now = time.time()
        if now - last_save_model_time >= 300:
            agent.save_model()
            last_save_model_time = now


def run_episodes(n_episode, env, agent, logger, monitor):
    # Run several episodes
    # 运行若干个episode
    for episode in range(n_episode):
        # Reset data at the beginning of an episode
        # 在episode开始时重置数据
        done = False
        collector = list()

        # Reset enviroment and get initial info
        # 重置环境, 并获取环境初始状态
        obs, state = env.reset(usr_conf=usr_conf)

        # Load the latest model and call it on demand; if in stand-alone mode, there is no need to load the remote model
        # 加载最新模型，按需调用；若训练采用单机模式，则无需加载远程模型，可不调用该函数
        agent.load_model(id="latest")

        # Run an episode loop
        # 运行一个episode循环
        while not done:
            # Agent performs inference, gets the predicted action for the next frame
            # 调用智能体预测函数，获取下一时刻的动作
            act_data = agent.predict(list_obs_data=[obs_data])[0]

            # Unpack ActData into action
            # 将智能体输出的ActData数据转换为符合环境数据协议要求的动作数据
            act = agent.action_process(act_data)

            # Interact with the environment, execute actions, get the next state
            # 调用环境step接口，与环境交互, 执行动作, 获取下一时刻的状态
            frame_no, _obs, score, terminated, truncated, _state = env.step(act)
            if _obs == None:
                break

            # Feature processing
            # 对环境返回的观测数据进行处理
            _obs_data = agent.observation_process(_obs, _state)

            # Disaster recovery
            # 容灾
            if truncated and frame_no == None:
                break

            # Calculate reward
            # 计算reward
            reward = reward_shaping(obs_data, _obs_data, state, _state)

            # Episode done signal
            # episode结束信号
            done = terminated or truncated

            # Construct sample
            # 构造样本
            frame = Frame(
                obs=obs_data.feature,
                _obs=_obs_data.feature,
                act=act,
                rew=reward,
                done=done,
            )
            collector.append(frame)

            # If the game is over, the sample is processed and sent to training
            # 如果episode结束，则进行样本处理，将样本送去训练
            if done:
                if len(collector) > 0:
                    collector = sample_process(collector)
                    # Return samples
                    # 返回样本数据, agent会调用agent.learn(g_data)进行训练
                    yield collector
                break

            # Status update
            # 状态更新
            obs_data = _obs_data
            obs = _obs
            state = _state
```

---

### 2.3.6 智能体开发

**开发目录**：`<智能体名称>/agent.py`

在完成模型和算法后，开发者还需要实现强化学习智能体，智能体使用模型进行决策、与环境交互并通过算法更新模型参数。

以下为实现强化学习智能体的核心函数介绍，有关函数的更多细节可以查阅分布式计算框架。

#### 核心函数介绍

**learn**

该函数输入为训练样本数据，开发者需要在该函数中调用算法消费训练样本进行训练。

当然，在不同的训练模式下，该函数使用方法有所不同：

- **单机训练**：开发者需要在训练工作流中手动调用该函数以进行一步训练。
- **分布式训练**：
  - 该函数作为训练函数会被循环执行，无需开发者手动调用。
  - 但该函数还作为样本发送函数，开发者需要在训练工作流中手动调用，以将样本发送至样本池。

```python
def learn(self, list_sample_data):
    self.algo.learn(list_sample_data)        # 调用算法消费训练样本进行训练 Call algorithm to train model
```

**Parameters**

| 参数名 | 介绍 |
|--------|------|
| list_sample_data | list类型，训练样本(SampleData)列表 |

---

#### predict

该方法通过调用模型进行预测，通常在训练时调用该方法，依策略的概率分布采样或引入随机概率。

```python
@predict_wrapper
def predict(self, list_obs_data, list_state):
    return [ActData]
```

**Parameters**

| 参数名 | 介绍 |
|--------|------|
| list_obs_data | list类型，观测数据(ObsData)列表 |
| list_state | 可选参数，list类型，环境返回的状态数据列表 |

**Return**

| 参数名 | 介绍 |
|--------|------|
| List(ActData) | list类型，开发者定义的动作数据(ActData)列表 |

---

#### exploit

该方法通过调用模型进行预测，通常在评估时调用该方法，选取策略中概率最高的动作或者策略认为最优的动作。

```python
@exploit_wrapper
def exploit(self, observation):
```

**Parameters**

| 参数名 | 介绍 |
|--------|------|
| observation | dict类型，环境观测字典，评估工作流中将原始的环境观测信息作为输入传入 agent.exploit()。 |

**Return**

| 参数名 | 介绍 |
|--------|------|
| action | list类型，动作列表，环境可以直接使用的动作指令 |

---

#### load_model

智能体通过该接口完成模型参数加载。在上文中提到，Actor会从模型池中获取最新模型参数文件，开发者需要手动调用load_model()函数，使智能体完成模型参数加载。

```python
@load_model_wrapper
def load_model(self, path=None, id="1"):
    # When loading the model, you can load multiple files,
    # and it is important to ensure that each filename matches the one used during the save_model process.
    # 加载模型, 可以加载多个文件, 注意每个文件名需要和save_model时保持一致
    model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
    self.model.load_state_dict(
        torch.load(model_file_path, map_location=self.device),
    )
```

**Parameters**

| 参数名 | 介绍 |
|--------|------|
| path | string类型，加载模型参数文件的路径，开发框架根据使用场景得到相应的路径, 并作为输入传入 load_model |
| id | string类型，模型参数文件的 id，使用 id 指定加载的模型参数文件 |

---

#### save_model

开发者可以通过该函数保存当前时刻的模型文件及智能体代码包，开发框架会将开发者需要保存的内容打包为zip格式的文件。

当开发者使用腾讯开悟客户端开发时，开发框架会在客户端指定目录下存储该zip文件。

当开发者使用腾讯开悟平台时，开发框架会将该zip文件存储在云端，开发者可以通过平台的训练管理模块查看每一个训练任务的zip文件，即模型。

```python
@save_model_wrapper
def save_model(self, path=None, id="1"):
    # To save the model, it can consist of multiple files,
    # and it is important to ensure that each filename includes the "model.ckpt-id" field.
    # 保存模型, 可以是多个文件, 需要确保每个文件名里包括了model.ckpt-id字段
    model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"

    # Copy the model's state dictionary to the CPU
    # 将模型的状态字典拷贝到CPU
    model_state_dict_cpu = {k: v.clone().cpu() for k, v in self.model.state_dict().items()}
    torch.save(model_state_dict_cpu, model_file_path)
```

**Parameters**

| 参数名 | 介绍 |
|--------|------|
| path | string类型，模型文件保存的路径，开发框架根据使用场景得到相应的路径, 并作为输入传入 save_model |
| id | string类型，模型文件的索引，开发框架获取到模型池中最新模型的索引, 并作为输入传入 save_model |

---

## 2.4 分布式计算框架

在强化学习项目的开发中，分布式计算框架是支撑大规模训练任务的核心基础设施。本开发框架提供了由腾讯王者荣耀团队自主研发的强化学习分布式计算框架KaiwuDRL，通过并行化计算、高效资源调度和分布式协同优化，显著提升智能体训练的效率与稳定性。

接下来，我们将详细介绍KaiwuDRL的系统架构与核心能力，帮助开发者进一步理解训练和评估工作流的运行逻辑。

### 总体架构

### 组件介绍

如上图所示，KaiwuDRL 的整体架构包括 Environment、Aisrv、Actor、Learner 等强化学习组件（均支持多实例并行运行）。此外，还集成了通信、日志、监控、对象存储等基础组件。

组件介绍如下表：

| 组件名称 | 功能描述 |
|----------|----------|
| Environment | 环境服务组件，负责运行强化学习环境，支持通过标准接口与环境交互，并返回环境的观测obs。 |
| Aisrv | 训练流程中枢，负责收集环境样本，运行训练、评估工作流，以及处理各个组件间的数据传输。 |
| Actor | 预测服务组件，负责响应Aisrv的预测请求，调用智能体 predict() 或 exploit() 函数生成动作决策结果。 |
| Learner | 训练服务组件，负责采集训练样本，调用智能体 learn() 函数完成梯度计算及模型迭代。 |
| MemoryPool | 样本存储组件，简称样本池。负责存储训练样本，接收 Aisrv 打包的训练样本，发往 Learner 用于智能体训练。 |
| ModelPool | 模型存储组件，简称模型池。负责存储模型参数文件，接收 Learner 产出的模型参数文件，将最新的模型参数文件发送给Actor。 |
| 日志 | 日志采集组件，负责记录强化学习系统中各个组件的运行日志，支持通过标准接口上报日志。 |
| 监控 | 监控采集组件，负责采集系统资源使用率、训练指标趋势等数据，支持通过标准接口上报数据指标。 |

MemoryPool和ModelPool仅在分布式训练时启用。

---

### 服务介绍

基于上述组件，KaiwuDRL 提供了预测服务和训练服务：

#### 预测服务

1. Aisrv → Environment：发送环境配置并创建新一局episode；
2. Environment → Aisrv：返回原始观测数据；
3. Aisrv → Actor：Aisrv基于原始观测数据，向Actor发送预测请求；
4. Actor：使用预测请求中的观测进行特征处理，智能体基于特征处理后的数据进行预测，并且将预测数据处理为环境可以识别的动作指令，发送给Aisrv；
5. Aisrv：使用动作指令与环境Environment进行交互，Environment返回新的观测；

#### 训练服务

1. Aisrv：预测服务不断产生轨迹数据，Aisrv完成样本处理，并发送至样本池；
2. Learner：从样本池按批采集样本进行训练，并将最新的模型参数同步至Actor；

---

### 工作流

KaiwuDRL提供了训练、评估工作流的接口函数，开发者可以按需灵活调用上述组件和服务，以实现模型的训练和评估。

#### 训练工作流

**开发目录**：`<智能体名称>/workflow/train_workflow.py`

**workflow**

训练工作流的核心函数，在workflow中可自定义训练流程。可以在智能体/工作流开发中查看详细的训练工作流代码示例。

```python
@attached
def workflow(envs, agents, logger=None, monitor=None):
```

**Parameters**

| 参数名 | 介绍 |
|--------|------|
| envs | list类型，环境列表，返回当前正在运行的环境。 |
| agents | list类型，智能体列表，通过调用开发者实现的 `<智能体名称>/agent.py` 实例化 Agent, 并作为输入传入 workflow。 |
| logger | Logger类型，框架提供的日志组件，接口与 python 的 logging 库一致。 |
| monitor | Monitor类型，框架提供的监控组件。 |

---

#### 评估工作流

在运行训练任务（训练工作流）并获得模型文件后，可以通过运行评估任务（评估工作流），对模型能力进行验证。

当开发者在使用腾讯开悟平台所提供的强化学习项目时，评估工作流由腾讯开悟官方实现，开发者无法修改。评估工作流会调用开发者自定义的agent.exploit()函数。

---

### Agent

KaiwuDRL提供了智能体相关的接口函数，开发者可以按需实现以下接口函数，并在训练工作流中调用。

**开发目录**：`<智能体名称>/agent.py`

#### learn

该函数输入为训练样本数据，开发者需要在该函数中调用算法消费训练样本进行训练。

当然，在不同的训练模式下，该函数使用方法有所不同：

- **单机训练**：开发者需要在训练工作流中手动调用该函数以进行一步训练。
- **分布式训练**：
  - 该函数作为训练函数会被循环执行，无需开发者手动调用。
  - 但该函数还作为样本发送函数，开发者需要在训练工作流中手动调用，以将样本发送至样本池。

```python
@learn_wrapper
def learn(self, list_sample_data):
```

**Parameters**

| 参数名 | 介绍 |
|--------|------|
| list_sample_data | list类型，训练样本(SampleData)列表，Learner从样本池按照配置项train_batch_size采样一批样本, 作为输入传入 learn() 函数。 |

**配置项**

**开发目录**: `/conf/configure_app.toml`

Learner在每一次执行learn()函数时，会从样本池采样一批样本作为输入。并按照开发者配置的频次dump_model_freq保存模型参数文件，模型同步服务将按照配置model_file_sync_per_minutes将模型参数文件推送至模型池。

Actor中的模型同步服务将按照配置model_file_sync_per_minutes，从模型池获取最新模型参数文件。

相关配置项如下：

```toml
[app]
# The time interval for executing the learn() function, configurable to throttle the Learner and balance sample production/consumption.
# 执行learn函数进行训练的时间间隔，可通过该配置让Learner休息以调节样本生产消耗比
learner_train_sleep_seconds = 0.001

# Replay buffer configurations
# 样本池容量
replay_buffer_capacity = 4096

# The ratio of the sample pool capacity that triggers training
# 当样本池中的样本占总容量的比例达到该值时，启动训练
preload_ratio = 1.0

# When new samples are added to the sample pool, the logic for removing old samples: reverb.selectors.Lifo, reverb.selectors.Fifo
# 当新样本加入样本池时，旧样本的移除逻辑，可选项：reverb.selectors.Lifo, reverb.selectors.Fifo
# reverb.selectors.Lifo：先进后出(Last In, First Out)
# reverb.selectors.Fifo：先进先出(First In, First Out)
reverb_remover = "reverb.selectors.Fifo"

# The sampling logic of the Learner from the sample pool: reverb.selectors.Fifo, reverb.selectors.Uniform
# Learner从样本池中采样的逻辑，可选项：reverb.selectors.Fifo, reverb.selectors.Uniform
# reverb.selectors.Uniform：Samples are selected uniformly at random from the replay buffer, with each sample having an equal probability of being chosen.
# reverb.selectors.Uniform：从回放缓冲区中随机均匀地选择样本，每个样本被选中的概率相同。
# reverb.selectors.Fifo：Samples are selected in the order they were added to the replay buffer.
# reverb.selectors.Fifo：按照先进先出从回放缓冲区中选择样本。
reverb_sampler = "reverb.selectors.Uniform"

# Training batch size limit for Learner
# Learner训练时样本批处理大小
train_batch_size = 2048

# Model dump frequency (steps)
# 训练间隔多少步输出模型参数文件
dump_model_freq = 1000

# The Learner pushes model updates, and the frequency at which Actors fetch the model (in minutes).
# Learner推送模型参数文件至模型池，以及Actor从模型池获取模型参数文件的频次（单位：分钟）
model_file_sync_per_minutes = 1

# he number of model updates pushed per learner iteration, and the maximum number of updates each actor can fetch at once (cap: 50).
# Learner每次推送模型参数文件，以及Actor每次获取模型参数文件的数量（上限：50）
modelpool_max_save_model_count = 1
```

---

#### predict

该方法通过调用模型进行预测，通常在训练时调用该方法，依策略的概率分布采样或引入随机概率。

```python
@predict_wrapper
def predict(self, list_obs_data, list_state):
    return [ActData]
```

**Parameters**

| 参数名 | 介绍 |
|--------|------|
| list_obs_data | list类型，观测数据(ObsData)列表 |
| list_state | 可选参数，list类型，环境返回的状态数据列表 |

**Return**

| 参数名 | 介绍 |
|--------|------|
| List(ActData) | list类型，开发者定义的动作数据(ActData)列表 |

---

#### exploit

该方法通过调用模型进行预测，通常在评估时调用该方法，选取策略中概率最高的动作或者策略认为最优的动作。

```python
@exploit_wrapper
def exploit(self, observation):
```

**Parameters**

| 参数名 | 介绍 |
|--------|------|
| observation | dict类型，环境观测字典，评估工作流中将原始的环境观测信息作为输入传入 agent.exploit()。 |

**Return**

| 参数名 | 介绍 |
|--------|------|
| action | list类型，动作列表，环境可以直接使用的动作指令 |

---

#### load_model

智能体通过该接口完成模型参数加载。在上文中提到，Actor会从模型池中获取最新模型参数文件，开发者需要手动调用load_model()函数，使智能体完成模型参数加载。

```python
@load_model_wrapper
def load_model(self, path=None, id="1"):
    # When loading the model, you can load multiple files,
    # and it is important to ensure that each filename matches the one used during the save_model process.
    # 加载模型, 可以加载多个文件, 注意每个文件名需要和save_model时保持一致
    model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
    self.model.load_state_dict(
        torch.load(model_file_path, map_location=self.device),
    )
```

**Parameters**

| 参数名 | 介绍 |
|--------|------|
| path | string类型，加载模型参数文件的路径，开发框架根据使用场景得到相应的路径, 并作为输入传入 load_model |
| id | string类型，模型参数文件的 id，使用 id 指定加载的模型参数文件 |

---

#### save_model

开发者可以通过该函数保存当前时刻的模型文件及智能体代码包，开发框架会将开发者需要保存的内容打包为zip格式的文件。

当开发者使用腾讯开悟客户端开发时，开发框架会在客户端指定目录下存储该zip文件。

当开发者使用腾讯开悟平台时，开发框架会将该zip文件存储在云端，开发者可以通过平台的训练管理模块查看每一个训练任务的zip文件，即模型。

```python
@save_model_wrapper
def save_model(self, path=None, id="1"):
    # To save the model, it can consist of multiple files,
    # and it is important to ensure that each filename includes the "model.ckpt-id" field.
    # 保存模型, 可以是多个文件, 需要确保每个文件名里包括了model.ckpt-id字段
    model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"

    # Copy the model's state dictionary to the CPU
    # 将模型的状态字典拷贝到CPU
    model_state_dict_cpu = {k: v.clone().cpu() for k, v in self.model.state_dict().items()}
    torch.save(model_state_dict_cpu, model_file_path)
```

**Parameters**

| 参数名 | 介绍 |
|--------|------|
| path | string类型，模型文件保存的路径，开发框架根据使用场景得到相应的路径, 并作为输入传入 save_model |
| id | string类型，模型文件的索引，开发框架获取到模型池中最新模型的索引, 并作为输入传入 save_model |

---

## 2.5 其他工具

### 加载预训练模型

本框架支持在已有的模型（称为预训练模型）基础上继续训练。预训练模型结构和继续训练的模型结构需保持一致。

- 当使用腾讯开悟平台时，在创建训练任务时，可以在弹窗中选择预训练模型以继续训练。
- 当使用腾讯开悟客户端时，需要将训练好的模型文件（客户端工作空间/ckpt路径下的pkl类型文件）放入代码包中，并在代码包conf/configure_app.toml中完成预训练模型相关的配置，配置说明如下：

```toml
# Whether to enable the preload model function. If enabled (true), the model specified by preload_model_id will be loaded as the initial model in the preload_model_dir directory; if disabled (false), no preloading will be performed.
# 是否启用加载预训练模型功能，若开启(true)，将在preload_model_dir目录下加载由preload_model_id指定的模型作为初始模型；若关闭(false)，则不加载预训练模型。
preload_model = false

# The relative path of the preloaded model folder (the variable name {agent_name} refers to the agent_algorithm name directory in the code package). It is only effective when preload_model=true. When the preload model function is enabled, you need to create a new ckpt folder under the agent_algorithm name directory in the code package and place the model file (.pkl) there.
# 预训练模型文件夹相对路径(变量名{agent_name}指代码包中agent_算法名目录)，仅在preload_model=true时生效；当开启加载预训练模型功能时，需要在代码包中agent_算法名目录下新建ckpt文件夹，将模型文件（.pkl）放置此即可。
preload_model_dir = "{agent_name}/ckpt"

# The identification ID of the preloaded model (here refers to the number of model training steps). This ID corresponds to the number of training steps recorded in the model file name. It only takes effect when preload_model=true.
# Note that it is forbidden to modify the original model file name, otherwise the model preloading process will fail.
# 预训练模型的标识ID（这里指模型训练步数），该ID对应模型文件名中的训练步数记录。仅在preload_model=true时生效。
# 注意，禁止修改原始模型文件名，否则将导致预训练模型加载失败。
preload_model_id = 1000
```

#### 客户端继续训练使用示例

以加载 40052步的DQN模型 为例：

- 解压模型压缩包，在ckpt路径下找到名为 model.ckpt-40052.pkl 的模型文件
- 将模型文件放到代码包路径下，例如在agent_dqn内新建一个ckpt目录，放入上述模型文件
- 在代码包conf/configure_app.toml中完成以下配置：

```toml
preload_model = true
preload_model_dir = "agent_dqn/ckpt"
preload_model_id = 40052
```

完成上述步骤后，Aisrv和Learner将加载预训练模型以继续训练。

---

### 加载对手模型功能

在使用腾讯开悟平台时，部分项目支持从平台的模型管理中加载自定义模型。 用户可以在 kaiwu.json 文件中配置期望用作对手模型的模型 ID。智能体通过 load_opponent_agent() 函数完成对手模型的加载。

该功能允许智能体加载指定的网络结构和参数，并根据模型文件中的实现调用 agent 的reset、predict、exploit等方法。 对手模型作为固定水平的自定义模型，可以在 PVP 任务中用于训练评估，以反映训练过程中模型水平的提高。

以下是load_opponent_agent()函数示例：

```python
@load_opponent_agent_wrapper
def load_opponent_agent(self, id="1"):
    # Framework provides loading opponent agent function, no need to implement function content
    # 框架提供的加载对手模型功能，无需实现函数内容
    pass
```

**Parameters**

| 参数名 | 介绍 |
|--------|------|
| id | string类型，平台模型管理的 模型id，使用 id 指定加载的模型文件 |

以下是kaiwu.json的配置示例：

```json
{
    "model_pool": [609, 608]
}
```

---

### 策略类型选择

本框架支持两种核心的训练范式：同策略（On-Policy）与异策略（Off-Policy）。此选择是训练流程的基础，将直接影响数据的使用方式、内存占用以及部分超参数的推荐设置。

| 特性 | On-Policy | Off-Policy |
|------|-----------|-------------|
| 数据来源 | 必须使用由当前正在训练的策略实时与环境交互产生的新数据。 | 可以使用历史经验数据，这些数据可能来自旧版本的策略或不同的探索策略。 |
| 数据使用 | 一次性为主。数据在用于一次策略更新后通常被丢弃，以确保训练数据与当前策略的一致性。 | 可重复利用。数据被存储在"经验回放缓冲区"中，可被多次采样用于训练。 |
| 主要优势 | 训练稳定，理论收敛性有保障，行为容易理解。 | 数据效率高，可充分利用每一次交互获得的数据，更适合现实世界中交互成本高的场景。 |

#### 配置参数详解

在代码包conf/configure_app.toml中，通过以下参数进行设置：

```toml
# Training paradigm: on-policy or off-policy
# 训练时采用on-policy, off-policy
algorithm_on_policy_or_off_policy = "on-policy"

# Model dump frequency (steps)
# Interval (in steps) for saving model parameter files.
# on_policy: set to 1 for optimal training.
# off_policy: recommend 1000 to reduce save frequency.
# 训练间隔多少步输出模型参数文件, on_policy时设置为1训练效果最好，off_policy时建议设置为1000, 减少模型保存频率
dump_model_freq = 1
```

---

### 2.5.1 监控与日志

本文介绍如何使用腾讯开悟平台的监控与日志功能，帮助您实时掌握训练状态、快速定位问题。

#### 监控

##### 查看监控面板

在训练管理页面，点击查看监控按钮，即可打开监控面板。

##### 监控面板组成

监控面板包含两个核心模块：

| 模块 | 功能 |
|------|------|
| 错误日志数量 | 展示各模块的错误日志统计，点击可查看详情 |
| 监控指标图 | 展示训练过程中的各类数据指标 |

##### 指标分类

监控指标分为四类：

| 分类 | 说明 |
|------|------|
| 基础指标（basic） | 训练进度相关的核心数据，如训练步数、预测次数等 |
| 硬件指标（hardware） | 资源使用情况，如 CPU、GPU、内存利用率 |
| 算法指标（algorithm） | 算法相关数据，不同算法的指标有所不同 |
| 环境指标（env） | 环境相关数据，不同环境的指标有所不同 |

##### 基础指标

| 面板名称 | 指标名称 | 说明 |
|----------|----------|------|
| 训练累计步数 | train_global_step | agent.learn() 的调用次数 |
| 预测累计次数 | predict_succ_cnt | agent.predict() 的调用次数 |
| 模型加载次数 | load_model_succ_cnt | agent.load_model() 成功调用的次数 |
| 样本接收次数 | sample_receive_cnt | 接收到的样本总数 |
| 已结束任务数 | episode_cnt | 已完成的 episode 数量 |
| 样本生产消耗比 | sample_production_and_consumption_ratio | 训练消耗样本数 / 采样生产样本数 |

##### 硬件指标

| 面板名称 | 指标名称 | 说明 |
|----------|----------|------|
| CPU 使用率 | aisrv_cpu_usage / learner_cpu_usage | 分别对应 aisrv 和 learner 进程 |
| GPU 使用率 | aisrv_gpu_usage / learner_gpu_usage | 分别对应 aisrv 和 learner 进程 |
| GPU 显存使用率 | gpu_memory | 显存占用百分比 |
| 内存使用率 | ram_usage | 容器内存占用，过高可能导致 OOM |

##### 算法指标

不同算法的指标各不相同，详见具体算法文档。

##### 环境指标

不同环境的指标各不相同，详见具体环境文档。

##### 自定义监控面板

平台支持在实验代码中自定义监控指标，配置并上报期望观测的业务数据。

**配置面板**

在算法配置目录下编辑监控配置文件：

文件路径：`agent_{算法名}/conf/monitor_builder.py`

配置示例：

```python
def build_monitor():
    monitor = MonitorConfigBuilder()

    config_dict = (
        monitor.title("项目名称")
        .add_group(group_name="算法指标", group_name_en="algorithm")
        .add_panel(
            name="累积回报",
            description="反映智能体能力的指标",
            type="line",
        )
        .add_metric(metrics_name="reward", expr="avg(reward{})")
        .end_panel()
        .end_group()
        .build()
    )
    return config_dict
```

**上报指标数据**

在代码中调用监控上报接口，代码示例：

```python
import os
from monitor import monitor

monitor_data = {
    "reward": 100.5
}
monitor.push_data({os.getpid(): monitor_data})
```

**配置参数说明**

| 配置项 | 字段 | 字段类型 | 说明 | 限制条件 |
|--------|------|----------|------|----------|
| 项目名称 | title | string | 监控面板名称（该字段不在监控页面进行展示，不建议修改） | 支持中英文、数字、=、+、/、@、#、_、-及空格，长度 1~100 字符 |
| 面板组 | group_name | string | 面板组名称 | 支持中英文、数字、_、-及空格，长度 1~20 字符 |
| | group_name_en | string | 面板组英文标识 | 支持英文、数字、_、-及空格，长度 1~50 字符 |
| 面板 | name | string | 面板名称 | 支持中英文、数字、_、-及空格，长度 1~20 字符 |
| | name_en | string | 面板英文标识 | 支持英文、数字、_、-及空格，长度 0~50 字符 |
| | description | string | 面板描述信息 | 支持中英文、数字、标点符号等，长度 0~200 字符 |
| | type | string | 图表类型 | 仅line（折线图）、stat（数值图）有效 |
| | unit | string | 数值单位，当 type 为 stat 时展示到指标后 | 仅 stat 类型有效 |
| 指标 | metrics_name | string | 指标显示名称 | 支持中英文、数字、_、-、{}及空格，长度 1~40 字符 |
| | expr | string | 指标查询表达式，支持指标变量（lable） | 使用 PromQL 的查询语法即可 |

**规格限制**：

- 折线图面板：指标数量限制为 20 个指标
- 数值图面板：指标数量限制为 2 个指标

##### 指标变量说明

如需按维度分组展示数据（如不同对手的胜率），可通过指标变量（label） 区分维度，并在配置中使用指标变量定义展示方式。

**第一步：上报带 label 的数据**

```python
# 上报玩家1的胜率指标
monitor.push_data({os.getpid(): win_rate, "player": "player1"})

# 上报玩家2的胜率指标
monitor.push_data({os.getpid(): win_rate, "player": "player2"})
```

**第二步：配置指标变量**

```python
.add_metric(
    metrics_name="win_rate_{player}",
    expr="avg(win_rate{}) by (player)"
)
```

说明：

- metrics_name字符串中的 {player} 会被实际的player1、player2替换；最终将在同一个面板上显示两条数据线：win_rate_player1 和 win_rate_player2

---

#### 日志

框架提供了统一的日志服务，帮助您记录和排查训练过程中的问题。

##### 日志格式

| 字段 | 说明 | 示例 |
|------|------|------|
| time | 时间戳 | 2024-09-18 19:33:04.813469 |
| level | 日志级别 | INFO / WARNING / ERROR |
| message | 日志内容 | kaiwu learner train count is 365676 |
| file | 源码文件 | on_policy_trainer.py |
| line | 代码行号 | 769 |
| module | 所属模块 | learner |
| process | 进程名 | on_policy_trainer |
| function | 函数名 | train_stat |
| stack | 错误堆栈 | 仅错误日志包含 |

**注意**：

1. 不要重写日志系统：重写后监控面板将无法统计错误日志数量
2. 日志流量限制：框架限制为 60 条/分钟，超出部分将被丢弃
