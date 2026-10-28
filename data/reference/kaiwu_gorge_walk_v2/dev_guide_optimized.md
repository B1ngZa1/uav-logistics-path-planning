开发指南\
项目简介

任务介绍
--------

峡谷漫步的目标是：训练一个智能体，让其在对地图不断地探索中学习移动策略，减少碰撞障碍物，以最少的步数**从起点走到终点**，可能会有附属任务——**收集宝箱**。

场景介绍
--------

峡谷漫步使用64\*64的网格化的地图（智能体每步移动的距离是一网格），地图中包含**起点**、**终点**、**道路**、**障碍物**和**宝箱**等元素。

  **元素**       **说明**
  -------------- ---------------------------------------------------------------------------------------------------------------------------
  **英雄**       环境中存在英雄单位，智能体可以控制英雄在地图中进行移动。本环境采用鲁班作为英雄角色。
  **起点**       任务开始时，英雄单位在起点出现。
  **终点**       任务设置的目的地，当英雄抵达终点时，任务结束。
  **道路**       英雄可以在道路中通过前、后、左、右四个方向移动进行试探，每次移动的距离是一个网格。
  **障碍物**     英雄遇到障碍物时无法继续前进，需要调整方向开启新的探索。
  **宝箱**       如果用户给任务配置了宝箱，则智能体可以通过控制英雄拾取宝箱增加积分，每个宝箱获得100积分，地图中共有10个可配置宝箱的点位。
  **视野范围**   智能体的视野范围是以自己为中心，5\*5范围内的所有格子。

![峡谷漫步地图场景示意图，展示64乘64网格、道路、障碍物与宝箱元素。](04_项目内容及任务分工\dev_guide_optimized_media/media/image1.png "图片 16"){width="6.645669291338582in"
height="5.770690069991251in"}

规则介绍
--------

总积分 = 终点积分 + 步数积分 + 宝箱积分

-   **终点积分**：到达终点即获得**150**积分。

-   **步数积分**：**(最大步数 - 完成步数) \*
    奖励系数0.2**。任务设置有最大步数，当智能体到达终点时，根据最大步数和任务完成步数计算步数积分。

-   **宝箱积分**：每获得一个宝箱，即可增加**100**积分。

注意：若在**最大步数**内没有走到终点（包括最大步数），则判定为任务超时。**超时任务的总积分为0**。

注意事项
--------

训练时，可选择如下算法：

-   dynamic\_programming

-   monte\_carlo

-   sarsa

-   q\_learning

-   diy

可选训练模式：

-   单机

-   分布式

**注意：当前实验提供的四个算法仅支持单机训练。若需要使用分布式训练，请在agent\_diy文件夹下按照开发框架文档自行完成算法开发。若算法与模式不匹配（如单机算法选分布式），训练将报错并自动停止。请正确选择算法与训练模式，确保任务正常运行。**

若是单机算法配置分布式训练模式，则在aisrv.log日志中会出现如下报错：

![单机算法配置分布式训练模式时的aisrv.log报错截图。](04_项目内容及任务分工\dev_guide_optimized_media/media/image2.png "图片 18"){width="6.716400918635171in"
height="3.62992125984252in"}

若是分布式算法配置单机训练模式，则在aisrv.log日志中会出现如下报错：

![分布式算法配置单机训练模式时的aisrv.log报错截图。](04_项目内容及任务分工\dev_guide_optimized_media/media/image3.png "图片 20"){width="6.700787401574803in"
height="3.631910542432196in"}

环境详述
========

环境配置
--------

在智能体和环境的交互中，首先会调用env.reset方法，该方法接受一个usr\_conf参数，这个参数通过读取train\_env\_conf.toml文件的内容来实现定制化的环境配置。因此，用户可以通过修改train\_env\_conf.toml文件中的内容来调整环境配置。

*\#
usr\_conf为代码包中的train\_env\_conf.toml配置文件，此处以agent\_ppo为例*\
usr\_conf = read\_usr\_conf("agent\_ppo/conf/train\_env\_conf.toml",
logger)\
env\_obs = env.reset(usr\_conf=usr\_conf)

train\_env\_conf.toml中包含以下配置信息：

  **数据名**             **数据类型**   **数据描述**                                                                                                                                        **默认值**
  ---------------------- -------------- --------------------------------------------------------------------------------------------------------------------------------------------------- ------------
  **start**              \[int,int\]    起点位置坐标                                                                                                                                        \[29,9\]
  **end**                \[int,int\]    终点位置坐标                                                                                                                                        \[11,55\]
  **treasure\_random**   bool           是否生成随机宝箱，false表示固定宝箱，true表示随机宝箱。开启固定，则使用treasure\_id生成固定宝箱。 若开启随机，则使用treasure\_count生成随机宝箱。   false
  **treasure\_count**    int            生成随机宝箱时的宝箱数量，仅在treasure\_random = true时生效，取值范围为0\~10。                                                                      0
  **treasure\_id**       list of int    生成固定宝箱时的宝箱编号，仅在treasure\_random = false时生效，取值范围为0\~9。                                                                      \[ \]
  **max\_step**          int            单局最大步数，取值范围为1\~2000。                                                                                                                   2000

train\_env\_conf.toml采用的默认配置：

\[env\_conf\]\
*\# 起点坐标*\
*\# 数组，包含两个整数元素\[x, z\]，每个坐标取值范围为0\~63。*\
*\# 起点坐标和终点坐标不能相同。*\
start = \[29, 9\]\
\
*\# 终点坐标*\
*\# 数组，包含两个整数元素\[x, z\]，每个坐标取值范围为0\~63。*\
*\# 起点坐标和终点坐标不能相同。*\
end = \[11, 55\]\
\
*\# 宝箱ID数组。每个ID必须是整数，取值范围：0\~9。*\
*\# 配置固定宝箱的ID。*\
treasure\_id = \[\]\
\
*\# 生成随机宝箱时的宝箱数量，仅在treasure\_random = true时生效。*\
*\# 整型，取值范围为0\~10。*\
treasure\_count = 0\
\
*\# 是否生成随机宝箱，布尔值，false表示固定宝箱，true表示随机宝箱。*\
*\# 若开启固定，则使用treasure\_id生成固定宝箱。*\
*\# 若开启随机，则使用treasure\_count生成随机宝箱。*\
treasure\_random = false\
\
*\# 整型，取值范围1\~2000。*\
*\# 最大步数。*\
max\_step = 2000

**补充说明**：

**环境配置仅在训练时生效**，请按上表所述进行train\_env\_conf.toml文件的配置。若配置错误，env.reset会在调用超时后返回Error，无法获取任务状态，训练不会正常进行，您可以通过监控界面查看**错误日志**进行排查。

**评估配置** 模型评估时，用户需要通过开悟平台创建评估任务并完成任务的环境配置。其中起点、终点位置固定为\[29,
9\]和\[11, 55\]，用户可以对宝箱及步数配置进行设置。

以下为每个**宝箱id**和**坐标**（原点为地图左下角）：

  **描述**   **宝箱0**    **宝箱1**   **宝箱2**   **宝箱3**    **宝箱4**    **宝箱5**    **宝箱6**    **宝箱7**    **宝箱8**    **宝箱9**
  ---------- ------------ ----------- ----------- ------------ ------------ ------------ ------------ ------------ ------------ ------------
  **坐标**   \[19, 14\]   \[9, 28\]   \[9, 44\]   \[42, 45\]   \[32, 23\]   \[49, 56\]   \[35, 58\]   \[23, 55\]   \[41, 33\]   \[54, 41\]

![宝箱ID与地图坐标的对照示意图。](04_项目内容及任务分工\dev_guide_optimized_media/media/image4.png "图片 22"){width="5.825951443569553in"
height="5.825951443569553in"}

环境数据
--------

在调用env.reset与env.step接口时，会返回环境当前的状态。其中env.reset接口返回obs数据，而env.step接口会额外返回reward数据。下面将对这些环境状态的各项数据及其含义进行详细的介绍，帮助用户更好的理解每个返回值的作用。

*\# usr\_conf为用户传入的环境配置*\
env\_obs = env.reset(usr\_conf=usr\_conf)\
\
*\# current\_action为用户传入的动作*\
env\_reward, env\_obs = env.step(current\_action)

环境返回信息
------------

### env\_obs

  **数据名**    **数据类型**   **数据描述**
  ------------- -------------- ------------------------------
  env\_id       str            环境ID
  frame\_no     int            当前环境实例运行时的帧号
  observation   Observation    环境实例的观测信息
  extra\_info   ExtraInfo      环境实例的额外信息（调试用）
  terminated    bool           是否游戏胜利（到达终点）
  truncated     bool           是否游戏中断（超时/撞墙）

### env\_reward

  **数据名**   **数据类型**   **数据描述**
  ------------ -------------- ------------------
  env\_id      str            环境ID
  frame\_no    int            当前帧号
  reward       int            当前步的即时奖励

### reward 计算规则：

-   **宝箱奖励**：收集到宝箱时获得

-   **胜利奖励**：到达终点时获得

-   **步数奖励**：到达终点时的额外奖励 = (max\_steps - step\_no) \*
    step\_bonus

观测信息(observation)
---------------------

为环境在交互中对智能体可见的信息，区别于环境内部状态信息。

  **数据名**     **数据类型**   **数据描述**
  -------------- -------------- ------------------
  step\_no       int            当前步数
  frame\_state   FrameState     当前帧的环境状态
  game\_info     GameInfo       全局统计信息

额外信息(extra\_info)
---------------------

为环境对训练框架和工作流提供的额外信息，主要用于调试。

  **数据名**        **数据类型**   **数据描述**
  ----------------- -------------- ------------------------
  result\_code      int            错误码，0表示正常
  result\_message   string         结果消息，正常时为"ok"

帧状态信息
----------

### frame\_state

  **数据名**   **数据类型**    **数据描述**
  ------------ --------------- -----------------------
  frame\_no    int             当前帧号
  hero         Hero            英雄信息
  organs       list of Organ   机关列表（终点+宝箱）

### hero

  **数据名**   **数据类型**   **数据描述**
  ------------ -------------- --------------
  pos          Position       英雄位置

### Position

  **数据名**   **数据类型**   **数据描述**
  ------------ -------------- ---------------
  x            int            x坐标（0-63）
  z            int            z坐标（0-63）

### Organ

  **数据名**   **数据类型**   **数据描述**
  ------------ -------------- -------------------------------
  sub\_type    int            类型：2=终点，1=宝箱
  config\_id   int            配置ID：22=终点，0-9=宝箱ID
  status       int            状态：1=可达/可获取，0=已收集
  pos          Position       机关位置
  dist         int            到该机关的离散化距离

### organs 列表说明：

-   第一个元素始终是终点（sub\_type=2, config\_id=22）

-   后续元素是已生成的宝箱（sub\_type=1, config\_id=0-9）

-   只有在 treasure\_id 中指定的宝箱才会出现

-   dist 为从英雄当前位置到该机关位置的最短路径步数（BFS）离散化结果，取值范围为 \[0,
    6\]，数值越大表示距离越远

-   dist 离散化区间：0（≤20）、1（21-35）、2（36-45）、3（46-52）、4（53-60）、5（61-70）、6（≥71）

统计信息
--------

### game\_info

  **数据名**         **数据类型**    **数据描述**
  ------------------ --------------- ------------------------------------------------------------------------------------------------------------------------------------
  score              int             当前步的得分
  total\_score       int             累计总分
  treasure\_count    int             已收集宝箱数量
  treasure\_score    int             宝箱累计得分
  view               int             视野大小（默认为2表示以智能体为中心5×5的视野范围）
  local\_view        list of int     局部视野一维数组，长度为(view\*2+1)²，默认25个元素。元素值：0=障碍物，1=可通行，2=起点，3=终点，4=宝箱。数组中间位置为英雄当前位置
  location\_memory   list of float   位置记忆（4096维），64×64地图展平为一维，索引=x\*64+z。元素取值范围\[0.0, 1.0\]，每走过一次增加0.1，最大1.0

**注意**：得分用于衡量模型在环境中的表现，也作为衡量强化学习训练产出模型的评价指标，与强化学习里的奖励reward 要区别开。

任务结束条件
------------

  **条件**   **字段**          **说明**
  ---------- ----------------- -----------------------------------------------------------
  胜利       terminated=True   英雄到达终点
  失败       truncated=True    超过最大步数、撞墙（如果配置了bump\_exit=True）、主动停止

动作空间
--------

峡谷漫步的动作空间设计比较简单，只有上下左右四个动作，智能体以网格为单位进行四个方向的移动，每次移动一格。

env.step(act) 里的 act 的合法取值为 {0, 1, 2, 3}，对应的动作如下：

action\_space:\
0: UP\
1: DOWN\
2: LEFT\
3: RIGHT

环境监控信息
------------

监控面板中包含了**env**模块，表示**环境指标**数据，详细说明如下。

  --------------------------------------------------------------------------------------------------------
  **指标名称**           **说明**
  ---------------------- ---------------------------------------------------------------------------------
  **score**              该面板包含两个指标：\
                         total\_score：任务结束时的总积分，若任务超时，则该局积分为0。\
                         treasure\_score：任务结束时收集到的宝箱奖励。

  **steps**              该面板包含两个指标：\
                         max\_steps：任务设置的最大步数。\
                         finished\_steps：任务结束时所用的步数。若任务超时，则该局完成步数等于最大步数。

  **treasure**           该面板包含两个指标：\
                         total\_treasures：任务设置的宝箱个数。\
                         collected\_treasures：任务结束时收集到的宝箱个数。

  **treasure\_random**   宝箱是否随机。若为0则表示宝箱位置固定，若为1则表示宝箱位置随机。
  --------------------------------------------------------------------------------------------------------

\
智能体详述
==========

特征处理
--------

特征处理的目标是将环境的原始观测数据（observation）转化为结构化、易于算法理解和利用的特征值。在本环境中，将智能体位置和宝箱状态组合编码为一个整数特征，用于Q表索引。本节将详细介绍特征的组成结构及其处理方式，帮助开发者理解特征设计背后的思路。

特征说明
--------

开发目录：智能体文件夹/agent.py文件中的observation\_process方法

代码包中的特征处理采用**组合编码**方式，将**位置信息**和**宝箱状态**编码为单一整数特征：

  **数据名**   **数据类型**   **数据描述**
  ------------ -------------- --------------------------------------------------
  feature      int            组合特征，由位置特征和宝箱状态二进制编码组合而成

特征计算公式：

$$feature = 1024 \times pos\_ feature + treasure\_\text{binary}\backslash n$$

其中：

pos\_feature：位置特征

treasure\_binary：宝箱状态的二进制编码

位置特征
--------

[*地图介绍*](https://tencentarena.com/docs/p-edu-gorge_walk_v2/37.1.3/guidebook/dev-guide/intro/#%E5%9C%B0%E5%9B%BE%E4%BB%8B%E7%BB%8D)里提到，峡谷漫步地图本质上是一个 **64×64** 的网格，智能体的位置可以用一个二维的坐标来表示。我们采用下面的方式将二维坐标等价转换为一维的位置特征值：

$$\text{pos}\_ feature = x \times 64 + z\backslash n$$

其中 x 和 z 分别从环境观测数据中获取：

pos = \[obs\["frame\_state"\]\["hero"\]\["pos"\]\["x"\],
obs\["frame\_state"\]\["hero"\]\["pos"\]\["z"\]\]\
pos\_feature = int(pos\[0\] \* 64 + pos\[1\])

位置特征的取值范围为 **\[0, 4095\]**（共4096个状态）。

宝箱状态信息
------------

峡谷漫步的宝箱总共有 **10** 个可能出现的位置，每个位置对应了一个宝箱id（0-9），详情见[*环境配置*](https://tencentarena.com/docs/p-edu-gorge_walk_v2/37.1.3/guidebook/dev-guide/env/#%E7%8E%AF%E5%A2%83%E9%85%8D%E7%BD%AE)。

宝箱状态从环境的 organs 列表中提取，通过遍历 sub\_type=1（宝箱类型）的机关获取：

treasure\_status = \[0\] \* 10\
for organ in obs\["frame\_state"\]\["organs"\]:\
if organ\["sub\_type"\] == 1: *\# 1=宝箱*\
treasure\_status\[organ\["config\_id"\]\] = int(organ\["status"\])

-   **1** 表示可以被收集

-   **0** 表示不可被收集（未生成或者已经收集过）

宝箱状态示例：

初始状态时，treasure\_status 为 \[0, 0, 0, 0, 0, 0, 0, 0, 0, 0\]

当环境启动时，会根据用户传入的 usr\_conf 生成宝箱，生成的宝箱状态会变为
1。比如前 5 个宝箱被生成了，那 treasure\_status 为 \[1, 1, 1, 1, 1, 0,
0, 0, 0, 0\]

随着环境的进行，宝箱2 和 宝箱4 被收集，则 treasure\_status 变为 \[1, 1,
0, 1, 0, 0, 0, 0, 0, 0\]

宝箱状态二进制编码
------------------

将10维的宝箱状态列表编码为一个整数，采用二进制编码方式：

$$\text{treasure}\_ binary = \sum_{i = 0}^{9}{\text{treasure}\_ status\lbrack i\rbrack \times}2^{i}\backslash n$$

treasure\_binary = sum(\[treasure\_status\[i\] \* (2\*\*i) for i in
range(10)\])

由于有10个宝箱，每个宝箱有2种状态（0或1），所以 treasure\_binary 的取值范围为 **\[0,
1023\]**（共1024种组合）。

组合特征
--------

最终的组合特征将位置特征和宝箱状态编码组合：

feature = int(1024 \* pos\_feature + treasure\_binary)

-   **1024** 是宝箱状态的可能组合数（2\^10）

    这种编码方式确保了每个（位置，宝箱状态）组合都有唯一的特征值

**注意**：如进行额外特征处理，则需要对Q表结构和算法方法进行相应的改动。

奖励设计
--------

开发目录：智能体文件夹/feature/definition.py文件中的reward\_shaping方法

奖励设计（Reward
Shaping）是强化学习中的关键环节，合理的奖励信号能够有效引导智能体学习。代码包中的奖励设计基于环境返回的得分信息：

def reward\_shaping(env\_reward, env\_obs):\
score = env\_obs\["observation"\]\["game\_info"\]\["score"\]\
terminated = env\_obs\["terminated"\]\
\
reward = 0\
\
*\# 获胜奖励（回合结束时的最终得分）*\
if terminated:\
reward += score\
\
*\# 回合期间收集宝箱的奖励*\
if score &gt; 0 and not terminated:\
reward += score\
\
return reward

奖励组成
--------

  **奖励类型**   **触发条件**                                     **奖励值**
  -------------- ------------------------------------------------ ------------------
  宝箱奖励       收集宝箱时（score &gt; 0 且 terminated=False）   环境返回的 score
  胜利奖励       到达终点时（terminated=True）                    环境返回的 score

**注意**：奖励是用于引导强化学习训练的信号，与环境的得分（score）概念不同。得分用于衡量模型表现，而奖励用于算法学习。

算法介绍
========

我们在峡谷漫步代码包中提供了4个算法 **dynamic\_programming**，**monte\_carlo**，**sarsa**，**q\_learning**，同时，我们还提供了一个**diy**模板算法文件夹，用户可在该文件夹中自定义算法实现。

Dynamic Programming 算法
------------------------

Dynamic
Programming基于贝尔曼方程，通过递归分解问题，将迷宫路径规划视为多阶段决策过程。通过值迭代或策略迭代算法，计算每个状态的最优值函数（V值）或最优策略。在已知环境完整模型和状态空间有限下，可以保证得到全局最优解。该算法缺点是计算复杂度较高，无法对未知环境进行建模。

Monte Carlo 算法
----------------

Monte Carlo
算法通过随机采样完整轨迹（从起点到终点），用经验平均回报估计状态值函数。无需环境模型，直接通过采样数据进行学习。原理简单，实现简单，无需环境模型，可适用于有较大随机性的环境。缺点是需大量采样，方差高，策略更新缓慢。

SARSA 算法
----------

SARSA（State-Action-Reward-State-Action）是一种On-Policy​的时序差分（Temporal
Difference,
TD）强化学习算法。其核心思想是通过智能体与环境交互生成的实际动作序列来更新Q值函数。与Q-Learning不同，SARSA严格遵循当前策略（如ε-贪婪策略）选择动作，即在更新Q值时依赖实际执行的下一动作，而非假设选择最优动作。

​Q-Learning 算法
----------------

Q-Learning
是一种Off-Policy​的强化学习算法，其目标是通过学习状态-动作值函数（Q
函数）来找到最优策略。Q 函数定义为在状态 s 下执行动作 a
后，智能体能获得的最大累积奖励期望值。通过最大化下一状态Q值更新当前Q表。利用贝尔曼最优方程，学习独立于当前策略的最优动作。它的缺点是：Q表存储限制（状态爆炸），无法处理连续空间。

算法监控信息
------------

监控面板中包含了**algorithm**模块，展示**算法指标**数据，以下是本环境中提供的**dynamic\_programming**、**monte\_carlo**、**sarsa**、**q\_learning**算法指标的详细说明。

  **指标名称**   **说明**
  -------------- ------------------------------------------------------------------------------------
  **reward**     最近10局平均每局的累积回报，反应了智能体的能力，正常训练情况下指标应该是震荡向上。

模型保存限制策略
----------------

为了避免用户保存模型的频率过于频繁，开悟平台对模型保存会有安全限制，不同的任务会有不同的限制，限制规则详情如下：

保存模型的频率限制: 2次/分钟

单个任务保存模型的次数限制：（不同算法的限制不同）

Dynamic Programming：10次

Monte Carlo：100次

Q-learning：100次

Sarsa：100次。

模型评估模式
------------

在峡谷漫步中，评估时用户需要在提交任务界面进行宝箱配置，提供的默认配置如下：

*\#环境配置*\
\[env\_conf\]\
*\# 宝箱id，数组，id取值范围：0\~9。*\
*\# 配置固定宝箱的id。*\
treasure\_id = \[\]\
\
*\# 生成随机宝箱时的宝箱数量，仅在treasure\_random =
true时生效，整型，取值范围为0\~10，默认值为0。*\
treasure\_count=0\
\
*\# 是否生成随机宝箱，布尔值，false表示固定宝箱，true表示随机宝箱。*\
*\# 若开启固定，则使用treasure\_id生成固定宝箱。*\
*\# 若开启随机，则使用treasure\_count生成随机宝箱。*\
*\# 默认值为false。*\
treasure\_random = false\
\
*\# 整型，取值范围1\~2000*\
*\# 最大步数*\
max\_step=2000

另外，训练模式时，用户一般使用agent.predict方法进行决策；而在评估模式时，平台会调用agent.exploit方法进行决策，一般情况下，模型在训练和评估时的决策会因算法不同和用户设计不同，而有不同的行为，这部分由用户定义和实现。

综述
====

欢迎来到腾讯开悟！

腾讯开悟强化学习开发框架是基于[*强化学习系统系列技术标准*](https://tencentarena.com/docs/p-edu-gorge_walk_v2/37.1.3/guidebook/taa-rl-fw/rl_standard/)打造的标准化开发套件。该框架为开发者提供了标准化的编程接口和丰富的工具集，支持开发者高效完成智能体开发、环境交互，以及模型的训练及预测流程。

训练流程简介
------------

本开发框架的完整训练流程如下图所示：

![腾讯开悟强化学习开发框架完整训练流程图。](04_项目内容及任务分工\dev_guide_optimized_media/media/image5.png "图片 24"){width="5.705123578302712in"
height="2.859722222222222in"}

如图，完整训练流程包含以下关键环节：

  -----------------------------------------------------------------------------------------------------------------------------
  **环节**                   **介绍**
  -------------------------- --------------------------------------------------------------------------------------------------
  **​智能体-环境循环交互**   - 智能体将环境提供的观测和奖励处理为符合预测函数输入要求的数据；\
                             - 调用预测函数，生成动作指令；\
                             - 将智能体输出的动作指令处理为符合环境输入要求的数据；\
                             - 环境执行动作后完成状态转移，并反馈新的观测数据和奖励数据；

  **样本处理**               - 每个环境有不同的开始与结束逻辑，智能体与环境从开始到结束的完整交互过程，称为episode；\
                             - 智能体与环境每一次交互产生的结构化数据，称为**样本**；一个episode产生的样本序列称为**轨迹**；\
                             - 对轨迹数据进行处理，转换为规范化**训练样本(SampleData)**；

  **​模型迭代优化**          - 基于训练样本，通过算法持续更新模型参数，实现策略优化；

  **智能体模型更新**         - 智能体加载最新模型，与环境继续循环交互；
  -----------------------------------------------------------------------------------------------------------------------------

该流程通过强化学习分布式计算框架提供的训练工作流实现。基于此，开发框架主要包含三大核心模块：

[*强化学习环境系统*](https://tencentarena.com/docs/p-edu-gorge_walk_v2/37.1.3/guidebook/taa-rl-fw/rl_env/)：提供标准的**强化学习环境接口**。开发者可以通过标准接口，实现智能体与环境的交互。

[*强化学习智能体开发套件*](https://tencentarena.com/docs/p-edu-gorge_walk_v2/37.1.3/guidebook/taa-rl-fw/rl_agent/info/)：提供标准的**强化学习智能体接口**，以及算法库、模型组件库等工具函数库。开发者可以通过工具函数库快速完成智能体的构建。

[*强化学习分布式计算框架*](https://tencentarena.com/docs/p-edu-gorge_walk_v2/37.1.3/guidebook/taa-rl-fw/distributed_computing_fw/)：提供标准接口，支持开发者按需实现训练工作流，运行单机或分布式的训练及评估任务。

代码包简介
----------

开发者可以通过腾讯开悟平台所提供的强化学习项目使用开发框架。一个强化学习项目的代码目录如下：

📦 根目录\
├── 📂 agent\
│ ├── 📂 algorithm\
│ └── 📄 \_\_init\_\_.py\
│ └── 📄 algorithm.py\
│ ├── 📂 conf\
│ └── 📄 \_\_init\_\_.py\
│ └── 📄 conf.py\
│ └── 📄 train\_env\_conf.toml\
│ ├── 📂 feature\
│ └── 📄 \_\_init\_\_.py\
│ └── 📄 definition.py\
│ └── 📄 preprocessor.py\
│ ├── 📂 model\
│ └── 📄 \_\_init\_\_.py\
│ └── 📄 model.py\
│ ├── 📂 workflow\
│ └── 📄 \_\_init\_\_.py\
│ └── 📄 train\_workflow.py\
│ ├── 📄 \_\_init\_\_.py\
│ └── 📄 agent.py\
├── 📂 conf\
│ ├── 📄 \_\_init\_\_.py\
│ ├── 📄 configure\_app.toml\
├── 📂 log\
└── 📄 train\_test.py

代码目录介绍：
--------------

  **目录名**           **介绍**
  -------------------- ----------------------------------------------------------------------------------------------------------------
  **agent/**           智能体子目录，智能体相关内容均集中于该目录，是开发者核心工作目录。
  **conf/**            配置文件目录，包含运行训练任务相关的配置，例如训练样本批处理大小batch\_size等。
  **log/**             日志目录，存放运行代码测试脚本时生成的日志文件。
  **train\_test.py**   代码正确性测试脚本，该脚本会使用当前代码包完成一步训练。建议开发者在启动训练任务前，确保代码已通过该脚本检测。

agent

  **目录/文件名**   **介绍**
  ----------------- ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  **algorithm/**    算法相关，开发者在该目录下完成算法实现，包含loss计算、模型优化等，详情见[*算法开发*](https://tencentarena.com/docs/p-edu-gorge_walk_v2/37.1.3/guidebook/taa-rl-fw/rl_agent/algorithm/)
  **feature/**      特征相关，开发者在该目录下完成数据结构定义和数据处理方法，以及样本处理和奖励计算，详情见[*数据处理与奖励设计*](https://tencentarena.com/docs/p-edu-gorge_walk_v2/37.1.3/guidebook/taa-rl-fw/rl_agent/feature/)
  **model/**        模型相关，开发者在该目录下完成模型实现。 详情见[*模型开发*](https://tencentarena.com/docs/p-edu-gorge_walk_v2/37.1.3/guidebook/taa-rl-fw/rl_agent/model/)
  **workflow/**     工作流目录，开发者在该目录下完成训练工作流的开发。 详情见[*工作流开发*](https://tencentarena.com/docs/p-edu-gorge_walk_v2/37.1.3/guidebook/taa-rl-fw/rl_agent/workflow/)
  **agent.py**      智能体核心代码文件，开发者在该文件中完成预测、训练等核心函数的实现。 详情见[*智能体开发*](https://tencentarena.com/docs/p-edu-gorge_walk_v2/37.1.3/guidebook/taa-rl-fw/rl_agent/agent/)

标准代码包中都存在一个agent\_diy子文件夹，该文件夹是预定义的智能体模板，可供开发者进行智能体的开发。

conf

  **文件名**                **介绍**
  ------------------------- --------------------------------------------------
  **configure\_app.toml**   训练任务相关的配置，包括样本大小、样本池大小等。

通过对训练流程和代码包的介绍，相信开发者能够对腾讯开悟开发框架建立了初步认知。

接下来，我们将详细介绍每个模块的功能及使用方式。

\
环境
====

在综述中提到，强化学习训练流程离不开智能体与环境的持续交互，本文将详细介绍强化学习环境系统的功能及标准接口函数。

概述
----

强化学习环境是基于输入动作，输出观测、奖励等反馈的功能模块，用于表达强化学习算法所求解的问题场景。

开发框架通过场景适配模块，对仿真器进行封装，将其特化的接口、协议转换为强化学习环境统一的接口和协议，供智能体调用。

强化学习环境系统主要提供如下功能：

-   接收配置信息，用于指定自身初始化方式，比如环境中各种元素的初始状态。

-   输出观测、奖励信息，可用于智能体预测、训练。

-   输出观测、奖励之外的其他信息，供强化学习系统相关组件使用以实现特定功能。其他信息可包括可视化数据、日志数据等，实现的功能包括环境可视化、运行状况监测等。

-   接收动作指令，完成状态转移并产生新的观测和奖励。

环境使用
--------

开发框架通过场景适配模块，将问题场景进行标准化封装，为开发者提供统一的交互接口与通信协议。由于环境之间存在差异，接口中所涉及的观测、奖励等信息的具体数据结构也有所不同，开发者需查阅所使用环境的官方数据协议文档以获取准确信息。

开发者可以在[*训练工作流*](https://tencentarena.com/docs/p-edu-gorge_walk_v2/37.1.3/guidebook/taa-rl-fw/distributed_computing_fw/#%E8%AE%AD%E7%BB%83%E5%B7%A5%E4%BD%9C%E6%B5%81)的workflow中获取到对应环境的实例，通过标准接口实现智能体与环境的交互。

核心函数介绍
------------

### reset(usr\_conf)

reset会将环境重置为环境配置文件中指定的状态，并且返回初始观测。

*\# usr\_conf为开发者传入的环境配置*\
obs, state = env.reset(usr\_conf = usr\_conf)

### Parameters

  **参数名**      **介绍**
  --------------- ------------------------
  **usr\_conf**   dict类型，环境配置文件

### Returns

  **参数名**   **介绍**
  ------------ ------------------------
  **obs**      dict类型，环境观测信息
  **state**    dict类型，环境全局信息

### step(act, stop\_game = false)

环境会执行传入的act动作指令，完成一次状态转移，并返回新的观测和奖励等信息。

frame\_no, \_obs, score, terminated, truncated, \_state = env.step(act,
stop\_game = false)

### Parameters

  **参数名**       **介绍**
  ---------------- ----------------------------
  **act**          dict类型，环境执行的动作
  **stop\_game**   bool类型，是否结束当前对局

### Returns

  **参数名**       **介绍**
  ---------------- --------------------------------------
  **frame\_no**    int类型，当前环境实例运行时的帧号
  **\_obs**        dict字典类型，当前帧的观测信息
  **score**        int类型，当前帧的奖励信息
  **terminated**   bool类型，当前环境实例是否结束
  **truncated**    bool类型，当前环境实例是否异常或中断
  **\_state**      dict字典类型，当前帧的全部状态信息

\
综述
====

智能体是强化学习系统中的核心模块，在[*开发框架综述*](https://tencentarena.com/docs/p-edu-gorge_walk_v2/37.1.3/guidebook/taa-rl-fw/intro/)中提到，完整训练流程包括：

  -----------------------------------------------------------------------------------------------------------------------------
  **环节**                   **介绍**
  -------------------------- --------------------------------------------------------------------------------------------------
  **​智能体-环境循环交互**   - 智能体将环境提供的观测和奖励处理为符合预测函数输入要求的数据；\
                             - 调用预测函数，生成动作指令；\
                             - 将智能体输出的动作指令处理为符合环境输入要求的数据；\
                             - 环境执行动作后完成状态转移，并反馈新的观测数据和奖励数据；

  **样本处理**               - 每个环境有不同的开始与结束逻辑，智能体与环境从开始到结束的完整交互过程，称为episode；\
                             - 智能体与环境每一次交互产生的结构化数据，称为**样本**；一个episode产生的样本序列称为**轨迹**；\
                             - 对轨迹数据进行处理，转换为规范化**训练样本(SampleData)**；

  **​模型迭代优化**          - 基于训练样本，通过算法持续更新模型参数，实现策略优化；

  **智能体模型更新**         - 智能体加载最新模型，与环境继续循环交互；
  -----------------------------------------------------------------------------------------------------------------------------

基于上述训练流程，我们将智能体的开发分为四个部分：

-   [*数据处理及奖励设计*](https://tencentarena.com/docs/p-edu-gorge_walk_v2/37.1.3/guidebook/taa-rl-fw/rl_agent/feature/)：介绍基于环境观测数据进行特征处理、样本处理和奖励设计的方法。

-   [*模型开发*](https://tencentarena.com/docs/p-edu-gorge_walk_v2/37.1.3/guidebook/taa-rl-fw/rl_agent/model/)：介绍模型开发接口及开发方法。

-   [*算法开发*](https://tencentarena.com/docs/p-edu-gorge_walk_v2/37.1.3/guidebook/taa-rl-fw/rl_agent/algorithm/)：介绍包括算法开发接口及开发方法。

-   [*工作流开发*](https://tencentarena.com/docs/p-edu-gorge_walk_v2/37.1.3/guidebook/taa-rl-fw/rl_agent/workflow/)：介绍开发者开发自定义训练工作流的方法。

接下来，将通过独立的章节对强化学习智能体开发套件中每个模块的功能及接口函数进行介绍。

\
特征处理
--------

环境返回的数据通常无法直接作为智能体预测和训练的输入，开发者需要完成特征处理、样本处理和奖励设计，确保数据结构与类型符合智能体的接口规范。

特征处理
--------

在特征处理时，开发者需要完成四个关键的开发工作，分别是**定义数据结构**、**观测处理**、**动作处理**。

定义数据结构
------------

开发目录：&lt;智能体文件夹&gt;/feature/definition.py

首先，开发者需要定义智能体可以使用的数据结构（类）。

开发框架已经预先定义好了三种数据类型：ObsData, ActData, SampleData。

ObsData和ActData分别表示智能体预测的输入和输出，将会由agent.predict()使用；

SampleData为训练样本的数据类型，训练样本将会被agent.learn()使用，进行模型训练。

核心函数介绍
------------

create\_cls

用于动态创建数据结构（类）。ObsData, ActData,
SampleData是训练流程必需的三类，但每一个类的数据结构包含哪些属性完全由开发者自定义，属性名称和属性数量没有限制。

ObsData = create\_cls("ObsData",\
feature=None,\
)\
ActData = create\_cls("ActData",\
action=None,\
prob=None,\
)\
SampleData = create\_cls("SampleData",\
npdata=None\
)

### Parameters

  **参数名**       **介绍**
  ---------------- ------------------------------------------
  **第一个参数**   字符串类型，类的名称
  **其余参数**     类的属性，默认值为None，由开发者自行定义

观测处理
--------

开发目录：&lt;智能体文件夹&gt;/agent.py

由于环境的reset和step接口返回的数据属于原始观测数据，无法直接作为智能体预测时的输入，开发者需要将这部分数据进行特征化。

核心函数介绍
------------

observation\_process

将环境返回的观测数据转换成ObsData类型数据。\
很多情况下，特征工程包含了大量的数值处理、数据转换和领域知识，我们建议将大量的特征处理代码在&lt;智能体文件夹&gt;/feature/preprocessor.py文件中实现，然后由于observation\_process进行调用。

def observation\_process(self, obs, state=None):\
return ObsData(feature=feature, legal\_act=legal\_actions)

### Parameters

  **参数名**   **介绍**
  ------------ --------------------------------------------------------
  **obs**      Observation类型，env.reset和env.step返回的环境观测数据
  **state**    EnvInfo类型，env.reset和env.step返回的环境状态数据

### Return

  **参数名**    **介绍**
  ------------- ------------------------------------------------------------------
  **ObsData**   开发者定义的ObsData类型的数据，将作为agent.predict()函数的输入。

动作处理
--------

开发目录：&lt;智能体文件夹&gt;/agent.py

由于环境的step接口的输入须要满足环境的特定数据协议，开发者需要将智能体预测的输出转换为符合环境step接口输入要求的数据。

核心函数介绍
------------

action\_process

将智能体预测输出的ActData类型数据转换成环境可以接收的动作数据.

def action\_process(self, act\_data):\
return act\_data.act

### Parameters

  **参数名**      **介绍**
  --------------- -------------------------------
  **act\_data**   开发者定义的ActData类型的数据

### Return

环境能处理的动作数据类型，作为env.step()的输入

奖励设计
--------

开发目录：&lt;智能体名称&gt;/feature/definition.py

这里的奖励特指强化学习中的Reward，注意要与环境反馈的Score进行区分。Score通常用于衡量智能体在环境中的实际表现。开发者在设计Reward时，有非常大的灵活性，不仅可以基于环境返回的观测信息，还可以加入开发者对问题的理解、经验或者知识。

核心函数介绍
------------

reward\_shaping

开发框架预设的奖励设计函数接口，开发者可以通过该函数实现复杂的奖励计算，在训练工作流中调用。

def reward\_shaping(obs, \_obs, state, \_state):\
return reward

### Parameters

参数个数和类型不限制，可以是环境信息、智能体信息、开发者的经验和知识等。

### Return

数值类型，计算出的reward值

样本处理
--------

开发目录：&lt;智能体文件夹&gt;/feature/definition.py

由于环境与智能体交互过程中产生的轨迹数据无法直接作为智能体训练时的输入，开发者需要将轨迹数据转换为训练样本数据。

核心函数介绍
------------

### sample\_process

将环境与智能体交互过程中产生的轨迹数据转换成开发者定义的SampleData类型数据。

@attached\
def sample\_process(self, list\_game\_data):\
return \[SampleData(\*\*i.\_\_dict\_\_) for i in list\_game\_data\]

### Parameters

  **参数名**             **介绍**
  ---------------------- -----------------------------------------------------------------------------------------
  **list\_game\_data**   list(Frame)类型， 使用开发者自定义的Frame作为输入，因为样本一般进行批处理，所以传入列表

### Return

  **参数名**                 **介绍**
  -------------------------- --------------------------------
  **list(SampleData)类型**   SampleData类型的数据组成的列表

为了支持分布式训练，样本数据需要进行网络传输，由于SampleData无法直接进行网络传输，需要先转换成Numpy的Array，待传输到对端之后再由np.Array转换成SampleData。

因此，开发者需要实现两个转换函数 SampleData2NumpyData和 NumpyData2SampleData，这两个函数互为反函数。

**注意**：由于这两个函数会被分布式计算框架调用，因此这两个函数的实现都必须包含一个装饰器@attached

### SampleData2NumpyData

将SampleData转换为NumpyData。

@attached\
def SampleData2NumpyData(g\_data):\
return g\_data.npdata

### Parameters

  **参数名**    **介绍**
  ------------- -----------------
  **g\_data**   SampleData 类型

### Return

Numpy.array类型

### NumpyData2SampleData

将NumpyData转换为SampleData。

@attached\
def NumpyData2SampleData(s\_data):\
return SampleData(npdata=s\_data)

### Parameters

  **参数名**    **介绍**
  ------------- ------------------
  **s\_data**   Numpy.array 类型

### Return type

SampleData类型

\
算法开发
--------

开发目录：&lt;智能体名称&gt;/algorithm/algorithm.py

在完成特征处理和奖励设计后，开发者还需要实现强化学习算法，以通过特定优化方法更新模型参数。

以下为实现强化学习算法的核心函数介绍，有关函数的更多细节可以查阅[*分布式计算框架*](https://tencentarena.com/docs/p-edu-gorge_walk_v2/37.1.3/guidebook/taa-rl-fw/distributed_computing_fw/#agent)

核心函数介绍
------------

### learn

实现强化学习优化算法的核心方法，该函数输入为训练样本数据，开发者需基于不同的算法完成相关实现，包括优化方法、损失计算等。

def learn(self, list\_sample\_data):\
"""\
Implementing the core method of the algorithm\
实现算法的核心方法\
"""\
loss = 0 *\# 基于不同算法实现loss计算 Calculate loss*\
loss.backward() *\# 计算梯度 Calculate gradient*\
self.optimizer.step() *\# 通过梯度下降等方法更新模型 Update weights*

### Parameters

  **参数名**               **介绍**
  ------------------------ ------------------------------------
  **list\_sample\_data**   list类型，训练样本(SampleData)列表

\
模型开发
--------

开发目录：&lt;智能体名称&gt;/model/model.py

一个强化学习模型是基于特征作为输入数据，输出策略的神经网络模型。

开发者需要在model.py文件中，实现神经网络模型。开发框架要求，模型类需继承 torch.nn.Module 类，即符合Pytorch模型的实现规范。

class Model(nn.Module):\
def \_\_init\_\_(self, state\_shape, action\_shape=0, softmax=False):\
super().\_\_init\_\_()

\
工作流开发
----------

训练工作流
----------

在完成智能体开发后，需要进一步实现由[*分布式计算框架*](https://tencentarena.com/docs/p-edu-gorge_walk_v2/37.1.3/guidebook/taa-rl-fw/distributed_computing_fw/)提供的训练工作流接口，使智能体和环境持续交互，收集训练样本，迭代模型参数，最终完成策略的优化。

核心函数介绍
------------

### workflow

通过该函数实现强化学习训练工作流，调用智能体和环境提供的接口，完成环境交互、样本收集和模型更新。

@attached\
def workflow(envs, agents, logger=None, monitor=None):

### Parameters

  **参数名**    **介绍**
  ------------- ----------------------------------------------------------------------------------------------------------------
  **envs**      list类型，环境列表，返回当前正在运行的环境。
  **agents**    list类型，智能体列表，通过调用开发者实现的 &lt;智能体名称&gt;/agent.py 实例化 Agent, 并作为输入传入 workflow。
  **logger**    Logger类型，框架提供的日志组件，接口与 python 的 logging 库一致。
  **monitor**   Monitor类型，框架提供的监控组件。

接下来，我们将通过一个训练工作流关键步骤的代码示例（具体实现由开发者完成），说明如何通过训练工作流实现完整训练流程。

@attached\
def workflow(envs, agents, logger=None, monitor=None):\
*\# Get the environment and agent*\
*\# 获取环境和智能体*\
env, agent = envs\[0\], agents\[0\]\
\
*\# Execute several epochs*\
*\# 执行若干次epoch*\
epoch\_num = 1000\
\
*\# Each epoch executes several episodes*\
*\# 每个epoch执行若干个episode*\
episode\_num\_every\_epoch = 1000\
\
*\# Training loop*\
*\# 训练循环*\
for epoch in range(epoch\_num):\
*\# After each episode, the trajectory data is converted into training
samples for training.*\
*\# 在每一个episode结束之后，将轨迹数据转换成训练样本进行训练*\
for g\_data in run\_episodes(episode\_num\_every\_epoch, env, agent,
logger, monitor):\
*\# Agent training. If single-machine training, the model is trained
directly; if distributed training, samples are sent to the
sample-pool.*\
*\#
agent进行训练。如果是单机训练，则直接对模型进行训练；如果是分布式训练，则将训练样本发送到样本池。*\
agent.learn(g\_data)\
*\# Ensure that the next training sample collected is new*\
*\# 清空g\_data，确保下一次搜集的训练样本是新的*\
g\_data.clear()\
\
*\# Save the model at intervals*\
*\# 依据时间间隔保存模型*\
now = time.time()\
if now - last\_save\_model\_time &gt;= 300:\
agent.save\_model()\
last\_save\_model\_time = now\
\
\
def run\_episodes(n\_episode, env, agent, logger, monitor):\
*\# Run several episodes*\
*\# 运行若干个episode*\
for episode in range(n\_episode):\
*\# Reset data at the beginning of an episode*\
*\# 在episode开始时重置数据*\
done = False\
collector = list()\
\
*\# Reset enviroment and get initial info*\
*\# 重置环境, 并获取环境初始状态*\
obs, state = env.reset(usr\_conf=usr\_conf)\
\
*\# Load the latest model and call it on demand; if in stand-alone mode,
there is no need to load the remote model*\
*\#
加载最新模型，按需调用；若训练采用单机模式，则无需加载远程模型，可不调用该函数*\
agent.load\_model(id="latest")\
\
*\# Run an episode loop*\
*\# 运行一个episode循环*\
while not done:\
*\# Agent performs inference, gets the predicted action for the next
frame*\
*\# 调用智能体预测函数，获取下一时刻的动作*\
act\_data = agent.predict(list\_obs\_data=\[obs\_data\])\[0\]\
\
*\# Unpack ActData into action*\
*\# 将智能体输出的ActData数据转换为符合环境数据协议要求的动作数据*\
act = agent.action\_process(act\_data)\
\
*\# Interact with the environment, execute actions, get the next state*\
*\# 调用环境step接口，与环境交互, 执行动作, 获取下一时刻的状态*\
frame\_no, \_obs, score, terminated, truncated, \_state = env.step(act)\
if \_obs == None:\
break\
\
*\# Feature processing*\
*\# 对环境返回的观测数据进行处理*\
\_obs\_data = agent.observation\_process(\_obs, \_state)\
\
*\# Disaster recovery*\
*\# 容灾*\
if truncated and frame\_no == None:\
break\
\
*\# Calculate reward*\
*\# 计算reward*\
reward = reward\_shaping(obs\_data, \_obs\_data, state, \_state)\
\
*\# Episode done signal*\
*\# episode结束信号*\
done = terminated or truncated\
\
*\# Construct sample*\
*\# 构造样本*\
frame = Frame(\
obs=obs\_data.feature,\
\_obs=\_obs\_data.feature,\
act=act,\
rew=reward,\
done=done,\
)\
collector.append(frame)\
\
*\# If the game is over, the sample is processed and sent to training*\
*\# 如果episode结束，则进行样本处理，将样本送去训练*\
if done:\
if len(collector) &gt; 0:\
collector = sample\_process(collector)\
*\# Return samples*\
*\# 返回样本数据, agent会调用agent.learn(g\_data)进行训练*\
yield collector\
break\
\
*\# Status update*\
*\# 状态更新*\
obs\_data = \_obs\_data\
obs = \_obs\
state = \_state

\
智能体开发
==========

开发目录：&lt;智能体名称&gt;/agent.py

在完成模型和算法后，开发者还需要实现强化学习智能体，智能体使用模型进行决策、与环境交互并通过算法更新模型参数。

以下为实现强化学习智能体的核心函数介绍，有关函数的更多细节可以查阅[*分布式计算框架*](https://tencentarena.com/docs/p-edu-gorge_walk_v2/37.1.3/guidebook/taa-rl-fw/distributed_computing_fw/#agent)

核心函数介绍
------------

### learn

该函数输入为训练样本数据，开发者需要在该函数中调用算法消费训练样本进行训练。

当然，在不同的训练模式下，该函数使用方法有所不同：

-   单机训练：开发者需要在训练工作流中手动调用该函数以进行一步训练。

-   分布式训练：

    该函数作为训练函数会被循环执行，无需开发者手动调用。

    但该函数还作为样本发送函数，开发者需要在训练工作流中手动调用，以将样本发送至样本池。

def learn(self, list\_sample\_data):\
self.algo.learn(list\_sample\_data) *\# 调用算法消费训练样本进行训练
Call algorithm to train model*

### Parameters

  **参数名**               **介绍**
  ------------------------ ------------------------------------
  **list\_sample\_data**   list类型，训练样本(SampleData)列表

### predict

该方法通过调用模型进行预测，通常在训练时调用该方法，依策略的概率分布采样或引入随机概率。

@predict\_wrapper\
def predict(self, list\_obs\_data, list\_state):\
return \[ActData\]

### Parameters

  **参数名**            **介绍**
  --------------------- --------------------------------------------
  **list\_obs\_data**   list类型，观测数据(ObsData)列表
  **list\_state**       可选参数，list类型，环境返回的状态数据列表

### Return

  **参数名**          **介绍**
  ------------------- ---------------------------------------------
  **List(ActData)**   list类型，开发者定义的动作数据(ActData)列表

### exploit

该方法通过调用模型进行预测，通常在评估时调用该方法，选取策略中概率最高的动作或者策略认为最优的动作。

@exploit\_wrapper\
def exploit(self, observation):

### Parameters

  **参数名**        **介绍**
  ----------------- ----------------------------------------------------------------------------------------
  **observation**   dict类型，环境观测字典，评估工作流中将原始的环境观测信息作为输入传入 agent.exploit()。

### Return

  **参数名**   **介绍**
  ------------ ------------------------------------------------
  **action**   list类型，动作列表，环境可以直接使用的动作指令

### load\_model

智能体通过该接口完成模型参数加载。在上文中提到，Actor会从模型池中获取最新模型参数文件，开发者需要手动调用load\_model()函数，使智能体完成模型参数加载。

@load\_model\_wrapper\
def load\_model(self, path=None, id="1"):\
*\# When loading the model, you can load multiple files,*\
*\# and it is important to ensure that each filename matches the one
used during the save\_model process.*\
*\# 加载模型, 可以加载多个文件,
注意每个文件名需要和save\_model时保持一致*\
model\_file\_path = f"{path}/model.ckpt-{str(id)}.pkl"\
self.model.load\_state\_dict(\
torch.load(model\_file\_path, map\_location=self.device),\
)

### Parameters

  **参数名**   **介绍**
  ------------ ----------------------------------------------------------------------------------------------------
  **path**     string类型，加载模型参数文件的路径，开发框架根据使用场景得到相应的路径, 并作为输入传入 load\_model
  **id**       string类型，模型参数文件的 id，使用 id 指定加载的模型参数文件

### save\_model

开发者可以通过该函数保存当前时刻的模型文件及智能体代码包，开发框架会将开发者需要保存的内容打包为zip格式的文件。

当开发者使用腾讯开悟客户端开发时，开发框架会在客户端指定目录下存储该zip文件。\
当开发者使用腾讯开悟平台时，开发框架会将该zip文件存储在云端，开发者可以通过平台的训练管理模块查看每一个训练任务的zip文件，即模型。

@save\_model\_wrapper\
def save\_model(self, list\_obs\_data, list\_state):\
*\# To save the model, it can consist of multiple files,*\
*\# and it is important to ensure that each filename includes the
"model.ckpt-id" field.*\
*\# 保存模型, 可以是多个文件,
需要确保每个文件名里包括了model.ckpt-id字段*\
model\_file\_path = f"{path}/model.ckpt-{str(id)}.pkl"\
\
*\# Copy the model's state dictionary to the CPU*\
*\# 将模型的状态字典拷贝到CPU*\
model\_state\_dict\_cpu = {k: v.clone().cpu() for k, v in
self.model.state\_dict().items()}\
torch.save(model\_state\_dict\_cpu, model\_file\_path)

### Parameters

  **参数名**   **介绍**
  ------------ ------------------------------------------------------------------------------------------------
  **path**     string类型，模型文件保存的路径，开发框架根据使用场景得到相应的路径, 并作为输入传入 save\_model
  **id**       string类型，模型文件的索引，开发框架获取到模型池中最新模型的索引, 并作为输入传入 save\_model

\
分布式计算框架
==============

在强化学习项目的开发中，​分布式计算框架是支撑大规模训练任务的核心基础设施。本开发框架提供了由腾讯王者荣耀团队自主研发的强化学习分布式计算框架KaiwuDRL，通过并行化计算、高效资源调度和分布式协同优化，显著提升智能体训练的效率与稳定性。

接下来，我们将详细介绍KaiwuDRL的系统架构与核心能力，帮助开发者进一步理解训练和评估工作流的运行逻辑。

总体架构

![KaiwuDRL分布式计算框架总体架构图。](04_项目内容及任务分工\dev_guide_optimized_media/media/image6.png "图片 26"){width="5.858267716535433in"
height="3.3616404199475065in"}

组件介绍

如上图所示，KaiwuDRL 的整体架构包括 Environment、Aisrv、Actor、Learner
等强化学习组件（均支持多实例并行运行）。此外，还集成了通信、日志、监控、对象存储等基础组件。

组件介绍如下表：

  **组件名称**   **功能描述**
  -------------- --------------------------------------------------------------------------------------------------------------------
  Environment    环境服务组件，负责运行强化学习环境，支持通过标准接口与环境交互，并返回环境的观测obs。
  Aisrv          训练流程中枢，负责收集环境样本，运行训练、评估工作流，以及处理各个组件间的数据传输。
  Actor          预测服务组件，负责响应Aisrv的预测请求，调用智能体 predict() 或 exploit() 函数生成动作决策结果。
  Learner        训练服务组件，负责采集训练样本，调用智能体 learn() 函数完成梯度计算及模型迭代。
  MemoryPool     样本存储组件，简称样本池。负责存储训练样本，接收 Aisrv 打包的训练样本，发往 Learner 用于智能体训练。
  ModelPool      模型存储组件，简称模型池。负责存储模型参数文件，接收 Learner 产出的模型参数文件，将最新的模型参数文件发送给Actor。
  日志           日志采集组件，负责记录强化学习系统中各个组件的运行日志，支持通过标准接口上报日志。
  监控           监控采集组件，负责采集系统资源使用率、训练指标趋势等数据，支持通过标准接口上报数据指标。

**MemoryPool和ModelPool仅在分布式训练时启用。**

服务介绍

基于上述组件，KaiwuDRL 提供了预测服务和训练服务：

预测服务

-   Aisrv → Environment：发送环境配置并创建新一局episode；

-   Environment → Aisrv：返回原始观测数据；

-   Aisrv → Actor：Aisrv基于原始观测数据，向Actor发送预测请求；

-   Actor：使用预测请求中的观测进行特征处理，智能体基于特征处理后的数据进行预测，并且将预测数据处理为环境可以识别的动作指令，发送给Aisrv；

-   Aisrv：使用动作指令与环境Environment进行交互，Environment返回新的观测；

训练服务

-   Aisrv：预测服务不断产生轨迹数据，Aisrv完成样本处理，并发送至样本池；

-   Learner：从样本池按批采集样本进行训练，并将最新的模型参数同步至Actor；

工作流

KaiwuDRL提供了训练、评估工作流的接口函数，开发者可以按需灵活调用上述组件和服务，以实现模型的训练和评估。

训练工作流
----------

开发目录：&lt;智能体名称&gt;/workflow/train\_workflow.py

### workflow

训练工作流的核心函数，在workflow中可自定义训练流程。可以在[*智能体/工作流开发*](https://tencentarena.com/docs/p-edu-gorge_walk_v2/37.1.3/guidebook/taa-rl-fw/rl_agent/workflow/)中查看详细的训练工作流代码示例。

@attached\
def workflow(envs, agents, logger=None, monitor=None):

### Parameters

  **参数名**    **介绍**
  ------------- ----------------------------------------------------------------------------------------------------------------
  **envs**      list类型，环境列表，返回当前正在运行的环境。
  **agents**    list类型，智能体列表，通过调用开发者实现的 &lt;智能体名称&gt;/agent.py 实例化 Agent, 并作为输入传入 workflow。
  **logger**    Logger类型，框架提供的日志组件，接口与 python 的 logging 库一致。
  **monitor**   Monitor类型，框架提供的监控组件。

评估工作流
----------

在运行训练任务（训练工作流）并获得模型文件后，可以通过运行评估任务（评估工作流），对模型能力进行验证。

当开发者在使用腾讯开悟平台所提供的强化学习项目时，评估工作流由腾讯开悟官方实现，开发者无法修改。评估工作流会调用开发者自定义的agent.exploit()函数。

### Agent

KaiwuDRL提供了智能体相关的接口函数，开发者可以按需实现以下接口函数，并在训练工作流中调用。

开发目录：&lt;智能体名称&gt;/agent.py

### learn

该函数输入为训练样本数据，开发者需要在该函数中调用算法消费训练样本进行训练。

当然，在不同的训练模式下，该函数使用方法有所不同：

-   单机训练：开发者需要在训练工作流中手动调用该函数以进行一步训练。

-   分布式训练：

    该函数作为训练函数会被循环执行，无需开发者手动调用。

    但该函数还作为样本发送函数，开发者需要在训练工作流中手动调用，以将样本发送至样本池。

@learn\_wrapper\
def learn(self, list\_sample\_data):

### Parameters

  **参数名**               **介绍**
  ------------------------ --------------------------------------------------------------------------------------------------------------------------
  **list\_sample\_data**   list类型，训练样本(SampleData)列表，Learner从样本池按照配置项train\_batch\_size采样一批样本, 作为输入传入 learn() 函数。

配置项
------

开发目录: /conf/configure\_app.toml

-   Learner在每一次执行learn()函数时，会从样本池采样一批样本作为输入。并按照开发者配置的频次dump\_model\_freq保存模型参数文件，模型同步服务将按照配置model\_file\_sync\_per\_minutes将模型参数文件推送至模型池。

-   Actor中的模型同步服务将按照配置model\_file\_sync\_per\_minutes，从模型池获取最新模型参数文件。

相关配置项如下：

\[app\]\
*\# The time interval for executing the learn() function, configurable
to throttle the Learner and balance sample production/consumption.*\
*\#
执行learn函数进行训练的时间间隔，可通过该配置让Learner休息以调节样本生产消耗比*\
learner\_train\_sleep\_seconds = 0.001\
\
*\# Replay buffer configurations*\
*\# 样本池容量*\
replay\_buffer\_capacity = 4096\
\
*\# The ratio of the sample pool capacity that triggers training*\
*\# 当样本池中的样本占总容量的比例达到该值时，启动训练*\
preload\_ratio = 1.0\
\
*\# When new samples are added to the sample pool, the logic for
removing old samples: reverb.selectors.Lifo, reverb.selectors.Fifo*\
*\#
当新样本加入样本池时，旧样本的移除逻辑，可选项：reverb.selectors.Lifo,
reverb.selectors.Fifo*\
*\# reverb.selectors.Lifo：先进后出(Last In, First Out)*\
*\# reverb.selectors.Fifo：先进先出(First In, First Out)*\
reverb\_remover = "reverb.selectors.Fifo"\
\
*\# The sampling logic of the Learner from the sample pool:
reverb.selectors.Fifo, reverb.selectors.Uniform*\
*\# Learner从样本池中采样的逻辑，可选项：reverb.selectors.Fifo,
reverb.selectors.Uniform*\
*\# reverb.selectors.Uniform：Samples are selected uniformly at random
from the replay buffer, with each sample having an equal probability of
being chosen.*\
*\#
reverb.selectors.Uniform：从回放缓冲区中随机均匀地选择样本，每个样本被选中的概率相同。*\
*\# reverb.selectors.Fifo：Samples are selected in the order they were
added to the replay buffer.*\
*\# reverb.selectors.Fifo：按照先进先出从回放缓冲区中选择样本。*\
reverb\_sampler = "reverb.selectors.Uniform"\
\
*\# Training batch size limit for Learner*\
*\# Learner训练时样本批处理大小*\
train\_batch\_size = 2048\
\
*\# Model dump frequency (steps)*\
*\# 训练间隔多少步输出模型参数文件*\
dump\_model\_freq = 1000\
\
*\# The Learner pushes model updates, and the frequency at which Actors
fetch the model (in minutes).*\
*\#
Learner推送模型参数文件至模型池，以及Actor从模型池获取模型参数文件的频次（单位：分钟）*\
model\_file\_sync\_per\_minutes = 1\
\
*\# he number of model updates pushed per learner iteration, and the
maximum number of updates each actor can fetch at once (cap: 50).*\
*\#
Learner每次推送模型参数文件，以及Actor每次获取模型参数文件的数量（上限：50）*\
modelpool\_max\_save\_model\_count = 1

### predict

该方法通过调用模型进行预测，通常在训练时调用该方法，依策略的概率分布采样或引入随机概率。

@predict\_wrapper\
def predict(self, list\_obs\_data, list\_state):\
return \[ActData\]

### Parameters

  **参数名**            **介绍**
  --------------------- --------------------------------------------
  **list\_obs\_data**   list类型，观测数据(ObsData)列表
  **list\_state**       可选参数，list类型，环境返回的状态数据列表

### Return

  **参数名**          **介绍**
  ------------------- ---------------------------------------------
  **List(ActData)**   list类型，开发者定义的动作数据(ActData)列表

### exploit

该方法通过调用模型进行预测，通常在评估时调用该方法，选取策略中概率最高的动作或者策略认为最优的动作。

@exploit\_wrapper\
def exploit(self, observation):

### Parameters

  **参数名**        **介绍**
  ----------------- ----------------------------------------------------------------------------------------
  **observation**   dict类型，环境观测字典，评估工作流中将原始的环境观测信息作为输入传入 agent.exploit()。

### Return

  **参数名**   **介绍**
  ------------ ------------------------------------------------
  **action**   list类型，动作列表，环境可以直接使用的动作指令

### load\_model

智能体通过该接口完成模型参数加载。在上文中提到，Actor会从模型池中获取最新模型参数文件，开发者需要手动调用load\_model()函数，使智能体完成模型参数加载。

@load\_model\_wrapper\
def load\_model(self, path=None, id="1"):\
*\# When loading the model, you can load multiple files,*\
*\# and it is important to ensure that each filename matches the one
used during the save\_model process.*\
*\# 加载模型, 可以加载多个文件,
注意每个文件名需要和save\_model时保持一致*\
model\_file\_path = f"{path}/model.ckpt-{str(id)}.pkl"\
self.model.load\_state\_dict(\
torch.load(model\_file\_path, map\_location=self.device),\
)

### Parameters

  **参数名**   **介绍**
  ------------ ----------------------------------------------------------------------------------------------------
  **path**     string类型，加载模型参数文件的路径，开发框架根据使用场景得到相应的路径, 并作为输入传入 load\_model
  **id**       string类型，模型参数文件的 id，使用 id 指定加载的模型参数文件

### save\_model

开发者可以通过该函数保存当前时刻的模型文件及智能体代码包，开发框架会将开发者需要保存的内容打包为zip格式的文件。

当开发者使用腾讯开悟客户端开发时，开发框架会在客户端指定目录下存储该zip文件。\
当开发者使用腾讯开悟平台时，开发框架会将该zip文件存储在云端，开发者可以通过平台的训练管理模块查看每一个训练任务的zip文件，即模型。

@save\_model\_wrapper\
def save\_model(self, path=None, id="1"):\
*\# To save the model, it can consist of multiple files,*\
*\# and it is important to ensure that each filename includes the
"model.ckpt-id" field.*\
*\# 保存模型, 可以是多个文件,
需要确保每个文件名里包括了model.ckpt-id字段*\
model\_file\_path = f"{path}/model.ckpt-{str(id)}.pkl"\
\
*\# Copy the model's state dictionary to the CPU*\
*\# 将模型的状态字典拷贝到CPU*\
model\_state\_dict\_cpu = {k: v.clone().cpu() for k, v in
self.model.state\_dict().items()}\
torch.save(model\_state\_dict\_cpu, model\_file\_path)

### Parameters

  **参数名**   **介绍**
  ------------ ------------------------------------------------------------------------------------------------
  **path**     string类型，模型文件保存的路径，开发框架根据使用场景得到相应的路径, 并作为输入传入 save\_model
  **id**       string类型，模型文件的索引，开发框架获取到模型池中最新模型的索引, 并作为输入传入 save\_model

其他功能

加载预训练模型
--------------

本框架支持在已有的模型（称为预训练模型）基础上继续训练。预训练模型结构和继续训练的模型结构需保持一致。

当使用腾讯开悟平台时，在创建训练任务时，可以在弹窗中选择预训练模型以继续训练。

当使用腾讯开悟客户端时，需要将训练好的模型文件（客户端工作空间/ckpt路径下的pkl类型文件）放入代码包中，并在代码包conf/configure\_app.toml中完成预训练模型相关的配置，配置说明如下：

*\# Whether to enable the preload model function. If enabled (true), the
model specified by preload\_model\_id will be loaded as the initial
model in the preload\_model\_dir directory; if disabled (false), no
preloading will be performed.*\
*\#
是否启用加载预训练模型功能，若开启(true)，将在preload\_model\_dir目录下加载由preload\_model\_id指定的模型作为初始模型；若关闭(false)，则不加载预训练模型。*\
preload\_model = false\
\
*\# The relative path of the preloaded model folder (the variable name
{agent\_name} refers to the agent\_algorithm name directory in the code
package). It is only effective when preload\_model=true. When the
preload model function is enabled, you need to create a new ckpt folder
under the agent\_algorithm name directory in the code package and place
the model file (.pkl) there.*\
*\#
预训练模型文件夹相对路径(变量名{agent\_name}指代码包中agent\_算法名目录)，仅在preload\_model=true时生效；当开启加载预训练模型功能时，需要在代码包中agent\_算法名目录下新建ckpt文件夹，将模型文件（.pkl）放置此即可。*\
preload\_model\_dir = "{agent\_name}/ckpt"\
\
*\# The identification ID of the preloaded model (here refers to the
number of model training steps). This ID corresponds to the number of
training steps recorded in the model file name. It only takes effect
when preload\_model=true.*\
*\# Note that it is forbidden to modify the original model file name,
otherwise the model preloading process will fail.*\
*\#
预训练模型的标识ID（这里指模型训练步数），该ID对应模型文件名中的训练步数记录。仅在preload\_model=true时生效。*\
*\# 注意，禁止修改原始模型文件名，否则将导致预训练模型加载失败。*\
preload\_model\_id = 1000

客户端继续训练使用示例
----------------------

以加载 40052步的DQN模型 为例：

解压模型压缩包，在ckpt路径下找到名为 model.ckpt-40052.pkl 的模型文件

将模型文件放到代码包路径下，例如在agent\_dqn内新建一个ckpt目录，放入上述模型文件

在代码包conf/configure\_app.toml中完成以下配置：

preload\_model = true\
preload\_model\_dir = "agent\_dqn/ckpt"\
preload\_model\_id = 40052

完成上述步骤后，Aisrv和Learner将加载预训练模型以继续训练。

加载对手模型功能
----------------

在使用腾讯开悟平台时，部分项目支持从平台的模型管理中加载自定义模型。
用户可以在 kaiwu.json 文件中配置期望用作对手模型的模型
ID。智能体通过 load\_opponent\_agent() 函数完成对手模型的加载。

该功能允许智能体加载指定的网络结构和参数，并根据模型文件中的实现调用
agent 的reset、predict、exploit等方法。
对手模型作为固定水平的自定义模型，可以在 PVP
任务中用于训练评估，以反映训练过程中模型水平的提高。

以下是load\_opponent\_agent()函数示例：

@load\_opponent\_agent\_wrapper\
def load\_opponent\_agent(self, id="1"):\
*\# Framework provides loading opponent agent function, no need to
implement function content*\
*\# 框架提供的加载对手模型功能，无需实现函数内容*\
pass

### Parameters

  **参数名**   **介绍**
  ------------ ---------------------------------------------------------------
  **id**       string类型，平台模型管理的 模型id，使用 id 指定加载的模型文件

以下是kaiwu.json的配置示例：

{\
"model\_pool": \[609, 608\]\
}

策略类型选择
------------

本框架支持两种核心的训练范式：**同策略（On-Policy）**与**异策略（Off-Policy）**。此选择是训练流程的基础，将直接影响数据的使用方式、内存占用以及部分超参数的推荐设置。

  **特性**   **On-Policy**                                                                            **Off-Policy**
  ---------- ---------------------------------------------------------------------------------------- ------------------------------------------------------------------------------------
  数据来源   必须使用由当前正在训练的策略实时与环境交互产生的**新数据**。                             可以使用**历史经验数据**，这些数据可能来自旧版本的策略或不同的探索策略。
  数据使用   **一次性为主**。数据在用于一次策略更新后通常被丢弃，以确保训练数据与当前策略的一致性。   **可重复利用**。数据被存储在“经验回放缓冲区”中，可被多次采样用于训练。
  主要优势   **训练稳定**，理论收敛性有保障，行为容易理解。                                           **数据效率高**，可充分利用每一次交互获得的数据，更适合现实世界中交互成本高的场景。

配置参数详解
------------

在代码包conf/configure\_app.toml中，通过以下参数进行设置：

*\# Training paradigm: on-policy or off-policy*\
*\# 训练时采用on-policy, off-policy*\
algorithm\_on\_policy\_or\_off\_policy = "on-policy"\
\
*\# Model dump frequency (steps)*\
*\# Interval (in steps) for saving model parameter files.*\
*\# on\_policy: set to 1 for optimal training.*\
*\# off\_policy: recommend 1000 to reduce save frequency.*\
*\# 训练间隔多少步输出模型参数文件,
on\_policy时设置为1训练效果最好，off\_policy时建议设置为1000,
减少模型保存频率*\
dump\_model\_freq = 1

\
监控与日志
==========

本文介绍如何使用腾讯开悟平台的监控与日志功能，帮助您实时掌握训练状态、快速定位问题。

监控
----

查看监控面板
------------

在**训练管理**页面，点击**查看监控**按钮，即可打开监控面板：

监控面板组成
------------

监控面板包含两个核心模块：

  **模块**           **功能**
  ------------------ ------------------------------------------
  **错误日志数量**   展示各模块的错误日志统计，点击可查看详情
  **监控指标图**     展示训练过程中的各类数据指标

指标分类
--------

监控指标分为四类：

  **分类**                    **说明**
  --------------------------- ------------------------------------------------
  **基础指标**（basic）       训练进度相关的核心数据，如训练步数、预测次数等
  **硬件指标**（hardware）    资源使用情况，如 CPU、GPU、内存利用率
  **算法指标**（algorithm）   算法相关数据，不同算法的指标有所不同
  **环境指标**（env）         环境相关数据，不同环境的指标有所不同

-   基础指标

  **面板名称**     **指标名称**                                  **说明**
  ---------------- --------------------------------------------- ------------------------------------
  训练累计步数     train\_global\_step                           agent.learn() 的调用次数
  预测累计次数     predict\_succ\_cnt                            agent.predict() 的调用次数
  模型加载次数     load\_model\_succ\_cnt                        agent.load\_model() 成功调用的次数
  样本接收次数     sample\_receive\_cnt                          接收到的样本总数
  已结束任务数     episode\_cnt                                  已完成的 episode 数量
  样本生产消耗比   sample\_production\_and\_consumption\_ratio   训练消耗样本数 / 采样生产样本数

-   硬件指标

  **面板名称**     **指标名称**                              **说明**
  ---------------- ----------------------------------------- --------------------------------
  CPU 使用率       aisrv\_cpu\_usage / learner\_cpu\_usage   分别对应 aisrv 和 learner 进程
  GPU 使用率       aisrv\_gpu\_usage / learner\_gpu\_usage   分别对应 aisrv 和 learner 进程
  GPU 显存使用率   gpu\_memory                               显存占用百分比
  内存使用率       ram\_usage                                容器内存占用，过高可能导致 OOM

-   算法指标

不同算法的指标各不相同，详见具体算法文档。

-   环境指标

不同环境的指标各不相同，详见具体环境文档。

自定义监控面板
--------------

平台支持在实验代码中自定义监控指标，配置并上报期望观测的业务数据。

配置面板
--------

在算法配置目录下编辑监控配置文件：

**文件路径**：agent\_{算法名}/conf/monitor\_builder.py

**配置示例**：

def build\_monitor():\
monitor = MonitorConfigBuilder()\
\
config\_dict = (\
monitor.title("项目名称")\
.add\_group(group\_name="算法指标", group\_name\_en="algorithm")\
.add\_panel(\
name="累积回报",\
description="反映智能体能力的指标",\
type="line",\
)\
.add\_metric(metrics\_name="reward", expr="avg(reward{})")\
.end\_panel()\
.end\_group()\
.build()\
)\
return config\_dict

上报指标数据
------------

在代码中调用监控上报接口，**代码示例**：

import os\
from monitor import monitor\
\
monitor\_data = {\
"reward": 100.5\
}\
monitor.push\_data({os.getpid(): monitor\_data})

配置参数说明
------------

  **配置项**   **字段**          **字段类型**   **说明**                                                 **限制条件**
  ------------ ----------------- -------------- -------------------------------------------------------- -----------------------------------------------------------------
  项目名称     title             string         监控面板名称（该字段不在监控页面进行展示，不建议修改）   支持中英文、数字、=、+、/、@、\#、\_、-及空格，长度 1\~100 字符
  面板组       group\_name       string         面板组名称                                               支持中英文、数字、\_、-及空格，长度 1\~20 字符
               group\_name\_en   string         面板组英文标识                                           支持英文、数字、\_、-及空格，长度 1\~50 字符
  面板         name              string         面板名称                                                 支持中英文、数字、\_、-及空格，长度 1\~20 字符
               name\_en          string         面板英文标识                                             支持英文、数字、\_、-及空格，长度 0\~50 字符
               description       string         面板描述信息                                             支持中英文、数字、标点符号等，长度 0\~200 字符
               type              string         图表类型                                                 仅line（折线图）、stat（数值图）有效
               unit              string         数值单位，当 type 为 stat 时展示到指标后                 仅 stat 类型有效
  指标         metrics\_name     string         指标显示名称                                             支持中英文、数字、\_、-、{}及空格，长度 1\~40 字符
               expr              string         指标查询表达式，支持指标变量（lable）                    使用 PromQL 的查询语法即可

**规格限制**：

折线图面板：指标数量限制为 20 个指标

数值图面板：指标数量限制为 2 个指标

指标变量说明

如需按维度分组展示数据（如不同对手的胜率），可通过指标变量（label）
区分维度，并在配置中使用指标变量定义展示方式。

**第一步：上报带 label 的数据**

*\# 上报玩家1的胜率指标*\
monitor.push\_data({os.getpid(): win\_rate, "player": "player1"})\
\
*\# 上报玩家2的胜率指标*\
monitor.push\_data({os.getpid(): win\_rate, "player": "player2"})

**第二步：配置指标变量**

.add\_metric(\
metrics\_name="win\_rate\_{player}",\
expr="avg(win\_rate{}) by (player)"\
)

**说明**：

metrics\_name字符串中的 {player} 会被实际的player1、player2替换；最终将在同一个面板上显示两条数据线：win\_rate\_player1 和 win\_rate\_player2

日志
----

框架提供了统一的日志服务，帮助您记录和排查训练过程中的问题。

日志格式
--------

  **字段**   **说明**   **示例**
  ---------- ---------- -------------------------------------
  time       时间戳     2024-09-18 19:33:04.813469
  level      日志级别   INFO / WARNING / ERROR
  message    日志内容   kaiwu learner train count is 365676
  file       源码文件   on\_policy\_trainer.py
  line       代码行号   769
  module     所属模块   learner
  process    进程名     on\_policy\_trainer
  function   函数名     train\_stat
  stack      错误堆栈   仅错误日志包含

注意：

**不要重写日志系统**：重写后监控面板将无法统计错误日志数量

**日志流量限制**：框架限制为 60 条/分钟，超出部分将被丢弃
