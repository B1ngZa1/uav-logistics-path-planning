<!-- image-->

# 《南京邮电大学学报(自然科学版)》网络首发论文

题目： 基于改进 MATD3 强化学习算法的多无人机通感一体系统能效优化

作者： 田峰，陆男，赵晓，吴辰轩

收稿日期： 2026-01-30

网络首发日期： 2026-04-21

引用格式： 田峰，陆男，赵晓，吴辰轩．基于改进 MATD3 强化学习算法的多无人机通感一体系统能效优化[J/OL]．南京邮电大学学报(自然科学版).https://link.cnki.net/urlid/32.1772.TN.20260421.1146.004

<!-- image-->

<!-- image-->

网络首发：在编辑部工作流程中，稿件从录用到出版要经历录用定稿、排版定稿、整期汇编定稿等阶段。录用定稿指内容已经确定，且通过同行评议、主编终审同意刊用的稿件。排版定稿指录用定稿按照期刊特定版式（包括网络呈现版式）排版后的稿件，可暂不确定出版年、卷、期和页码。整期汇编定稿指出版年、卷、期、页码均已确定的印刷或数字出版的整期汇编稿件。录用定稿网络首发稿件内容必须符合《出版管理条例》和《期刊出版管理规定》的有关规定；学术研究成果具有创新性、科学性和先进性，符合编辑部对刊文的录用要求，不存在学术不端行为及其他侵权行为；稿件内容应基本符合国家有关书刊编辑、出版的技术标准，正确使用和统一规范语言文字、符号、数字、外文字母、法定计量单位及地图标注等。为确保录用定稿网络首发的严肃性，录用定稿一经发布，不得修改论文题目、作者、机构名称和学术内容，只可基于编辑规范进行少量文字的修改。

出版确认：纸质期刊编辑部通过与《中国学术期刊（光盘版）》电子杂志社有限公司签约，在《中国学术期刊（网络版）》出版传播平台上创办与纸质期刊内容一致的网络版，以单篇或整期出版形式，在印刷出版之前刊发论文的录用定稿、排版定稿、整期汇编定稿。因为《中国学术期刊（网络版）》是国家新闻出版广电总局批准的网络连续型出版物（ISSN 2096-4188，CN 11-6037/Z），所以签约期刊的网络版上网络首发论文视为正式出版。

# 基于改进 MATD3 强化学习算法的多无人机通感一体系统能效优化

田 峰，陆 男，赵 晓，吴辰轩（南京邮电大学通信与信息工程学院 江苏 南京 210003）

摘 要：当前无人机辅助的通信感知一体化（ISAC）研究受到广泛关注。针对多无人机 ISAC系统资源受限和用户动态漫游等问题，首先，本文考虑了多无人机通信感知和无人机轨迹等模型，并构建系统能效最大化为目标的无人机飞行轨迹和波束成形联合优化问题；其次，设计“两步法”联合优化框架：提出一种基于无人机服务容量的 K-means 自适应无人机与用户关联算法，实现动态实时优化匹配；提出改进的多智能体双延迟深度确定性策略梯度算法，利用波束成形“功率——相位”分离动作降低搜索维度，并在集中式训练分布式执行框架下实现对分式能量效率目标的稳定高效优化。仿真结果表明，所提方法在动态用户场景下不仅显著提升了系统的能量效率，而且实现了无人机轨迹和波束成形的联合优化。

关键词：无人机；通信感知一体化；改进的MATD3算法；轨迹优化；波束成形  
中图分类号：TN929.5  
文献标志码：A

# Optimization on Energy Efficiency of Multi-UAV ISAC Systems Based on Improved MATD3

Tian Feng, Lu Nan, Zhao Xiao, Wu Chenxuan

(College of Telecommunications and Information Engineering, Nanjing University of Posts and Telecommunications, Nanjing 210003, China)

Abstract: At present, research on unmanned aerial vehicle (UAV) aided integrated communication and sensing (ISAC) has attracted widespread attention. For solving the problems of resource constraints in multi-UAV ISAC systems and dynamic user roaming, this paper firstly considers models of communication, sensing and UAV-trajectory, and constructs UAV flight trajectory and beamforming with the goal of maximizing system energy efficiency. Secondly, a "two-step" joint optimization framework is designed: a K-means adaptive UAV and user association algorithm is proposed to achieve dynamic real-time optimization matching; and an improved multi-agent twin delayed deep deterministic policy gradient (MATD3) is proposed to reduce the search dimension and achieves stable and efficient optimal solution. Simulation results show that the proposed method not only significantly improves the energy efficiency of the system, but also achieves joint optimization of UAV trajectory and beamforming.

Key words: UAV; ISAC; Improved MATD3; Trajectory design; Beamforming

随着无线通信技术的飞速发展，通信感知一体化 技 术 （ Integrated Sensing and Communication,ISAC）逐渐成为“6G 愿景”中的六大关键技术之一。ISAC 系统将通信功能与感知功能进行融合，集成于同一软硬件平台共享频谱资源，不仅显著提升了频谱与硬件资源的利用率[1][2]，也为实现高精度环境感知与低延迟通信的协同优化提供了可能。同时，与传统的地面基站相比，无人机（UnmannedAerial Vehicles, UAV）具备高机动性、快速部署、视距（Line of Sight, LoS）链路优越等特点，将其引入 ISAC 系统，可进一步拓展低空应用的巨大潜力[3][4]。

目前，国内外研究团队已针对无人机通信感知一体化的需求，开展了一系列相关研究。其中，文献[3]提出了一种无人机辅助的自适应 ISAC 机制，通过调整时长交替优化波束成形和无人机飞行轨迹。文献[5-6]提出了涉及用户调度、发射功率和无人机轨迹的联合优化问题最大化安全数据传输速率。文献[7]研究了多无人机 ISAC 系统的通信感知资源分配问题，对分解子问题进行迭代求解提高通感性能。上述优化问题一般考虑采用传统的凸优化、交替方向乘子法或启发式算法等进行求解。然而，随着系统网络规模的扩大与环境动态性的增强，此类问题呈现出高维度、非线性和多约束等特征，属于 NP-Hard 问题，通过常规的数学规划方法难以有效求解或优化性能欠佳。

因此，越来越多的研究者逐渐转向数据驱动的智能优化方法，尤其是强化学习（ReinforcementLearning, RL）技术[8][9]，用于解决多无人机协同场景下的相关问题。强化学习通过智能体与环境的持续交互实现策略自主优化，其早期基于表格的 Q学习等方法适用于状态空间有限的任务解决序列决策问题，而深度强化学习（Deep RL, DRL）算法如深度 Q 网络（Deep Q-Network, DQN）、近端策略优化（Proximal Policy Optimization, PPO）、深度确定性策略梯度（Deep Deterministic Policy Gradient,DDPG）等进一步拓展了其在连续控制、高维观测等复杂场景中的应用能力，已成功应用于无人机轨迹规划、自动驾驶以及机器人控制等多个领域[10][11]。针对多无人机协同 ISAC 系统，可考虑多智能机间协同决策与计算的多智能体强化学习（Multi-Agent RL, MARL）。MARL 将每架无人机视为独立智能体，支持分布式执行与局部观测，可有效降低计算负担和协调多智能体之间的关系。文献[11]利用多智能体深度 Q 网络来处理离散用户关联决策，同时提出一种多智能体近端策略优化算法来实现连续的优化轨迹和功率控制。文献[12]提出基于多智能体学习的航迹规划与资源分配算法对每架无人机的发射功率、资源分配和导航进行优化。文献[13]提出混合奖励多智能体邻近策略优化算法，将复杂的组合奖励分解为团队奖励和个人奖励实现多无人机系统优化。

值得关注的是，现有多无人机通感一体化研究大多聚焦于通信速率、感知精度等性能指标的优化，而无人机飞行能耗与轨迹规划、悬停时间等密切相关，其能量效率问题尚未得到充分重视[14][15]。因此，如何通过多智能协同决策提升无人机通感一体化系统的能量效率，已成为实现系统长期可靠运行的关键问题。本文考虑动态用户漫游场景下的多无人机通信感知一体化系统的通信能量效率优化问题。首先，建立了多无人机 ISAC 系统下的通信与感知模型、无人机飞行轨迹与用户漫游模型和能量效率模型，并以最大化系统能量效率为目标，构建了一个涉及无人机——用户关联、无人机轨迹和波束成形的联合优化问题；其次，提出针对该混合整数非线性规划问题求解的两步法：第一步设计基于无人机服务容量的K-means自适应无人机与用户关联算法，实现无人机与用户间的动态实时匹配；第二步在固定用户关联的基础上，采用改进的多智能体双延迟深度确定性策略梯度算法，利用波束成形“功率——相位”分离动作空间降低搜索维度，同时引入了多约束惩罚机制构建复合奖励函数，在集中式训练分布式执行框架下实现对分式能量效率目标的稳定高效优化。最后，通过仿真验证了所提方法在动态用户场景下的有效性，结果表明该方法不仅能够显著提升系统能效，还能够同步优化无人机轨迹、服务覆盖范围及波束成形性能。

本文内容分为四个部分。第一部分构建了由多架无人机、通信用户与感知用户组成的多无人机ISAC 系统模型，并给出相关通信、感知及能量消耗模型。第二部分在此基础上建立了以系统能量效率最大化为目标的联合优化问题。第三部分设计了基于服务容量的K-means自适应无人机与用户关联算法和改进的 MATD3 算法求解该问题。最后通过仿真实验对所提算法的性能进行验证与分析。

## 1 系统模型与问题建立

## 1.1 系统模型

本文考虑一个多无人机协作场景下的通信感知一体化系统，如图 1所示。该系统包含N 架无人机， $K _ { c }$ 个通信用户以及 $K _ { s }$ 个感知用户，且每架无人机和地面用户分别配备 个线性均匀阵列天线和单个收发天线。每架无人机可分别向地面通信用户收发信号实现上下行通信，同时向地面感知用户发送探测信号进行感知，地面用户在规定的服务区内随机漫游活动。假定所有无人机都在高度为H 的同一平面上飞行，因此无人机 、通信用户 $k _ { c }$ 、感知 用 户 $k _ { s }$ 在 时 隙 的 位 置 分 别 表 示 为${ \bf u } _ { n } \left( t \right) = ( x _ { n } \left( t \right) , y _ { n } \left( t \right) , H )$ ${ \bf q } _ { k _ { c } } \left( t \right) = \left( x _ { k _ { c } } \left( t \right) , y _ { k _ { c } } \left( t \right) , 0 \right)$ 、$q _ { k _ { s } } \left( t \right) = \left( x _ { k _ { s } } \left( t \right) , y _ { k _ { s } } \left( t \right) , 0 \right)$ ，其中，t 表 示时隙，$t \in [ 1 , . . . , T ]$ 且 $n \in [ 1 , . . . , N ]$ $k _ { c } \in [ 1 , . . . , K _ { c } ]$ $k _ { s } \in [ 1 , . . . , K _ { s } ]$ 。

<!-- image-->  
图1 系统模型

## 1.2 通信与感知模型

由于无人机与地面用户之间的信道以LoS链路为主，该系统中的通信模型采用自由空间路径损耗模型，本文侧重考虑下行通信，无人机 到用户 $K _ { c }$ 的信道增益表示为：

$$
h _ { n , k _ { c } } = h ( \mathbf { u } _ { n } , q _ { k _ { c } } ) = \sqrt { \alpha d ^ { - 2 } ( u _ { n } , q _ { k _ { c } } ) } a ( u _ { n } , q _ { k _ { c } } )\tag{1}
$$

其中， $\alpha = ( [ G _ { _ T } \cdot G _ { c } \cdot \lambda ^ { 2 } ] / [ ( 4 \pi ) ^ { 2 } ] )$ 是参考距离1m处的信道增益， $G _ { T }$ 和 $G _ { c }$ 分别是通信用户的发射天线增益和接收天线增益； $a ( u _ { n } , q _ { k _ { c } } )$ 是朝向用户$K _ { c }$ 的阵列响应向量，可表示为：

$$
a ( u _ { n } , q _ { k _ { c } } ) = \left[ 1 , e ^ { j 2 \pi d \cos ( \varphi ^ { D } ) / \lambda } , \cdots , e ^ { j 2 \pi ( M - 1 ) d \cos ( \varphi ^ { D } ) / \lambda } \right] ^ { T }\tag{2}
$$

其中， 和 分别是天线空间和波长。

$$
\varphi ^ { D } = \operatorname { a r c c o s } ( [ z _ { n } - z _ { k _ { c } } ] / [ \sqrt { \left( x _ { n } - x _ { k _ { c } } \right) ^ { 2 } + \left( y _ { n } - y _ { k _ { c } } \right) ^ { 2 } + \left( z _ { n } - z _ { k _ { c } } \right) ^ { 2 } } ] )\tag{3}
$$

其中， $\varphi ^ { D }$ 为无人机与通信用户之间的俯仰角角度。

考虑有限的通信感知服务容量，每架无人机被分配给一定数量的通信和感知用户，且每个用户只能与其中一架无人机匹配，设 $\mathcal { S } _ { n , k } [ t ] = 1 \in \left\{ 0 , 1 \right\}$ 表示在时隙 无人机 和目标用户 之间的关联系数，$\delta _ { n , k } [ t ] \in \{ 0 , 1 \}$ ， $\delta _ { n , k } [ t ] = 1$ 表示用户 在时隙 与无人机 相关联， $\delta _ { n , k } [ t ] = 0$ 表示用户k 在时隙 与无人机 不关联，因此用户关联约束可表示为：

$$
\sum _ { n = 1 } ^ { N } \delta _ { n , k } \left[ t \right] = 1 \forall \mathbf { k } \in \mathcal { K }\tag{4}
$$

其中， 表示为用户， $\mathcal { K } = [ 1 , . . . , K ]$ $K = K _ { c } + K _ { s }$

无人机 的发射的通信信号为：

$$
{ x _ { n } } = \sum _ { k _ { c } = 1 } ^ { K _ { c } } \delta _ { n , k _ { c } } w _ { n , k _ { c } } s _ { n , k _ { c } }\tag{5}
$$

其中 $\mathbf { s } _ { n , k _ { c } }$ ∈和 $\mathbf { w } _ { n , k _ { c } [ t ] }$ 分别表示具有 个符号的信号和无人机 处用户 $k _ { c }$ 的波束成形矢量。

用户 $k _ { c }$ 接收到的信号表示为：

$$
y _ { k _ { c } } = \sum _ { n = 1 } ^ { N } \delta _ { n , k _ { c } } \left( h _ { n , k _ { c } } w _ { n , k _ { c } } s _ { n , k _ { c } } + \sum _ { l \neq k _ { c } } h _ { n , k _ { c } } w _ { n , l } s _ { n , l } + \sum _ { m \neq n , l \neq k _ { c } } h _ { m , k _ { c } } w _ { m , l } s _ { m , l } + n _ { 0 } \right)\tag{6}
$$

其中， $n _ { 0 } \sim \mathcal { C N } ( 0 , \sigma ^ { 2 } )$ 表示加性高斯白噪声， $\sigma ^ { 2 }$ 表示噪声功率， $h _ { n , k _ { c } }$ 表示无人机 与其相关联的用户$k _ { c }$ 之间的信道增益， $h _ { m , k _ { c } }$ 表示其它无人机（非无人机 ）与用户 $k _ { c }$ 之间的信道增益， $s _ { n , \mathrm { l } }$ 和 $w _ { n , 1 }$ 分别表示无人机 发送给其他通信用户的信号及其对应的波束成形矢量。

因此，通信用户 $k _ { c }$ 不仅受到同一无人机服务区域内其它链路的干扰，还受到其他无人机服务区域内链路的干扰，因此，用户 $k _ { c }$ 的接收信干噪比 $\gamma _ { k _ { c } }$ 为：

$$
\gamma _ { k _ { c } } = \sum _ { n = 1 } ^ { N } \delta _ { n , k _ { c } } \left( 1 + \frac { \left| h _ { n , k _ { c } } w _ { n , k _ { c } } \right| ^ { 2 } } { \sum _ { l \neq k _ { c } } \left| h _ { n , k _ { c } } w _ { n , l } \right| ^ { 2 } + \sum _ { l \neq k _ { c } } \left| h _ { m , k _ { c } } w _ { m , l } \right| ^ { 2 } + \sigma ^ { 2 } } \right)
$$

根据通信带宽B，用户 $k _ { c }$ 可达和速率可表示为：

(7)

$$
R _ { k _ { c } } = B \log _ { 2 } \left( 1 + \gamma _ { k _ { c } } \right)\tag{8}
$$

为保证无人机与通信用户之间的数据传输质量，可设置无人机 时隙下行通信数据速率阈值为$R _ { \mathrm { m i n } }$ ，即：

$$
R _ { k _ { c } } \left( t \right) \geq R _ { \operatorname* { m i n } }\tag{9}
$$

类似通信信道增益，无人机 和目标感知用户$k _ { s }$ 之间的信道增益表示为：

$$
h _ { n , k _ { s } } = h ( u _ { n } , q _ { k _ { s } } ) = \sqrt { \beta d ^ { - 4 } ( u _ { n } , q _ { k _ { s } } ) } a ( u _ { n } , q _ { k _ { s } } )\tag{10}
$$

其中， $\beta = ( [ G _ { T } \cdot G _ { S } \cdot \eta \cdot \lambda ^ { 2 } ] / [ ( 4 \pi ) ^ { 3 } ] )$ 是参考距离 1m处的信道功率， $G _ { \scriptscriptstyle T }$ 和 $G _ { s }$ 分别是感知用户的发射天线增益和接收天线增益； 表示感知用户的雷达截面的平均值， 是载波的波长； $a ( u _ { n } , q _ { k _ { s } } )$ 是朝向用户 $k _ { s }$ 的阵列响应向量，可表示为：

$$
a ( u _ { n } , q _ { k _ { s } } ) = \left[ 1 , e ^ { j 2 \pi d \cos ( \varphi ^ { D ^ { \scriptstyle * } } ) / \lambda } , \cdots , e ^ { j 2 \pi ( M - 1 ) d \cos ( \varphi ^ { D ^ { \scriptstyle * } } ) / \lambda } \right] ^ { T }\tag{11}
$$

$$
\varphi ^ { D ^ { \prime } } = \operatorname { a r c c o s } ( [ z _ { n } - z _ { k _ { s } } ] / [ \sqrt { \left( x _ { n } - x _ { k _ { s } } \right) ^ { 2 } + \left( y _ { n } - y _ { k _ { s } } \right) ^ { 2 } + \left( z _ { n } - z _ { k _ { s } } \right) ^ { 2 } } ] )\tag{12}
$$

其中， $\varphi ^ { D ^ { \prime } }$ 无人机 到其关联的传感用户 $k _ { s }$ 的俯仰角角度。

系统可将通信信号用于对地面用户的感测，并考虑将无人机n到其关联感知用户 $k _ { s }$ 的发射波束图增益作为衡量感知性能的标准，即感知信号到目标方向的功率 $\Gamma _ { k _ { s } }$ ，可表示为：

$$
\Gamma _ { k _ { s } } = \mathbb { E } \bigg [ \bigg | a ^ { H } \left( u _ { n } , q _ { k _ { s } } \right) x _ { n } \bigg | ^ { 2 } \bigg ] = a ^ { H } \left( u _ { n } , q _ { k _ { s } } \right) \left( \sum _ { k _ { c } = 1 } ^ { K _ { c } } \delta _ { n , k _ { s } } w _ { n , k _ { c } } w _ { n , k _ { c } } ^ { H } \right) a \left( u _ { n } , q _ { k _ { s } } \right) \left( \sum _ { k _ { s } = 1 } ^ { K _ { c } } \frac { 1 } { K _ { n , k _ { s } } } w _ { n , k _ { s } } w _ { n , k _ { c } } ^ { H } \right) ,\tag{13}
$$

其中， $a ( u _ { n } , q _ { k _ { s } } )$ 是朝向用户 $k _ { s }$ 的阵列响应向量，$w _ { n , k _ { c } }$ 表示无人机 n 对感知用户 $k _ { s }$ 的波束成形矢量，$\delta _ { n , k _ { s } }$ 表示在 t 时隙无人机 n 和感知用户 $k _ { s }$ 之间的关联系数。

为满足感知性能的要求，针对不同传感用户的波束图增益考虑最小阈值和回波距离$\mathrm { d } ( u _ { n } - q _ { k _ { s } } )$ ，对感知性能有如下要求：

$$
\Gamma _ { k _ { s } } \ \ge \mathbf { d } ^ { 2 } ( u _ { n } - q _ { k _ { s } } ) \tilde { \Gamma }\tag{14}
$$

## 1.3 无人机运动轨迹与用户漫游模型

在每个时隙t，ISAC信号被同时发送到通信用户和感知用户，对于任意一架无人机 ，其位置满足如下约束条件:

$$
x ^ { l } \leq x _ { n } [ t ] \leq x ^ { u }\tag{15}
$$

$$
y ^ { l } \le y _ { n } [ t ] \le y ^ { u }\tag{16}
$$

其中， $( x ^ { l } , y ^ { l } )$ 和 $( x ^ { u } , y ^ { u } )$ 分别是无人机运动轨迹的下界和上界，且无人机每个时隙的最大飞行速度为$\nu _ { m a x }$ ，无人机之间的距离大于安全距离 $d _ { \mathrm { m i n } }$ 。无人机的方向角表示为 $\boldsymbol { \hat { o } } _ { n } ( n )$ ，无人机的飞行速度表示为v(t)。

考虑用户在无人机服务区内的随机移动过程，采用随机漫游模型来模拟用户位置的变化。在任何离散时隙 ，用户沿着关于正态分布 $N ( 0 , V _ { \mathrm { m a x } } ^ { 2 } )$ 的和 y轴随机移动，其中 $V _ { \mathrm { m a x } }$ 是用户移动的最大速度。由于用户位置的不断变化，从式子（7）、（8）和（13）可以看出，不适当的用户关联都会对通信和感知用户的可实现速率和波束图增益产生负面影响。因此，无人机必须不时更新其用户关联关系，来获得系统的最佳性能。

## 1.4 能量效率模型

由上述无人机运动轨迹与用户漫游模型可见，无人机动态运动与地面用户进行通信感知，其能量消耗主要包括飞行能耗和通感能耗，后者所占比例小得多，可以忽略不计。本文考虑旋翼无人机模型，因此 时隙中的能耗表示为：

$$
E _ { f } \left( t \right) = \left[ P _ { 0 } \left( 1 + \frac { 3 \Vert \mathbf { \Omega } \nu ( t ) \Vert ^ { 2 } } { U _ { t i p } ^ { 2 } } \right) + \frac { 1 } { 2 } d _ { r } \rho s A \Vert \mathbf { \Omega } \nu ( t ) \Vert ^ { 3 } + P _ { 1 } \left( \sqrt { 1 + \frac { \Vert \mathbf { \Omega } \nu ( t ) \Vert ^ { 4 } } { 4 \nu _ { 0 } ^ { 4 } } } - \frac { \Vert \mathbf { \Omega } \nu ( t ) \Vert ^ { 2 } } { 2 \nu _ { 0 } ^ { 2 } } \right) ^ { \frac { 1 } { 2 } } \right] \tau\tag{17}
$$

其中， $P _ { 0 }$ 和 $P _ { 1 }$ 分别对应无人机在悬停静止状态时的叶片轮廓功率和诱导功率， $U _ { t i p }$ 和 $\nu _ { 0 }$ 分别表示旋翼无人机的叶尖角速度和平均旋翼诱导速度， $d _ { r }$ 表示机身阻力比， $\rho$ 表示空气密度， 表示旋翼的健壮性系数，A表示无人机的旋翼盘面积，τ表示时隙的长度。

因此，可将能量效率定义为所有通信用户的总数据速率与所有无人机的能量消耗的比率 $\eta _ { E E } ( t )$ ，可表示为：

$$
\eta _ { E E } \left( t \right) = \frac { \tau \displaystyle \sum _ { k _ { c } = 1 } ^ { K _ { c } } R _ { k _ { c } } \left( t \right) } { \displaystyle \sum _ { n = 1 } ^ { N } E _ { n } ^ { f } \left( t \right) }\tag{18}
$$

## 1.5 优化问题构建

综上，本文的目标为最大化多无人机通信感知一体化系统的能量效率，在满足通信速率约束、感知波束图增益约束、最大发射功率约束和无人机飞行约束的情况下，同时考虑无人机的飞行方向、飞行速度、波束成形矢量和用户关联系数的联合优化。因此，该优化问题表述为：

$$
\operatorname* { m a x } _ { \hat { \sigma } _ { n } ( n ) , \nu _ { n } ( t ) , w _ { n } ( t ) , \delta _ { n , k } [ t ] } \sum _ { t = 1 } ^ { T } \eta _ { E E } ( t )\tag{19}
$$

$$
s . t \hat { \sigma } _ { n } ( n ) \in [ 0 , 2 \pi )\tag{20}
$$

$$
\nu _ { n } ( t ) \in [ 0 , \nu _ { \operatorname* { m a x } } ]\tag{21}
$$

$$
\sum _ { n = 1 } ^ { N } \delta _ { n , k } \left[ t \right] = 1 \forall \mathbf { k } \in \mathcal { K }\tag{22}
$$

$$
\| \mathbf { \nabla } w _ { n } ( t ) \| ^ { 2 } \leq P _ { n } ^ { \operatorname* { m a x } } \quad \forall n \in \mathcal { N }
$$

$$
R _ { k _ { c } } \left( t \right) \geq R _ { \operatorname* { m i n } }\tag{23}
$$

(24)

$$
\Gamma _ { k _ { s } } \ge d ^ { 2 } ( u _ { n } - q _ { k _ { s } } ) \tilde { \Gamma } \quad \forall n \in \mathcal { N } , \forall k _ { s } \in \mathcal { K } _ { s }\tag{25}
$$

$$
\begin{array} { r l } { \Vert u _ { m } ( t ) - u _ { n } ( t ) \Vert \geq d _ { \operatorname* { m i n } } } & { { } \forall m , n \in \mathcal { N } } \end{array}\tag{26}
$$

$$
x ^ { l } \leq x _ { n } ( t ) \leq x ^ { u }\tag{27}
$$

$$
y ^ { l } \leq y _ { n } ( t ) \leq y ^ { u } \quad \forall n \in \mathcal N\tag{28}
$$

其中，优化目标（19）表示系统的能量效率，（20）、（21）分别表示无人机 的飞行方向与飞行速度限制，（22）是用户关联约束，其中 $\mathcal { K } = [ 1 , . . . , K ]$ ，确保一个地面用户只与一架无人机相关联，（23）表示分配给用户的传输功率不超过每架无人机的最大功率， 表示无人机， $\mathcal { N } = [ 1 , . . . , N ]$ ，（24）表示保证无人机与用户之间的通信速率满足最小阈值要求，（25）表示保证每个被感知的用户在 时隙满足最小感知功率阈值 $\tilde { \Gamma }$ ， $\mathcal { K } _ { s }$ 表示感知用户，

束和飞行范围约束。由于该优化问题是一个混合整数非线性规划（MINLP）问题，目标函数为分式形式，约束条件高度耦合且非凸，传统的优化方法难以有效解决此类问题，同时，由于波束成形矢量使得动作空间急剧膨胀，导致传统强化学习算法收敛性能较差，因此本文对强化学习算法进行改进来求解该问题。

## 2 优化问题求解

针对上述问题带来的挑战，本文提出针对该混合整数非线性规划问题求解的两步法。第一步，提出一种基于无人机服务容量的K-means自适应无人机与用户关联算法，对地面用户进行聚类确定无人机与用户之间的实时动态匹配关系；第二步，采用改进的多智能体双延迟深度确定性策略梯度

（Multi-Agent Twin Delayed Deep DeterministicPolicy Gradient，MATD3）算法，实现多无人机轨迹与波束成形的联合优化。最终实现系统能量效率最大化。

## 2.1 无人机与用户关联算法

在该场景下，用户进行实时漫游会导致其空间分布随时间不断变化，若无人机与用户之间的匹配关系长期保持不变，将难以适应信道状态和服务需求的实时变化，因此本节提出了一种基于无人机服务容量的 K-means 自适应用户关联算法，在提供通信和感知服务之前确定无人机所服务的地面用户。

K-means 是一种常见的聚类算法，在本文中，无人机与用户之间的匹配关系是后续确定系统性能的重要条件，然而传统 K-means[16]用户关联方法仅考虑基于距离来进行分配，未考虑到无人机有限的通信和感知服务容量（能力）问题，可能导致部分无人机通感任务负载过重，从而影响系统整体的通信与感知性能。因此本文在传统的 K-means 算法上引入了服务容量约束，通过显式地考虑无人机在通信与感知任务中的有限服务供给容量，自适应地进行用户匹配关联。具体而言，当某一无人机的通感任务负载超过一定的可服务容量时，相关用户将被重新分配至服务能力可行且距离次优的无人机，从而保证系统的服务可行性和通感任务负载均衡，以达到整体的性能最优。

对于服务区域内的用户 ，我们为其定义一个通信感知服务需求：

$$
d _ { \mathrm { k } } = { \left\{ \begin{array} { l l } { R _ { k } ^ { \mathrm { m i n } } , } & { i \in k _ { c } } \\ { \lambda G _ { k } ^ { \mathrm { m i n } } , } & { i \in k _ { s } } \end{array} \right. }\tag{29}
$$

其中 $R _ { k } ^ { \mathrm { { m i n } } }$ 为通信用户的最小速率需求， $G _ { k } ^ { \mathrm { m i n } }$ 为感知用户最小波束增益需求， 归一化系数。

对于每架无人机n，定义其最大服务容量为$C _ { n } ^ { \mathrm { m a x } }$ ，表示单架无人机在一个用户关联周期内可支持的最大综合通感任务负载，无人机当前负载即为$C _ { n } = \sum _ { k \in K } d _ { k }$ 当 $C _ { n } \leq C _ { n } ^ { \mathrm { m a x } }$ 时，当前关联满足服务可行性，当 $C _ { n } > C _ { n } ^ { \mathrm { m a x } }$ 时，说明无人机上可分配服务已超载，我们对与无人机 关联的用户，按照其与无人机的距离从远到近排序，依次选取距离较远的用户，为该用户寻找第二优且服务未超载的无人机，

$$
n ^ { \prime } = \arg \operatorname* { m i n } _ { { m \neq n } } \left\| q _ { k } - u _ { m } \right\|\tag{30}
$$

且满足：

$$
C _ { n ^ { \prime } } + d _ { k } \leq C _ { n ^ { \prime } } ^ { \mathrm { m a x } }\tag{31}
$$

通过这种方式将用户重新关联至新的无人机 $n ^ { \prime }$ ，并更新 $C _ { n }$ 与 $C _ { n } .$ ，直至所有无人机满足容量约束。

## 2.2 改进的 MATD3 强化学习算法

基于上一节中实现的无人机与用户之间的优化匹配关系，本节进一步提出了一种改进的多智能体双延迟深度确定性策略梯度算法实现多无人机轨迹与波束成形的联合优化问题。在求解该问题时，传统的优化方法难以同时考虑较多的耦合约束条件，而强化学习算法往往会遇到高维波束成形动作导致收敛缓慢甚至不稳定以及分式能量效率目标易引发过估计的问题，因此本文算法在 MATD3算法上进行了改进，通过设计“相位——功率”分离的动作表示方式，将原始高维复数波束成形向量解耦为两个低维实数变量，将原本在复数域进行的高维连续优化问题，转化为实数域内的搜索问题，显著降低了动作空间维度与策略学习难度。此外，针对奖励目标的设计问题，本文将瞬时系统能量效率作为主要奖励项，引入了多约束惩罚机制构建复合奖励函数，通过在策略更新过程中显式感知能效收益与约束条件之间的权衡关系，结合集中式训练分布式执行（centralized training with decentralizedexecution, CTDE）、双 Critic 网络与延迟更新机制，避免了处理分式规划时的迭代求解震荡问题，使得该算法在动态用户漫游场景下仍保持良好的收敛性能。

考虑将每架无人机视为一个单独的智能体，多无人机 ISAC 系统可建模为一个马尔可夫决策过程，用一个四元组 , , , 'S 表示。

## （1）状态空间 ：

对于第 架无人机，在时隙 的局部观测 $o _ { n } ( t )$ 定义为： $o _ { n } \left( t \right) = \left\{ q _ { n } \left( t \right) , h _ { n , k _ { c } } , h _ { n , k _ { s } } \right\} , \delta _ { n , k } \left( t \right) = 1$ ，包括无人机的位置信息、无人机与通信用户之间的通信信道增益、无人机与感知用户之间的感知信道增益。全局状态 $S ( t ) = \left\{ o _ { 1 } ( t ) , . . . , o _ { N } ( t ) \right\}$ 包含所有智能体的联合观测信息。

## （2）动作空间 ：

第 个智能体的动作向量 $a _ { n } ( t )$ 定义为：

$$
a _ { n } ( t ) = \left\{ \hat { \sigma } _ { n } ( t ) , \nu _ { n } ( t ) , \Phi _ { n } ( t ) , \mathbf { P } _ { n } ( t ) \right\}\tag{32}
$$

其中 $\Phi ( t ) \in [ 0 , 2 \pi ) ^ { M \times 1 }$ 表示波束成形的相位偏移矩阵， $\mathbf { P } _ { m } \left( t \right) \in [ 0 , P _ { \operatorname* { m a x } } ]$ 表示功率分配因子。通过这种降维方式，我们将实际执行的波束成形矢量 $w _ { n } \left( t \right)$ 重构为 $w _ { n } ( t ) = \sqrt { p _ { n } ( t ) } \cdot e ^ { j \phi _ { n } ( t ) }$ ，将复杂的复数域搜索转化为有界的实数域搜索，显著降低了学习难度。

## （3）奖励函数 ：

针对优化目标为分式且存在非凸约束的问题，将约束条件转化为惩罚项，并设计复合奖励函数。智能体 在 时刻的瞬时奖励 $r _ { n } \left( t \right)$ 设计如下：

$$
\begin{array} { r } { r _ { n } \left( t \right) = \alpha _ { 1 } \cdot \eta _ { E E } \left( t \right) - \alpha _ { 2 } \cdot \rho _ { \mathrm { s } } \left( t \right) - \alpha _ { 3 } \cdot \rho _ { c } \left( t \right) - \alpha _ { 4 } \cdot \rho _ { f } \left( t \right) } \end{array}\tag{33}
$$

其中， $\eta _ { E E } ( t )$ 为归一化后的瞬时系统能量效率，此外该式包含三个惩罚项部分， $\rho _ { \mathrm { s } } ( t ) \setminus \rho _ { c } ( t ) \setminus \rho _ { f } ( t )$ 分别为感知约束惩罚、通信约束惩罚以及防碰撞惩罚。 $\alpha _ { 1 } , \alpha _ { 2 } , \alpha _ { 3 } , \alpha _ { 4 }$ 均为权重系数，用于平衡各目标的重要性。定义 为折扣因子， $\gamma \in ( 0 , 1 )$ ，通过最大化长期累积奖励 $\sum \gamma ^ { t } r _ { n } ( t )$ ，智能体能够在动态环境中不断调整其动作策略，最终实现系统整体性能的最优化。

（4）更新的状态空间 $S ^ { \prime }$

在时隙 ，各无人机智能体基于当前状态采取联合动作 后，系统环境随之发生演化，进入下一时隙 ，对应的下一状态记为

MATD3 是一种改进的 Actor-Critic（AC）算法，主要用于解决连续动作空间中的非平稳多智能体问题。它具有分散式 Actor 网络、集中式 Critic 网络、经验回放缓存区和额外噪声以确保稳定高效的学习，其中每个智能体包括 Actor 模块和 Critic 模块。Actor 模块实现智能体与环境的交互，直接输出确定性动作；Critic 模块对 Actor 模块的策略进行评估并指导策略提升。MATD3 模型的训练与执行框架如图 2所示。

<!-- image-->  
图2 MATD3模型的训练与执行框架

为使智能体更容易表现出合作行为，在训练过程中，Critic 模块需要获取其他智能体的观测和动作信息，而在执行阶段，智能体仅根据自身的观测来选择动作。

为了解决分式目标函数极易导致的 Q 值过估计问题，每个智能体 的 Critic 网络有两个 值函数 $Q _ { i } ^ { 1 } ( s , a _ { 1 } , . . . a _ { N } )$ 和 $Q _ { i } ^ { 2 } ( s , a _ { 1 } , . . . a _ { N } )$ 。

在计算目标 值时，取两个 值函数的最小值，使用所有智能体的下一个动作 $a _ { j }$

$$
\begin{array} { r } { y _ { i } = r _ { i } + \gamma \operatorname* { m i n } _ { j = 1 , 2 } Q _ { * } ^ { j } ( s ^ { \prime } , a _ { 1 ^ { \prime } } , . . . , a _ { N ^ { \prime } } ) } \end{array}\tag{34}
$$

其中， $( Q _ { * _ { i } } ^ { 1 } , Q _ { * _ { i } } ^ { 2 } )$ 是目标网络， $a _ { j }$ '是通过目标 Actor网络 $\pi _ { j }$ 生成的动作。这种截断机制有效抑制了误差累积，使得针对能量效率的策略评估更加稳健。

Critic 网络通过最小化下式来学习价值函数：

$$
L ( \theta ^ { Q } ) = \mathbb { E } _ { ( s , a , s ^ { \prime } , r ) \sim D } \bigg [ \Big ( Q _ { \theta ^ { Q } } ( s , a _ { 1 } , \cdots , a _ { n } ) - \Big ( r + \gamma Q _ { \theta ^ { Q ^ { \mathcal { O } } } } ( s ^ { \prime } , \pi _ { i } ( s _ { 1 ^ { \prime } } ) , \cdots , \pi _ { i } ( s _ { n ^ { \prime } } ) ) \Big ) \Big ) ^ { 2 } \bigg ]\tag{35}
$$

其中， $\theta ^ { Q }$ 和 $\theta ^ { Q ^ { \prime } }$ 分别是 Critic 网络和目标 Critic 网络的参数， $\pi _ { i }$ 是智能体 采取的策略，D是经验池。

此外，每个智能体中采用分散的 Actor 网络用于根据本地观察做出行为。Actor 网络的更新也是通过最大化 Critic 网络的 值来进行的，Actor 策略的梯度可以通过最大化下式来更新其参数 $\theta ^ { \pi _ { i } }$

$$
\begin{array} { r } { \nabla _ { \theta ^ { \pi _ { i } } } J ( \theta ^ { \pi _ { i } } ) \approx \mathbb { E } _ { s \sim D , \epsilon \sim \mathcal { N } } \biggl [ \nabla _ { a } Q _ { \theta ^ { \sigma } } ( s , a _ { 1 } , \cdots , a _ { n } ) \Big | _ { a _ { i } = \pi _ { \theta ^ { \pi _ { i } } } ( s _ { i } ) + \epsilon } \nabla _ { \theta ^ { \pi _ { i } } } \pi _ { \theta ^ { \pi _ { i } } } ( s _ { i } ) \biggr ] } \end{array}\tag{36}
$$

其中， $\boldsymbol { J } ( \boldsymbol { \theta } ^ { \pi _ { i } } )$ 是智能体 的策略梯度。

考虑到波束成形对相位误差较为敏感，为了减少估计偏差，本文考虑在计算目标 值时，给每个智能体的动作 $a _ { i }$ 加入噪声 $a _ { i } = \pi _ { i } ( s _ { i } ) + \epsilon$ ，ε是为了增强智能体探索的策略噪声。这样可以避免策略过拟合某些特定的动作，从而提高探索性和稳定性。

为了进一步提高稳定性，该算法采用了延迟更新策略，Actor 网络的更新频率低于 Critic 网络：Critic 网络每更新两次，Actor 网络更新一次。此外，目 标 网 络 的 参 数 更 新 也 较 慢 ， 遵 循$\theta _ { i ^ { \prime } } \gets \mu \theta _ { i } + ( 1 - \mu ) \theta _ { i ^ { \prime } }$ ，其中 $\mu$ 是软更新的速率参数。

改进的 MATD3 强化学习算法的伪代码如下：

1：设定步数 $T _ { \mathrm { m a x } }$ ，批量大小 、经验缓冲池 $D :$   
初始化 Actor 网络 和双 Critic 网络 $\mu _ { \theta _ { n } }$ $\mu _ { \phi _ { n , 1 } } ~ \mu _ { \phi _ { n , 2 } }$ 及目标网  
络的网络参数  
2：for each episode do  
$^ { 3 } \mathfrak { s } \mathfrak { s }$ 初始化环境状态信息  
${ \mathbf { 4 } } ;$ 初始化无人机和用户位置信息；  
${ \mathfrak { s } } _ { \mathfrak { s } }$ for t from 1 to $T _ { \mathrm { m a x } }$ do  
${ \mathfrak { f } } { \mathfrak { z } }$ 计算 $h _ { n , k _ { c } } , h _ { n , k _ { s } }$ ·  
$\mathbf { 7 } { \ : } ;$ 每 个 无 人 机 根 据 局 部 观 测 选 择 动 作 ：  
$a _ { i } ( t ) = \mu _ { \theta _ { n } } \left( o _ { i } ( t ) \right) + \epsilon$   
${ \mathbf { 8 } } \colon$ 根据 $\mathbf { \delta } \Phi _ { n } , \mathbf { P } _ { n }$ 重构波束成形矢量 $\mathbf { w } _ { n } = \sqrt { p _ { n } } \cdot e ^ { j \phi }$   
${ \bf 9 } ;$ 执行动作，得到即时奖励 ( )r t 和新的状态 $S ^ { \prime }$   
$\mathbf { 1 0 } { \ : } $ 将 存入经验池  
11： 更新状态 $o _ { i }$   
12： for i from 1 to N do  
$\mathbf { 1 3 : }$ 从 中随机采样一小批数量为 的样本；  
14： 根据（33）计算损失函数，用于更新两个 Critic  
网络  
15： if t mod d = 0 then  
16： 根据（34）更新Actor网络；  
17： 根据 $\theta _ { i ^ { \prime } } \gets \mu \theta _ { i } + ( 1 - \mu ) \theta _ { i ^ { \prime } }$ 进行软更新目标参数  
18： end if  
19： end for  
20： end for  
21：end for

对于强化学习算法，计算复杂度通常从模型训练阶段和模型执行阶段两个方面进行分析。本文所提算法在训练阶段的总体计算复杂度可表示为$\mathcal { O } \biggl ( E \cdot \frac { D } { B } \cdot L \cdot N \cdot M \biggr )$ ，其中 表示训练轮数， 表示时间步数， 表示批量样本大小， 表示神经网络单层的计算复杂度， 表示网络层数， 表示智能体的数量。在执行阶段，该算法的复杂度为$\mathcal { O } \big ( ( O _ { M } ( m ) + O _ { A } ( m ) ) \cdot L \big )$ ，其中 $O _ { M } ( m )$ 为神经元数量为 的矩阵乘法计算复杂度， $O _ { A } ( m )$ 为激活函数计算的复杂度， 为网络层数。

## 3 仿真结果与分析

## 3.1 仿真参数设置

本节通过仿真实验评估本文提出的算法的性能，模拟一个 500m×500m 的方形服务区，我们规定当无人机的位置超出指定服务区时无人机将会被重置到服务区的边界点。具体仿真参数如表 2 所示。

表2 仿真参数设置
<table><tr><td>参数</td><td>参数值</td></tr><tr><td>时隙长度τ</td><td>0.1s</td></tr><tr><td>无人机最大速度  $\nu _ { \mathrm { m a x } }$ </td><td>5m/s</td></tr><tr><td>无人机的飞行高度H</td><td>150m</td></tr><tr><td>无人机的最大发射功率  $P _ { \mathrm { m a x } }$ </td><td>20dBm</td></tr><tr><td>无人机的阵列天线数量M</td><td>4</td></tr><tr><td>无人机数量N</td><td>3</td></tr><tr><td>通信用户数量  $K _ { c }$ </td><td>6</td></tr><tr><td>感知用户数量  $K _ { s }$ </td><td>3</td></tr><tr><td>用户移动的最大速度  $V _ { \mathrm { m a x } }$ </td><td>0.5m/s</td></tr><tr><td>带宽B</td><td>15MHz</td></tr><tr><td>载频  $f _ { c }$ </td><td>2GHz</td></tr><tr><td>AWGN功率  $\sigma$ </td><td>-110mdBs</td></tr><tr><td> $\tilde { \Gamma }$  感知波束增益阈值</td><td>-60dBm</td></tr><tr><td>最小通信速率  $R _ { \mathrm { m i n } }$ </td><td>5Mbps</td></tr><tr><td>参考信道增益  $\alpha , \beta$ </td><td>-60dBm</td></tr><tr><td>经验缓存池D</td><td>36000</td></tr><tr><td>批样样本S</td><td>54</td></tr><tr><td>学习率α</td><td>1e-4</td></tr><tr><td>折扣因子γ</td><td>0.96</td></tr></table>

## 3.2 仿真结果展示

图3展示了四种不同算法的累计奖励的性能比较，分别为本文改进的 MATD3 算法、普通的MATD3 算法、MADDPG 算法以及地面 ISAC 算法，其中地面 ISAC 场景中使用三个配备了四个天线的基站代替无人机，在该场景下将消耗能量设定为固定值。可以看到与普通的 MATD3 算法和 MADDPG算法相比，本文提出的改进 MATD3 算法在训练初期虽然同样存在一定震荡，但其震荡幅度更小，且收敛速度更快，并且算法的效果最为显著，在训练轮数达到 200 之后始终保持最高的奖励水平，这主要是由于我们对动作空间的改进显著降低了波束成形搜索空间的难度，探索噪声对策略的扰动被有效抑制，从而提升了学习过程的稳定性。仿真结果表明本文算法能够在满足通信与感知用户需求的同时获得最大的能量效率，其次是 MATD3 算法，最后是 MADDPG 算法。此外，地面 ISAC 基准方案由于缺乏智能优化的资源分配能力以及受限于地面信道的路径损耗，其性能始终维持在较低的水平。

<!-- image-->  
图3 四种不同算法的累计奖励的性能比较

图 4 显示了第 1000 轮时用户聚类的奖励。可以看到，经过第 150 个时隙时本文所提的关联算法显示出明显的优势，这是因为用户位置的移动，原有的用户关联失效，无人机无法为用户提供正常服务，这将会触发奖励函数中的惩罚项，导致算法获得的奖励不断下降，这也表明了对用户进行重关联任务的重要性。而对于进行重关联的用户，能量效率在每个周期内呈现出先增加后降低的趋势，这是由于用户位置随时间不断变化，初始关联结果不再是最优的。

<!-- image-->  
图4 用户关联算法的性能比较

图5展示了系统的通信和速率与无人机数量的变化关系。总体而言，随着无人机数量从 3 架增加至 6 架，所有方案的系统和速率均呈现出单调递增的态势，这验证了增加空中接入点能够通过空间复用增益显著提升系统容量。在所有对比方案中，所提改进的 MATD3 算法在不同无人机密度下始终保持着最高的和速率性能。这说明本文采用的集中式训练分布式执行（CTDE）框架能够有效促进多无人机间的协作。相反，地面 ISAC 方案由于固定的部位位置和有限的视距链路概率，无法充分利用空间自由度，因此在提升系统容量方面表现最差。

<!-- image-->  
图5无人机数量与系统的通信和速率的变化关系

图6进一步评估了用户最大移动速度对系统能量效率的影响，以验证所提算法在动态环境下的鲁棒性。仿真结果表明，随着用户移动速度的增加，所有算法的系统能效均出现不同程度的下降，这是因为用户位置的快速变化导致信道状态信息（CSI）过时，进而降低了波束成形的准确性。即便在用户以最大速度移动的高动态场景下，文章所提算法依然能够维持较高的能效水平。这种优越的适应性得益于本文提出的基于资源感知的K-means自适应用户关联机制，使得系统能够动态地更新用户关联并实时调整无人机轨迹以跟随移动用户。相比之下，传统的静态关联策略无法及时适应拓扑的剧烈变化，导致通信质量与感知增益急剧恶化，从而引起系统能效的显著衰减。

<!-- image-->  
图6 用户最大移动速度与系统能量效率的关系

图7展示了系统归一化的能量效率与无人机数量的变化关系。从图上可以看出，随着无人机数量从 3 架增加至 7 架，所有方案的系统和速率均呈现出递增的态势，这是因为随着无人机数量的增加，更多的无人机为更多地面用户提供定位辅助和通信服务，无人机可以更好地进行探索，相应的系统能量效率也逐渐增加。其中所提 MATD3 算法始终优于其它算法，因为该算法能够在保证通感性能的同时，通过优化无人机飞行轨迹与波束成形性能，达到了能量效率的最大化。

<!-- image-->  
图7 无人机数量与系统能量效率的关系

## 4 结束语

本文研究了基于改进 MATD3 强化学习算法的多无人机通感一体化系统能量效率优化问题。首先，构建了多无人机通感系统模型和最大化系统能量效率联合优化问题；其次，设计了基于无人机服务容量的 K-means 无人机与用户关联算法，同时提出了改进的 MATD3 算法，最终实现多无人机轨迹与波束成形的协同优化。仿真结果表明，所提方法在动态用户场景下能够有效提升系统通信能量效率，同时能够同步优化无人机轨迹、服务覆盖范围及波束成形性能。

## 参考文献：

[1] CUI Y H, LIU F, JING X J, et al. Integrating sensing and communications for ubiquitous IoT: applications, trends, and challenges[J]. IEEE Network, 2021, 35(5): 158-167.

[2] 宋雨欣，张晖，汤萍萍，等. 6G 车联网场景下基于通感 一体的波束赋形算法 [J]. 南京邮电大学学报 (自然科 学版), 2025, 45 (6): 32-42. SONG Yuxin, ZHANG Hui, TANG Pingping, et al. Beamforming algorithm based on integrated sensing and communication in 6G vehicle networking scenarios [J]. Journal of Nanjing University of Posts and Telecommunications (Natural Science), 2025, 45 (6): 32-42. (in Chinese)

[3] DENG C L, FANG X M, WANG X B. Beamforming design and trajectory optimization for UAV-empowered adaptable integrated sensing and communication[J]. IEEE Transactions on Wireless Communications, 2023, 22(11): 8512-8526.

[4] MENG K T, WU Q Q, MA S D, et al. Throughput maximization for UAV-enabled integrated periodic sensing and communication[J]. IEEE Transactions on Wireless Communications, 2023, 22(1): 671-687.

[5] LIU Y M, LIU X, LIU Z C, et al. Secure rate maximization for ISAC-UAV assisted communication amidst multiple

eavesdroppers[J]. IEEE Transactions on Vehicular Technology, 2024, 73(10): 15843-15847.

[6] YANG W Y, LIU X, LIU Z C. Joint trajectory and resource allocation optimization for mobile vehicles in UAV assisted secure ISAC system[C]// 2024 IEEE 24th International Conference on Communication Technology (ICCT), Chengdu, China, 2024: 321-325.

[7] REZAEI O, NAGHSH M N, KARBASI S M, et al. Multi-UAV enabled integrated sensing and wireless powered communication: a robust multi-objective approach[J]. IEEE Transactions on Vehicular Technology, DOI: 10.1109/TVT.2025.3641304.

[8] DWIPUTRO D W, HENDRAWAN. Machine learning approach for integrated sensing and communication in 6G radio standard: a systematic literature review[C]// 2025 19th International Conference on Telecommunication Systems, Services, and Applications (TSSA), Yogyakarta, Indonesia, 2025: 1-6.

[9] AMHAZ A, ELHATTAB M, ASSI C, et al. UAV-assisted NOMA for enhanced ISAC performance using deep deterministic policy gradient[C]// GLOBECOM 2024 - 2024 IEEE Global Communications Conference, Cape Town, South Africa, 2024: 67-72.

[10] XU Y T, WANG J C, ZHANG R C, et al. Mixture of experts for decentralized generative AI and reinforcement learning in wireless networks: a comprehensive survey[J]. IEEE Communications Surveys & Tutorials, 2026, 28: 4051-4085.

[11] WANG Z W, ZHAO J W, HE F, et al. A hierarchical MADRL framework for joint resource allocation and trajectory design in UAV enabled multi-static ISAC networks[J]. IEEE Wireless Communications Letters, 2026, 15: 1325-1329.

[12] ZHANG T, ZHU K, ZHENG S Q, et al. Trajectory design and power control for joint radar and communication enabled multi-UAV cooperative detection systems[J]. IEEE Transactions on Communications, 2023, 71(1): 158-172.

[13] GAO Q, ZHONG R K, SHIN H, et al. MARL-based UAV trajectory and beamforming optimization for ISAC system[J]. IEEE Internet of Things Journal, 2024, 11(24): 40492-40505.

[14] QI X H, YUAN M X, ZHANG Q Y, et al. Joint power-trajectory-scheduling optimization in a mobile UAV-enabled network via alternating iteration[J]. China Communications, 2022, 19(1): 136-152.

[15] ZENG Y, XU J, ZHANG R. Energy minimization for wireless communication with rotary-wing UAV[J]. IEEE Transactions on Wireless Communications, 2019, 18(4): 2329-2345.

[16] LIU F H, CHEN H Q, MIAO J S, et al. Energy efficiency optimization for UAV-assisted cellular networks: a periodic clustering-based MATD3 approach[C]// 2024 IEEE Global Communications Conference (GLOBECOM), Cape Town, South Africa, 2024: 229-234.