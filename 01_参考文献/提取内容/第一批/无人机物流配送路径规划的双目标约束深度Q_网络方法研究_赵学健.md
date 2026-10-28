<!-- image-->  
小型微型计算机系统Journal of Chinese Computer SystemsISSN 1000-1220,CN 21-1106/TP

# 《小型微型计算机系统》网络首发论文

题目： 无人机物流配送路径规划的双目标约束深度 Q 网络方法研究

作者： 赵学健，夏郡，王恩良

DOI： 10.20009/j.cnki.21-1106/TP.2026-0101

收稿日期： 2026-03-04

网络首发日期： 2026-05-20

引用格式： 赵学健，夏郡，王恩良．无人机物流配送路径规划的双目标约束深度 Q 网络方法研究[J/OL]．小型微型计算机系统.https://doi.org/10.20009/j.cnki.21-1106/TP.2026-0101

<!-- image-->

<!-- image-->

网络首发：在编辑部工作流程中，稿件从录用到出版要经历录用定稿、排版定稿、整期汇编定稿等阶段。录用定稿指内容已经确定，且通过同行评议、主编终审同意刊用的稿件。排版定稿指录用定稿按照期刊特定版式（包括网络呈现版式）排版后的稿件，可暂不确定出版年、卷、期和页码。整期汇编定稿指出版年、卷、期、页码均已确定的印刷或数字出版的整期汇编稿件。录用定稿网络首发稿件内容必须符合《出版管理条例》和《期刊出版管理规定》的有关规定；学术研究成果具有创新性、科学性和先进性，符合编辑部对刊文的录用要求，不存在学术不端行为及其他侵权行为；稿件内容应基本符合国家有关书刊编辑、出版的技术标准，正确使用和统一规范语言文字、符号、数字、外文字母、法定计量单位及地图标注等。为确保录用定稿网络首发的严肃性，录用定稿一经发布，不得修改论文题目、作者、机构名称和学术内容，只可基于编辑规范进行少量文字的修改。

出版确认：纸质期刊编辑部通过与《中国学术期刊（光盘版）》电子杂志社有限公司签约，在《中国学术期刊（网络版）》出版传播平台上创办与纸质期刊内容一致的网络版，以单篇或整期出版形式，在印刷出版之前刊发论文的录用定稿、排版定稿、整期汇编定稿。因为《中国学术期刊（网络版）》是国家新闻出版广电总局批准的网络连续型出版物（ISSN 2096-4188，CN 11-6037/Z），所以签约期刊的网络版上网络首发论文视为正式出版。

DOI：10.20009/j.cnki.21-1106/TP.2026-0101

# 无人机物流配送路径规划的双目标约束深度 Q 网络方法研究

赵学健 1,2，夏 郡 1，王恩良 2

1（南京邮电大学 现代邮政学院，南京 210003)

2（南京邮电大学 宽带无线通信与传感网络技术教育部重点实验室，南京 210003）

E-mail：zhaoxj@njupt.edu.cn

摘 要： 低空经济与城市即时配送需求的增长，展现出多无人机物流配送在提升末端效率方面具有显著潜力.然而在障碍密集、禁飞区域频发且存在能量补给约束的三维城市空域中，多无人机路径规划与动态避障同时面临高维离散决策、稀疏回报、价值估计不稳定以及强安全约束难以全过程保证等挑战.为此，本文面向体素化三维城市环境构建多无人机取送任务的序列决策模型，提出一种双目标约束深度Q网络（DTC-DQN）.该方法在Double DQN基础上：引入双目标网络下界估计以抑制 Q 值高估并提升收敛稳定性；设计面向多智能体轨迹一致性的 n 步回报缓冲以加速回报传播并避免时间片段混入；融合 Dueling 架构与优先级经验回放以增强特征表达与样本利用效率；并将动作掩码与安全屏蔽机制嵌入决策与目标计算全过程，从源头限制动作选择于可行集合，保障全过程安全与任务可达.实验在不同规模三维栅格实例上开展对比，结果表明：相较 DQN、Double DQN 与 Dueling DQN 等基线，所提方法在收敛速度、策略稳定性与任务完成成功率方面均表现更优，并在时延与能量损耗指标上取得更好的综合权衡.研究结果为复杂约束下的多无人机城市物流路径规划提供了可扩展的学习式求解框架.

关键词：无人机系统；多无人机协同；三维路径规划；深度强化学习；深度Q网络

中图分类号：TP301

文献标识码：A

# Research on Dual-target Constrained Deep Q-network Method for UAV Logistics Distribution Path Planning

ZHAO Xuejian 1,2，XIA Jun 1，WANG Enliang 2

1（School of Modern Posts,Nanjing University of Posts and Telecommunications，Nanjing 210003，China）

2 (Key Lab of Broadband Wireless Communication and Sensor Network Technology,Ministry of Education,Nanjing University of Posts and Telecommunications， Nanjing 210003，China）

Abstract: Driven by the booming low-altitude economy and urban instant delivery demand,multi-Unmanned Aerial Vehicle (multi-UAV) logistics has great potential for boosting last-mile delivery efficiency.However,in 3D urban airspace with dense obstacles,frequent no-fly zones and energy constraints,multi-UAV path planning and dynamic obstacle avoidance face critical challenges:high-dimensional discrete decision-making,sparse rewards,unstable value estimation,and difficulty in ensuring full-process safety constraints.To address these issues,this paper constructs a sequential decision-making model for multi-UAV pickup and delivery tasks in voxelized 3D urban environments,and proposes a Dual-Target Constrained Deep Q-Network (DTC-DQN).Built on Double Q-learning,this method integrates dual target networks for lower-bound estimation to mitigate Q-value overestimation,a multi-agent trajectory-consistent n-step return buffer,Dueling architecture with prioritized experience replay,and full-process action masking and safety shielding to guarantee feasible actions and flight safety.Comparative experiments on 3D grid instances of varying scales show that the proposed method outperforms classic baselines including DQN,Double DQN and Dueling DQN in convergence speed,policy stability and task success rate,and achieves a better comprehensive trade-off between delay and energy consumption.This work provides a scalable learning-based framework for multi-UAV urban logistics path planning under complex constraints.

Keywords:UAV systems;multi-UAV coordination;3D path planning;deep reinforcement learning;DQN

## 0 引 言

随着全球电子商务的迅猛发展与城市化进程的持续深化，城市物流配送体系正面临日益严峻的压力.传统地面配送模式受限于城市交通拥堵、高昂人力成本及环境污染等问题，其效率与成本瓶颈愈发凸显.无人机物流凭借其高机动性、对地面交通的天然规避能力以及高度自动化潜力，有望显著提升配送效率、降低运营成本，并推动未来智慧城市交通体系的深刻变革[1].然而，单一无人机难以同时兼顾远距离飞行与多订单承载，仅能基于局部感知进行决策，因此，将无人机配送从概念验证推进到城市级应用，核心挑战集中体现在复杂三维城市环境下的多无人机协同路径规划与动态避障.

目前，多无人机路径规划方法可以分为传统算法和基于智能优化算法的方法两大类[2].传统算法基于数学证明或逻辑过程以获得严格最优或高效可行结果的确定性解，但其计算复杂度随问题规模呈指数级增长，难以满足应急物流中的实时决策需求[3].基于智能优化理论的智能优化算法，能够在复杂的环境约束和任务需求下，有效处理多无人机路径规划中的高维优化问题[4].其中，启发式与元启发式算法在求解大规模车辆路径问题上展现出一定潜力，但这些方法通常依赖于固定的启发式规则，对环境动态变化的适应性较差，且难以在多个目标之间实现自适应权衡.相比之下，深度强化学习为解决此类高维、动态、多目标的序列决策问题提供了新范式[5].通过与环境的试错交互，深度强化学习智能体能够自主学习从环境观测到最优动作的映射策略，无需预先构建精确的环境模型[6].这种数据驱动的学习能力使其天然适于处理随机扰动与动态变化，并可通过精心设计的奖励函数隐式学习多目标间的复杂平衡[7].

深度Q网络(DQN)作为基于智能优化算法的方法之一，其利用神经网络从高维输入中评估离散动作价值的核心机制与无人机依据传感器获取的局部环境信息选择飞行指令高度契合，通过将任务进度、能耗、安全与时效等多目标量化并融合为统一的奖励信号，能够学习到一个内在地优化长期累积回报的决策策略，实现比硬编码规则更灵活的自适应平衡[8].此外，结合参数共享的集中训练分散执行范式，所有无人机可以共享同一策略网络，而执行时仅依赖自身局部观测，显著提升了样本效率并确保了系统的可扩展性[9].然而，原始DQN及其基础变体在应对复杂物流配送任务时仍存在明显局限：标准DQN架构受Q值过高估计问题困扰，导致策略学习过程振荡且易收敛至次优解；在广阔且障碍密布的三维空间中，稀疏的任务完成信号与低效的探索行为会浪费大量训练样本，严重延缓学习进程；传统强化学习通过惩罚机制“事后”学习避障约束，无法在训练与执行全过程中保证绝对的飞行安全与操作可行性[10].

针对上述挑战，本文提出一种双目标约束深度 Q网络：Dual-Target Constrained Deep Q-Network（DTC-DQN），主要创新点如下：

（1）提出基于双目标网络下界的稳定价值评估机制.引入两套独立更新的目标网络，在计算时序差分目标时取其Q值最小值，有效抑制Q值高估，提升收敛质量.

（2）将硬约束掩码与安全屏蔽机制深度嵌入 DQN 全流程.通过“先验+屏蔽”的约束嵌入方案，在每一决策点利用动作掩码将选择限制于物理与规则允许的动作子集，并引入安全屏蔽机制，在特定状态下动态修改动作空间以强制引导安全行为，从而保障任务可行性与过程安全性.

（3）设计面向多智能体轨迹一致性的N步回报缓冲策略.为提升样本效率，为每架无人机维护独立的 N 步经验缓冲区，确保计算 N 步回报时经验片段来源于同一智能体的连续轨迹，避免多智能体并行采样导致的时间维度污染，保证估计无偏性.

（4）构建融合多种技术的协同学习架构.将 Dueling 网络、优先级经验回放、双目标网络与 N 步回报有机整合，形成高效、稳定的端到端学习闭环.

## 1 相关工作

无人机路径规划的理论基础可追溯至移动机器人路径规划与网络覆盖优化领域.传统主流方法中，针对物流取送任务，Deng 等人将无线电地图与环境地图相结合以刻画空间通信质量分布，在改进的 Dijkstra 搜索框架下构建可搜索空间，并通过混合遗传算法求解能耗最小的配送策略 [11].围绕三维离散空间规划，Yoo等人针对障碍物密集导致搜索效率下降与路径质量波动的问题，将“障碍物密度”引入三维航迹优化过程，从而增强算法对复杂障碍分布的适应性[12].此类方法在可解释性与可复现性方面具有优势，但其性能往往高度依赖离散化精度与先验地图质量，在三维大尺度空间中易出现状态空间膨胀，导致在线计算开销显著上升.

在连续空间的实时避障与多机协同控制方面，基于势场的反应式规划因计算代价低、易于在线实现而被广泛采用.Lu等人提出融合LSTM、DQN与人工势场法的路径规划方法，并结合“孪生”思路面向复杂场景开展建模与验证，试图以学习式策略增强对环境变化的响应能力 [13].面向多无人机编队任务，Pan等人在势场构造中引入线吸引势与角排斥势，使无人机在保持期望编队间距的同时实现障碍规避[14].除局部避障外，研究也逐步将路径规划与群体层任务决策进行耦合建模：Guo等人提出“无人机群路径规划—任务分配”联动的组合优化框架，将路径规划输出作为任务分配输入以实现两者关联[15]；S.-W.Cho 等人面向海上搜救的多无人机覆盖路径规划任务，采用六边形网格分解并构建MILP 模型以刻画覆盖约束，实现覆盖任务的结构化表达[16]；为缓解 MILP 在规模扩展条件下的计算瓶颈，Hajkarim等人引入MARPO算法，在小规模算例中保持与 MILP接近的精度的同时，在更大规模下提升可扩展性与计算效率[17].上述模型驱动与势场类方法对环境先验与模型设定依赖较强，势场方法亦可能受局部极值影响产生振荡或不可达现象，在动态障碍与临时禁飞等非平稳因素频发时往往需要高频重规划，因而难以稳定支撑多无人机的实时协同运行.

为增强对复杂约束与高维搜索空间的适应能力，启发式与元启发式算法逐渐成为研究重点.Wen 等人为降低多无人机三维协同曲线路径规划的计算复杂度并提升收敛效率，提出基于混沌精英自适应遗传算法的三维协同曲线路径规划方法，通过改进初始化与遗传算子提升搜索效率与鲁棒性[18]；Wu等人面向动态环境下的无人机航迹规划，提出基于改进约束差分进化的动态路径规划方法，以支持路径的在线更新与约束处理[19].尽管进化类方法具有先验依赖低、全局搜索能力强、易于处理多目标权衡等优点，但其迭代式求解通常伴随较高计算开销且对参数设置敏感，在强动态环境中若缺乏增量更新与快速可行性修复机制，仍难稳定满足多无人机在线协同规划的实时性与安全性要求.

近年来，机器学习尤其是深度强化学习被用于建立从局部观测到控制动作的端到端映射，以提升在动态与不确定环境中的在线适应性与决策效率，并逐步从单机规划拓展到编队协同与群体任务.Qian 等人面向复杂环境下异构无人机编队实时轨迹规划，设计 CNN-LSTM 轨迹预测模型并提出在线规划方法，通过预训练的预测网络替代传统物理模型进行轨迹预测，结合MPC思想与多目标粒子群优化对规划代价进行评估与优化，使生成轨迹更贴近可跟踪性需求[20]；Huang等人则将群智能优化与深度学习结合，通过数据驱动模型与群体搜索机制互补提升复杂环境适应性[21].在多无人机协同任务层面，Liu等人提出基于动作掩码的多智能体近端策略优化算法，用于三维环境下多无人机协同搜索移动目标，通过高—低空协同架构、视场编码及动作掩码避碰机制提升收敛效率并增强协同安全性[22]；Dong 等人面向多障碍环境提出深度强化学习路径规划方法，以端到端决策实现避障与到达目标[23]；Gao等人提出相关经验学习以提升复杂未知环境中的经验利用效率[24]；Demir等人面向野火侦察等大尺度灾害场景提出基于深度强化学习的多无人机轨迹规划方法，在任务建模层面通过区域划分将轨迹规划转化为对子区域的周期性访问任务，以适配能量约束与连通性等限制[25].学习式方法在环境适应性与在线推理速度方面展现出潜力，但在价值估计稳定性、样本利用效率以及强约束场景下的安全可行性保障方面仍存在共性瓶颈.

综上，现有研究难以同时兼顾多无人机物流配送所要求的实时协同、能量约束下的长期效益以及全过程可验证安全.尤其在城市三维空间中，动态禁飞区、空中冲突规避与续航补能策略使可行解空间呈现强耦合与强非平稳特征，既需要稳定高效的价值评估与样本利用机制，也需要在决策层面嵌入硬约束以保证飞行安全与任务可行性.基于上述研究缺口，本文进一步在 Double DQN 框架内引入更稳健的价值估计、面向多智能体轨迹一致性的回报建模以及动作级安全约束嵌入机制，以期为复杂物流场景提供可扩展且可落地的多无人机路径规划方案.

## 2 DTC-DQN 解决无人机路径规划与动态避障问题

## 2.1 问题分析

在低空三维城市物流配送场景中，多无人机路径规划与动态避障具有典型的高维、强约束与强耦合特征.一方面，体素化三维空域使动作空间呈离散组合爆炸，障碍物与禁飞区导致可行解空间高度非凸；多无人机并行执行取送任务使路径规划与冲突规避强耦合，且受电量约束与充电桩补给决策影响，长期收益需要跨时间尺度优化.此外，城市配送任务的关键奖励（取件/投递成功）具有明显稀疏性，若缺乏有效的回报传播机制与样本利用策略，将导致训练效率低下并表现出收敛缓慢.基于价值函数的深度强化学习方法在处理离散动作与局部观测方面具有优势，但经典DQN仍易受Q值过高估计与目标漂移影响，训练过程易出现振荡并收敛至次优策略；仅依赖惩罚项的“软约束学习”难以保证全过程安全与可行动作的可达性.综上，本文在 Double DQN 框架内进一步引入更稳健的目标估计、面向轨迹一致性的 n步回报、优先级经验回放与 Dueling 结构，并将动作掩码与安全屏蔽作为硬约束嵌入决策与学习过程，以在样本效率、收敛稳定性与安全可行性之间实现统一.

## 2.2 场景与任务建模

本文为低空三维城市环境中的多无人机物流配送问题.将空域离散为尺寸为 $G _ { _ x } \times G _ { _ y } \times G _ { _ z }$ 的体素栅格，建筑体及不可飞越结构构成静态障碍体素集合 ，由政府或场域管理规则产生的禁飞体素构成集合（位于地面或低层），无人机可在充电桩上方悬停充电，使得路径规划同时受空间可行性与能量可持续性约束.问题模型如图1所示：

<!-- image-->  
图1 低空 3D 物流配送场景问题模型  
Fig 1 Low-altitude 3D logistics distribution scenario problem model

系统包含 架同构无人机并采用参数共享策略，任务集合 $T = \left\{ 1 , . . . , N \right\}$ 中每个任务 由取件点与投递点二元组$( p _ { i } , d _ { i } )$ 与软截止时间 $L _ { i }$ 构成；当任务完成时刻超过 $L _ { i }$ 时施加软延误惩罚.为贴近低空实际，环境在执行动作后以小概率引入水平风扰动，使位置转移具有随机性；，同时电量按单位动作消耗进行累积更新，用于刻画续航与补能决策.

回合按离散时间步 $t = 0 , 1 , \ldots$ 推进.无人机 在时刻 的空间位置为 $\boldsymbol { x } _ { t } ^ { k } = ( x , y , z )$ ，电量为 ${ b } _ { t } ^ { k }$ ，其局部感知为以自身为中心、半径为 $( r _ { x } , r _ { y } , r _ { z } )$ 的三维邻域占据体素矩阵，并融合当前子目标（取件或投递）相对方向的归一化向量以及电量、截止时间等标量特征.该建模以局部密集观测支持端到端策略学习，同时通过动作掩码和“安全屏蔽”嵌入硬约束，确保路径规划与避障的可行性.

在训练范式上采用集中训练、分散执行与参数共享：训练阶段汇聚所有无人机交互样本更新共享网络参数，执行阶段每架无人机仅基于自身局部观测独立选动作.由于单机仅见局部观测，严格意义上问题具有部分可观测性；实现中将“局部观测向量”作为状态输入，得到一组同构的近似MDP，用于 DQN 系列离散动作价值学习.

对任一无人机 ，其状态向量由局部占据体素展平的二值向量 $o _ { t } ^ { k }$ 与自身和目标特征向量 $f _ { t } ^ { k }$ 拼接得到：

$$
s _ { t } ^ { k } = [ o _ { t } ^ { k } ; u _ { t } ^ { k } ]\tag{1}
$$

其中 $o _ { t } ^ { k }$ 为局部邻域占据体素块的二值展平表示， ${ \boldsymbol u } _ { t } ^ { k }$ 包含当前子目标方向 ${ \boldsymbol g } _ { t } ^ { \kappa }$ 、归一化电量 ${ b } _ { t } ^ { k }$ 及与截止时间相关的量，从而使网络在不访问全局地图的前提下具备朝向目标推进、兼顾续航与时效的决策依据.

动作空间采用 27 个三维离散原语，可表示为：

$$
a _ { t } ^ { k } = [ \Delta x , \Delta y , \Delta z ] , \Delta x , \Delta y , \Delta z \in \{ - 1 , 0 , 1 \}\tag{2}
$$

其中(0,0,0)为悬停，其余 26 个动作覆盖相邻体素的面、棱、角方向移动.为保证硬可行性，实际可选动作限制为可行集合 ${ A _ { t } } ^ { k } \subseteq A$ ：若动作导致越界、进入 $O \cup F$ ，或电量不足以支付动作能耗，则该动作被实时剔除.实现上以动作掩码$m _ { t } ^ { k } ( a ) \in \{ 0 , 1 \}$ 表示可行性，并在动作选择与目标计算时中统一以 $A _ { t } ^ { \ k } = \{ a \vert m _ { t } ^ { k } ( a ) = 1 \}$ 约束计算与评估过程，从源头避免在不可行动作上浪费探索.

奖励设计同时覆盖任务推进、能耗、安全与时效，并通过稠密项提升学习可用信号.单步奖励写为:

$$
\begin{array} { l } { { { r _ { t } ^ { k } } { = } { \lambda _ { d } } { \Delta { D _ { t } } ^ { k } } { - } { \lambda _ { e } } } E ( { a _ { t } ^ { k } } ) - { \lambda _ { b } } I [ { b _ { t + 1 } } ^ { k } { < } { b _ { t h } } ] - { \lambda _ { \nu } } I [ \nu i o l a t e _ { t } ^ { k } ] }  \\ { { { - } { \lambda _ { n } } I [ n e a r _ { t } ^ { k } ] + R _ { p } I [ p i c k u p ] + R _ { d } I [ d e l i \nu e r ] } } \end{array}\tag{3}
$$

其中 为指示函数，条件成立取 1，否则取 $0 . \Delta D _ { t } ^ { k }$ 表示相对当前子目标的距离缩短量.为使三维栅格中不同方向具有一致度量，距离采用切比雪夫范数： $D _ { \infty } ( x , g ) = \left\| x - g \right\|$ 并定义 $\Delta D _ { t } ^ { k } = D _ { \infty } ( x _ { t } ^ { k } , g _ { t } ^ { k } ) - D _ { \infty } ( x _ { t + 1 } ^ { k } , g _ { t } ^ { k } )$ ，其中 ${ g } _ { t } ^ { k }$ 为当前子目标点. $E ( a _ { t } ^ { k } )$ 是平移、爬升与悬停的能耗线性组合近似；为硬违规事件； $n e a r _ { t } ^ { k }$ 为近失事件.为抑制拥挤场景下近失惩罚的重复叠加导致回报尺度失真，实现中采用“每无人机每步至多计一次近失惩罚”的上限约束.超参数取值与代码一致： $\alpha = 1 . 0 , \eta = 0 . 0 1 , \rho = 0 . 0 1$ .文中给出的关键系数示例为：低电量惩罚系数与违规惩罚分别为 0.05 与 3.0，取件与投递奖励分别为10与30.

## 2.3 Double DQN 改进策略

为提升价值估计稳定性与样本效率，本文在 DoubleDQN 框架上集成 Dueling 结构、优先级经验回放（PER）、n 步回报，以及双目标网络下界目标，并将动作掩码/安全屏蔽贯穿于动作选择与目标计算.在线网络 $Q _ { \theta } ( s , a )$ 采用两层全连接骨干并引入 Dueling 分解，以降低在多动作空间中因优势项漂移导致的训练不稳定：

$$
Q _ { \theta } ( s , a ) = V _ { \theta } ( s ) + A _ { \theta } ( s , a ) - \frac { 1 } { \left| \mathrm { \bf A } \right| } \sum _ { a } A _ { \theta } ( s , a ^ { ' } )\tag{4}
$$

在目标构造上，首先在可行动作子集上用在线网络选取下一步贪婪动作：

$$
a _ { t + 1 } ^ { \mathrm { a } } = \arg \operatorname* { m a x } _ { a \in \mathbf { A } _ { t + 1 } } Q _ { \theta } ( s _ { t + 1 } , a )\tag{5}
$$

随后由两套软更新的目标网络 $Q _ { \theta _ { 1 } ^ { - } } , Q _ { \theta _ { 2 } ^ { - } }$ 评估并取下界构造TD目标：

$$
y _ { t } = r _ { t } + \gamma ( 1 - \mathrm { d o n e } _ { t + 1 } ) \operatorname* { m i n } \Bigl ( Q _ { _ { \theta _ { 1 } ^ { - } } } ( s _ { t + 1 } , a _ { t + 1 } ^ { * } ) , Q _ { _ { \theta _ { 2 } ^ { - } } } ( s _ { t + 1 } , a _ { t + 1 } ^ { * } ) \Bigr )\tag{6}
$$

其中 为折扣因子， 为终止标记.该“选择—评估解耦+双目标取最小值”的组合同时抑制Q值过高估计与目标漂移.

经验回放采用PER，以TD误差 $\delta _ { i } = y _ { i } - Q _ { \theta } ( s _ { i } , a _ { i } )$ 定义采样概率与重要性采样权重：

$$
P ( i ) = \frac { ( \mid \delta i \mid \varepsilon ) \alpha } { \sum _ { j } ( \left| \delta j \right| + \varepsilon ) ^ { \alpha } } , w _ { i } = \frac { ( N \cdot P ( i ) ) ^ { - \beta } } { m a x ( N \cdot P ( j ) ) ^ { - \beta } }\tag{7}
$$

并使用加权均方误差最小化：

$$
L ( \theta ) = \frac { 1 } { B } \sum _ { i = 1 } ^ { B } w _ { i } ( Q _ { \theta } ( s _ { i } , a _ { i } ) - y _ { i } ) ^ { 2 }\tag{8}
$$

其中 为批量大小、 为回放池容量尺度， $\alpha$ 控制优先级强度， $\beta$ 用于渐进校正采样偏差.目标网络采用软更新以稳定训练：

$$
\theta _ { j } ^ { - }  \tau \theta + ( 1 - \tau ) \theta _ { j } ^ { - } , j \in \{ 1 , 2 \}\tag{9}
$$

引入n步回报以加速稀疏任务信号下的回报传播.对每架无人机维护独立的 n 步缓冲，确保 n 步序列来自同一智能体的连续轨迹，避免多智能体并行采样在时间维度的交叉混入.n步回报与目标定义为：

$$
\begin{array} { c } { { \displaystyle R _ { t } ^ { ( n ) } = \sum _ { j = 0 } ^ { n - 1 } \gamma ^ { j } r _ { t + j } } } \\ { { \displaystyle y _ { t } = R _ { t } ^ { ( n ) } + \gamma ^ { n } ( 1 - d o n e _ { t + n } ) m \underline { { { i } } } n { \cal Q } _ { \theta _ { j } ^ { - } } \Big ( s _ { t + n } , a r g \underline { { { m a x } } } \underline { { { Q } } } _ { \theta } ( s _ { t + n } , a ) \Big ) } } \end{array}\tag{10}
$$

探索采用线性退火的 $\varepsilon - g r e e d y$ ，随机动作与贪婪动作均限制在 ${ \mathbf A } _ { t } ^ { k }$ 内，使探索过程不破坏硬可行性前提.

## 2.4 约束嵌入与执行机制

本文将安全与可行性保障拆为动作层硬约束与执行层冲突/扰动处理两类机制并行实现.动作掩码基于越界、障碍、禁飞与电量可达性实时生成可行动作集合 ，保证策略始终在可行子集中选择；在此基础上引入安全屏蔽（shield），当电量低于阈值 时进一步禁止增大到最近充电桩距离的动作，以在资源紧张状态下强制引导回桩补能.

近失事件以切比雪夫距离阈值刻画，设置“每无人机每步至多一次惩罚”的计数上限，用于稳定拥挤空域下的奖励尺度；多机对同一体素的占用冲突在执行阶段采用确定性消解策略以保证状态转移可解析.环境以小概率水平风扰动引入随机位移，提高策略对动态扰动的鲁棒性.通过“掩码 +屏蔽 + 冲突规则 + 随机扰动”的组合，学习过程聚焦于可行空间内的智能路径规划与局部避障，而非在大量不可行动作上进行低效试错.

图 2 展示了本文所提出的双目标约束深度 Q 网络算法整体框架，涵盖环境建模、状态感知、硬约束嵌入、决策网络与训练机制，以及安全协同路径输出五个核心模块.

<!-- image-->  
图2 算法整体框架图  
Fig 2 Overall algorithm framework diagram

## 3 实验

实验选取在三种规模的三维体素栅格上进行，每一规模使用三个独立随机种子（42、43、44）生成实例，以降低结果对单一地图的偶然依赖；每个实例训练 40 回合，最大步长为 500.无人机能量模型以平移、爬升、悬停分别消耗1.0/0.6/0.3 单位成本；电量阈值为 12，桩上充电速率为 1.5.奖励系数与方法论一致，任务延误罚为 1.0，近失罚在每无人机每步仅计一次.

网络结构对所有算法保持一致，均采用两层全连接骨干、ReLU 激活；优化器为 Adam，学习率 $2 \times 1 0 ^ { - 4 }$ ，批量大小 64，目标网络软更新系数 0.005；经验池容量 120000，最小热身步数 5000，以避免早期噪声主导训练.探索退火采用线性日程，起始与结束分别为 1.0 与 0.05，总退火步数12000.DTC-DQN 使用三步回报，双目标网络取下界，PER的 与 分别为 0.6 与从 0.4 到 1.0 的线性上调；重要性权重按归一化方式修正偏差.为了公正比较，DoubleDQN 与DuelingDQN 也启用动作掩码与安全屏蔽，而不引入 PER或双目标；DQN 则不使用 Double 与 Dueling 结构.

实验环境配置包括：GPU NVIDIA RTX 5090D；CPUAMD Ryzen 9 9950x 16-Core Processor；Windows11 操作系统 23H2 版本；Python 版本 3.10.

## 3.1 收敛性实验分析

为评估所提算法的收敛速度与稳定性，选取典型问题实例，问题规模描述与参数设置如表1所示：

表1 问题参数数据表  
Table 1 Problem Parameter Data Table
<table><tr><td>参数</td><td>说明</td><td>数值</td></tr><tr><td>Small-Case</td><td>三维体素栅格上宽高</td><td>22*22*6</td></tr></table>

<table><tr><td>Medium-Case</td><td>三维体素栅格上宽高</td><td>32*32*8</td></tr><tr><td>Large-Case</td><td>三维体素栅格上宽高</td><td>42*42*10</td></tr><tr><td>K</td><td>可选无人机数量</td><td>3/6/10</td></tr><tr><td>M</td><td>取投任务数量</td><td>6/12/20</td></tr><tr><td>Obstacle</td><td>障碍密度</td><td>6%-10%</td></tr><tr><td>Prohibit</td><td>禁飞区域密度</td><td>3%-5%</td></tr><tr><td>C</td><td>充电桩数量</td><td>2/3/4</td></tr><tr><td>Rmax</td><td>无人机视野区域</td><td>5*5*2</td></tr></table>

分别对 DTC-DQN、Dueling DQN、Double DQN、DQN 等算法独立运行 20 次，记录每次运行中的最优适应度随迭 代的变化过程.将所有运行的收敛曲线取平均，并统计标准 差，绘制平均收敛曲线及波动区间.

<!-- image-->

<!-- image-->

<!-- image-->  
图3 收敛曲线对比实验  
Fig 3 Delay and energy consumption comparison experiment

三种仿真实验规模的实验结果如图 3 所示.相比 DoubleDQN，所提 DTC-DQN 在训练早期即可获得更快的回报增长速度，并在中后期达到更高的平台值，同时其标准差带明显收敛，表现出更强的训练稳定性与更小的策略波动.表明在相同交互预算下，DTC-DQN能够更高效地从稀疏任务回报与稠密避障信号中提炼有效策略，并在复杂三维约束下形成更可靠的飞行决策.

相比之下 DQN 由于目标网络单一且易受高估与目标漂 移影响，曲线波动更大、无法收敛；Dueling DQN对价值表 达有所改善，缺乏更强的样本优先利用与快速回报传播机 制，其稳定性与性能仍受到限制.

综上，实验结果表明DTC-DQN在多无人机三维路径规划与动态避障任务中具备更快的收敛速度、更优的策略质量与更强的稳定性.

## 3.2 成功率实验分析

为评估各算法在强约束三维空域中的任务可达性与安全执行能力，本文引入成功率（Success Rate,SR）指标.一回合（episode）被判定为成功需同时满足：在最大步长限制内完成该回合分配的所有任务；执行过程中未发生越界、进入静态障碍体素集合或禁飞体素集合等违规事件；未因能量耗尽导致任务中止.在给定测试集与随机种子下，成功率定义为：

$$
S R = \frac { N _ { s u c } } { N _ { t o t a l } }\tag{11}
$$

其中 $N _ { s u c }$ 为成功回合数量， $N _ { t o t a l }$ 为总评估回合数量.该指标能够从“是否可持续完成任务”的角度综合反映路径质量、冲突规避能力以及能量与时效决策的合理性.

<!-- image-->  
图4 成功率对比实验  
Fig 4 Convergent curve comparison experiment

实验结果如图4所示，DTC-DQN在三种规模实例上均取得最高或接近最高的成功率，并且随着问题规模增大，其优势更加明显.这表明在障碍与禁飞区域密度提高、无人机数量与任务数量增加时，所提方法仍能保持较好的可扩展性与可行性控制能力.

为评估所提 DTC-DQN 在实际运行中可能遇到的非理想条件，我们在小型规模仿真实验额外引入了基于 Dryden湍流模型生成连续风速场扰动，湍流强度 $\sigma = 1 . 5 m / s$ ，风阻系数k=0.02.无人机运动学模型中加入风速扰动项，能耗同时按风阻二次型增加：

表2风场扰动实验结果  
Table2 Wind field disturbance experiment results
<table><tr><td>实验组</td><td>成功率</td><td>增加路径长度</td></tr><tr><td>理想条件</td><td>91%</td><td></td></tr><tr><td>风场扰动</td><td>87%</td><td>7%</td></tr></table>

结果如表2所示：在风场扰动下，任务成功率为 87%，较无扰动下降 4%；证明动作掩码与安全屏蔽机制在扰动下仍能有效保证全过程安全性.额外路径长度平均增加7%，均在可接受范围.上述结果表明 DTC-DQN 对典型环境扰动具有较好的鲁棒性.

## 3.3 消融实验分析

为验证所提核心机制和改进模块对系统性能的贡献，本文设计消融实验：保持其他条件不变，将完整系统与去除该机制的系统对比，考察算法在中等规模问题下的奖励函数值.此替换方式确保各实验组框架一致，准确评估每个模块独立贡献.实验组设置如表3所示：

表3消融实验设置  
Table3 Ablation experiment setup
<table><tr><td>实验组 模型描述</td></tr><tr><td>DTC-DQN 全部改进模块，完整的DTC-DQN算法</td></tr><tr><td>去除“双目标网络下界（clipped）”机制，改 Ablation-1</td></tr><tr><td>为单目标网络或标准DoubleDQN目标 去除“n步回报与轨迹一致性缓冲”机制，改 Ablation-2</td></tr><tr><td>为1步TD目标 去除“优先级经验回放（PER）”机制，经验</td></tr></table>

<!-- image-->  
图5 消融实验奖励函数值对比  
Fig 5 Comparison of reward function values in ablation experiments

从图5可见，DTC-DQN在所有对照组中均取得最优的目标函数表现，说明各改进模块对性能提升具有互补贡献.其中，Ablation-1 表现训练过程更易出现价值高估导致的策略振荡，最终策略质量下降，验证了双目标下界估计对稳定学习的重要性.Ablation-2在训练回报上升效果较差，表明在稀疏任务奖励条件下，n 步回报对快速信用分配与缩短学习“冷启动期”具有关键作用.Ablation-3导致收敛效果降低且后期平台值受限，说明优先级采样能够更有效地利用关键转移样本、提升样本效率.Ablation-4通常表现出更高的失败率与更差的目标函数结果，其根本原因在于硬约束缺失后智能体在训练阶段会频繁经历不可行动作与高风险状态，造成大量无效交互与早期终止回合，从而降低有效学习信号占比并损害最终策略可行性.

## 3.4 时延与能量损失实验

本节从任务时效与能耗两个角度验证所提 DTC-DQN在多无人机路径规划与动态避障中的工程有效性.考虑城市物流配送对准时性与续航的双重要求，分别统计平均时延与单位任务能耗.平均时延定义为所有任务完成时间相对软截止时间的超期量平均值：

$$
D e l a y = \frac { 1 } { | T | } { \sum _ { j \in T } } \operatorname* { m a x } ( 0 , t _ { j } ^ { f i n i s h } - d _ { j } )\tag{12}
$$

单位任务能耗定义为回合内总能量消耗与完成任务数之比：

$$
E n e r g y = \frac { \sum _ { t } \sum _ { i } e _ { i , t } } { \mid T _ { d o n e } \mid }\tag{13}
$$

其中 $t _ { j } ^ { f i n i s h }$ 为任务 $j ^ { j }$ 的完成时刻， $d _ { j }$ 为其软截止时间， $e _ { i , t }$ 表示无人机 在时间步 的能量消耗.两指标均为越小越好.

<!-- image-->  
图6 时延与能量损失对比实验  
Fig 6 Delay and energy consumption comparison experiment

如图6所示DTC-DQN在平均时延与单位任务能耗两项指标上均优于对比算法，体现出更好的“准时性—能耗”综合权衡能力.其原因在于通过稳定价值估计与 n 步回报更快学习到面向子目标的有效推进策略，减少无效绕行与局部振荡，从而缩短路径长度并降低悬停/反复调整带来的能耗；同时，动作掩码与安全屏蔽降低了因进入高风险区域而产生的回避机动与冲突消解代价，使得多无人机在拥挤空域中仍能保持较高的通行效率.相比之下，其他方法由于训练不稳定或样本利用不足，在复杂场景中更易产生保守绕行或回退动作，导致平均完成时间上升并叠加更多能量消耗.

## 4 总结

本文面向低空三维城市物流配送中的多无人机路径规划与动态避障问题，针对离散高维决策、稀疏回报、价值估计不稳定及强安全约束难以全过程保证等关键挑战，构建了体素化三维场景与取送任务的序列决策模型，并提出DTC-DQN框架.方法层面，本文通过双目标网络下界估计抑制 Q 值高估、提升收敛稳定性；通过面向多智能体轨迹一致性的n步回报缓冲加速回报传播并避免时间片段污染；融合Dueling结构与优先级经验回放增强表达能力与样本利用效率；并将动作掩码与安全屏蔽机制嵌入动作选择与目标计算过程，实现可行性约束的前置化与全过程安全保障.实验结果表明，所提方法在不同规模三维实例上相较 DQN、Double DQN 与 Dueling DQN 等基线取得更快的收敛速度、更高的成功率与更优的时延—能耗权衡，并在与近期代表性方法的对比中表现出良好的稳定性与可扩展性.

未来工作将从以下方向进一步拓展：一是引入更精细的飞行动力学与气象扰动模型，提升策略在真实低空环境中的泛化能力和算法适应性；二是将任务分配、充电调度与路径规划进行更紧耦合的联合学习，以进一步提升大规模场景下的系统效率；三是探索连续动作与层级决策结构，以减少体素离散带来的分辨率—计算开销折中，并提升轨迹平滑性与可执行性；四是结合通信约束与协同感知机制，研究部分可观测条件下的协同避障与鲁棒控制.

## 5 参考文献

[1]. Garg V,Niranjan S,Gligor P D.Drones in last-mile delivery:a systematic review on efficiency,accessibility,and sustainability[J].Transportation Research Part D:Transport and Environment,2023,123:103831.

[2]. Zhai L,Wu H,Lai L,et al.Intelligent optimization algorithms for multi-UAV path planning:a comprehensive review[J].IEEE Access,2025,13:101106-101130.

[3]. Murray C C,Raj R.The multiple flying sidekicks traveling salesman problem:parcel delivery with multiple drones[J].Transportation Research Part C:Emerging Technologies,2020,110:368-398.

[4]. Otto A,Agatz N,Campbell J,et al.Optimization approaches for civil applications of unmanned aerial vehicles (UAVs) or aerial drones:a survey[J].Networks,2018,72(4):411-458.

[5]. Arulkumaran K,Deisenroth M P,Brundage M,et al.Deep reinforcement learning:a brief survey[J].IEEE Signal Processing Magazine,2017,34(6):26-38.

[6]. Sutton R S,Barto A G.Reinforcement learning:an introduction[M].Cambridge,MA:MIT Press,1998.

[7]. Roijers D M,Vamplew P,Whiteson S,et al.A survey of multi-objective sequential decision-making[J].Journal of Artificial Intelligence Research,2013,48:67-113.

[8]. Mnih V, Kavukcuoglu K, Silver D,et al.Human-level control through deep reinforcement learning[J].Nature,2015,518(7540):529-533.

[9]. Christianos F,Papoudakis G,Rahman A,et al.Scaling multi-agent reinforcement learning with selective parameter

sharing[C]//Proceedings of the 38th International Conference on Machine Learning,Virtual Event,PMLR,2021.

[10]. Van Hasselt H,Guez A,Silver D.Deep reinforcement learning with double q-learning[C]//Proceedings of the AAAI Conference on Artificial Intelligence,2016,30(1).

[11]. Deng Z,Wu F,Xu Y,et al.Energy minimization for radio map-based UAV pickup and delivery logistics system[J].IEEE Transactions on Vehicular Technology,2024,73(11):17893-17898.

[12]. Yoo Y D,Moon J H.Study on a-star algorithm-based 3D path optimization method considering density of obstacles[J].Aerospace,2025,12(2):85.

[13]. Lu Y,Wang X,Yang Y,et al.LSTM-DQN-APF path planning algorithm empowered by twins in complex scenarios[J].Applied Sciences,2025,15(8):4565.

[14]. Pan Z,Zhang C,Xia Y,et al.An improved artificial potential field method for path planning and formation control of the multi-UAV systems[J].IEEE Transactions on Circuits and Systems II:Express Briefs,2022,69(3):1129-1133.

[15]. Guo C,Huang L,Tian K.Combinatorial optimization for UAV swarm path planning and task assignment in multi-obstacle battlefield environment[J].Applied Soft Computing,2025,171:112773.

[16]. Cho S W,Park J H,Park H J,et al.Multi-UAV coverage path planning based on hexagonal grid decomposition in maritime search and rescue[J].Mathematics,2021,10(1):83.

[17]. Hajkarim S M H,Sujit P B,Menon P P.Optimized mission planning for heterogeneous uncrewed vehicle teams[J].IEEE Access,2024,12:147387-147399.

[18]. Wen C,Dong W,Xie W.3D cooperative curved path planning method for multi-UAV based on CEA-GA[J].Journal of Beijing University of Aeronautics and Astronautics,2023,49(11):3086-3099.

[19]. Wu W,Guo X,Zhou S.Dynamic path planning based on improved constrained differential evolution algorithm[J].Control and Decision,2020,35(10):2381-2390.

[20]. Qian W,Yi W,Yuan S,et al.Control-oriented real-time trajectory planning for heterogeneous UAV formations[J].Drones,2025,9(2):78.

[21]. Yang J,Huang X.Intelligent route planning method for UAV based on swarm intelligence and deep learning technology[J].Computing and Informatics,2024,43(4):874-899.

[22]. Liu Y,Li X,Wang J,et al.Reinforcement-learning-based multi-UAV cooperative search for moving targets in 3D scenarios[J].Drones,2024,8(8):378.

[23]. Zhang S,Li Y,Dong Q.Autonomous navigation of UAV in multi-obstacle environments based on a deep reinforcement learning approach[J].Applied Soft Computing,2022,115:108194.

[24]. Hu Z,Gao X,Wan K,et al.Relevant experience learning:a deep reinforcement learning method for UAV autonomous

motion planning in complex unknown environments[J].Chinese Journal of Aeronautics,2021,34(12):187-204.

[25]. Demir K,Tumen V,Kosunalp S,et al.A deep reinforcement learning algorithm for trajectory planning of swarm UAV fulfilling wildfire