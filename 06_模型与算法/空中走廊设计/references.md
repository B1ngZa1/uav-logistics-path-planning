# 参考文献与规范来源

## A. 法规、运行与安全框架

1. 中国政府网：《无人驾驶航空器飞行管理暂行条例》（国务院、中央军委令第761号，2023-06-28 公布，2024-01-01 施行）。该条例明确境内无人驾驶航空器飞行活动须遵守条例，并涉及分类管理、实名登记、运行合格证、无线电和网络信息安全等要求；本文将这些作为“必须核验的合规接口”，不把示例高度当作批准高度。原文：[中国政府网](https://app.www.gov.cn/govdata/gov/202306/28/504796/article.html)。

2. ICAO, *Manual on Remotely Piloted Aircraft Systems (RPAS), Doc 10019*, 1st ed., 2015. 内容覆盖 RPAS 进入非隔离空域、通信与控制、空中交通管理、detect-and-avoid、安全管理和法律议题；本文借其安全论证边界，不复制其未针对本项目的具体运行批准条件。出版页：[ICAO Store](https://store.icao.int/en/manual-on-remotely-piloted-aircraft-systems-rpas-doc-10019)。

## B. 强化学习与多智能体算法

3. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., and Klimov, O., “Proximal Policy Optimization Algorithms,” arXiv:1707.06347, 2017. 论文页：[arXiv](https://arxiv.org/abs/1707.06347)。用于 PPO clipped surrogate、GAE 接口和集中 critic 的基础说明。

4. Yu, C., Velu, A., Vinitsky, E., Gao, J., Wang, Y., Bayen, A., and Wu, Y., “The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games,” NeurIPS 2022. 论文：[NeurIPS proceedings PDF](https://proceedings.neurips.cc/paper_files/paper/2022/file/9c1535a02f0ce079433344e14d910597-Paper-Datasets_and_Benchmarks.pdf)；实现：[marlbenchmark/on-policy](https://github.com/marlbenchmark/on-policy)。论文明确区分 MAPPO 的集中 value function 与 IPPO 的局部 value function，并讨论参数共享、GAE 和 value clipping。

5. Lowe, R., Wu, Y., Tamar, A., Harb, J., Abbeel, P., and Mordatch, I., “Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments,” NeurIPS 2017. [DOI/论文入口](https://papers.nips.cc/paper_files/paper/2017/hash/68a9750337a418a86fe06c1991a1d64c-Abstract.html)。用于 CTDE actor–critic 接口和多机集中训练/分散执行的背景。

## C. 图搜索、多智能体路径与冲突消解

6. Sharon, G., Stern, R., Felner, A., and Sturtevant, N. R., “Conflict-Based Search for Optimal Multi-Agent Pathfinding,” *Artificial Intelligence*, 219, 40–66, 2015. DOI: [10.1016/j.artint.2014.11.006](https://doi.org/10.1016/j.artint.2014.11.006)。用于高层冲突树 + 低层单机搜索、时间/位置约束和 reservation table 的依据。

7. Huang, T., Koenig, S., and Dilkina, B., “Learning to Resolve Conflicts for Multi-Agent Path Finding with Conflict-Based Search,” *AAAI*, 35(13), 11246–11253, 2021. DOI: [10.1609/aaai.v35i13.17341](https://doi.org/10.1609/aaai.v35i13.17341)。用于说明冲突选择可以由学习器辅助，但硬冲突约束仍由规划器维护。

8. Hart, P. E., Nilsson, N. J., and Raphael, B., “A Formal Basis for the Heuristic Determination of Minimum Cost Paths,” *IEEE Transactions on Systems Science and Cybernetics*, 4(2), 100–107, 1968. DOI: [10.1109/TSSC.1968.300136](https://doi.org/10.1109/TSSC.1968.300136)。用于 A* 单机候选路线的启发式搜索基础。

## D. 项目内部输入与参数溯源

9. `00_项目管理/参数一致性规范.md`：项目主配置基线，采用 `z=[10,50] m`、`dt=0.1 s`、UAV 间 10 m、障碍物 5 m 等值。

10. `00_项目管理/参数一致性审查报告_2026-07-13.md`：记录 L1/L2 与旧 3D 体系的分层差异，以及 Critic 激活函数、奖励惩罚等未完全收敛项。

11. `06_模型与算法/算法文档/三维空域模型设计.md`、`状态空间设计.md`、`动作空间设计.md`、`奖励函数设计.md`：走廊模型复用的坐标、体素、状态、动作和奖励语义来源。

12. `07_仿真与控制代码/Python仿真/config.py`、`environment/airspace.py`、`environment/uav_dynamics.py`、`environment/collision_detector.py`：实现接口与已知默认值冲突的核对来源；本文不修改这些文件。

## 引用边界

- [1] 是当前中国境内运行的法规入口，但不提供本模型所需的具体航路/层带批准值；运行前必须按机型、重量、空域和任务类别核验。
- [2] 是国际 RPAS 安全框架，不等价于中国法律或项目认证依据。
- [3]–[8] 支持算法结构与求解思想，不证明任何特定城市的真实飞行安全性。
- 所有“10 m、5 m、18–48 m、0.2 s 通信延迟、0.8 链路裕度”等未明确标为法规的数值均是示例或项目仿真假设。
