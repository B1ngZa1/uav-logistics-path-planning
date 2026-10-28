# 知识卡片：基于深度强化学习的旋翼无人机敏捷飞行控制与运动规划

---

## 一、文献基本信息

| 字段 | 内容 |
|:---|:---|
| 标题 | 基于深度强化学习的旋翼无人机敏捷飞行控制与运动规划 |
| 英文标题 | Agile Quadrotor Flight Control and Motion Planning with Deep Reinforcement Learning |
| 作者 | 陈时雨 |
| 导师 | 楼云江 教授（主）、李衍杰 教授（副） |
| 学位 | 工学博士 |
| 学科 | 控制科学与工程 |
| 单位 | 哈尔滨工业大学（深圳） |
| 答辩时间 | 2024年5月 |
| 论文类型 | 博士学位论文（学术型） |
| 分类号 | TP242.6 |

**关键词：** 旋翼无人机、深度强化学习、敏捷飞行控制、运动规划、安全性修正

---

## 二、核心内容概述

### 2.1 研究背景与动机

四旋翼无人机具有垂直起降快、动作范围广、运营成本低、飞行精度高及敏捷灵活性强等优点，在搜索救援、智能交通、电路巡检、快递运输、农药喷洒、民用航拍等领域广泛应用。然而，随着应用场景复杂度提升，对无人机自主飞行控制提出了更高要求：

- **敏捷性与精度**：传统PID算法存在积分饱和、响应速度慢、鲁棒性差等缺陷
- **模型不确定性**：实际系统高度非线性，精确动力学模型难以获取
- **实时响应**：传统分层轨迹规划+跟踪控制方法计算复杂，无法实时动态响应
- **安全性保证**：空间受限场景下飞行安全性至关重要

### 2.2 研究目标

以四旋翼无人机为研究对象，基于强化学习理论与深度神经网络技术，分别从底层飞行控制、运动规划、仿真-实物迁移、飞行安全性四个方面提升无人机自主飞行性能。

### 2.3 四大研究贡献总览

| 序号 | 贡献 | 核心方法 | 关键创新 |
|:---:|:---|:---|:---|
| 1 | 底层敏捷飞行控制 | 端到端DRL控制策略 | 模型不确定性下的鲁棒控制 |
| 2 | 间隙穿越运动规划 | 安全意识探索+端到端策略 | 位置/姿态/方向安全奖励 |
| 3 | 仿真-实物迁移 | 动力学随机化+Sim2Sim2Real | 截断高斯分布参数调整 |
| 4 | 安全飞行控制 | 即插即用ASPC修正器 | MPC框架+在线自适应调节 |

---

## 三、理论框架与方法详解

### 3.1 第2章：无人机动力学模型与仿真平台

#### 3.1.1 无人机系统硬件平台

| 组件 | 型号 | 关键参数 |
|:---|:---|:---|
| 机架 | HSKRC-XL6 | 正"X"型，4mm碳纤维，130g |
| 无刷电机 | SmooX-2306 | 2280KV，850W，33g |
| 桨叶 | GF5040 | 5英寸三叶桨，5g |
| 机载计算机 | Nvidia Jetson TX2 | 6核ARM 2.0GHz，256核CUDA，8GB内存，154g |
| 底层飞控 | TMOTOR-F722 | STM32F722，216MHz，8.4g |
| 视觉模块 | Intel Realsense T265 | 自带V-SLAM，200Hz，55g |
| 电调 | TMOTOR-F55A | 四合一，55A持续，17.5g |
| 电池 | ACE-4S | 2000mAh，120C放电，215g |

**推重比：** $r_{t2w} = 4.8$（单电机最大推力11N）

#### 3.1.2 电机推力与力矩模型

**电机推力模型**（二次多项式）：

$$f_i = T_a \Omega_i^2 + T_b \Omega_i + T_c \quad (2\text{-}1)$$

辨识结果：$T_a = 2.992 \times 10^{-7}$, $T_b = 1.689 \times 10^{-4}$, $T_c = -5.805 \times 10^{-2}$

**电机力矩模型**（线性关系）：

$$\tau_i^M = k_{\tau f} f_i, \quad i \in \{1,2,3,4\} \quad (2\text{-}2)$$

力矩-推力比率：$k_{\tau f} = 0.016$

**电机滞后响应模型**（一阶惯性）：

$$\Omega_t = c \Omega_{t-1} + (1-c)\Omega_t^d, \quad c = e^{-\delta_M / \Delta t} \quad (2\text{-}3)$$

电机响应时间间隔 $\delta_M = 0.033$ s

#### 3.1.3 动力学方程

**旋转矩阵**（$z_B \to y_B \to x_B$ 旋转顺序）：

$$\mathbf{R} = \begin{bmatrix} \cos\psi\cos\theta & \cos\psi\sin\theta\sin\phi - \sin\psi\cos\phi & \cos\psi\sin\theta\cos\phi + \sin\psi\sin\phi \\ \sin\psi\cos\theta & \sin\psi\sin\theta\sin\phi + \cos\psi\cos\phi & \sin\psi\sin\theta\cos\phi - \cos\psi\sin\phi \\ -\sin\theta & \cos\theta\sin\phi & \cos\theta\cos\phi \end{bmatrix} \quad (2\text{-}6)$$

**牛顿-欧拉动力学方程**：

$$\dot{\mathbf{p}} = \mathbf{v} \quad (2\text{-}9)$$

$$\dot{\mathbf{v}} = \frac{\mathbf{R}\mathbf{F}}{m} - \mathbf{g} \quad (2\text{-}10)$$

$$\dot{\mathbf{R}} = \mathbf{R}[\boldsymbol{\omega}]_\times \quad (2\text{-}11)$$

$$\dot{\boldsymbol{\omega}} = \mathbf{I}^{-1}(\boldsymbol{\tau} - \boldsymbol{\omega} \times (\mathbf{I}\boldsymbol{\omega})) \quad (2\text{-}12)$$

**四阶龙格-库塔状态更新**：

$$\mathbf{s}_{t+1}^q = \mathbf{s}_t^q + f_{RK4}(\mathbf{s}_t^q, \mathbf{u}_t, \Delta t) \quad (2\text{-}14)$$

其中 $f_{RK4}(\cdot) = \frac{1}{6}\Delta t(K_1 + 2K_2 + 2K_3 + K_4)$

#### 3.1.4 仿真平台架构

基于3D游戏引擎Unity搭建，具备：
- 前端：Unity 3D渲染 + 机体坐标系 + 飞行轨迹可视化
- 后端：C++动力学系统（RK4状态更新）+ ZeroMQ实时通信
- API：Reset/Step/Done函数 + Pybind11（C++与Python转换）
- **并行环境**：多环境并行计算，加速数据采样

| 仿真平台 | 渲染引擎 | 动力学 | 保真度 | RL API | 并行环境 | 效率 |
|:---|:---|:---|:---|:---|:---|:---|
| RotorS | Gazebo | 复杂 | 高 | 无 | 无 | 低 |
| AirSim | Unreal | 复杂 | 高 | 无 | 有 | 中 |
| **本文** | **Unity** | **简单** | **低** | **有** | **有** | **高** |

---

### 3.2 第3章：考虑模型不确定性的无人机敏捷飞行控制

#### 3.2.1 深度强化学习理论基础

**MDP定义：** $\langle \mathcal{S}, \mathcal{A}, \mathcal{R}, P, \rho_0, \gamma \rangle$

**最优策略目标：**

$$\pi^* = \arg\max_\pi \mathbb{E}_{\pi,P}\left[\sum_{k=0}^{\infty} \gamma^k r_{t+k}\right] \quad (3\text{-}1)$$

**状态值函数的贝尔曼方程：**

$$V^\pi(\mathbf{s}) = \mathbb{E}_{\bar{\mathbf{u}} \sim \pi, \mathbf{s}' \sim P}[r(\mathbf{s}, \bar{\mathbf{u}}) + \gamma V^\pi(\mathbf{s}')] \quad (3\text{-}4)$$

**动作值函数的贝尔曼方程：**

$$Q^\pi(\mathbf{s}, \bar{\mathbf{u}}) = \mathbb{E}_{\mathbf{s}' \sim P}[r(\mathbf{s}, \bar{\mathbf{u}}) + \gamma \mathbb{E}_{\bar{\mathbf{u}}' \sim \pi}[Q^\pi(\mathbf{s}', \bar{\mathbf{u}}')]] \quad (3\text{-}5)$$

**贝尔曼最优性方程：**

$$V^*(\mathbf{s}) = \max_{\bar{\mathbf{u}}} \mathbb{E}_{\mathbf{s}' \sim P}[r(\mathbf{s}, \bar{\mathbf{u}}) + \gamma V^*(\mathbf{s}')] \quad (3\text{-}11)$$

#### 3.2.2 策略优化方法体系

| 方法类别 | 代表算法 | 适用场景 | 核心思想 |
|:---|:---|:---|:---|
| 基于值函数 | DP / MC / TD / SARSA / Q-Learning | 离散动作空间 | 通过值函数迭代改进策略 |
| 基于策略梯度 | REINFORCE / TRPO / PPO | 连续动作空间 | 直接沿策略梯度方向更新参数 |
| Actor-Critic | A2C / SAC / DDPG | 连续动作空间 | 结合值函数估计与策略梯度 |

**策略梯度定理：**

$$\nabla_\vartheta J(\pi_\vartheta) = \mathbb{E}_{\tau \sim \pi_\vartheta}\left[\sum_{t=0}^{T} \nabla_\vartheta \log \pi_\vartheta(\bar{\mathbf{u}}_t | \mathbf{s}_t) \hat{A}^{\pi}(\mathbf{s}_t, \bar{\mathbf{u}}_t)\right] \quad (3\text{-}54)$$

**PPO截断目标函数：**

$$L^{\pi_{\vartheta_k}}(\vartheta) = \mathbb{E}\left[\min\left(\rho_t(\vartheta)\hat{A}^{\pi}, \text{clip}(\rho_t(\vartheta), 1-\zeta, 1+\zeta)\hat{A}^{\pi}\right)\right] \quad (3\text{-}63)$$

其中 $\rho_t(\vartheta) = \frac{\pi_\vartheta(\bar{\mathbf{u}}_t|\mathbf{s}_t)}{\pi_{\vartheta_k}(\bar{\mathbf{u}}_t|\mathbf{s}_t)}$，截断参数 $\zeta = 0.2$

#### 3.2.3 端到端敏捷飞行控制框架

**状态空间**（18维，原文第58页明确指出 "18维向量"）：

$$\mathbf{s}_t = \hat{\mathbf{s}}_t^q - \mathbf{s}_t^d = [\mathbf{e}_t^p, \mathbf{e}_t^R, \mathbf{e}_t^v, \mathbf{e}_t^\omega]^T \in \mathbb{R}^{18} \quad (3\text{-}41)$$

包含：位置误差(3) + 旋转矩阵误差(9) + 线速度误差(3) + 角速度误差(3) = 18维
（观测状态 $\hat{\mathbf{s}}_t^q$ 与目标状态 $\mathbf{s}_t^d$ 的误差向量）

**动作空间**（4维连续）：

$$\bar{\mathbf{u}}_t = [\bar{f}_1, \bar{f}_2, \bar{f}_3, \bar{f}_4]^T, \quad \bar{f}_i \in [-1, 1]$$

端到端映射：观测状态 $\to$ 归一化电机推力

**奖励函数设计：**

$$r_t = \beta_p r_t^p + \beta_\Omega r_t^\Omega + \beta_v r_t^v + \beta_\omega r_t^\omega + \beta_u r_t^u$$

| 分量 | 公式 | 作用 |
|:---|:---|:---|
| 位置奖励 $r_t^p$ | $\exp(-\|\mathbf{e}_t^p\|_2^2)$ | 最小化位置误差 |
| 姿态奖励 $r_t^\Omega$ | $\exp(-\|\mathbf{e}_t^\Omega\|_2^2)$ | 最小化姿态误差 |
| 线速度奖励 $r_t^v$ | $\exp(-\|\mathbf{e}_t^v\|_2^2)$ | 最小化速度误差 |
| 角速度奖励 $r_t^\omega$ | $\exp(-\|\mathbf{e}_t^\omega\|_2^2)$ | 最小化角速度误差 |
| 振荡惩罚 $r_t^u$ | $-\|\bar{\mathbf{u}}_t - \bar{\mathbf{u}}_{t-1}\|_\infty$ | 抑制机体抖动 |

**模型不确定性处理：**

训练时对无人机动力学参数施加随机化：
- 质量 $m \sim \mathcal{U}(m_{\min}, m_{\max})$
- 臂长 $l \sim \mathcal{U}(l_{\min}, l_{\max})$
- 电机响应时间 $\delta_M \sim \mathcal{U}(\delta_{M,\min}, \delta_{M,\max})$
- 力矩-推力比 $k_{\tau f} \sim \mathcal{U}(k_{\tau f,\min}, k_{\tau f,\max})$

状态观测噪声：多元高斯 $\boldsymbol{\mu}_N = [0,0,0]^T$，$\boldsymbol{\Sigma} = \text{diag}(0.1, 0.1, 0.1)$ + 均匀分布 $\boldsymbol{\mu}_U = [0.1, 0.1, 0.1]^T$

#### 3.2.4 策略网络结构

4层全连接网络（原文第62页表3-2）：
- 输入层：18维状态向量
- 隐藏层1：256个神经元 + ReLU
- 隐藏层2：256个神经元 + ReLU
- 输出层：4维动作（Tanh激活）+ 多元高斯采样

#### 3.2.5 仿真实验结果

**测试任务1：随机初始化起飞至悬停**
- 10次不同随机初始状态测试
- 最大欧拉角误差超过2 rad
- 所有误差在200个时间步（4秒）内收敛至0
- 最大角速度达到12 rad/s

**测试任务2：不同速度轨迹跟踪**（圆形 + 双纽线）
- 速度：1 m/s、3 m/s、6 m/s
- 相比PID：RL策略超调更小、响应速度快10 s、6 m/s高速时PID出现大跟踪误差

**测试任务3：随机扰动鲁棒性**
- 噪声概率0.3：RL与PID均可完成，RL超调更小
- 噪声概率0.5：PID无法完成6 m/s高速跟踪，RL仍可完成
- 结论：RL策略在随机扰动环境下具备极强抗干扰能力

---

### 3.3 第4章：无人机间隙穿越的实时动态响应运动规划

#### 3.3.1 端到端间隙穿越框架

**状态空间**（24维）：

$$\mathbf{s}_t = [\hat{\mathbf{p}}_t, \hat{\boldsymbol{\Theta}}_t, \hat{\mathbf{v}}_t, \hat{\boldsymbol{\omega}}_t, \hat{\mathbf{a}}_t, \mathbf{p}_g, \boldsymbol{\Theta}_g, d_t^p, d_t^n, \theta_t^n]^T \in \mathbb{R}^{24}$$

| 分量 | 维度 | 含义 |
|:---|:---|:---|
| $\hat{\mathbf{p}}_t$ | 3 | 无人机位置估计 |
| $\hat{\boldsymbol{\Theta}}_t$ | 3 | 欧拉角姿态估计 |
| $\hat{\mathbf{v}}_t$ | 3 | 线速度估计 |
| $\hat{\boldsymbol{\omega}}_t$ | 3 | 角速度估计 |
| $\hat{\mathbf{a}}_t$ | 3 | 线加速度估计 |
| $\mathbf{p}_g$ | 3 | 狭窄间隙中心位置 |
| $\boldsymbol{\Theta}_g$ | 3 | 狭窄间隙姿态（俯仰角 $\theta_g$） |
| $d_t^p$ | 1 | 无人机到间隙平面距离 |
| $d_t^n$ | 1 | 无人机到间隙法向量距离 |
| $\theta_t^n$ | 1 | 速度方向与间隙法向量夹角 |

**动作空间**（4维）：

$$\bar{\mathbf{u}}_t = [\bar{f}_1, \bar{f}_2, \bar{f}_3, \bar{f}_4]^T, \quad \bar{f}_i \in [-1, 1]$$

#### 3.3.2 具有安全意识探索的奖励函数

**总奖励：**

$$r_t = \beta_p r_t^p + \beta_s r_t^s + \beta_u r_t^u \quad (4\text{-}20)$$

**（1）位置奖励：**

$$r_t^p = \exp(-e_t^g), \quad e_t^g = \|\hat{\mathbf{p}}_t - \mathbf{p}_g\|_2^2 \quad (4\text{-}11, 4\text{-}12)$$

**（2）安全意识奖励（三个分量）：**

位置安全奖励（引导无人机从间隙中心方向靠近）：

$$r_t^{sp} = -(h_t^p)^2 \left[1 - \exp\left(-\varepsilon(d_t^n)^2\right)\right] \quad (4\text{-}13)$$

其中权重 $h_t^p = \max\left[1 - \frac{d_t^p}{d_0^p}, 0\right]$，激活距离 $d_0^p = 3$ m

姿态安全奖励（近距离时调整姿态对齐间隙）：

$$r_t^{s\Omega} = -(h_t^\Omega)^2 \left[1 - \exp\left(-\varepsilon e_t^\Omega\right)\right] \quad (4\text{-}16)$$

其中姿态误差 $e_t^\Omega = a_t^z \tan\theta_g - a_t^x$，激活距离 $d_0^\Omega = 1$ m

飞行方向安全奖励（约束速度方向）：

$$r_t^{sv} = \min\left[1 - \frac{\theta_t^n}{\theta_0^n}, 0\right] \quad (4\text{-}17)$$

**（3）振荡惩罚：**

$$r_t^u = \|\bar{\mathbf{u}}_t - \bar{\mathbf{u}}_{t-1}\|_\infty \quad (4\text{-}19)$$

#### 3.3.3 训练超参数

| 超参数 | 符号 | 值 |
|:---|:---|:---|
| 折扣因子 | $\gamma$ | 0.99 |
| GAE权重 | $\lambda$ | 0.95 |
| 单次训练时间步 | $T$ | 250 |
| 样本集合容量 | $\|D\|$ | 100 |
| 策略网络学习率 | $\alpha_\vartheta$ | $3 \times 10^{-4}$ |
| 值函数网络学习率 | $\alpha_\varphi$ | $3 \times 10^{-4}$ |
| PPO截断参数 | $\zeta$ | 0.2 |
| 位置奖励权重 | $\beta_p$ | 3.6 |
| 安全意识奖励权重 | $\beta_s$ | 0.01 |
| 振荡惩罚权重 | $\beta_u$ | -0.012 |
| 单步仿真时间 | $\Delta t$ | 0.02 s |

#### 3.3.4 仿真实验结果

**测试1：三间隙穿越**
- 三个间隙位置：$[0,3,4]$, $[1,6,7]$, $[0,9,5]$；俯仰角：$22.5°$, $-45°$, $0°$
- 穿越时间：0.94s / 2.4s / 3.9s，总跨度9m仅需4s
- 平均位置均方误差：0.021 m
- 最大线速度：7.43 m/s，最大角速度：5.86 rad/s

**测试2：多样化场景泛化（未训练过的场景）**

| 场景 | 俯仰角 | 位置均方误差 (m) | 俯仰角误差 (rad) |
|:---|:---|:---|:---|
| 单倾斜间隙 | 30° | 0.058 | 0.06 |
| 单倾斜间隙 | 60° | 0.072 | 0.1 |
| 多水平间隙1 | 0° | 0.032 | 0.09 |
| 多水平间隙2 | 0° | 0.034 | 0.1 |
| 多水平间隙3 | 0° | 0.026 | 0.02 |

**测试3：安全意识探索碰撞率对比（100次重复）**

| 方法 | 平均碰撞率 |
|:---|:---|
| 无安全意识探索 | 11% |
| **有安全意识探索** | **2.7%** |

**测试4：与相关方法对比**

| 方法 | 轨迹规划耗时 | 附加控制器 | 随机环境 | 响应时间 | $e_t^p$ (m) | $e_t^\Omega$ (rad) |
|:---|:---|:---|:---|:---|:---|:---|
| 基于优化[83] | 1.05s/次 | 需要 | 否 | 1.05s | 0.127 | 0.13 |
| 课程学习[91] | / | 需要 | 否 | 0.02s | - | - |
| 好奇心[92] | / | 需要 | 部分 | 0.01s | 0.034 | 0.04 |
| **本文** | **/** | **不需要** | **是** | **0.003s** | **0.046** | **0.07** |

核心优势：响应频率300+ Hz，无需附加控制器，支持随机多样化场景

---

### 3.4 第5章：基于动力学随机化的仿真-实物迁移

#### 3.4.1 Sim-to-Sim-to-Real迁移机制

**三阶段流程：**

1. **低保真仿真训练**（Unity）：简单动力学 + 并行环境 $\to$ 快速策略收敛
2. **高保真仿真验证**（RotorS/Gazebo）：精确动力学 $\to$ 策略性能评估与参数调整
3. **实物部署**：可迁移准则判断 $\to$ 安全迁移至真实系统

**动力学参数随机化（截断高斯分布）：**

$$m \in \mathcal{G}_m(\bar{\mu}_m, \bar{\sigma}_m, m_{\min}, m_{\max}) \quad (5\text{-}1)$$
$$l \in \mathcal{G}_l(\bar{\mu}_l, \bar{\sigma}_l, l_{\min}, l_{\max}) \quad (5\text{-}2)$$
$$\delta_M \in \mathcal{G}_{\delta_M}(\bar{\mu}_{\delta_M}, \bar{\sigma}_{\delta_M}, \delta_{M,\min}, \delta_{M,\max}) \quad (5\text{-}3)$$
$$k_{\tau f} \in \mathcal{G}_{k_{\tau f}}(\bar{\mu}_{k_{\tau f}}, \bar{\sigma}_{k_{\tau f}}, k_{\tau f,\min}, k_{\tau f,\max}) \quad (5\text{-}4)$$

**方差线性调节：**

$$\bar{\sigma}_{i,k+1} = \bar{\sigma}_{i,k} + 0.1 O_i \quad (5\text{-}5)$$

量级：$O_m = 1.0$, $O_l = 0.1$, $O_{\delta_M} = O_{k_{\tau f}} = 0.01$

**可迁移准则（总体性能指标）：**

$$k^* = \arg\min_k J_k(\mu_{\text{RMSE}}^k, \sigma_{\text{RMSE}}^k) \in \mathcal{P} \quad (5\text{-}7)$$

其中 $J_k := \mu_{\text{RMSE}}^k + \alpha_\sigma \sigma_{\text{RMSE}}^k$，位置RMSE：

$$p_{\text{RMSE}} = \sqrt{\frac{\sum_{t=1}^T \|\mathbf{e}_t^p\|_2^2}{T}} \quad (5\text{-}6)$$

#### 3.4.2 实物实验结果

- 底层飞行控制：双纽线轨迹跟踪（实物），成功迁移
- 运动规划：水平/倾斜狭窄间隙穿越（实物），成功迁移
- 机载计算机100Hz + 底层飞控板500Hz实时闭环
- 通过Sim-to-Sim-to-Real验证，预测并确认了迁移可行性

---

### 3.5 第6章：基于模型预测修正的安全飞行控制

#### 3.5.1 自适应安全性预测修正器（ASPC）

**核心思想：** 即插即用模块，对任意RL控制器输出的潜在不安全控制动作进行安全性验证与修正

**优化问题：**

$$\min_{\mathbf{u}_t^{\text{safe}}} \|\mathbf{u}_t^{RL} - \mathbf{u}_t^{\text{safe}}\|_2^2 \quad (6\text{-}4)$$

$$\text{s.t.} \quad \mathbf{s}_{t+n+1}^q = \mathbf{s}_{t+n}^q + f_{\text{dyn}}(\mathbf{s}_{t+n}^q, \mathbf{u}_{t+n}^{\text{safe}}) \quad (6\text{-}5)$$
$$\mathbf{s}_{t+n}^q \in \mathcal{S} \quad (6\text{-}6)$$
$$\mathbf{u}_{t+n}^{\text{safe}} \in \mathcal{U} \quad (6\text{-}7)$$
$$\mathbf{s}_{t+N}^q \in \mathcal{S}_{\text{end}} \quad (6\text{-}8)$$

**安全控制动作限制集合**（凸多面体）：

$$\mathcal{U} := \{\mathbf{u} \in \mathbb{R}^4 | H\mathbf{u} \leq \mathbf{u}_{\text{lim}}\} \quad (6\text{-}9)$$

其中 $\mathbf{u}_{\text{lim}} = [F_{\min}, \boldsymbol{\omega}_{\min}, F_{\max}, \boldsymbol{\omega}_{\max}]^T$

参数：$F_{\min}=2$ N, $F_{\max}=20$ N, $\boldsymbol{\omega}_{\min}=[-3,-3,-2]^T$ rad/s, $\boldsymbol{\omega}_{\max}=[3,3,2]^T$ rad/s

**安全状态集合**（凸多面体 + 在线自适应）：

$$\mathcal{S} := \{\mathbf{s}^q \in \mathbb{R}^{15} | B\mathbf{s}^q \leq \mathbf{s}_t^{\text{lim}}\} \quad (6\text{-}10)$$

**在线自适应调节：**

$$\mathbf{s}_t^{\text{lim}} = [\mathbf{p}_t^{\min}, \mathbf{p}_t^{\max}]^T \quad (6\text{-}11)$$

随着环境特征变化实时调节安全边界，实现对不同场景的自适应

**安全性验证结果：**

$$\mathbf{u}_t^{\text{safe}} = \begin{cases} = \mathbf{u}_t^{RL} & \text{RL is safe} \\ \to \mathbf{u}_t^{RL} & \text{RL is unsafe} \end{cases} \quad (6\text{-}12)$$

**实现细节：**
- 预测步数 $N = 20$
- 求解工具：ACADO + qpOASES
- 计算频率：100 Hz
- RL控制器输出：合推力 + 角速度 $\mathbf{u}^{RL} = [F, \boldsymbol{\omega}]^T$
- 底层角速度控制器：Body-rate控制器（500 Hz）

#### 3.5.2 安全飞行控制实验

**仿真测试（三种受限环境）：**
- 大厅（高度受限）：无人机垂直上升至安全高度边界后自动悬停
- 走廊（宽度受限）：路径点设在安全范围外，无人机沿边界安全飞行
- 房间（空间受限）：双纽线轨迹超出安全边界，ASPC实时修正

**实物测试：** 空间受限场景下双纽线轨迹跟踪，无人机始终在安全边界内飞行

---

## 四、创新点总结

### 4.1 四大核心创新

| 创新 | 方法名称 | 关键特征 |
|:---|:---|:---|
| 创新1 | 考虑模型不确定性的敏捷飞行控制策略 | 端到端映射(状态→电机推力)、动力学参数随机化训练、多元噪声注入 |
| 创新2 | 基于安全意识探索的间隙穿越策略 | 位置/姿态/方向三重安全奖励、距离自适应激活机制、300+ Hz实时响应 |
| 创新3 | Sim-to-Sim-to-Real迁移机制 | 截断高斯分布参数化、高低保真仿真过渡、可迁移准则(均值+方差) |
| 创新4 | 自适应安全性预测修正器(ASPC) | 即插即用MPC框架、在线自适应安全边界、最小化敏捷性损失 |

### 4.2 方法优势对比

| 指标 | 传统PID | 基于优化 | 现有RL方法 | **本文RL方法** |
|:---|:---|:---|:---|:---|
| 模型依赖 | 低 | 高 | 低 | **低** |
| 敏捷性 | 差 | 中 | 较好 | **优** |
| 鲁棒性 | 差 | 中 | 中 | **强** |
| 实时响应 | 快 | 慢 | 快 | **极快(300+Hz)** |
| 泛化能力 | 无 | 弱 | 弱 | **强** |
| 安全保证 | 无 | 部分 | 无 | **有(ASPC)** |
| Sim-to-Real | 不适用 | 需精确模型 | 需精确模型 | **无需精确模型** |

---

## 五、与本项目的关联分析

### 5.1 技术映射关系

| 本项目（无人机配送） | 本文（敏捷飞行控制） | 关联度 |
|:---|:---|:---|
| 无人机配送调度 | 无人机底层飞行控制 | ★★★★★ |
| 城市3D环境避障 | 狭窄间隙穿越运动规划 | ★★★★☆ |
| 多约束飞行(禁飞区/建筑) | 安全状态集合约束(ASPC) | ★★★★☆ |
| Sim-to-Real迁移 | 动力学随机化+Sim2Sim2Real | ★★★★☆ |
| 多机协同(CTDE) | 本文仅单机 | ★★☆☆☆ |

### 5.2 可借鉴技术

1. **端到端控制策略**：可将配送无人机的"状态→电机推力"映射用于L3/L4层级
2. **动力学参数随机化训练**：提升配送策略对模型不确定性的鲁棒性
3. **安全意识探索奖励设计**：位置安全奖励(式4-13)可用于配送中的建筑避障引导
4. **ASPC安全修正器**：即插即用模块，可直接用于配送无人机的禁飞区/建筑安全保证
5. **Sim-to-Sim-to-Real迁移机制**：可迁移准则可用于验证配送策略是否可部署到实物

### 5.3 差异与局限

1. **单机 vs 多机**：本文仅研究单架无人机，本项目需多机协同（CTDE）
2. **简单任务 vs 复杂配送**：间隙穿越为单目标任务，外卖配送涉及多订单、时间窗口
3. **连续控制 vs 离散决策**：本文动作空间为连续推力，L1/L2层级为离散网格动作
4. **无通信约束**：本文不考虑多机通信延迟/中断

---

## 六、参考价值评级

| 维度 | 评级 | 说明 |
|:---|:---|:---|
| 理论深度 | ★★★★★ | 完整的DRL理论推导（值函数→策略梯度→Actor-Critic→PPO），极具教学价值 |
| 工程实践 | ★★★★★ | 从仿真到实物的完整链路，含硬件平台搭建、电机辨识、飞控部署 |
| 仿真实现 | ★★★★★ | Unity仿真平台+RL API设计+并行环境，可直接参考搭建配送仿真 |
| 算法创新 | ★★★★☆ | ASPC安全修正器设计精巧，安全意识探索奖励设计有启发性 |
| 与项目相关度 | ★★★★☆ | 底层飞行控制直接可用，但缺少多机协同和复杂配送任务 |

**总体评级：** ★★★★★（强烈推荐阅读，尤其第2-3章的仿真平台搭建和DRL控制框架设计）

---

*生成时间：2026-07-13*
*数据来源：陈时雨博士论文全文提取（哈尔滨工业大学，2024年5月）*
