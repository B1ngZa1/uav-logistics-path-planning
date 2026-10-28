# -*- coding: utf-8 -*-
"""
================================================================================
agent.py — 学生DIY智能体 / Student DIY Agent
================================================================================
职责 / Responsibility:
    学生自行实现的强化学习智能体骨架。
    集成模型(Model)、算法(Algorithm)、预处理器(Preprocessor)三大组件。
    Skeleton for a reinforcement learning agent to be implemented by students.
    Integrates Model, Algorithm, and Preprocessor components.

参考 / Reference:
    - agent_dqn/agent.py (DQN Agent 完整实现 / full DQN agent reference)
    - agent_q_learning/agent.py (Q-learning Agent 完整实现)
    - agent_monte_carlo/agent.py (Monte Carlo Agent 完整实现)
    - KaiWu DRL 框架的 Agent 接口规范

领域适配 / Domain Adaptation:
    - 无人机配送: 50维观测, 8维离散动作
    - Drone delivery: 50-dim observation, 8 discrete actions

核心方法说明 / Core Method Descriptions:
    reset()              — 回合开始时调用, 重置回合状态, 递增episodes计数
                           Called at episode start; resets state, increments episodes
    predict()            — 训练时动作选择 (带探索, 如epsilon-greedy)
                           Training action selection (with exploration, e.g. epsilon-greedy)
    exploit()            — 评估时动作选择 (纯贪心/确定性, 无探索)
                           Evaluation action selection (purely greedy/deterministic)
    learn()              — 从样本中学习, 更新模型参数
                           Learn from samples, update model parameters
    save_model()         — 保存模型到文件
                           Save model to file
    load_model()         — 从文件加载模型
                           Load model from file
    observation_process()— 环境观测 → ObsData (委托给Preprocessor)
                           Raw env obs → ObsData (delegates to Preprocessor)
    action_process()     — ActData → 可执行动作 (委托给Preprocessor)
                           ActData → executable action (delegates to Preprocessor)
    reward_shaping()     — 奖励塑形 (委托给definition.reward_shaping)
                           Reward shaping (delegates to definition.reward_shaping)
================================================================================
"""

import os
from typing import Dict, Any, Optional

import numpy as np

from agent_diy.conf.conf import Config
from agent_diy.model.model import Model
from agent_diy.algorithm.algorithm import Algorithm
from agent_diy.feature.preprocessor import Preprocessor
from agent_diy.feature.definition import (
    ObsData,
    ActData,
    SampleData,
    reward_shaping as _reward_shaping,
)


class Agent:
    """
    学生DIY智能体 — 无人机外卖配送 / Student DIY Agent — Drone Food Delivery

    学生需要完成以下任务:
      1. 设计并实现神经网络模型 (model/model.py)
      2. 设计并实现RL算法 (algorithm/algorithm.py)
      3. 在本类中连接各组件, 实现完整的智能体逻辑
      4. 在 workflow/train_workflow.py 中实现训练工作流

    Students need to:
      1. Design and implement neural network model (model/model.py)
      2. Design and implement RL algorithm (algorithm/algorithm.py)
      3. Wire up components in this class for complete agent logic
      4. Implement training workflow (workflow/train_workflow.py)

    Usage / 用法::

        agent = Agent()
        env_obs, _ = env.reset()
        agent.reset(env_obs)
        obs_data = agent.observation_process(env_obs)

        while not done:
            act_data = agent.predict([obs_data])
            act = agent.action_process(act_data)
            _, reward, terminated, truncated, env_obs = env.step(act)
            shaped_reward = agent.reward_shaping(reward)
            # ... 收集样本 ...
            obs_data = next_obs_data

        agent.learn(train_data)
    """

    def __init__(
        self,
        agent_type: str = "player",
        device: Any = None,
        logger: Any = None,
        monitor: Any = None,
    ) -> None:
        """
        初始化学生DIY智能体 / Initialize student DIY agent.

        TODO: 在此完成组件初始化 / Complete component initialization here

        Args / 参数:
            agent_type (str): 智能体类型 / Agent type ("player" for single-agent)
            device: 计算设备 / Compute device (cpu / cuda)
            logger: 日志记录器 / Logger instance
            monitor: 监控实例 / Monitor instance
        """
        self.agent_type = agent_type
        self.logger = logger
        self.monitor = monitor

        # ============================================================
        # 超参数引用 / Hyper-parameter References
        # ============================================================
        # TODO: 从 Config 中读取超参数 / Read hyperparameters from Config
        self.gamma = Config.GAMMA
        self.learning_rate = Config.LEARNING_RATE
        self.action_size = Config.ACTION_SIZE

        # ============================================================
        # 模型 / Model
        # ============================================================
        # TODO: 创建模型实例 (完成 model/model.py 后取消注释)
        #       Create model instance (uncomment after completing model/model.py)
        self.model = Model()

        # ============================================================
        # 算法 / Algorithm
        # ============================================================
        # TODO: 创建算法实例 (完成 algorithm/algorithm.py 后取消注释)
        #       Create algorithm instance (uncomment after completing algorithm/algorithm.py)
        self.algorithm = Algorithm()

        # ============================================================
        # 预处理器 / Preprocessor
        # ============================================================
        self.preprocessor = Preprocessor()

        # ============================================================
        # 探索策略 / Exploration Strategy
        # ============================================================
        # TODO: 设置探索参数 / Set exploration parameters
        self._epsilon = Config.EPSILON_START
        self.epsilon_min = Config.EPSILON_MIN
        self.epsilon_decay = Config.EPSILON_DECAY

        # ============================================================
        # 训练状态 / Training State
        # ============================================================
        self.train_steps = 0      # 累计训练步数 / Cumulative training steps
        self.episodes = 0         # 已完成回合数 / Completed episodes

    # ============================================================
    # 日志辅助 / Logging Helper
    # ============================================================

    def _log(self, level: str, message: str) -> None:
        """
        安全日志输出 / Safe logging output.

        Args / 参数:
            level (str): 日志级别 (info, warning, error) / Log level
            message (str): 日志消息 / Log message
        """
        if self.logger is None:
            return
        log_method = getattr(self.logger, level, None) or getattr(self.logger, "info", None)
        if log_method is not None:
            log_method(message)

    # ============================================================
    # reset — 回合重置 / Episode Reset
    # ============================================================

    def reset(self, env_obs: Optional[Dict[str, Any]] = None) -> None:
        """
        回合开始时调用 — 重置回合状态, 更新回合计数, 衰减探索率。
        Called at episode start — reset episode state, update episode counter,
        decay exploration rate.

        学生需要在此:
          - 递增 episodes 计数 / Increment episodes counter
          - 可选择在此衰减 epsilon (每回合衰减), 或在 predict() 中每步衰减
          - Optionally decay epsilon here (per-episode) or in predict() (per-step)
          - 重置任何回合级别的状态 / Reset any episode-level state

        TODO: 实现回合重置逻辑 / Implement episode reset logic

        Args / 参数:
            env_obs: 环境初始观测 (可选) / Initial environment observation (optional)
        """
        self.episodes += 1
        # TODO: 在此添加回合重置逻辑 / Add episode reset logic here
        # 示例: 每回合衰减 epsilon
        # self._epsilon = max(self.epsilon_min, self._epsilon * self.epsilon_decay)
        pass

    # ============================================================
    # predict — 训练时动作选择 / Training Action Selection
    # ============================================================

    def predict(self, list_obs_data):
        """
        训练时动作选择 — 通常使用 epsilon-greedy 策略。
        Training action selection — typically uses epsilon-greedy policy.

        学生需要实现探索-利用权衡:
          - 以概率 epsilon: 随机探索 (均匀从合法动作中选择)
          - 以概率 (1 - epsilon): 贪心利用 (选择模型预测的最优动作)

        Students need to implement explore-exploit tradeoff:
          - With probability epsilon: random exploration among legal actions
          - With probability (1 - epsilon): greedy exploitation (model's best action)

        TODO: 实现训练时动作选择逻辑 / Implement training action selection logic

        Args / 参数:
            list_obs_data: ObsData列表 (使用第一个元素) / List of ObsData (use first)

        Returns / 返回:
            ActData: 包含选中动作的 ActData 对象 / ActData with selected action
        """
        obs_data = list_obs_data[0]

        # TODO: 实现 epsilon-greedy 动作选择 / Implement epsilon-greedy action selection
        # 步骤参考 / Step reference:
        # 1. 获取合法动作列表 / Get legal action list
        # 2. 随机数 vs epsilon 比较 / Compare random number vs epsilon
        # 3. 探索: 随机选合法动作 / Explore: random legal action
        # 4. 利用: 选模型预测的最佳动作 / Exploit: model's best action
        # 5. 递增 train_steps / Increment train_steps

        self.train_steps += 1
        # 占位符: 总是选动作0 / Placeholder: always pick action 0
        return ActData(act=0, val=None)

    # ============================================================
    # exploit — 评估时动作选择 / Evaluation Action Selection
    # ============================================================

    def exploit(self, env_obs: Dict[str, Any]):
        """
        评估时动作选择 — 纯贪心/确定性策略, 不探索。
        Evaluation action selection — purely greedy/deterministic, no exploration.

        学生需要实现确定性策略:
          - 始终选择模型预测的最佳合法动作
          - 不使用 epsilon 探索

        Students need to implement deterministic policy:
          - Always select model's best legal action
          - No epsilon exploration

        TODO: 实现评估时动作选择逻辑 / Implement evaluation action selection logic

        Args / 参数:
            env_obs: 环境观测字典 / Environment observation dict

        Returns / 返回:
            ActData: 包含最优动作的 ActData / ActData with best action
        """
        obs_data = self.observation_process(env_obs)

        # TODO: 实现贪心动作选择 / Implement greedy action selection
        # 步骤参考 / Step reference:
        # 1. 使用模型计算所有动作的值 / Use model to compute action values
        # 2. 在合法动作中选择值最大的 / Select best among legal actions
        # 3. 随机打破平局 / Randomly break ties

        # 占位符: 总是选动作0 / Placeholder: always pick action 0
        return ActData(act=0, val=None)

    # ============================================================
    # learn — 学习 / Learning
    # ============================================================

    def learn(self, list_sample_data):
        """
        从经验样本中学习 — 委托给 Algorithm.learn()。
        Learn from experience samples — delegates to Algorithm.learn().

        学生需要:
          - 将样本传递给算法进行学习
          - 返回学习统计信息 (loss, epsilon, learn_count 等)

        Students should:
          - Pass samples to algorithm for learning
          - Return learning statistics (loss, epsilon, learn_count, etc.)

        TODO: 实现学习逻辑 / Implement learning logic

        Args / 参数:
            list_sample_data: 样本数据列表 / List of sample data
                每个样本包含: obs, act, rew, next_obs, done

        Returns / 返回:
            dict: 学习统计信息 / Learning statistics
                - loss: 损失值 (如有) / Loss value (if applicable)
                - epsilon: 当前探索率 / Current epsilon
                - train_steps: 累计训练步数 / Cumulative training steps
        """
        if list_sample_data is None or len(list_sample_data) == 0:
            return {}

        # TODO: 委托给 algorithm.learn() / Delegate to algorithm.learn()
        # stats = self.algorithm.learn(list_sample_data)

        # 返回统计信息 / Return statistics (placeholder)
        return {}

    # ============================================================
    # save_model — 保存模型 / Model Saving
    # ============================================================

    def save_model(self, path: Optional[str] = None, id: str = "latest") -> None:
        """
        保存模型到文件 / Save model to file.

        学生需要实现:
          - 创建输出目录 / Create output directory
          - 保存模型权重和训练元数据 (epsilon, train_steps, episodes)
          - Save model weights and training metadata

        TODO: 实现模型保存逻辑 / Implement model saving logic

        Args / 参数:
            path (Optional[str]): 保存目录路径 / Save directory path
            id (str): 检查点标识符 / Checkpoint identifier (default: "latest")
        """
        if path is None:
            path = "outputs/models"
        os.makedirs(path, exist_ok=True)
        # TODO: 在此保存模型 / Save model here
        # 参考: self.algorithm.save_model(filepath)
        self._log("info", f"Model save placeholder: path={path}, id={id}")

    # ============================================================
    # load_model — 加载模型 / Model Loading
    # ============================================================

    def load_model(self, path: Optional[str] = None, id: str = "latest") -> None:
        """
        从文件加载模型 / Load model from file.

        学生需要实现:
          - 检查文件是否存在 / Check if file exists
          - 加载模型权重和训练元数据 / Load model weights and training metadata
          - 文件不存在时静默回退 / Silently fallback if file not found

        TODO: 实现模型加载逻辑 / Implement model loading logic

        Args / 参数:
            path (Optional[str]): 加载目录路径 / Load directory path
            id (str): 检查点标识符 / Checkpoint identifier (default: "latest")
        """
        if path is None:
            path = "outputs/models"
        # TODO: 在此加载模型 / Load model here
        # 参考: self.algorithm.load_model(filepath)
        self._log("info", f"Model load placeholder: path={path}, id={id}")

    # ============================================================
    # observation_process — 观测处理 / Observation Processing
    # ============================================================

    def observation_process(self, env_obs):
        """
        处理环境观测, 生成 ObsData — 委托给 Preprocessor.observation_process()。
        Process environment observation → ObsData — delegates to Preprocessor.

        TODO: 实现观测处理 (通常只需委托即可)
              Implement observation processing (usually just delegation)

        Args / 参数:
            env_obs: 环境观测字典 / Environment observation dict

        Returns / 返回:
            ObsData: 包含 feature 和 legal_act 的结构化数据
        """
        # TODO: 检查并完善预处理逻辑 / Review and improve preprocessing logic
        return self.preprocessor.observation_process(env_obs)

    # ============================================================
    # action_process — 动作处理 / Action Processing
    # ============================================================

    def action_process(self, act_data):
        """
        处理动作数据, 提取可执行的动作 — 委托给 Preprocessor.action_process()。
        Process action data → executable action — delegates to Preprocessor.

        TODO: 实现动作处理 (通常只需委托即可)
              Implement action processing (usually just delegation)

        Args / 参数:
            act_data: ActData 对象 / ActData object

        Returns / 返回:
            int: 可执行的动作索引 (0-7) / Executable action index
        """
        # TODO: 检查并完善动作处理逻辑 / Review and improve action processing logic
        return self.preprocessor.action_process(act_data)

    # ============================================================
    # reward_shaping — 奖励塑形 / Reward Shaping
    # ============================================================

    def reward_shaping(
        self,
        reward: float,
        env_obs: Optional[Dict[str, Any]] = None,
        prev_env_obs: Optional[Dict[str, Any]] = None,
        action: Optional[int] = None,
    ) -> float:
        """
        奖励塑形 — 委托给 definition.reward_shaping()。
        Reward shaping — delegates to definition.reward_shaping().

        学生可在 feature/definition.py 中的 reward_shaping() 函数添加自定义逻辑。
        Students may add custom logic in reward_shaping() in feature/definition.py.

        Args / 参数:
            reward (float): 原始环境奖励 / Raw environment reward
            env_obs: 当前环境观测 (可选) / Current observation (optional)
            prev_env_obs: 前一环境观测 (可选) / Previous observation (optional)
            action: 执行的动作 (可选) / Action taken (optional)

        Returns / 返回:
            float: 塑形后的奖励 / Shaped reward
        """
        return _reward_shaping(reward, env_obs)

    # ============================================================
    # 属性 / Properties
    # ============================================================

    @property
    def epsilon(self) -> float:
        """当前探索率 / Current exploration rate"""
        return self._epsilon

    @epsilon.setter
    def epsilon(self, value: float) -> None:
        self._epsilon = float(value)

    def __repr__(self) -> str:
        return (f"Agent(epsilon={self.epsilon:.4f}, "
                f"episodes={self.episodes}, "
                f"train_steps={self.train_steps})")
