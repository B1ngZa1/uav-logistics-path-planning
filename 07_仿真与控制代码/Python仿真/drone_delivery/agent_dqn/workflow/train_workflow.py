# -*- coding: utf-8 -*-
"""
================================================================================
workflow/train_workflow.py — DQN 训练工作流 / DQN Training Workflow
================================================================================
职责 / Responsibility:
    定义 DQN 单机训练工作流和回合运行器, 驱动 DQN 智能体在无人机配送
    环境中进行训练和评估。
    Defines single-machine DQN training workflow and episode runner,
    driving the DQN agent to train and evaluate in the drone delivery environment.

参考 / Reference:
    - agent_q_learning/workflow/train_workflow.py (KaiWu workflow 模式)
    - agent_monte_carlo/workflow/train_workflow.py
    - KaiWu DRL 框架的 workflow() 函数签名

DQN 训练特点 / DQN Training Characteristics:
    - 异策略 (off-policy): 每步学习, 使用 replay buffer
    - 每步学习: 每次环境交互后立即通过 algorithm.learn() 更新
      使用经验回放缓冲区存储历史样本, 从中采样 minibatch 计算 Double DQN loss
    - Off-policy: learns every step using replay buffer
    - Per-step learning: update immediately after each env interaction
      Uses replay buffer to store history; samples minibatches for Double DQN loss

训练循环 / Training Loop::

    reset → observation_process →
        [predict → step → construct_sample → learn] × N →
    end of episode

环境接口 / Environment Interface (Simple2DDeliveryEnv):
    step(action) → (obs, reward, terminated, truncated, info)
================================================================================
"""

import time
from typing import Dict, Any, Optional, List

import numpy as np

from conf.conf import config
from feature.definition import SampleData, SampleManager


# =============================================================================
# workflow — 训练工作流入口 / Training Workflow Entry Point
# =============================================================================

def workflow(envs, agents, logger=None, monitor=None):
    """
    DQN 单机训练工作流 — 无人机配送。
    Single-machine DQN training workflow — Drone Delivery.

    通过 orchestrating 环境与智能体交互, 执行完整的训练流程:
    从初始化 → 训练循环 → 评估 → 模型保存。
    Orchestrates environment-agent interaction to execute full training pipeline:
    init → training loop → evaluation → model saving.

    遵循 Kaiwu 框架的 workflow() 函数签名。
    Follows Kaiwu framework's workflow() function signature.

    DQN 学习特点: 每步学习 (非每回合), 使用 replay buffer。
    DQN learning: per-step (not per-episode), using replay buffer.

    Args / 参数:
        envs: 环境对象或环境列表 / Environment object or list of envs
        agents: 智能体对象或智能体列表 / Agent object or list of agents
        logger: 日志记录器 (可选) / Logger instance (optional)
        monitor: 监控器 (可选) / Monitor instance (optional)

    Yields / 产出:
        dict: 每个回合的训练统计 / Per-episode training statistics
    """
    # 统一处理为单个对象 / Normalize to single objects
    if len(envs) == 0 or len(agents) == 0:
        raise ValueError("envs and agents must not be empty")

    env = envs[0] if isinstance(envs, (list, tuple)) else envs
    agent = agents[0] if isinstance(agents, (list, tuple)) else agents

    # 创建回合运行器 / Create episode runner
    runner = EpisodeRunner(env=env, agent=agent, logger=logger, monitor=monitor)

    # 训练参数 / Training parameters
    num_episodes = config.NUM_EPISODES if hasattr(config, "NUM_EPISODES") else 5000
    eval_interval = config.EVAL_INTERVAL if hasattr(config, "EVAL_INTERVAL") else 100
    save_interval = config.MODEL_SAVE_INTERVAL if hasattr(config, "MODEL_SAVE_INTERVAL") else 500

    # 训练开始日志 / Training start log
    if logger:
        logger.info(
            f"[DQN] Starting training: {num_episodes} episodes, "
            f"eval_interval={eval_interval}, save_interval={save_interval}"
        )
        logger.info(
            f"[DQN] Config: GAMMA={config.GAMMA}, LR={config.LEARNING_RATE}, "
            f"BATCH={config.BATCH_SIZE}, BUFFER={config.BUFFER_SIZE}, "
            f"EPS_START={config.EPSILON_START}, EPS_DECAY={config.EPSILON_DECAY}, "
            f"DEVICE={config.DEVICE}"
        )
    else:
        print(
            f"[DQN] Starting training: {num_episodes} episodes, "
            f"eval_interval={eval_interval}"
        )

    start_time = time.time()

    for ep in range(num_episodes):
        # ---- 运行一个训练回合 / Run one training episode ----
        stats = runner.run_episode()

        # 汇报给监控器 / Report to monitor
        if monitor:
            monitor.put({f"train_{k}": v for k, v in stats.items()})

        # ---- 定期评估与日志 / Periodic evaluation and logging ----
        if (ep + 1) % eval_interval == 0:
            # 计算最近窗口的平均指标 / Compute average metrics in recent window
            avg_r = runner.get_recent_avg_reward(window=eval_interval)
            avg_d = (
                np.mean(runner.episode_deliveries[-eval_interval:])
                if runner.episode_deliveries
                else 0
            )

            # 运行评估 / Run evaluation
            eval_episodes = config.EVAL_EPISODES if hasattr(config, "EVAL_EPISODES") else 10
            eval_metrics = runner.evaluate(num_episodes=eval_episodes)

            elapsed = time.time() - start_time
            log_msg = (
                f"[DQN] Ep {ep + 1:5d}/{num_episodes} | "
                f"AvgR: {avg_r:8.1f} | "
                f"AvgD: {avg_d:.1f} | "
                f"eps: {agent.epsilon:.4f} | "
                f"loss: {agent.last_loss:.4f} | "
                f"buf: {agent.buffer_size} | "
                f"EvalR: {eval_metrics['avg_reward']:.1f} | "
                f"Time: {elapsed:.1f}s"
            )
            if logger:
                logger.info(log_msg)
            else:
                print(log_msg)

            if monitor:
                monitor.put({f"eval_{k}": v for k, v in eval_metrics.items()})

            # 保存最佳模型 / Save best model
            if avg_r > runner.best_avg_reward:
                runner.best_avg_reward = avg_r
                agent.save_model(id="best")
                if logger:
                    logger.info(f"[DQN] New best model saved: avg_reward={avg_r:.1f}")

        # ---- 定期保存检查点 / Periodic checkpoint save ----
        if (ep + 1) % save_interval == 0:
            agent.save_model(id=f"ep{ep + 1}")

        # 产出每回合统计 / Yield per-episode stats
        yield stats

    # ---- 训练结束 / Training finished ----
    agent.save_model(id="latest")
    elapsed = time.time() - start_time

    if logger:
        logger.info(f"[DQN] Training complete. Total time: {elapsed:.1f}s")
        logger.info(
            f"[DQN] Final: epsilon={agent.epsilon:.4f}, "
            f"steps={agent.total_steps}, "
            f"buffer={agent.buffer_size}, "
            f"learn_count={agent.learn_count}"
        )
    else:
        print(f"\n[DQN] Training complete. Total time: {elapsed:.1f}s")

    # 最终评估 / Final evaluation
    final_eval = runner.evaluate(num_episodes=100)
    final_msg = (
        f"[DQN] Final evaluation (100 episodes): "
        f"AvgR={final_eval['avg_reward']:.1f}, "
        f"AvgD={final_eval['avg_deliveries']:.1f}, "
        f"SuccessRate={final_eval['success_rate']:.1%}"
    )
    if logger:
        logger.info(final_msg)
    else:
        print(final_msg)


# =============================================================================
# EpisodeRunner — 回合运行器 / Episode Runner
# =============================================================================

class EpisodeRunner:
    """
    DQN 回合运行器 — 管理单个训练/评估回合的完整生命周期。
    Episode Runner — manages the full lifecycle of a single training/eval episode.

    职责 / Responsibilities:
        1. run_episode(): 执行一个训练回合 (epsilon-greedy + 每步DQN学习)
           Run one training episode (epsilon-greedy + per-step DQN learning)
        2. evaluate(): 执行多个评估回合 (纯贪心策略, 不更新模型)
           Run multiple evaluation episodes (pure greedy, no model update)

    DQN 训练循环 / DQN Training Loop::

        reset → observation_process →
            [predict → step → construct_sample → learn] × N →
        end of episode

    环境接口 / Environment interface:
        step(action) → (obs, reward, terminated, truncated, info)
    """

    def __init__(self, env, agent, logger=None, monitor=None):
        """
        初始化回合运行器 / Initialize episode runner.

        Args / 参数:
            env: 环境对象 (Simple2DDeliveryEnv) / Environment object
            agent: DQN Agent 实例 / DQN Agent instance
            logger: 日志记录器 (可选) / Logger instance (optional)
            monitor: 监控器 (可选) / Monitor instance (optional)
        """
        self.env = env
        self.agent = agent
        self.logger = logger
        self.monitor = monitor

        # 训练统计 / Training statistics
        self.episode_rewards: List[float] = []
        """回合奖励历史 / Episode reward history"""

        self.episode_deliveries: List[int] = []
        """回合配送完成数历史 / Episode delivery count history"""

        self.episode_steps: List[int] = []
        """回合步数历史 / Episode step count history"""

        self.best_avg_reward = -float("inf")
        """最佳平均奖励 / Best average reward"""

    # -------------------------------------------------------------------------
    # run_episode — 运行一个训练回合
    # -------------------------------------------------------------------------

    def run_episode(self, seed: Optional[int] = None) -> Dict[str, float]:
        """
        执行一个完整的 DQN 训练回合 (每步学习)。
        Run one complete DQN training episode (learn per step).

        DQN 循环 / DQN Loop::

            env_obs, info = env.reset()
            agent.reset(env_obs)
            obs_data = agent.observation_process(env_obs)

            while not done:
                act_data = agent.predict([obs_data])           # epsilon-greedy
                act = agent.action_process(act_data)
                env_obs, reward, terminated, truncated, _ = env.step(act)
                shaped_reward = agent.reward_shaping(reward)
                next_obs_data = agent.observation_process(env_obs)
                done = terminated or truncated

                # 构造样本 / Construct sample
                sample = SampleData(
                    obs=obs_data.feature, act=act,
                    rew=shaped_reward, next_obs=next_obs_data.feature,
                    done=done, legal_act=obs_data.legal_act,
                )
                # 每步学习 (DQN 核心特点) / Per-step learning (DQN key feature)
                agent.learn([sample])
                obs_data = next_obs_data

        Args / 参数:
            seed: 环境随机种子 / Environment random seed (None for automatic)

        Returns / 返回:
            回合统计字典 / Episode statistics dict:
                - reward: 累计奖励 / Cumulative reward
                - deliveries: 配送完成数 / Deliveries completed
                - steps: 回合步数 / Steps taken
                - epsilon: 当前探索率 / Current epsilon
                - loss: 最近 loss / Most recent loss
                - buffer_size: 缓冲区大小 / Buffer size
        """
        # ---- 环境重置 / Environment reset ----
        if seed is not None:
            env_obs, info = self.env.reset(seed=seed)
        else:
            env_obs, info = self.env.reset()

        # ---- 智能体回合开始 / Agent episode start ----
        self.agent.reset(env_obs)

        # ---- 特征处理 / Feature processing ----
        obs_data = self.agent.observation_process(env_obs)

        # ---- 回合统计 / Episode statistics ----
        episode_reward = 0.0
        episode_steps = 0
        done = False
        last_loss = 0.0

        # 最大步数限制 / Max steps guard
        max_steps = config.MAX_STEPS if hasattr(config, "MAX_STEPS") else 800

        # ============================================================
        # DQN 主循环 / DQN Main Loop (每步学习 / learn every step)
        # ============================================================
        while not done:
            # 动作选择 (epsilon-greedy) / Action selection
            act_data = self.agent.predict([obs_data])
            act = self.agent.action_process(act_data)

            # 环境步进 / Environment step
            env_obs, reward, terminated, truncated, info = self.env.step(act)

            # 奖励塑形 / Reward shaping
            shaped_reward = self.agent.reward_shaping(reward)

            # 下一观测处理 / Next observation processing
            next_obs_data = self.agent.observation_process(env_obs)

            # 终止判断 / Termination check
            done = terminated or truncated

            # ---- 构造样本 / Construct sample ----
            sample = SampleData(
                obs=obs_data.feature,
                act=act,
                rew=shaped_reward,
                next_obs=next_obs_data.feature,
                done=done,
                legal_act=obs_data.legal_act,
            )

            # ---- 每步学习 (DQN 核心特点) / Per-step learning (DQN key feature) ----
            learn_stats = self.agent.learn([sample])
            last_loss = learn_stats.get("loss", 0.0)

            # 更新状态 / Update state
            obs_data = next_obs_data
            episode_reward += shaped_reward
            episode_steps += 1

            # 步数超限安全防护 / Step limit safety guard
            if episode_steps >= max_steps and not done:
                done = True

        # ---- 统计配送完成数 / Count deliveries completed ----
        deliveries = self._count_deliveries(env_obs)

        # ---- 记录统计 / Record statistics ----
        self.episode_rewards.append(episode_reward)
        self.episode_deliveries.append(deliveries)
        self.episode_steps.append(episode_steps)

        return {
            "reward": float(episode_reward),
            "deliveries": int(deliveries),
            "steps": int(episode_steps),
            "epsilon": float(self.agent.epsilon),
            "loss": float(last_loss),
            "buffer_size": int(self.agent.buffer_size),
        }

    # -------------------------------------------------------------------------
    # evaluate — 评估 / Evaluation
    # -------------------------------------------------------------------------

    def evaluate(self, num_episodes: int = 10) -> Dict[str, float]:
        """
        执行贪婪评估 (纯利用, 不更新模型)。
        Run greedy evaluation (pure exploitation, no model updates).

        评估期间不进行探索, 使用贪心策略选择动作, 不更新网络。
        No exploration during evaluation; greedy action selection only;
        network is NOT updated.

        Args / 参数:
            num_episodes: 评估回合数 / Number of evaluation episodes

        Returns / 返回:
            评估指标字典 / Evaluation metrics dict:
                - avg_reward: 平均累计奖励 / Average total reward
                - avg_deliveries: 平均配送完成数 / Average deliveries per episode
                - avg_steps: 平均回合步数 / Average episode steps
                - success_rate: 全部订单完成占比 / Proportion of full completions
        """
        total_reward = 0.0
        total_deliveries = 0
        total_steps = 0
        successes = 0

        max_steps = config.MAX_STEPS if hasattr(config, "MAX_STEPS") else 800
        num_orders = config.NUM_ORDERS if hasattr(config, "NUM_ORDERS") else 5

        for ep in range(num_episodes):
            # 环境重置 / Environment reset
            env_obs, info = self.env.reset(seed=10000 + ep)
            done = False
            ep_steps = 0

            while not done:
                # 贪心动作选择 (无探索, 使用 exploit)
                # Greedy action selection (no exploration, using exploit)
                act_data = self.agent.exploit(env_obs)
                action = self.agent.action_process(act_data)

                # 环境步进 / Environment step
                env_obs, reward, terminated, truncated, info = self.env.step(action)
                total_reward += float(reward)
                total_steps += 1
                ep_steps += 1
                done = terminated or truncated

                # 步数安全防护 / Step limit guard
                if ep_steps >= max_steps and not done:
                    done = True

            # 统计 / Statistics
            deliveries = self._count_deliveries(env_obs)
            total_deliveries += deliveries
            if deliveries >= num_orders:
                successes += 1

        return {
            "avg_reward": total_reward / num_episodes,
            "avg_deliveries": total_deliveries / num_episodes,
            "avg_steps": total_steps / num_episodes,
            "success_rate": successes / num_episodes,
        }

    # -------------------------------------------------------------------------
    # 辅助方法 / Helper Methods
    # -------------------------------------------------------------------------

    def get_recent_avg_reward(self, window: int = 100) -> float:
        """
        获取最近 N 回合的平均奖励。
        Get average reward over the last N episodes.

        Args / 参数:
            window: 滑动窗口大小 / Sliding window size

        Returns / 返回:
            最近 N 回合的平均奖励 / Average reward of last N episodes
        """
        if len(self.episode_rewards) < window:
            return float(np.mean(self.episode_rewards)) if self.episode_rewards else 0.0
        return float(np.mean(self.episode_rewards[-window:]))

    @staticmethod
    def _count_deliveries(env_obs: Dict[str, Any]) -> int:
        """
        从环境观测中提取已完成的配送数量。
        Extract number of completed deliveries from environment observation.

        处理 Simple2DDeliveryEnv 的观测格式。
        Handles Simple2DDeliveryEnv observation format.

        Args / 参数:
            env_obs: 环境观测字典 / Environment observation dict

        Returns / 返回:
            完成的配送数量 / Number of completed deliveries
        """
        # 解包可能的 {"observation": {...}} 包装 / Unwrap possible wrapper
        obs = env_obs.get("observation", env_obs) if isinstance(env_obs, dict) else {}

        order_status = obs.get("order_status", None)
        if order_status is None:
            return 0

        # 计算完成的订单数 / Count completed orders
        if isinstance(order_status, np.ndarray):
            return int(order_status.sum())
        if isinstance(order_status, (list, tuple)):
            return sum(int(s) for s in order_status)
        return 0
