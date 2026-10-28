#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
无人机配送 — Q-learning训练工作流模块
Drone Delivery — Q-learning Training Workflow Module

职责 / Responsibility:
    定义单机训练工作流和回合运行器，驱动Q-learning智能体在无人机配送
    环境中进行训练和评估。
    Defines single-machine training workflow and episode runner, driving the
    Q-learning agent to train and evaluate in the drone delivery environment.

参考 / Reference:
    kaiwu_gorge_walk_v2/agent_q_learning/workflow/train_workflow.py

领域适配 / Domain Adaptation:
    Gorge Walk treasure collection → Drone pickup-delivery:
    - hero_pos → drone_pos (x,y grid)
    - treasure → delivery_order
    - treasure_state (10-bit binary) → order_status_mask (5-bit)
    - endpoint → depot at [30,10]
    - max_step=1000 → max_steps=500
    - treasure_score=100 → delivery_reward=+100
    - end_score=150 → safe_return_bonus=+500
"""

import time
import numpy as np
from typing import Dict, List, Any, Optional

from agent_q_learning.conf.conf import Config
from agent_q_learning.feature.definition import SampleManager


def workflow(envs, agents, logger=None, monitor=None):
    """
    单机训练工作流 — 无人机配送Q-learning。
    Single-machine training workflow — Drone Delivery Q-learning.

    通过 orchestrating 环境与智能体交互，执行完整的训练流程：
    从初始化 → 训练循环 → 评估 → 模型保存。
    Orchestrates environment-agent interaction to execute full training pipeline:
    init → training loop → evaluation → model saving.

    遵循Kaiwu框架的workflow()函数签名。
    Follows Kaiwu framework's workflow() function signature.

    Args:
        envs: 环境对象或环境列表 / Environment object or list of envs
        agents: 智能体对象或智能体列表 / Agent object or list of agents
        logger: 日志记录器 (可选) / Logger instance (optional)
        monitor: 监控器 (可选) / Monitor instance (optional)

    Yields:
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
    num_episodes = Config.NUM_EPISODES
    eval_interval = Config.EVAL_INTERVAL
    save_interval = eval_interval  # 保存间隔与评估间隔一致 / Save interval same as eval

    # 训练开始日志 / Training start log
    if logger:
        logger.info(
            f"Starting Q-learning training: {num_episodes} episodes, "
            f"eval_interval={eval_interval}"
        )
        logger.info(
            f"Config: GAMMA={Config.GAMMA}, LR={Config.LEARNING_RATE}, "
            f"EPS={Config.EPSILON_START}, EPS_DECAY={Config.EPSILON_DECAY}"
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
            eval_metrics = runner.evaluate(num_episodes=Config.EVAL_EPISODES)

            elapsed = time.time() - start_time
            if logger:
                logger.info(
                    f"Episode {ep+1}/{num_episodes} | "
                    f"AvgReward: {avg_r:.1f} | "
                    f"AvgDeliveries: {avg_d:.1f}/{Config.NUM_ORDERS} | "
                    f"Epsilon: {agent.epsilon:.4f} | "
                    f"Eval: R={eval_metrics['avg_reward']:.1f} "
                    f"D={eval_metrics['avg_deliveries']:.1f} | "
                    f"Time: {elapsed:.1f}s"
                )
            else:
                print(
                    f"Ep {ep+1:5d} | "
                    f"AvgR: {avg_r:8.1f} | "
                    f"AvgD: {avg_d:.1f}/{Config.NUM_ORDERS} | "
                    f"eps: {agent.epsilon:.4f} | "
                    f"Eval: R={eval_metrics['avg_reward']:.1f} "
                    f"D={eval_metrics['avg_deliveries']:.1f}"
                )

            if monitor:
                monitor.put({f"eval_{k}": v for k, v in eval_metrics.items()})

            # 保存最佳模型 / Save best model
            if avg_r > runner.best_avg_reward:
                runner.best_avg_reward = avg_r
                agent.save_model(id="best")
                if logger:
                    logger.info(f"New best model saved: avg_reward={avg_r:.1f}")

        # ---- 定期保存检查点 / Periodic checkpoint save ----
        if (ep + 1) % save_interval == 0:
            agent.save_model(id=f"ep{ep+1}")

        # 产出每回合统计 / Yield per-episode stats
        yield stats

    # ---- 训练结束 / Training finished ----
    agent.save_model(id="latest")
    elapsed = time.time() - start_time

    if logger:
        logger.info(f"Training complete. Total time: {elapsed:.1f}s")
        logger.info(
            f"Final epsilon: {agent.epsilon:.4f}, "
            f"Q-table size: {agent.q_table_size} states"
        )

    # 最终评估 / Final evaluation
    final_eval = runner.evaluate(num_episodes=100)
    if logger:
        logger.info(
            f"Final evaluation (100 episodes): "
            f"AvgR={final_eval['avg_reward']:.1f}, "
            f"AvgD={final_eval['avg_deliveries']:.1f}/{Config.NUM_ORDERS}, "
            f"SuccessRate={final_eval['success_rate']:.1%}"
        )
    else:
        print(
            f"\nFinal evaluation (100 episodes): "
            f"AvgR={final_eval['avg_reward']:.1f}, "
            f"AvgD={final_eval['avg_deliveries']:.1f}/{Config.NUM_ORDERS}"
        )


class EpisodeRunner:
    """
    回合运行器 — 管理单个训练/评估回合的完整生命周期。
    Episode Runner — manages the full lifecycle of a single training/eval episode.

    职责 / Responsibilities:
        1. run_episode(): 执行一个训练回合 (epsilon-greedy + Q-learning更新)
           Run one training episode (epsilon-greedy + Q-learning update)
        2. evaluate(): 执行多个评估回合 (纯贪心策略，不更新Q表)
           Run multiple evaluation episodes (pure greedy, no Q-table update)

    Q-learning 训练循环 / Q-learning training loop::

        reset → observation_process → [predict → step → add_sample → loop] → learn

    环境接口 / Environment interface (Simple2DDeliveryEnv):
        step(action) → (obs, reward, terminated, truncated, info)
    """

    def __init__(self, env, agent, logger=None, monitor=None):
        """
        初始化回合运行器。
        Initialize episode runner.

        Args:
            env: 环境对象 (Simple2DDeliveryEnv) / Environment object
            agent: Q-learning Agent 实例 / Q-learning Agent instance
            logger: 日志记录器 (可选) / Logger instance (optional)
            monitor: 监控器 (可选) / Monitor instance (optional)
        """
        self.env = env
        self.agent = agent
        self.logger = logger
        self.monitor = monitor

        # 训练统计 / Training statistics
        self.episode_rewards: List[float] = []
        self.episode_deliveries: List[int] = []
        self.episode_steps: List[int] = []
        self.best_avg_reward = -float("inf")

    def run_episode(self, seed: Optional[int] = None) -> Dict[str, float]:
        """
        执行一个完整的Q-learning训练回合。
        Run one complete Q-learning training episode.

        标准Q-learning循环 / Standard Q-learning loop::

            env_obs, info = env.reset()
            agent.reset(env_obs)
            obs_data = agent.observation_process(env_obs)

            while not done:
                act_data = agent.predict([obs_data])      # epsilon-greedy
                act = agent.action_process(act_data)
                _, reward, terminated, truncated, env_obs = env.step(act)
                shaped_reward = agent.reward_shaping(reward)
                next_obs_data = agent.observation_process(env_obs)
                done = terminated or truncated

                collector.add(obs=obs_data, _obs=next_obs_data,
                              act=act, rew=shaped_reward, done=done)
                obs_data = next_obs_data

            collector.process_last_frame()
            agent.learn(collector.get_game_data())         # Q-learning update

        Args:
            seed: 环境随机种子 / Environment random seed (None for automatic)

        Returns:
            回合统计字典 / Episode statistics dict:
                - reward: 累计奖励 / Cumulative reward
                - deliveries: 配送完成数 / Deliveries completed
                - steps: 回合步数 / Steps taken
                - epsilon: 当前探索率 / Current epsilon
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

        # ---- 样本收集器 / Sample collector ----
        collector = SampleManager()

        # ---- 回合统计 / Episode statistics ----
        episode_reward = 0.0
        episode_steps = 0
        done = False

        # ============================================================
        # Q-learning主循环 / Q-learning main loop
        # ============================================================
        while not done:
            # 动作选择 (epsilon-greedy) / Action selection
            act_data = self.agent.predict([obs_data])
            act = self.agent.action_process(act_data)

            # 环境步进 / Environment step
            # step() 返回: (obs, reward, terminated, truncated, info)
            env_obs, reward, terminated, truncated, info = self.env.step(act)

            # 奖励塑形 / Reward shaping
            shaped_reward = self.agent.reward_shaping(reward)

            # 下一观测处理 / Next observation processing
            next_obs_data = self.agent.observation_process(env_obs)

            # 终止判断 / Termination check
            done = terminated or truncated

            # 收集样本 / Collect sample
            collector.add(
                obs=obs_data,
                _obs=next_obs_data,
                act=act,
                rew=shaped_reward,
                done=done,
            )

            # 更新状态 / Update state
            obs_data = next_obs_data
            episode_reward += shaped_reward
            episode_steps += 1

            # 步数超限安全防护 / Step limit safety guard
            if episode_steps >= Config.MAX_STEPS and not done:
                done = True

        # ============================================================
        # 学习更新 / Learning update
        # ============================================================
        collector.process_last_frame()
        train_data = collector.get_game_data()

        if train_data and len(train_data) > 0:
            self.agent.learn(train_data)

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
        }

    def evaluate(self, num_episodes: int = 10) -> Dict[str, float]:
        """
        执行贪婪评估 (纯利用，不更新Q表)。
        Run greedy evaluation (pure exploitation, no Q-table updates).

        评估期间不进行探索，使用贪心策略选择动作，不更新Q表。
        No exploration during evaluation; greedy action selection only;
        Q-table is NOT updated.

        Args:
            num_episodes: 评估回合数 / Number of evaluation episodes

        Returns:
            评估指标字典 / Evaluation metrics dict:
                - avg_reward: 平均累计奖励 / Average total reward
                - avg_deliveries: 平均配送完成数 / Average deliveries per episode
                - avg_steps: 平均回合步数 / Average episode steps
                - success_rate: 全部5单完成占比 / Proportion of full completions
        """
        total_reward = 0.0
        total_deliveries = 0
        total_steps = 0
        successes = 0

        # 保存训练epsilon / Save training epsilon
        saved_epsilon = self.agent.epsilon

        for ep in range(num_episodes):
            # 环境重置 / Environment reset
            env_obs, info = self.env.reset(seed=10000 + ep)
            done = False

            while not done:
                # 贪心动作选择 (无探索) / Greedy action selection (no exploration)
                action = self.agent.exploit(env_obs)

                # 环境步进 / Environment step
                # step() 返回: (obs, reward, terminated, truncated, info)
                env_obs, reward, terminated, truncated, info = self.env.step(action)
                total_reward += float(reward)
                total_steps += 1
                done = terminated or truncated

                # 步数安全防护 / Step limit guard
                if total_steps % Config.MAX_STEPS == 0 and not done:
                    done = True

            # 统计 / Statistics
            deliveries = self._count_deliveries(env_obs)
            total_deliveries += deliveries
            if deliveries >= Config.NUM_ORDERS:
                successes += 1

        # 恢复epsilon / Restore epsilon (not needed since exploit doesn't change it,
        # but the agent's predict might have side effects)
        self.agent.epsilon = saved_epsilon

        return {
            "avg_reward": total_reward / num_episodes,
            "avg_deliveries": total_deliveries / num_episodes,
            "avg_steps": total_steps / num_episodes,
            "success_rate": successes / num_episodes,
        }

    def get_recent_avg_reward(self, window: int = 100) -> float:
        """
        获取最近N回合的平均奖励。
        Get average reward over the last N episodes.

        Args:
            window: 滑动窗口大小 / Sliding window size

        Returns:
            最近N回合的平均奖励 / Average reward of last N episodes
        """
        if len(self.episode_rewards) < window:
            return (
                float(np.mean(self.episode_rewards))
                if self.episode_rewards
                else 0.0
            )
        return float(np.mean(self.episode_rewards[-window:]))

    # ============================================================
    # 辅助方法 (Helper)
    # ============================================================

    @staticmethod
    def _count_deliveries(env_obs: Dict[str, Any]) -> int:
        """
        从环境观测中提取已完成的配送数量。
        Extract number of completed deliveries from environment observation.

        处理 Simple2DDeliveryEnv 的观测格式。
        Handles Simple2DDeliveryEnv observation format.

        Args:
            env_obs: 环境观测字典 / Environment observation dict

        Returns:
            完成的配送数量 (0-5) / Number of completed deliveries
        """
        # 解包可能的 {"observation": {...}} 包装 / Unwrap possible wrapper
        obs = env_obs.get("observation", env_obs) if isinstance(env_obs, dict) else {}

        order_status = obs.get("order_status", None)
        if order_status is None:
            return 0

        # 计算完成的订单数 (状态位为1的个数) / Count completed orders
        if isinstance(order_status, np.ndarray):
            return int(order_status.sum())
        if isinstance(order_status, (list, tuple)):
            return sum(int(s) for s in order_status)
        return 0
