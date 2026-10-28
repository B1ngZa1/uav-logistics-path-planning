#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
================================================================================
workflow/train_workflow.py -- A2C Training Workflow / A2C 训练工作流
================================================================================

Responsibility / 职责:
    Implements standard A2C training loop with on-policy batch learning.
    实现标准 A2C 训练循环, 包含同策略批量学习:
    - EpisodeRunner class: manages single-episode env interaction and data collection
    - workflow() function: top-level training entry point, drives multi-episode training
    - A2C: collect full episode trajectory, compute returns, learn at episode end
    - A2C 特性: 收集完整 episode 轨迹, 计算回报, 回合末尾执行一次批量学习

Reference / 参考:
    - kaiwu_infinity_valley/agent_a2c/workflow/train_workflow.py
    - agent_q_learning/workflow/train_workflow.py (EpisodeRunner pattern)
    - Mnih et al., "Asynchronous Methods for Deep RL", ICML 2016

Domain Adaptation / 领域适配:
    - Random2DDeliveryEnv environment (Level 2: 2D random merchants)
    - A2C on-policy: collect full trajectory per episode, learn once at episode end
    - Smoke test compatible agent interface

Kaiwu Integration / Kaiwu 集成:
    - Exported as agent_a2c.workflow.train_workflow.workflow per algo_conf TOML
    - Signature: workflow(envs, agents, logger=None, monitor=None, *args, **kwargs)
================================================================================
"""

import os
import time
import logging
from typing import Any, Dict, List, Optional

import numpy as np

from agent_a2c.agent import Agent
from agent_a2c.conf.conf import Config
from agent_a2c.feature.definition import SampleData
from agent_a2c.conf.monitor_builder import monitor_config as mc

logger = logging.getLogger("train_workflow")
logger.setLevel(logging.INFO)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(h)


# =============================================================================
# EpisodeRunner -- Single Episode Controller / 单回合控制器
# =============================================================================

class EpisodeRunner:
    """
    A2C Single Episode Controller / A2C 单回合控制器

    Responsibilities / 职责:
        1. Reset env and get initial obs / 重置环境并获取初始观测
        2. Agent predicts action -> env executes -> collect trajectory data
        3. At episode end: compute discounted returns -> create SampleData -> learn
        4. 回合终止时: 计算折扣回报 -> 创建 SampleData -> 批量学习

    Usage / 用法::

        runner = EpisodeRunner(env, agent)
        stats = runner.run_episode(seed=42)
    """

    def __init__(self, env, agent: Agent, max_steps: int = Config.MAX_STEPS) -> None:
        """
        Initialize episode runner. / 初始化回合控制器。

        Args / 参数:
            env: Random2DDeliveryEnv instance / 环境实例
            agent: A2C Agent instance / A2C Agent 实例
            max_steps: Max steps per episode / 单回合最大步数
        """
        self._env = env
        self._agent = agent
        self._max_steps = max_steps

    def run_episode(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Run a complete training episode. / 运行一个完整的训练回合。

        A2C Process / A2C 流程:
            1. Reset env and agent / 重置环境和 Agent
            2. Step-by-step: predict -> env.step -> collect transition
            3. At episode end: compute returns -> create SampleData -> agent.learn()
            4. 回合结束时: 计算回报 -> 构造样本 -> agent.learn()

        Args / 参数:
            seed: Random seed / 随机种子

        Returns / 返回:
            Dict with: episode_reward, episode_length, total_orders, completed,
                       on_time, failed, completion_rate, on_time_rate,
                       loss, policy_loss, value_loss, entropy, gradient_norm
        """
        # ---- Reset / 重置 ----
        obs, info = self._env.reset(seed=seed)
        self._agent.reset(obs)

        # ---- Trajectory buffers / 轨迹缓冲区 ----
        obs_list: List[np.ndarray] = []
        next_obs_list: List[np.ndarray] = []
        action_list: List[int] = []
        reward_list: List[float] = []
        done_list: List[bool] = []
        legal_act_list: List[np.ndarray] = []
        log_prob_list: List[float] = []
        value_list: List[float] = []

        episode_reward: float = 0.0
        last_info = info

        # ---- Episode loop / 回合循环 ----
        for step_idx in range(self._max_steps):
            # Step 1: Agent processes observation and predicts action
            obs_data = self._agent.observation_process(obs)
            act_data = self._agent.predict([obs_data])
            action = self._agent.action_process(act_data)

            # Step 2: Environment executes action
            next_obs, reward, terminated, truncated, info = self._env.step(action)
            shaped_reward = self._agent.reward_shaping(reward, obs)

            # Step 3: Collect transition data
            obs_list.append(obs_data.feature)
            next_obs_list.append(
                self._agent.observation_process(next_obs).feature
            )
            action_list.append(action)
            reward_list.append(shaped_reward)
            done_list.append(terminated or truncated)
            legal_act_list.append(obs_data.legal_act)
            log_prob_list.append(act_data.log_prob)
            value_list.append(act_data.val)

            episode_reward += shaped_reward
            last_info = info

            # Step 4: Advance observation
            obs = next_obs

            # Step 5: Check termination
            if terminated or truncated:
                break

        ep_length = step_idx + 1

        # ---- Compute discounted returns / 计算折扣回报 ----
        returns = self._compute_returns(reward_list, done_list, Config.GAMMA)

        # ---- Build SampleData list / 构造样本列表 ----
        samples: List[SampleData] = []
        for i in range(len(obs_list)):
            samples.append(SampleData(
                obs=obs_list[i],
                act=action_list[i],
                rew=reward_list[i],
                next_obs=next_obs_list[i],
                done=done_list[i],
                legal_act=legal_act_list[i],
                log_prob=log_prob_list[i],
                val=value_list[i],
            ))

        # ---- A2C batch learning / A2C批量学习 ----
        learn_stats = self._agent.learn(samples)

        # ---- Build statistics / 汇总统计 ----
        total_orders = last_info.get("total_orders", 0)
        completed = last_info.get("completed", 0)
        on_time = last_info.get("on_time", 0)
        failed = last_info.get("failed", 0)

        stats = {
            "episode_reward": float(episode_reward),
            "episode_length": int(ep_length),
            "total_orders": int(total_orders),
            "completed": int(completed),
            "on_time": int(on_time),
            "failed": int(failed),
            "completion_rate": float(completed / total_orders) if total_orders > 0 else 0.0,
            "on_time_rate": float(on_time / total_orders) if total_orders > 0 else 0.0,
            "loss": float(learn_stats.get("loss", 0.0)),
            "policy_loss": float(learn_stats.get("policy_loss", 0.0)),
            "value_loss": float(learn_stats.get("value_loss", 0.0)),
            "entropy": float(learn_stats.get("entropy", 0.0)),
            "gradient_norm": float(learn_stats.get("grad_norm", 0.0)),
        }

        return stats

    @staticmethod
    def _compute_returns(rewards: List[float], dones: List[bool],
                         gamma: float) -> List[float]:
        """
        Compute n-step discounted returns backward.
        向后递推计算 n-step 折扣回报。

        G_t = r_t + gamma * G_{t+1}
        When done_t = True, reset cumulative return to 0.
        当 done_t = True 时, 重置累计回报为 0。

        Args / 参数:
            rewards: Reward list (T,) / 奖励列表
            dones: Done flags (T,) / 终止标记
            gamma: Discount factor / 折扣因子

        Returns / 返回:
            Discounted returns (T,) / 折扣回报列表
        """
        T = len(rewards)
        rets = [0.0] * T
        R = 0.0
        for t in reversed(range(T)):
            if dones[t]:
                R = 0.0
            R = rewards[t] + gamma * R
            rets[t] = R
        return rets


# =============================================================================
# workflow -- Top-Level Training Entry / 顶层训练入口
# =============================================================================

def workflow(
    envs=None,
    agents=None,
    logger=None,
    monitor=None,
    *args,
    **kwargs,
) -> Agent:
    """
    A2C Training Workflow / A2C 训练工作流

    Kaiwu-compatible entry point. Signature matches algo_conf_drone_delivery.toml:
        train_workflow = "agent_a2c.workflow.train_workflow.workflow"

    流程 / Process:
        1. Extract env and agent from envs/agents lists
           从 envs/agents 列表中提取环境和 Agent
        2. Create EpisodeRunner / 创建 EpisodeRunner
        3. Loop: run episodes -> log stats -> periodic model save
           循环: 运行回合 -> 记录统计 -> 定期保存模型

    Args / 参数:
        envs: List of environment instances / 环境实例列表
        agents: List of Agent instances / Agent 实例列表
        logger: Logger instance (optional) / 日志记录器 (可选)
        monitor: Monitor instance (optional) / 监控实例 (可选)

    Returns / 返回:
        Agent: Trained agent instance / 训练完成的 Agent
    """
    # Extract env and agent from Kaiwu-style args
    env = envs[0] if envs else None
    agent = agents[0] if agents else None

    if env is None:
        raise ValueError("No environment provided. Pass envs=[env] to workflow().")

    if agent is None:
        agent = Agent(device=Config.DEVICE)

    num_episodes = kwargs.get("num_episodes", Config.NUM_EPISODES)
    save_dir = kwargs.get("save_dir", "outputs/models")
    log_interval = kwargs.get("log_interval", 20)
    seed = kwargs.get("seed", 42)

    os.makedirs(save_dir, exist_ok=True)

    runner = EpisodeRunner(env=env, agent=agent, max_steps=Config.MAX_STEPS)

    # Training history / 训练历史
    history: Dict[str, List[float]] = {
        "episode_rewards": [],
        "episode_lengths": [],
        "completion_rates": [],
        "on_time_rates": [],
        "losses": [],
        "policy_losses": [],
        "value_losses": [],
    }

    _log = logger.info if logger else print
    _log("=" * 60)
    _log("A2C Training Workflow Started / A2C 训练开始")
    _log(f"  Episodes: {num_episodes}  |  State Dim: {Config.STATE_DIM}")
    _log(f"  Actions: {Config.ACTION_SIZE}  |  Hidden Dim: {Config.HIDDEN_DIM}")
    _log(f"  Gamma: {Config.GAMMA}  |  LR: {Config.LR}")
    _log(f"  Entropy Coef: {Config.ENTROPY_COEF}  |  Value Coef: {Config.VALUE_COEF}")
    _log(f"  Max Grad Norm: {Config.MAX_GRAD_NORM}")
    _log("=" * 60)

    start_time = time.time()
    last_save_time = start_time
    last_report_time = start_time

    for ep in range(1, num_episodes + 1):
        ep_seed = seed + ep if seed is not None else None
        stats = runner.run_episode(seed=ep_seed)

        # Record history / 记录历史
        history["episode_rewards"].append(stats["episode_reward"])
        history["episode_lengths"].append(stats["episode_length"])
        history["completion_rates"].append(stats.get("completion_rate", 0.0))
        history["on_time_rates"].append(stats.get("on_time_rate", 0.0))
        history["losses"].append(stats["loss"])
        history["policy_losses"].append(stats["policy_loss"])
        history["value_losses"].append(stats["value_loss"])

        # Periodic logging / 定期日志
        if ep % log_interval == 0 or ep == 1:
            elapsed = time.time() - start_time
            recent_r = history["episode_rewards"][-log_interval:]
            avg_reward = np.mean(recent_r) if recent_r else 0.0

            recent_c = history["completion_rates"][-log_interval:]
            avg_c = np.mean(recent_c) if recent_c else 0.0

            recent_l = history["losses"][-log_interval:]
            avg_loss = np.mean(recent_l) if recent_l else 0.0

            _log(
                f"Ep {ep:>5d}/{num_episodes} | "
                f"Reward: {avg_reward:>8.2f} | "
                f"Completion: {avg_c:>6.2%} | "
                f"Orders: {stats['completed']}/{stats['total_orders']} | "
                f"Loss: {avg_loss:.4f} | "
                f"PLoss: {stats['policy_loss']:.4f} | "
                f"VLoss: {stats['value_loss']:.4f} | "
                f"Entropy: {stats['entropy']:.4f} | "
                f"Time: {elapsed:.1f}s"
            )

        # Monitor reporting (Kaiwu style) / 监控上报
        now = time.time()
        if monitor is not None and (now - last_report_time >= 60):
            try:
                monitor.put_data({os.getpid(): {
                    "reward": stats["episode_reward"],
                    "policy_loss": stats["policy_loss"],
                    "value_loss": stats["value_loss"],
                    "entropy": stats["entropy"],
                    "gradient_norm": stats["gradient_norm"],
                }})
            except Exception:
                pass
            last_report_time = now

        # Periodic model save / 定期保存模型
        if now - last_save_time >= 1800:  # every 30 minutes
            agent.save_model(save_dir, id=f"ep{ep}")
            _log(f"  Model saved at episode {ep}")
            last_save_time = now

    # ---- Training complete / 训练完成 ----
    total_time = time.time() - start_time
    final_path = os.path.join(save_dir, "model.ckpt-latest.pkl")
    agent.save_model(save_dir, id="latest")

    _log("=" * 60)
    _log(f"A2C Training Complete / A2C 训练完成 -- {total_time:.1f}s")
    _log(f"  Total train steps: {agent.train_steps}")

    last_100 = history["episode_rewards"][-100:]
    if last_100:
        _log(f"  Avg Reward (last 100): {np.mean(last_100):.2f}")

    last_100_c = history["completion_rates"][-100:]
    if last_100_c:
        _log(f"  Avg Completion (last 100): {np.mean(last_100_c):.2%}")

    _log(f"  Final model: {final_path}")
    _log("=" * 60)

    return agent


# =============================================================================
# CLI Entry Point / 命令行入口
# =============================================================================

if __name__ == "__main__":
    """
    Direct training run / 可直接运行训练.

    Usage / 用法::
        python -m agent_a2c.workflow.train_workflow
    """
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    try:
        from environment.random_delivery_env import Random2DDeliveryEnv
        env = Random2DDeliveryEnv()
        agent = Agent()
        trained = workflow(envs=[env], agents=[agent], num_episodes=Config.NUM_EPISODES)
        print("A2C Training finished successfully! / A2C 训练完成!")
    except ImportError as e:
        logger.warning(f"Cannot import env: {e}. Use workflow(envs, agents) from external script.")
