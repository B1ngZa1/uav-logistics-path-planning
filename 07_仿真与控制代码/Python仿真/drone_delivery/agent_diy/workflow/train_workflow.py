# -*- coding: utf-8 -*-
"""
================================================================================
workflow/train_workflow.py — 学生DIY训练工作流 / Student DIY Training Workflow
================================================================================
职责 / Responsibility:
    学生自行实现的训练工作流骨架。
    驱动智能体在无人机配送环境中进行训练和评估。
    Skeleton for a training workflow to be implemented by students.
    Drives the agent to train and evaluate in the drone delivery environment.

参考 / Reference:
    - agent_dqn/workflow/train_workflow.py (DQN 完整工作流)
    - agent_q_learning/workflow/train_workflow.py (Q-learning 完整工作流)
    - agent_monte_carlo/workflow/train_workflow.py (Monte Carlo 完整工作流)

领域适配 / Domain Adaptation:
    - 无人机配送环境: step() 返回 (obs, reward, terminated, truncated, info)
    - Drone delivery env: step() returns (obs, reward, terminated, truncated, info)

典型训练循环 / Typical Training Loop::

    for episode in range(num_episodes):
        env_obs, info = env.reset()
        agent.reset(env_obs)
        obs_data = agent.observation_process(env_obs)

        while not done:
            act_data = agent.predict([obs_data])         # 动作选择
            act = agent.action_process(act_data)
            env_obs, reward, terminated, truncated, _ = env.step(act)
            shaped_reward = agent.reward_shaping(reward)
            next_obs_data = agent.observation_process(env_obs)
            done = terminated or truncated

            # 收集样本 (根据算法, 可能每步或每回合学习)
            # Collect samples (learn per step or per episode depending on algorithm)
            collector.add(obs=obs_data, _obs=next_obs_data,
                          act=act, rew=shaped_reward, done=done, ret=0.0)
            obs_data = next_obs_data

        collector.process_last_frame()
        agent.learn(collector.get_game_data())   # 学习更新
        yield stats                               # 产出训练统计

提示 / Tips:
    - DQN类方法: 每步学习 (异策略/off-policy, 使用 replay buffer)
      DQN-style: learn every step (off-policy, use replay buffer)
    - 策略梯度: 每回合学习 (同策略/on-policy, 收集完整轨迹)
      Policy gradient: learn per episode (on-policy, collect full trajectory)
================================================================================
"""

import time
from typing import Dict, Any, Optional, List

import numpy as np

from agent_diy.conf.conf import Config
from agent_diy.feature.definition import SampleManager


def workflow(envs, agents, logger=None, monitor=None):
    """
    学生自行实现的训练工作流 / Training workflow to be implemented by students

    遵循 Kaiwu 框架的 workflow() 函数签名。
    Follows Kaiwu framework's workflow() function signature.

    TODO: 实现训练循环 / Implement training loop

    Args / 参数:
        envs: 环境对象或环境列表 / Environment object or list of envs
        agents: 智能体对象或智能体列表 / Agent object or list of agents
        logger: 日志记录器 (可选) / Logger instance (optional)
        monitor: 监控器 (可选) / Monitor instance (optional)

    Yields / 产出:
        dict: 每个回合的训练统计 / Per-episode training statistics
    """
    # ============================================================
    # 1. 输入验证 / Input Validation
    # ============================================================
    if len(envs) == 0 or len(agents) == 0:
        raise ValueError("envs and agents must not be empty")

    env = envs[0] if isinstance(envs, (list, tuple)) else envs
    agent = agents[0] if isinstance(agents, (list, tuple)) else agents

    # ============================================================
    # 2. 训练参数 / Training Parameters
    # ============================================================
    # TODO: 从 Config 读取训练参数 / Read training params from Config
    num_episodes = Config.NUM_EPISODES
    max_steps = Config.MAX_STEPS
    eval_interval = Config.EVAL_INTERVAL

    # ============================================================
    # 3. 训练循环 / Training Loop
    # ============================================================
    # TODO: 在此实现完整的训练循环 / Implement full training loop here
    #
    # 建议结构 / Suggested structure:
    #   for ep in range(num_episodes):
    #       # ---- 环境重置 / Environment reset ----
    #       env_obs, info = env.reset()
    #       agent.reset(env_obs)
    #       obs_data = agent.observation_process(env_obs)
    #
    #       # ---- 样本收集器 / Sample collector ----
    #       collector = SampleManager()
    #       episode_reward = 0.0
    #
    #       # ---- 回合循环 / Episode loop ----
    #       for step in range(max_steps):
    #           # 动作选择 / Action selection
    #           act_data = agent.predict([obs_data])
    #           act = agent.action_process(act_data)
    #
    #           # 环境步进 / Environment step
    #           env_obs, reward, terminated, truncated, info = env.step(act)
    #           shaped_reward = agent.reward_shaping(reward)
    #           done = terminated or truncated
    #
    #           # 下一观测处理 / Next observation processing
    #           next_obs_data = agent.observation_process(env_obs)
    #
    #           # 收集样本 / Collect sample
    #           collector.add(obs=obs_data, _obs=next_obs_data,
    #                         act=act, rew=shaped_reward, done=done, ret=0.0)
    #
    #           obs_data = next_obs_data
    #           episode_reward += shaped_reward
    #
    #           if done:
    #               break
    #
    #       # ---- 学习更新 / Learning update ----
    #       collector.process_last_frame()
    #       train_data = collector.get_game_data()
    #       if train_data:
    #           agent.learn(train_data)
    #
    #       # ---- 定期评估与日志 / Periodic eval & logging ----
    #       if (ep + 1) % eval_interval == 0:
    #           # ... 评估逻辑 ...
    #           pass
    #
    #       # ---- 产出统计 / Yield statistics ----
    #       yield {"reward": episode_reward, "episode": ep, ...}

    # 占位符实现 / Placeholder implementation:
    if logger:
        logger.info(
            f"[DIY] Starting training: {num_episodes} episodes "
            f"(placeholder — students need to implement the training loop)"
        )
    else:
        print(
            f"[DIY] Starting training: {num_episodes} episodes "
            f"(placeholder — students need to implement the training loop)"
        )

    start_time = time.time()

    # TODO: 实现训练循环 / Implement training loop
    # 以下为占位符循环, 学生需替换为完整实现
    # Below is a placeholder loop; students should replace with full implementation
    for ep in range(num_episodes):
        # ---- 环境重置 / Environment reset ----
        env_obs, _ = env.reset()
        agent.reset(env_obs)

        # ---- 回合循环 (占位符) / Episode loop (placeholder) ----
        done = False
        episode_reward = 0.0
        while not done:
            # TODO: 实现完整的回合循环 / Implement full episode loop
            obs_data = agent.observation_process(env_obs)
            act_data = agent.predict([obs_data])
            act = agent.action_process(act_data)
            env_obs, reward, terminated, truncated, _ = env.step(act)
            shaped_reward = agent.reward_shaping(reward)
            done = terminated or truncated
            episode_reward += shaped_reward
            break  # 占位符: 每回合只跑一步 / Placeholder: only one step per episode

        # ---- 产出统计 / Yield statistics ----
        yield {
            "episode": ep,
            "reward": episode_reward,
            "status": "TODO: implement full training loop",
        }

    elapsed = time.time() - start_time
    if logger:
        logger.info(f"[DIY] Training placeholder completed in {elapsed:.1f}s")
    else:
        print(f"[DIY] Training placeholder completed in {elapsed:.1f}s")
