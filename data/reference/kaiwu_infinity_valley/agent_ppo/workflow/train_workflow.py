#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import time
import os
import numpy as np
from agent_ppo.feature.definition import (
    SampleManager,
    reward_shaping,
)
from tools.train_env_conf_validate import read_usr_conf
from tools.metrics_utils import get_training_metrics
from common_python.utils.workflow_disaster_recovery import handle_disaster_recovery


def workflow(envs, agents, logger=None, monitor=None, *args, **kwargs):
    last_save_model_time = time.time()
    env = envs[0]
    agent = agents[0]

    # Read and validate configuration file
    # 配置文件读取和校验
    usr_conf = read_usr_conf("agent_ppo/conf/train_env_conf.toml", logger)
    if usr_conf is None:
        logger.error(f"usr_conf is None, please check agent_ppo/conf/train_env_conf.toml")
        return

    # Create EpisodeRunner instance
    # 创建 EpisodeRunner 实例
    episode_runner = EpisodeRunner(
        env=env,
        agent=agent,
        usr_conf=usr_conf,
        logger=logger,
        monitor=monitor,
    )

    while True:
        # Run episodes and collect data
        # 运行对局并收集数据
        for training_data in episode_runner.run_episodes():
            agent.send_sample_data(training_data)
            training_data.clear()

            # save model file
            # 保存model文件
            now = time.time()
            if now - last_save_model_time >= 1800:
                agent.save_model()
                last_save_model_time = now


class EpisodeRunner:
    def __init__(self, env, agent, usr_conf, logger, monitor):
        self.env = env
        self.agent = agent
        self.usr_conf = usr_conf
        self.logger = logger
        self.monitor = monitor
        self.episode_cnt = 0
        self.last_report_monitor_time = 0
        self.last_get_training_metrics_time = 0

    def run_episodes(self):
        # Single episode process
        # 单局流程
        while True:
            # Retrieving training metrics
            # 获取训练中的指标
            now = time.time()
            if now - self.last_get_training_metrics_time >= 60:
                training_metrics = get_training_metrics()
                self.last_get_training_metrics_time = now
                if training_metrics is not None:
                    self.logger.info(f"training_metrics is {training_metrics}")

            # Reset the environment and get the initial state
            # 重置环境, 并获取初始状态
            env_obs = self.env.reset(usr_conf=self.usr_conf)

            # Disaster recovery
            # 容灾
            if handle_disaster_recovery(env_obs, self.logger):
                continue

            # At the start of each episode, support loading the latest model file
            # The call will load the latest model from a remote training node
            # 每次对局开始时, 支持加载最新model文件, 该调用会从远程的训练节点加载最新模型
            self.agent.reset(env_obs)
            self.agent.load_model(id="latest")

            # Feature processing
            # 特征处理
            obs_data, remain_info = self.agent.observation_process(env_obs)
            obs = env_obs["observation"]

            # Initialize sample collector
            # 初始化样本收集器
            collector = SampleManager()

            # Episode variables
            # 对局变量
            self.episode_cnt += 1
            done = False
            step = 0
            bump_cnt = 0
            total_reward = 0
            total_reward_list = [0] * 6

            self.logger.info(f"Episode {self.episode_cnt} start, usr_conf is {self.usr_conf}")

            while not done:
                # Agent performs inference, gets the predicted action for the next frame
                # Agent 进行推理, 获取下一帧的预测动作
                act_data = self.agent.predict(list_obs_data=[obs_data])
                act_data = act_data[0]

                # Unpack ActData into action
                # ActData 解包成动作
                act = self.agent.action_process(act_data)

                # Interact with the environment, execute actions, get the next state
                # 与环境交互, 执行动作, 获取下一步的状态
                env_reward, env_obs = self.env.step(act)

                # Disaster recovery
                # 容灾
                if handle_disaster_recovery(env_obs, self.logger):
                    break

                frame_no = env_obs["frame_no"]
                _obs = env_obs["observation"]
                terminated = env_obs["terminated"]
                truncated = env_obs["truncated"]
                step += 1

                # Feature processing
                # 特征处理
                _obs_data, _remain_info = self.agent.observation_process(env_obs)
                _obs_data.frame_no = frame_no

                total_score = _obs["env_info"]["total_score"]

                # Calculate reward
                # 计算 reward
                if _obs is None:
                    reward = 0
                else:
                    (reward, reward_list, is_bump,) = reward_shaping(
                        frame_no,
                        total_score,
                        terminated,
                        truncated,
                        remain_info,
                        _remain_info,
                        obs,
                        _obs,
                    )
                    for i in range(len(reward_list)):
                        total_reward_list[i] += reward_list[i]

                    # Wall bump behavior statistics
                    # 撞墙行为统计
                    bump_cnt += is_bump

                done = terminated or truncated

                # Determine environment over
                # 判断环境结束
                if done:
                    self.logger.info(f"Episode ended at step {step}, with a total score of {total_score}.")
                    # Add a penalty for the termination of the environment.
                    # 增加环境结束的惩罚
                    reward -= min((self.usr_conf["env_conf"]["max_step"] - step) / 100, 10)

                # Construct sample
                # 构造样本
                total_reward += reward
                prob, value, action = act_data.prob, act_data.value, act_data.action
                obs_feature = np.array(obs_data.feature)
                collector.add(
                    obs=obs_feature,
                    legal_action=np.array(obs_data.legal_action),
                    sub_action=np.array(obs_data.sub_action_mask).flatten()[0]
                    if hasattr(obs_data, "sub_action_mask")
                    else 0,
                    action=action,
                    prob=prob,
                    reward=reward,
                    reward_sum=0,
                    advantage=0,
                    value=value.flatten()[0],
                    next_value=0,
                    done=done,
                    is_train=1,
                )

                # If the environment is over, the sample is processed and sent to training
                # 如果环境结束，则进行样本处理，将样本送去训练
                if done:
                    now = time.time()
                    if now - self.last_report_monitor_time >= 60 and self.monitor:
                        monitor_data = {
                            "reward": total_reward,
                            "episode_cnt": self.episode_cnt,
                        }
                        self.monitor.put_data({os.getpid(): monitor_data})
                        self.last_report_monitor_time = now

                    if collector.count > 0:
                        collector.process_last_frame()
                        training_data = collector.get_game_data()
                        yield training_data
                    break

                # Status update
                # 状态更新
                obs_data = _obs_data
                remain_info = _remain_info
