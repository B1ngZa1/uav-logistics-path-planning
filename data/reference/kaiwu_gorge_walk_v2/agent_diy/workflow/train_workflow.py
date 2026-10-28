#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import os
import time

from common_python.utils.common_func import Frame
from common_python.utils.workflow_disaster_recovery import handle_disaster_recovery
from tools.metrics_utils import get_training_metrics
from tools.train_env_conf_validate import read_usr_conf

from agent_diy.conf.conf import Config
from agent_diy.feature.definition import reward_shaping, sample_process


def _cfg(name, default):
    return getattr(Config, name, default)


def workflow(envs, agents, logger=None, monitor=None, *args, **kwargs):
    try:
        usr_conf = read_usr_conf("agent_diy/conf/train_env_conf.toml", logger)
        if usr_conf is None:
            logger.error("usr_conf is None, please check agent_diy/conf/train_env_conf.toml")
            return

        env, agent = envs[0], agents[0]
        monitor_data = {"reward": 0, "loss": 0, "epsilon": agent.epsilon, "win_rate": 0}
        last_report_monitor_time = 0
        last_get_training_metrics_time = 0
        last_save_model_time = time.time()

        total_reward = 0.0
        episode_count = 0
        win_count = 0
        latest_loss = 0.0

        logger.info("Start DIY DQN training...")
        start_time = time.time()

        episodes = _cfg("EPISODES", 10000)
        save_model_interval = _cfg("SAVE_MODEL_INTERVAL", 300)

        for episode in range(episodes):
            now = time.time()
            if now - last_get_training_metrics_time > 15:
                last_get_training_metrics_time = now
                training_metrics = get_training_metrics()
                if training_metrics:
                    logger.info(f"training_metrics is {training_metrics}")

            env_obs = env.reset(usr_conf=usr_conf)
            if handle_disaster_recovery(env_obs, logger):
                continue

            obs_data = agent.observation_process(env_obs)
            done = False
            episode_reward = 0.0

            while not done:
                act_data = agent.predict(list_obs_data=[obs_data])[0]
                current_action = agent.action_process(act_data)

                next_env_reward, next_env_obs = env.step(current_action)
                if handle_disaster_recovery(next_env_obs, logger):
                    break

                terminated = bool(next_env_obs["terminated"])
                truncated = bool(next_env_obs["truncated"])
                done = terminated or truncated

                agent.record_transition(env_obs, current_action, next_env_obs)
                next_obs_data = agent.observation_process(next_env_obs)
                reward = reward_shaping(next_env_reward, next_env_obs, prev_env_obs=env_obs, action=current_action)

                sample = Frame(
                    state=obs_data.feature,
                    action=current_action,
                    reward=reward,
                    next_state=next_obs_data.feature,
                    done=done,
                )
                learn_result = agent.learn(sample_process([sample]))
                if learn_result:
                    latest_loss = float(learn_result.get("loss", latest_loss))

                agent.decay_epsilon()
                episode_reward += reward
                total_reward += reward
                env_obs = next_env_obs
                obs_data = next_obs_data

                if terminated:
                    win_count += 1

            episode_count += 1
            now = time.time()
            win_rate = win_count / float(episode + 1)

            if now - last_report_monitor_time > 15:
                avg_reward = total_reward / max(episode_count, 1)
                logger.info(
                    f"Episode: {episode + 1}, Episode Reward: {episode_reward:.3f}, "
                    f"Avg Reward: {avg_reward:.3f}, Loss: {latest_loss:.6f}, "
                    f"Epsilon: {agent.epsilon:.4f}, Win Rate: {win_rate:.3f}"
                )
                monitor_data.update(
                    {
                        "reward": avg_reward,
                        "loss": latest_loss,
                        "epsilon": agent.epsilon,
                        "win_rate": win_rate,
                    }
                )
                if monitor:
                    monitor.put_data({os.getpid(): monitor_data})
                total_reward = 0.0
                episode_count = 0
                last_report_monitor_time = now

            if now - last_save_model_time > save_model_interval:
                logger.info(f"Saving Model at Episode: {episode + 1}")
                agent.save_model(id=str(episode + 1))
                last_save_model_time = now

            if win_rate > 0.9 and episode > 200:
                logger.info(f"DIY DQN training converged at episode {episode + 1}")
                break

        logger.info(f"Training Time for {episode + 1} episodes: {time.time() - start_time} s")
        agent.save_model(id=str(episode + 1))

    except Exception as e:
        raise RuntimeError(f"workflow error: {e}")
