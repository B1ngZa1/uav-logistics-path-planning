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
from collections import deque
from agent_q_learning.feature.definition import sample_process, reward_shaping
from tools.train_env_conf_validate import read_usr_conf
from tools.metrics_utils import get_training_metrics
from common_python.utils.workflow_disaster_recovery import handle_disaster_recovery


def workflow(envs, agents, logger=None, monitor=None, *args, **kwargs):
    """
    Training workflow for improved Q-Learning agent with:
    - Proper epsilon decay across episodes (NOT reset each episode)
    - Experience replay with Double Q-Learning
    - Learning rate scheduling
    - Validation episodes every 50 training episodes
    - Enhanced convergence criteria with treasure tracking
    - Rate-limited model saving
    """
    try:
        usr_conf = read_usr_conf("agent_q_learning/conf/train_env_conf.toml", logger)
        if usr_conf is None:
            logger.error("usr_conf is None, please check agent_q_learning/conf/train_env_conf.toml")
            return

        env, agent = envs[0], agents[0]
        EPISODES = agent.episodes
        env_conf = usr_conf.get("env_conf", usr_conf) if isinstance(usr_conf, dict) else {}
        if env_conf.get("treasure_random", False):
            expected_treasures = int(env_conf.get("treasure_count", 0))
        else:
            expected_treasures = len(env_conf.get("treasure_id", []))

        # Monitor state
        last_report_monitor_time = 0
        last_get_training_metrics_time = 0

        logger.info("Start Training...")
        logger.info(f"Epsilon schedule: start={agent.epsilon}, min={agent.epsilon_min}, "
                    f"decay={agent.epsilon_decay}")
        start_time = time.time()
        last_save_model_time = start_time
        save_count = 0

        # Training statistics with moving window for convergence detection
        total_reward = 0
        episode_count = 0
        win_count = 0
        episode_steps_total = 0
        loss_total = 0.0
        td_error_total = 0.0
        loss_count = 0
        recent_wins = deque(maxlen=100)
        recent_treasures = deque(maxlen=100)

        for episode in range(EPISODES):
            # Retrieve training metrics every 15 seconds
            if time.time() - last_get_training_metrics_time > 15:
                last_get_training_metrics_time = time.time()
                training_metrics = get_training_metrics()
                if training_metrics:
                    logger.info(f"training_metrics is {training_metrics}")

            # Reset environment
            env_obs = env.reset(usr_conf=usr_conf)

            if handle_disaster_recovery(env_obs, logger):
                continue

            agent.reset_episode()

            # Process initial observation
            obs_data = agent.observation_process(env_obs)

            # Episode loop — epsilon is NOT reset here (managed by agent across episodes)
            done = False
            episode_reward = 0
            episode_steps = 0

            while not done:
                # Select action using current epsilon (decays across episodes, not within)
                act_data = agent.predict(list_obs_data=[obs_data])
                act_data = act_data[0]

                current_action = agent.action_process(act_data)

                # Execute action
                next_env_reward, next_env_obs = env.step(current_action)

                if handle_disaster_recovery(next_env_obs, logger):
                    break

                terminated, truncated = next_env_obs["terminated"], next_env_obs["truncated"]

                agent.record_transition(env_obs, current_action, next_env_obs)

                # Process next observation
                next_obs_data = agent.observation_process(next_env_obs)

                # Calculate shaped reward
                reward = reward_shaping(
                    next_env_reward,
                    next_env_obs,
                    prev_env_obs=env_obs,
                    action=current_action,
                )

                done = terminated or truncated
                if terminated:
                    obs_container = (
                        next_env_obs.get("observation", next_env_obs)
                        if isinstance(next_env_obs, dict)
                        else {}
                    )
                    game_info = obs_container.get("game_info") or obs_container.get("env_info") or {}
                    treasures = float(game_info.get("treasure_count", 0))
                    full_success = treasures >= expected_treasures
                    win_count += int(full_success)
                    recent_wins.append(int(full_success))
                    recent_treasures.append(treasures)
                elif truncated:
                    recent_wins.append(0)
                    obs_container = (
                        next_env_obs.get("observation", next_env_obs)
                        if isinstance(next_env_obs, dict)
                        else {}
                    )
                    game_info = obs_container.get("game_info") or obs_container.get("env_info") or {}
                    recent_treasures.append(float(game_info.get("treasure_count", 0)))

                # Create sample as dict with done flag (replaces Frame for done pipeline support)
                sample = {
                    "state": obs_data.feature,
                    "action": current_action,
                    "reward": reward,
                    "next_state": next_obs_data.feature,
                    "done": done,
                }

                samples = sample_process([sample])
                learn_result = agent.learn(samples)

                loss_total += float(learn_result.get("loss", 0.0))
                td_error_total += float(learn_result.get("td_error", 0.0))
                loss_count += 1

                total_reward += reward
                episode_reward += reward
                episode_steps += 1
                env_obs = next_env_obs
                obs_data = next_obs_data

            # End of episode: decay epsilon and update counters
            agent.decay_epsilon()
            episode_count += 1
            episode_steps_total += episode_steps
            now = time.time()

            # Validation episodes every 50 training episodes
            if episode > 0 and episode % 50 == 0:
                val_success = 0
                val_total_steps = 0
                val_total_treasures = 0.0
                val_total_score = 0.0
                val_runs = 0
                for _ in range(5):
                    val_env_obs = env.reset(usr_conf=usr_conf)
                    if handle_disaster_recovery(val_env_obs, logger):
                        continue
                    agent.reset_episode()
                    val_runs += 1
                    val_done = False
                    val_ep_steps = 0
                    while not val_done:
                        val_action = agent.exploit(val_env_obs)
                        val_env_reward, val_env_obs = env.step(val_action)
                        if handle_disaster_recovery(val_env_obs, logger):
                            break
                        val_terminated = val_env_obs.get("terminated", False)
                        val_truncated = val_env_obs.get("truncated", False)
                        val_done = val_terminated or val_truncated
                        val_ep_steps += 1
                        if val_done:
                            val_total_steps += val_ep_steps
                            val_obs_container = (
                                val_env_obs.get("observation", val_env_obs)
                                if isinstance(val_env_obs, dict)
                                else {}
                            )
                            val_game_info = (
                                val_obs_container.get("game_info")
                                or val_obs_container.get("env_info")
                                or {}
                            )
                            treasures = float(val_game_info.get("treasure_count", 0))
                            if val_terminated and treasures >= expected_treasures:
                                val_success += 1
                            val_total_treasures += treasures
                            val_total_score += float(val_game_info.get("total_score", 0))

                val_divisor = max(1, val_runs)
                logger.info(
                    f"Validation: success={val_success / val_divisor:.0%}, "
                    f"steps={val_total_steps / val_divisor:.0f}, "
                    f"treasures={val_total_treasures / val_divisor:.1f}, "
                    f"score={val_total_score / val_divisor:.0f}"
                )

            # Convergence requires reaching the goal with every configured treasure.
            recent_win_rate = sum(recent_wins) / max(1, len(recent_wins))
            avg_treasures = sum(recent_treasures) / max(1, len(recent_treasures))
            is_converged = (
                len(recent_wins) >= 100
                and recent_win_rate >= 0.90
                and avg_treasures >= expected_treasures
                and episode >= 200
            )

            # Report progress every 15 seconds or on convergence
            if is_converged or now - last_report_monitor_time > 15:
                avg_reward = total_reward / max(1, episode_count)
                avg_steps = episode_steps_total / max(1, episode_count)
                avg_loss = loss_total / max(1, loss_count)
                avg_td_error = td_error_total / max(1, loss_count)
                logger.info(
                    f"Episode: {episode + 1}, Epsilon: {agent.epsilon:.4f}, "
                    f"Avg Reward: {avg_reward:.2f}, "
                    f"Recent Win Rate: {recent_win_rate:.2%}, "
                    f"LR: {agent.algorithm._current_lr():.4f}"
                )
                monitor_data = {
                    "reward": avg_reward,
                    "win_rate": recent_win_rate,
                    "epsilon": agent.epsilon,
                    "loss": avg_loss,
                    "td_error": avg_td_error,
                    "episode_steps": avg_steps,
                    "treasure_count": avg_treasures,
                }
                if monitor:
                    monitor.put_data({os.getpid(): monitor_data})

                total_reward = 0
                episode_count = 0
                win_count = 0
                episode_steps_total = 0
                loss_total = 0.0
                td_error_total = 0.0
                loss_count = 0
                last_report_monitor_time = now

            if is_converged:
                logger.info(
                    f"Training Converged at Episode: {episode + 1}, "
                    f"Win Rate: {recent_win_rate:.2%}, "
                    f"Avg Treasures per Win: {avg_treasures:.1f}, "
                    f"Epsilon: {agent.epsilon:.4f}"
                )
                break

            # Model saving with rate limiting (>= 30s between saves, max 100 saves)
            if time.time() - last_save_model_time >= 30 and save_count < 100:
                logger.info(f"Saving Model at Episode: {episode + 1}")
                agent.save_model()
                last_save_model_time = time.time()
                save_count += 1

        end_time = time.time()
        logger.info(f"Training completed: {episode + 1} episodes in {end_time - start_time:.1f}s")
        agent.episodes = episode + 1

        # Save final model (respect platform save limit)
        if save_count < 100:
            agent.save_model()
            save_count += 1

    except Exception as e:
        raise RuntimeError(f"workflow error: {e}")
