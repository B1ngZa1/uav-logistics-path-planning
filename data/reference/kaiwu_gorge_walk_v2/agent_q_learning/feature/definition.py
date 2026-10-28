#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import numpy as np


def _observation(env_obs):
    return env_obs.get("observation", env_obs) if isinstance(env_obs, dict) else {}


def _game_info(env_obs):
    obs = _observation(env_obs)
    return obs.get("game_info") or obs.get("env_info") or {}


def _hero_pos(env_obs):
    hero = _observation(env_obs).get("frame_state", {}).get("hero", {})
    pos = hero.get("pos", {})
    return int(pos.get("x", 0)), int(pos.get("z", 0))


def _organs(env_obs):
    return _observation(env_obs).get("frame_state", {}).get("organs", [])


def _total_score(env_obs):
    return float(_game_info(env_obs).get("total_score", 0.0))


def _reward_value(env_reward):
    if env_reward is None:
        return None
    if isinstance(env_reward, dict):
        if "reward" not in env_reward:
            return None
        return float(env_reward["reward"])
    if hasattr(env_reward, "reward"):
        return float(env_reward.reward)
    try:
        return float(env_reward)
    except (TypeError, ValueError):
        return None


def _target_dist(env_obs):
    for organ in _organs(env_obs):
        if organ.get("sub_type") == 2:
            return float(organ.get("dist", 6))
    return 6.0


def _nearest_treasure_dist(env_obs):
    distances = [
        float(organ.get("dist", 6))
        for organ in _organs(env_obs)
        if organ.get("sub_type") == 1 and int(organ.get("status", 0)) == 1
    ]
    return min(distances) if distances else None


def _active_treasure_count(env_obs):
    return sum(
        1
        for organ in _organs(env_obs)
        if organ.get("sub_type") == 1 and int(organ.get("status", 0)) == 1
    )


def _treasure_count(env_obs):
    return float(_game_info(env_obs).get("treasure_count", 0.0))


def _treasure_score(env_obs):
    return float(_game_info(env_obs).get("treasure_score", 0.0))


def _exploration_bonus(prev_obs, curr_obs):
    """
    Bonus for visiting previously unvisited cells.

    Uses location_memory to detect novel positions. Returns a small bonus
    when the agent steps into a cell it hasn't visited before, encouraging
    map exploration early in training.
    """
    prev_memory = _game_info(prev_obs).get("location_memory", [])
    curr_memory = _game_info(curr_obs).get("location_memory", [])
    if not prev_memory or not curr_memory:
        return 0.0
    x, z = _hero_pos(curr_obs)
    idx = x * 64 + z
    if idx < len(curr_memory):
        prev_val = float(prev_memory[idx]) if idx < len(prev_memory) else 0.0
        curr_val = float(curr_memory[idx])
        if prev_val < 0.05 and curr_val >= 0.05:
            return 2.0
    return 0.0


def sample_process(list_game_data):
    return [
        {
            "state": frame["state"] if isinstance(frame, dict) else frame.state,
            "action": frame["action"] if isinstance(frame, dict) else frame.action,
            "reward": frame["reward"] if isinstance(frame, dict) else frame.reward,
            "next_state": frame["next_state"] if isinstance(frame, dict) else frame.next_state,
            "done": frame["done"] if isinstance(frame, dict) else getattr(frame, "done", False),
        }
        for frame in list_game_data
    ]


def reward_shaping(env_reward, env_obs, prev_env_obs=None, action=None):
    """
    Shaped reward for Q-Learning in Gorge Walk.

    Reward components:
    - Step penalty (-0.2): matches the evaluation step value
    - Treasure collection: strong positive signal (200 per treasure)
    - Endpoint reached with all treasures: large bonus (500)
    - Early endpoint without all treasures: penalty
    - Distance shaping: guides agent toward treasures first, then endpoint
    - Wall collision: penalty for actions that don't move
    - Exploration bonus: small reward for visiting new cells
    """
    terminated = bool(env_obs.get("terminated", False))
    truncated = bool(env_obs.get("truncated", False))

    # Base step penalty to encourage efficiency
    reward = -0.2

    # The environment reward is already an immediate signal. Only fall back to
    # cumulative total_score when no reward value is available. game_info.score
    # is the current-step score and must never be differenced across frames.
    raw_reward = _reward_value(env_reward)
    if raw_reward is not None:
        reward += raw_reward
    elif prev_env_obs is not None:
        reward += _total_score(env_obs) - _total_score(prev_env_obs)

    # Treasure collection handling
    if prev_env_obs is not None:
        treasure_count_delta = max(0.0, _treasure_count(env_obs) - _treasure_count(prev_env_obs))
        treasure_score_delta = max(0.0, _treasure_score(env_obs) - _treasure_score(prev_env_obs))
        active_delta = max(0, _active_treasure_count(prev_env_obs) - _active_treasure_count(env_obs))

        if treasure_count_delta > 0 or treasure_score_delta > 0 or active_delta > 0:
            reward += 50.0 * max(treasure_count_delta, active_delta, 1)
            reward += 0.5 * treasure_score_delta

    # Terminal rewards
    if terminated:
        remaining_treasures = _active_treasure_count(env_obs)
        if remaining_treasures > 0:
            reward -= 150.0 * remaining_treasures
        else:
            reward += 200.0

    # Truncation penalty
    if truncated and not terminated:
        reward -= 100.0

    # Distance-based shaping
    if prev_env_obs is not None:
        prev_goal = _target_dist(prev_env_obs)
        curr_goal = _target_dist(env_obs)
        prev_treasure = _nearest_treasure_dist(prev_env_obs)
        curr_treasure = _nearest_treasure_dist(env_obs)

        # When treasures exist, prioritize approaching them
        if prev_treasure is not None and curr_treasure is not None:
            if curr_treasure < prev_treasure:
                reward += 4.0
            elif curr_treasure > prev_treasure:
                reward -= 2.0
        elif prev_treasure is None:
            # Only after all treasures are collected, incentivize endpoint approach
            if curr_goal < prev_goal:
                reward += 3.0
            elif curr_goal > prev_goal:
                reward -= 1.5

    # Wall collision / no-move penalty
    if prev_env_obs is not None:
        if action is not None and _hero_pos(prev_env_obs) == _hero_pos(env_obs) and not terminated:
            reward -= 3.0

    # Exploration bonus for visiting new cells
    if prev_env_obs is not None:
        reward += _exploration_bonus(prev_env_obs, env_obs)

    return float(np.clip(reward, -100.0, 300.0))
