#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import math

import numpy as np


FEATURE_SIZE = 55
MAP_SIZE = 64
ACTION_SIZE = 4
ACTION_DELTAS = {
    0: (0, 1),   # UP
    1: (0, -1),  # DOWN
    2: (-1, 0),  # LEFT
    3: (1, 0),   # RIGHT
}


def _observation(env_obs):
    if isinstance(env_obs, dict):
        return env_obs.get("observation", env_obs)
    return {}


def _frame_state(env_obs):
    return _observation(env_obs).get("frame_state", {})


def _game_info(env_obs):
    obs = _observation(env_obs)
    return obs.get("game_info") or obs.get("env_info") or {}


def _hero_pos(env_obs):
    hero = _frame_state(env_obs).get("hero", {})
    pos = hero.get("pos", {})
    return int(pos.get("x", 0)), int(pos.get("z", 0))


def _organs(env_obs):
    return _frame_state(env_obs).get("organs", [])


def _score(env_obs):
    info = _game_info(env_obs)
    return float(info.get("score", 0.0))


def _reward_value(env_reward):
    if env_reward is None:
        return 0.0
    if isinstance(env_reward, (int, float, np.integer, np.floating)):
        return float(env_reward)
    if isinstance(env_reward, dict):
        for key in ("reward", "score", "total_score"):
            if key in env_reward:
                return _reward_value(env_reward[key])
        for value in env_reward.values():
            parsed = _reward_value(value)
            if parsed != 0.0:
                return parsed
        return 0.0
    if isinstance(env_reward, (list, tuple, np.ndarray)):
        if len(env_reward) == 0:
            return 0.0
        parsed_values = [_reward_value(value) for value in env_reward]
        non_zero = [value for value in parsed_values if value != 0.0]
        return non_zero[0] if non_zero else 0.0
    try:
        return float(env_reward)
    except (TypeError, ValueError):
        return 0.0


def _total_score(env_obs):
    info = _game_info(env_obs)
    return float(info.get("total_score", info.get("score", 0.0)))


def _target_organ(env_obs):
    for organ in _organs(env_obs):
        if organ.get("sub_type") == 2:
            return organ
    return None


def _target_dist(env_obs):
    target = _target_organ(env_obs)
    if target and "dist" in target:
        return float(target["dist"])
    x, z = _hero_pos(env_obs)
    target_pos = target.get("pos", {}) if target else {}
    tx, tz = int(target_pos.get("x", 0)), int(target_pos.get("z", 0))
    return min(6.0, (abs(tx - x) + abs(tz - z)) / 12.0)


def _local_view(env_obs):
    info = _game_info(env_obs)
    view = info.get("local_view", [])
    if view is None:
        return []
    return list(view)


def position_from_obs(env_obs):
    return _hero_pos(env_obs)


def _neighbor_indexes(local_view):
    if not local_view:
        return {}
    width = int(math.sqrt(len(local_view)))
    if width * width != len(local_view) or width < 3:
        return {}
    center = len(local_view) // 2
    return {
        0: center - width,
        1: center + width,
        2: center - 1,
        3: center + 1,
    }


def _pad(values, size, fill=0.0):
    values = list(values)
    if len(values) >= size:
        return values[:size]
    return values + [fill] * (size - len(values))


def legal_actions_from_obs(env_obs):
    local_view = _local_view(env_obs)
    if not local_view:
        return list(range(ACTION_SIZE))

    neighbors = _neighbor_indexes(local_view)
    if not neighbors:
        return list(range(ACTION_SIZE))

    legal = []
    for action, index in neighbors.items():
        if 0 <= index < len(local_view) and int(local_view[index]) != 0:
            legal.append(action)
    return legal or list(range(ACTION_SIZE))


def _priority_target(env_obs):
    x, z = _hero_pos(env_obs)
    active_treasures = []
    target = None
    for organ in _organs(env_obs):
        if organ.get("sub_type") == 2:
            target = organ
        elif organ.get("sub_type") == 1 and int(organ.get("status", 0)) == 1:
            dist = float(organ.get("dist", 6))
            pos = organ.get("pos", {})
            manhattan = abs(int(pos.get("x", x)) - x) + abs(int(pos.get("z", z)) - z)
            active_treasures.append((dist, manhattan, organ))

    if active_treasures:
        active_treasures.sort(key=lambda item: (item[0], item[1]))
        return active_treasures[0][2]
    return target


def heuristic_action_scores(env_obs, legal_actions=None, blocked_actions=None):
    legal_actions = legal_actions or list(range(ACTION_SIZE))
    blocked_actions = blocked_actions or set()
    x, z = _hero_pos(env_obs)
    info = _game_info(env_obs)
    local_view = _local_view(env_obs)
    neighbors = _neighbor_indexes(local_view)
    memory = info.get("location_memory", [])

    scores = np.full(ACTION_SIZE, -1e9, dtype=np.float32)
    target = _priority_target(env_obs)
    target_pos = target.get("pos", {}) if target else {}
    tx, tz = int(target_pos.get("x", x)), int(target_pos.get("z", z))

    current_dist = abs(tx - x) + abs(tz - z)
    for action in legal_actions:
        if action in blocked_actions:
            scores[action] = -50.0
            continue

        dx, dz = ACTION_DELTAS.get(action, (0, 0))
        nx, nz = x + dx, z + dz
        next_dist = abs(tx - nx) + abs(tz - nz)
        score = float(current_dist - next_dist)

        if local_view and action in neighbors:
            index = neighbors[action]
            if 0 <= index < len(local_view):
                cell = int(local_view[index])
                if cell == 3:
                    score += 8.0
                elif cell == 4:
                    score += 5.0
                elif cell == 0:
                    score -= 100.0

        if memory and 0 <= nx < MAP_SIZE and 0 <= nz < MAP_SIZE:
            score -= 1.5 * float(memory[nx * MAP_SIZE + nz])

        scores[action] = score

    if np.max(scores) <= -50.0:
        for action in legal_actions:
            if scores[action] <= -50.0:
                scores[action] = 0.0

    return scores


def extract_vector_feature(env_obs):
    obs = _observation(env_obs)
    info = _game_info(env_obs)
    x, z = _hero_pos(env_obs)
    x = max(0, min(MAP_SIZE - 1, x))
    z = max(0, min(MAP_SIZE - 1, z))

    target = _target_organ(env_obs)
    target_pos = target.get("pos", {}) if target else {}
    tx, tz = int(target_pos.get("x", x)), int(target_pos.get("z", z))
    goal_dx = (tx - x) / float(MAP_SIZE - 1)
    goal_dz = (tz - z) / float(MAP_SIZE - 1)
    goal_dist = min(1.0, _target_dist(env_obs) / 6.0)

    treasure_status = [0.0] * 10
    treasure_dist = [1.0] * 10
    nearest = None
    for organ in _organs(env_obs):
        if organ.get("sub_type") != 1:
            continue
        config_id = int(organ.get("config_id", -1))
        if not 0 <= config_id < 10:
            continue
        status = float(organ.get("status", 0))
        dist = min(1.0, float(organ.get("dist", 6)) / 6.0)
        treasure_status[config_id] = status
        treasure_dist[config_id] = dist
        if status > 0 and (nearest is None or dist < nearest[0]):
            nearest = (dist, organ)

    if nearest is None:
        nearest_dx, nearest_dz, nearest_dist = 0.0, 0.0, 1.0
    else:
        nearest_pos = nearest[1].get("pos", {})
        nearest_dx = (int(nearest_pos.get("x", x)) - x) / float(MAP_SIZE - 1)
        nearest_dz = (int(nearest_pos.get("z", z)) - z) / float(MAP_SIZE - 1)
        nearest_dist = nearest[0]

    local_view = [float(i) / 4.0 for i in _pad(_local_view(env_obs), 25)]

    memory = info.get("location_memory", [])
    memory_features = []
    if memory:
        coords = [(x, z), (x, z + 1), (x, z - 1), (x - 1, z), (x + 1, z)]
        for mx, mz in coords:
            if 0 <= mx < MAP_SIZE and 0 <= mz < MAP_SIZE:
                memory_features.append(float(memory[mx * MAP_SIZE + mz]))
            else:
                memory_features.append(1.0)
    else:
        memory_features = [0.0] * 5

    step_no = float(obs.get("step_no", 0.0))
    features = [
        x / float(MAP_SIZE - 1),
        z / float(MAP_SIZE - 1),
        goal_dx,
        goal_dz,
        goal_dist,
        *treasure_status,
        *treasure_dist,
        nearest_dx,
        nearest_dz,
        nearest_dist,
        *local_view,
        *memory_features,
        min(1.0, step_no / 2000.0),
        min(1.0, _score(env_obs) / 500.0),
        min(1.0, _total_score(env_obs) / 1000.0),
        min(1.0, float(info.get("treasure_count", 0.0)) / 10.0),
        min(1.0, float(info.get("treasure_score", 0.0)) / 1000.0),
    ]

    return np.asarray(_pad(features, FEATURE_SIZE), dtype=np.float32)


def _nearest_treasure_dist(env_obs):
    distances = [
        float(organ.get("dist", 6))
        for organ in _organs(env_obs)
        if organ.get("sub_type") == 1 and int(organ.get("status", 0)) == 1
    ]
    return min(distances) if distances else None


def sample_process(list_game_data):
    return [
        {
            "state": frame.state,
            "action": frame.action,
            "reward": frame.reward,
            "next_state": frame.next_state,
            "done": frame.done,
        }
        for frame in list_game_data
    ]


def reward_shaping(env_reward, env_obs, prev_env_obs=None, action=None):
    terminated = bool(env_obs.get("terminated", False))
    truncated = bool(env_obs.get("truncated", False))

    reward = -0.05
    score_delta = _score(env_obs) - (_score(prev_env_obs) if prev_env_obs is not None else 0.0)
    raw_reward = _reward_value(env_reward)
    if raw_reward != 0.0:
        score_delta = max(score_delta, raw_reward)
    reward += score_delta / 10.0

    if terminated:
        reward += 20.0
    if truncated and not terminated:
        reward -= 20.0

    if prev_env_obs is not None:
        prev_goal = _target_dist(prev_env_obs)
        curr_goal = _target_dist(env_obs)
        if curr_goal < prev_goal:
            reward += 0.5
        elif curr_goal > prev_goal:
            reward -= 0.2

        prev_treasure = _nearest_treasure_dist(prev_env_obs)
        curr_treasure = _nearest_treasure_dist(env_obs)
        if prev_treasure is not None and curr_treasure is not None:
            if curr_treasure < prev_treasure:
                reward += 0.2
            elif curr_treasure > prev_treasure:
                reward -= 0.1

        if action is not None and _hero_pos(prev_env_obs) == _hero_pos(env_obs) and not terminated:
            reward -= 1.0

    return float(np.clip(reward, -30.0, 30.0))
