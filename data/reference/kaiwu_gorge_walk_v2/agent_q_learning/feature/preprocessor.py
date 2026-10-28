#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Feature helpers for Gorge Walk Q-Learning.

This module keeps the official position + treasure-state encoding and adds an
exact graph planner for evaluation.  With at most ten treasures, Held-Karp
dynamic programming is small enough to choose the score-optimal subset and
visit order while preserving enough steps to reach the endpoint.
"""


import json
import math
import os
from collections import deque

import numpy as np


MAP_SIZE = 64
TREASURE_COUNT = 10
TREASURE_STATE_SIZE = 2**TREASURE_COUNT
ACTION_SIZE = 4
ACTION_DELTAS = {
    0: (0, 1),   # UP
    1: (0, -1),  # DOWN
    2: (-1, 0),  # LEFT
    3: (1, 0),   # RIGHT
}
ENDPOINT_CONFIG_ID = 22
MAX_STEP = 2000
TREASURE_VALUE = 100.0
STEP_VALUE = 0.2
ENDPOINT_VALUE = 150.0

_MAP_GRAPH = None
_PATH_CACHE = {}


def observation(env_obs):
    return env_obs.get("observation", env_obs) if isinstance(env_obs, dict) else {}


def frame_state(env_obs):
    return observation(env_obs).get("frame_state", {})


def game_info(env_obs):
    obs = observation(env_obs)
    return obs.get("game_info") or obs.get("env_info") or {}


def organs(env_obs):
    return frame_state(env_obs).get("organs", [])


def hero_position(env_obs):
    hero = frame_state(env_obs).get("hero", {})
    pos = hero.get("pos", {})
    return int(pos.get("x", 0)), int(pos.get("z", 0))


def step_no(env_obs):
    obs = observation(env_obs)
    try:
        return int(obs.get("step_no", env_obs.get("frame_no", 0)))
    except (AttributeError, TypeError, ValueError):
        return 0


def _state_id(x, z):
    return int(x) * MAP_SIZE + int(z)


def _map_file_candidates():
    feature_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(feature_dir, "..", ".."))
    cwd = os.getcwd()
    return [
        os.path.join(cwd, "conf", "map_data", "F_level_1.json"),
        os.path.join(cwd, "conf", "map_data", "F_level_0.json"),
        os.path.join(project_root, "conf", "map_data", "F_level_1.json"),
        os.path.join(project_root, "conf", "map_data", "F_level_0.json"),
    ]


def _load_map_graph():
    global _MAP_GRAPH
    if _MAP_GRAPH is not None:
        return _MAP_GRAPH

    _MAP_GRAPH = {}
    for path in _map_file_candidates():
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                map_data = json.load(f)
        except (OSError, ValueError):
            continue

        graph = {}
        for state_text, actions in map_data.items():
            state = int(state_text)
            edges = []
            for action_text, transition in actions.items():
                action = int(action_text)
                next_state = int(transition[0])
                if next_state != state:
                    edges.append((action, next_state))
            graph[state] = edges
        if graph:
            _MAP_GRAPH = graph
            break

    return _MAP_GRAPH


def _transition_state(state, action):
    for edge_action, next_state in _load_map_graph().get(int(state), []):
        if edge_action == int(action):
            return next_state
    return None


def transition_position(position, action):
    """Return the map transition for an action, or None when it is blocked."""
    state = _state_id(*position)
    next_state = _transition_state(state, action)
    if next_state is None:
        return None
    return int(next_state // MAP_SIZE), int(next_state % MAP_SIZE)


def _shortest_path(start_state, goal_state):
    start_state, goal_state = int(start_state), int(goal_state)
    if start_state == goal_state:
        return None, 0

    key = (start_state, goal_state)
    if key in _PATH_CACHE:
        return _PATH_CACHE[key]

    graph = _load_map_graph()
    if not graph:
        return None, None

    queue = deque([(start_state, None, 0)])
    seen = {start_state}
    while queue:
        state, first_action, dist = queue.popleft()
        for action, next_state in graph.get(state, []):
            if next_state in seen:
                continue
            next_first_action = action if first_action is None else first_action
            if next_state == goal_state:
                result = (int(next_first_action), dist + 1)
                if len(_PATH_CACHE) > 50000:
                    _PATH_CACHE.pop(next(iter(_PATH_CACHE)))
                _PATH_CACHE[key] = result
                return result
            seen.add(next_state)
            queue.append((next_state, next_first_action, dist + 1))

    result = (None, None)
    _PATH_CACHE[key] = result
    return result


def _path_distance(start_state, goal_state):
    return _shortest_path(start_state, goal_state)[1]


def shortest_path_action(start_position, goal_position):
    """Return (first_action, distance) for two map coordinates."""
    return _shortest_path(_state_id(*start_position), _state_id(*goal_position))


def clamp_dist(value, default=6):
    try:
        value = int(value)
    except (TypeError, ValueError):
        value = default
    return max(0, min(6, value))


def official_base_feature(env_obs):
    """
    Official base feature:
    feature = 1024 * pos_feature + treasure_binary
    pos_feature = x * 64 + z
    treasure_binary = binary encoding of 10 treasure statuses.
    """
    x, z = hero_position(env_obs)
    x = max(0, min(MAP_SIZE - 1, x))
    z = max(0, min(MAP_SIZE - 1, z))
    pos_feature = int(x * MAP_SIZE + z)

    treasure_status = [0] * TREASURE_COUNT
    for organ in organs(env_obs):
        if organ.get("sub_type") != 1:
            continue
        config_id = int(organ.get("config_id", -1))
        if 0 <= config_id < TREASURE_COUNT:
            treasure_status[config_id] = int(organ.get("status", 0))

    treasure_binary = sum(treasure_status[i] * (2**i) for i in range(TREASURE_COUNT))
    return int(TREASURE_STATE_SIZE * pos_feature + treasure_binary)


def endpoint_dist_feature(env_obs):
    """
    Terminal BFS discrete distance. The guide defines dist in range [0, 6].
    The terminal is normally organs[0], but we also check sub_type == 2.
    """
    all_organs = organs(env_obs)
    if all_organs:
        first = all_organs[0]
        if first.get("sub_type") == 2 or first.get("config_id") == 22:
            return clamp_dist(first.get("dist", 6))

    for organ in all_organs:
        if organ.get("sub_type") == 2:
            return clamp_dist(organ.get("dist", 6))
    return 6


def nearest_treasure_dist_feature(env_obs):
    """
    Nearest uncollected treasure distance encoded with 3 bits.
    Values 0-6 are valid BFS discrete distances; value 7 means no active treasure.
    """
    nearest = 7
    for organ in organs(env_obs):
        if organ.get("sub_type") != 1 or int(organ.get("status", 0)) != 1:
            continue
        nearest = min(nearest, clamp_dist(organ.get("dist", 6)))
    return int(nearest)


def active_treasures(env_obs, ignored_treasures=None):
    ignored_treasures = set(ignored_treasures or [])
    result = []
    for organ in organs(env_obs):
        if organ.get("sub_type") != 1 or int(organ.get("status", 0)) != 1:
            continue
        config_id = int(organ.get("config_id", -1))
        if config_id in ignored_treasures:
            continue
        pos = organ.get("pos", {})
        x, z = hero_position(env_obs)
        tx, tz = int(pos.get("x", x)), int(pos.get("z", z))
        result.append(
            {
                "config_id": config_id,
                "dist": clamp_dist(organ.get("dist", 6)),
                "x": tx,
                "z": tz,
                "manhattan": abs(tx - x) + abs(tz - z),
            }
        )
    return result


def active_treasure_mask(env_obs):
    """Encode currently collectible treasures as a stable ten-bit mask."""
    mask = 0
    for treasure in active_treasures(env_obs):
        config_id = int(treasure["config_id"])
        if 0 <= config_id < TREASURE_COUNT:
            mask |= 1 << config_id
    return mask


def endpoint_target(env_obs):
    x, z = hero_position(env_obs)
    for organ in organs(env_obs):
        if organ.get("sub_type") == 2:
            pos = organ.get("pos", {})
            tx, tz = int(pos.get("x", x)), int(pos.get("z", z))
            return {
                "config_id": int(organ.get("config_id", ENDPOINT_CONFIG_ID)),
                "dist": clamp_dist(organ.get("dist", 6)),
                "x": tx,
                "z": tz,
                "manhattan": abs(tx - x) + abs(tz - z),
                "is_endpoint": True,
            }
    return {
        "config_id": ENDPOINT_CONFIG_ID,
        "dist": 6,
        "x": x,
        "z": z,
        "manhattan": 0,
        "is_endpoint": True,
    }


def _is_better_path(candidate, incumbent, treasures):
    """Keep the first deterministic route when a DP state has equal cost."""
    if incumbent is None or candidate[0] < incumbent[0]:
        return True
    return False


def _is_better_plan(candidate, incumbent):
    """Maximize score utility, then prefer more treasure, fewer steps and IDs."""
    if incumbent is None:
        return True
    candidate_key = (
        round(candidate["utility"], 9),
        candidate["treasure_count"],
        -candidate["steps"],
    )
    incumbent_key = (
        round(incumbent["utility"], 9),
        incumbent["treasure_count"],
        -incumbent["steps"],
    )
    if candidate_key != incumbent_key:
        return candidate_key > incumbent_key
    return tuple(candidate["target_ids"]) < tuple(incumbent["target_ids"])


def optimal_route(env_obs, max_step=MAX_STEP):
    """Compute the score-optimal reachable treasure route ending at the goal.

    The completion step equal to ``max_step`` is treated as a timeout by the
    environment, so a route must use strictly fewer than the remaining steps.
    """
    graph = _load_map_graph()
    if not graph:
        return None

    current_position = hero_position(env_obs)
    current_state = _state_id(*current_position)
    endpoint = endpoint_target(env_obs)
    endpoint_state = _state_id(endpoint["x"], endpoint["z"])
    direct_action, direct_distance = _shortest_path(current_state, endpoint_state)
    if direct_distance is None:
        return None

    remaining_steps = max(0, int(max_step) - step_no(env_obs))
    route_budget = remaining_steps - 1
    if direct_distance > route_budget:
        return None

    treasures = sorted(active_treasures(env_obs), key=lambda item: item["config_id"])
    treasure_states = [_state_id(item["x"], item["z"]) for item in treasures]
    collected_count = int(game_info(env_obs).get("treasure_count", 0) or 0)

    best_plan = {
        "targets": [dict(endpoint)],
        "target_ids": [ENDPOINT_CONFIG_ID],
        "steps": direct_distance,
        "treasure_count": 0,
        "utility": -STEP_VALUE * direct_distance,
        "first_action": direct_action,
    }

    # (mask, last_index) -> (path_steps_from_current, route_indices)
    dp = {}
    for index, treasure_state in enumerate(treasure_states):
        _, distance = _shortest_path(current_state, treasure_state)
        if distance is not None:
            dp[(1 << index, index)] = (distance, (index,))

    for mask in range(1, 1 << len(treasures)):
        for last_index in range(len(treasures)):
            current = dp.get((mask, last_index))
            if current is None:
                continue

            path_steps, route_indices = current
            distance_to_end = _path_distance(treasure_states[last_index], endpoint_state)
            if distance_to_end is not None:
                total_steps = path_steps + distance_to_end
                if total_steps <= route_budget:
                    target_ids = [treasures[index]["config_id"] for index in route_indices]
                    candidate = {
                        "targets": [dict(treasures[index]) for index in route_indices] + [dict(endpoint)],
                        "target_ids": target_ids + [ENDPOINT_CONFIG_ID],
                        "steps": total_steps,
                        "treasure_count": len(route_indices),
                        "utility": TREASURE_VALUE * len(route_indices) - STEP_VALUE * total_steps,
                        "first_action": _shortest_path(current_state, treasure_states[route_indices[0]])[0],
                    }
                    if _is_better_plan(candidate, best_plan):
                        best_plan = candidate

            for next_index, next_state in enumerate(treasure_states):
                if mask & (1 << next_index):
                    continue
                distance = _path_distance(treasure_states[last_index], next_state)
                if distance is None:
                    continue
                next_key = (mask | (1 << next_index), next_index)
                candidate = (path_steps + distance, route_indices + (next_index,))
                if _is_better_path(candidate, dp.get(next_key), treasures):
                    dp[next_key] = candidate

    final_step = step_no(env_obs) + best_plan["steps"]
    best_plan["projected_score"] = (
        ENDPOINT_VALUE
        + TREASURE_VALUE * (collected_count + best_plan["treasure_count"])
        + STEP_VALUE * (int(max_step) - final_step)
    )
    best_plan["remaining_steps"] = remaining_steps
    best_plan["active_mask"] = active_treasure_mask(env_obs)
    best_plan["endpoint_position"] = (endpoint["x"], endpoint["z"])
    return best_plan


def priority_target(env_obs, ignored_treasures=None):
    """Compatibility wrapper returning the first exact-plan target."""
    route = optimal_route(env_obs)
    if route is not None and route["targets"]:
        target = dict(route["targets"][0])
        target["path_action"] = route["first_action"]
        target["path_dist"] = _path_distance(
            _state_id(*hero_position(env_obs)),
            _state_id(target["x"], target["z"]),
        )
        target["is_endpoint"] = target["config_id"] == ENDPOINT_CONFIG_ID
        return target
    return endpoint_target(env_obs)


def q_learning_feature(env_obs):
    """
    Official position + treasure-mask feature used by observation_process.
    """
    return official_base_feature(env_obs)


def state_size():
    return MAP_SIZE * MAP_SIZE * TREASURE_STATE_SIZE


def legal_actions_from_obs(env_obs):
    """
    Filter invalid actions using game_info.local_view.

    action_space: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT.
    local_view: 0=obstacle, non-zero=walkable/special cell.
    """
    x, z = hero_position(env_obs)
    map_actions = [action for action, _ in _load_map_graph().get(_state_id(x, z), [])]
    if map_actions:
        return map_actions

    local_view = game_info(env_obs).get("local_view", [])
    if not local_view:
        return list(range(ACTION_SIZE))

    width = int(math.sqrt(len(local_view)))
    if width * width != len(local_view) or width < 3:
        return list(range(ACTION_SIZE))

    center = len(local_view) // 2
    neighbor_index = {
        0: center - width,
        1: center + width,
        2: center - 1,
        3: center + 1,
    }
    legal_actions = [
        action
        for action, index in neighbor_index.items()
        if 0 <= index < len(local_view) and int(local_view[index]) != 0
    ]
    return legal_actions or list(range(ACTION_SIZE))


def heuristic_action_scores(
    env_obs,
    legal_actions=None,
    blocked_actions=None,
    ignored_treasures=None,
    recent_positions=None,
    escape_mode=False,
):
    """
    Small policy prior for Q-learning:
    - prefer endpoint-safe shortest paths to opportunistic treasures
    - fall back to endpoint shortest path when time is tight
    - avoid local obstacles, blocked directions, and short loops
    - when escape_mode is enabled, temporarily de-emphasize the current treasure
    """
    legal_actions = legal_actions or list(range(ACTION_SIZE))
    blocked_actions = blocked_actions or set()
    recent_positions = set(recent_positions or [])
    local_view = game_info(env_obs).get("local_view", [])
    x, z = hero_position(env_obs)
    target = priority_target(env_obs, ignored_treasures)
    current_state = _state_id(x, z)
    target_state = _state_id(target["x"], target["z"])
    planned_action, path_dist = _shortest_path(current_state, target_state)
    current_dist = path_dist
    if current_dist is None:
        current_dist = abs(target["x"] - x) + abs(target["z"] - z)
    memory = game_info(env_obs).get("location_memory", [])

    scores = np.full(ACTION_SIZE, -1e9, dtype=np.float32)
    for action in legal_actions:
        if action in blocked_actions:
            scores[action] = -50.0
            continue

        dx, dz = ACTION_DELTAS.get(action, (0, 0))
        nx, nz = x + dx, z + dz
        next_state = _transition_state(current_state, action)
        if next_state is not None:
            next_dist = _path_distance(next_state, target_state)
        else:
            next_dist = None
        if next_dist is None:
            next_dist = abs(target["x"] - nx) + abs(target["z"] - nz)
        score = float(current_dist - next_dist)
        if planned_action is not None and action == planned_action:
            score += 12.0

        if (nx, nz) in recent_positions:
            score -= 6.0
        elif escape_mode:
            score += 2.0

        # Local-view direct signal: treasure/endpoint in adjacent cell should win.
        if local_view:
            width = int(math.sqrt(len(local_view)))
            if width * width == len(local_view) and width >= 3:
                center = len(local_view) // 2
                neighbor_index = {
                    0: center - width,
                    1: center + width,
                    2: center - 1,
                    3: center + 1,
                }
                index = neighbor_index.get(action)
                if index is not None and 0 <= index < len(local_view):
                    cell = int(local_view[index])
                    if cell == 4:
                        score += -4.0 if escape_mode else 8.0
                    elif cell == 3:
                        score += 4.0
                    elif cell == 0:
                        score -= 100.0

        if memory and 0 <= nx < MAP_SIZE and 0 <= nz < MAP_SIZE:
            score -= 1.5 * float(memory[nx * MAP_SIZE + nz])

        if not target.get("is_endpoint", False) and not escape_mode:
            score *= 2.0
        elif escape_mode:
            score -= 0.5 * abs(current_dist - next_dist)

        scores[action] = score

    return scores
