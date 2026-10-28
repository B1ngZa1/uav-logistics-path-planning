#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""Hybrid exact-planning and tabular Q-learning agent."""


import os

import numpy as np
from kaiwudrl.interface.agent import BaseAgent
from common_python.utils.common_func import create_cls

from agent_q_learning.algorithm.algorithm import Algorithm, SparseQTable
from agent_q_learning.conf.conf import Config
from agent_q_learning.feature.preprocessor import (
    active_treasure_mask,
    endpoint_target,
    heuristic_action_scores,
    hero_position,
    legal_actions_from_obs,
    optimal_route,
    q_learning_feature,
    shortest_path_action,
    step_no,
    transition_position,
)


ObsData = create_cls("ObsData", feature=None, legal_actions=None, raw_obs=None)
ActData = create_cls("ActData", act=None)

CHECKPOINT_VERSION = 2
STATE_ENCODING = "official_position_treasure_mask_v1"


class Agent(BaseAgent):
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None) -> None:
        self.logger = logger

        self.state_size = Config.STATE_SIZE
        self.action_size = Config.ACTION_SIZE
        self.learning_rate = Config.LEARNING_RATE
        self.gamma = Config.GAMMA

        self.epsilon = Config.EPSILON_START
        self.epsilon_min = Config.EPSILON_MIN
        self.epsilon_decay = Config.EPSILON_DECAY
        self.episodes = Config.EPISODES

        self.algorithm = Algorithm(
            gamma=self.gamma,
            learning_rate=self.learning_rate,
            state_size=self.state_size,
            action_size=self.action_size,
            replay_capacity=Config.REPLAY_CAPACITY,
            batch_size=Config.BATCH_SIZE,
            target_update_freq=Config.TARGET_UPDATE_FREQ,
            lr_min=Config.LR_MIN,
            lr_decay_steps=Config.LR_DECAY_STEPS,
            tau=Config.TARGET_SOFT_TAU if Config.USE_SOFT_UPDATE else None,
        )

        self.blocked_actions_by_pos = {}
        self.route_plan = None
        self.route_active_mask = None
        self.route_endpoint_position = None
        self.expected_position = None
        self.last_step_no = None

        super().__init__(agent_type, device, logger, monitor)

    def _log(self, level, message):
        if self.logger is None:
            return
        log_method = getattr(self.logger, level, None) or getattr(self.logger, "info", None)
        if log_method is not None:
            log_method(message)

    def reset_episode(self):
        """Clear all navigation state that must not leak across episodes."""
        self.blocked_actions_by_pos.clear()
        self.route_plan = None
        self.route_active_mask = None
        self.route_endpoint_position = None
        self.expected_position = None
        self.last_step_no = None

    def _ensure_episode(self, env_obs):
        current_step = step_no(env_obs)
        is_new_episode = (
            self.last_step_no is not None
            and (current_step < self.last_step_no or (current_step == 0 and self.last_step_no != 0))
        )
        if is_new_episode:
            self.reset_episode()
        self.last_step_no = current_step

    def _invalidate_route(self):
        self.route_plan = None
        self.route_active_mask = None
        self.route_endpoint_position = None
        self.expected_position = None

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def predict(self, list_obs_data):
        obs_data = list_obs_data[0]
        action = self._epsilon_greedy(
            state=obs_data.feature,
            epsilon=self.epsilon,
            legal_actions=obs_data.legal_actions,
            env_obs=getattr(obs_data, "raw_obs", None),
        )
        return [ActData(act=action)]

    def exploit(self, env_obs):
        """Use the exact map plan; use Q-values only when planning is unavailable."""
        self._ensure_episode(env_obs)
        obs_data = self.observation_process(env_obs)
        legal_actions = obs_data.legal_actions or list(range(self.action_size))
        action = self._best_action(obs_data.feature, legal_actions, env_obs=env_obs)
        return self.action_process(ActData(act=action))

    def _epsilon_greedy(self, state, epsilon=0.1, legal_actions=None, env_obs=None):
        legal_actions = legal_actions or list(range(self.action_size))
        if np.random.rand() <= epsilon:
            self._invalidate_route()
            return int(np.random.choice(legal_actions))
        return self._best_action(state, legal_actions, env_obs=env_obs)

    def _masked_q_values(self, state, legal_actions):
        q_values = self.algorithm.Q[state, :].copy()
        for illegal_action in set(range(self.action_size)) - set(legal_actions):
            q_values[illegal_action] = -1e9
        return q_values

    def _planned_action(self, env_obs, legal_actions):
        self._ensure_episode(env_obs)
        current_position = hero_position(env_obs)
        active_mask = active_treasure_mask(env_obs)
        endpoint = endpoint_target(env_obs)
        endpoint_position = (endpoint["x"], endpoint["z"])

        if self.expected_position is not None and current_position != self.expected_position:
            self._invalidate_route()

        if (
            self.route_plan is None
            or self.route_active_mask != active_mask
            or self.route_endpoint_position != endpoint_position
        ):
            self.route_plan = optimal_route(env_obs)
            self.route_active_mask = active_mask
            self.route_endpoint_position = endpoint_position

        if self.route_plan is None or not self.route_plan.get("targets"):
            return None

        target = self.route_plan["targets"][0]
        target_position = (int(target["x"]), int(target["z"]))
        action, distance = shortest_path_action(current_position, target_position)
        if distance is None or action is None or action not in legal_actions:
            self._invalidate_route()
            return None

        self.expected_position = transition_position(current_position, action)
        return int(action)

    def _best_action(self, state, legal_actions, env_obs=None):
        if env_obs is not None:
            planned_action = self._planned_action(env_obs, legal_actions)
            if planned_action is not None:
                return planned_action

        q_values = self._masked_q_values(state, legal_actions)
        legal_q = q_values[legal_actions]
        if not np.allclose(legal_q, legal_q[0]):
            return int(np.argmax(q_values))

        if env_obs is not None:
            heuristic_scores = heuristic_action_scores(
                env_obs,
                legal_actions=legal_actions,
                blocked_actions=self.blocked_actions_by_pos.get(hero_position(env_obs), set()),
            )
            return int(np.argmax(heuristic_scores))
        return int(legal_actions[0])

    def learn(self, list_sample_data):
        return self.algorithm.learn(list_sample_data)

    def sync_target_network(self):
        self.algorithm.sync_target()

    def observation_process(self, env_obs):
        legal_actions = legal_actions_from_obs(env_obs)
        blocked_actions = self.blocked_actions_by_pos.get(hero_position(env_obs), set())
        filtered_actions = [action for action in legal_actions if action not in blocked_actions]
        return ObsData(
            feature=q_learning_feature(env_obs),
            legal_actions=filtered_actions or legal_actions,
            raw_obs=env_obs,
        )

    def action_process(self, act_data):
        return int(max(0, min(self.action_size - 1, act_data.act)))

    def _legal_actions_from_obs(self, env_obs):
        return legal_actions_from_obs(env_obs)

    def record_transition(self, prev_env_obs, action, next_env_obs):
        prev_position = hero_position(prev_env_obs)
        next_position = hero_position(next_env_obs)
        if prev_position == next_position and not next_env_obs.get("terminated", False):
            self.blocked_actions_by_pos.setdefault(prev_position, set()).add(int(action))
            self._invalidate_route()
        else:
            self.blocked_actions_by_pos.get(prev_position, set()).discard(int(action))
            if self.expected_position is not None and next_position != self.expected_position:
                self._invalidate_route()
        self.last_step_no = step_no(next_env_obs)

    def save_model(self, path=None, id="1"):
        path = path or "ckpt"
        os.makedirs(path, exist_ok=True)
        model_file_path = f"{path}/model.ckpt-{str(id)}.npy"
        payload = {
            "format_version": CHECKPOINT_VERSION,
            "state_encoding": STATE_ENCODING,
            "q_table": self.algorithm.Q.to_serializable(),
            "epsilon": float(self.epsilon),
            "train_steps": int(self.algorithm.train_steps),
        }
        np.save(model_file_path, payload, allow_pickle=True)
        self._log("info", f"save model {model_file_path} successfully")

    def load_model(self, path=None, id="1"):
        path = path or "ckpt"
        model_file_path = f"{path}/model.ckpt-{str(id)}.npy"
        try:
            payload = np.load(model_file_path, allow_pickle=True)
            if isinstance(payload, np.ndarray) and payload.shape == ():
                payload = payload.item()
            if not isinstance(payload, dict):
                raise ValueError("checkpoint payload is not a dictionary")
            if payload.get("format_version") != CHECKPOINT_VERSION:
                raise ValueError("checkpoint format version is incompatible")
            if payload.get("state_encoding") != STATE_ENCODING:
                raise ValueError("checkpoint state encoding is incompatible")

            q_table = SparseQTable.from_serializable(payload.get("q_table"), self.action_size)
            if any(state < 0 or state >= self.state_size for state in q_table.data):
                raise ValueError("checkpoint contains state IDs outside the official state space")

            self.algorithm.Q = q_table
            self.epsilon = max(
                self.epsilon_min,
                min(Config.EPSILON_START, float(payload.get("epsilon", self.epsilon))),
            )
            self.algorithm.train_steps = max(0, int(payload.get("train_steps", 0)))
            self.algorithm.sync_target(tau=None)
            self._log("info", f"load model {model_file_path} successfully")
        except FileNotFoundError:
            self.algorithm.Q = SparseQTable(action_size=self.action_size)
            self.algorithm.sync_target(tau=None)
            self._log("warning", f"File {model_file_path} not found; using a fresh Q table")
        except (TypeError, ValueError, KeyError) as error:
            self.algorithm.Q = SparseQTable(action_size=self.action_size)
            self.algorithm.sync_target(tau=None)
            self._log(
                "warning",
                f"Ignoring incompatible checkpoint {model_file_path}: {error}; using a fresh Q table",
            )
