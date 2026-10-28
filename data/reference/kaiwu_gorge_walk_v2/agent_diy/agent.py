#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import math
import os

import numpy as np
import torch
from kaiwudrl.interface.agent import BaseAgent
from common_python.utils.common_func import create_cls

from agent_diy.algorithm.algorithm import Algorithm
from agent_diy.conf.conf import Config
from agent_diy.feature.definition import (
    FEATURE_SIZE,
    extract_vector_feature,
    heuristic_action_scores,
    legal_actions_from_obs,
    position_from_obs,
)
from agent_diy.model.model import Model

ObsData = create_cls("ObsData", feature=None, legal_actions=None, raw_obs=None)
ActData = create_cls("ActData", act=None)


def _cfg(name, default):
    return getattr(Config, name, default)


class Agent(BaseAgent):
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None) -> None:
        self.logger = logger
        self.action_size = _cfg("ACTION_SIZE", 4)
        self.epsilon = _cfg("EPSILON_START", 1.0)
        self.total_env_steps = 0
        self.epsilon_start = _cfg("EPSILON_START", 1.0)
        self.epsilon_end = _cfg("EPSILON_END", 0.05)
        self.epsilon_decay_steps = _cfg("EPSILON_DECAY_STEPS", 50000)
        self.last_pos = None
        self.last_action = None
        self.blocked_actions_by_pos = {}

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # The DIY feature extractor outputs FEATURE_SIZE values. Keep this stable even if
        # an old conf.py from the template is accidentally loaded by the platform.
        self.observation_shape = FEATURE_SIZE
        self.model = Model(self.observation_shape, self.action_size).to(self.device)
        self.target_model = Model(self.observation_shape, self.action_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=_cfg("LEARNING_RATE", 1e-3))
        self.algorithm = Algorithm(
            model=self.model,
            target_model=self.target_model,
            optimizer=self.optimizer,
            gamma=_cfg("GAMMA", 0.98),
            batch_size=_cfg("BATCH_SIZE", 128),
            replay_buffer_size=_cfg("REPLAY_BUFFER_SIZE", 20000),
            warmup_size=_cfg("WARMUP_SIZE", 1000),
            target_update_freq=_cfg("TARGET_UPDATE_FREQ", 200),
            grad_clip_norm=_cfg("GRAD_CLIP_NORM", 10.0),
            device=self.device,
            logger=logger,
        )

        super().__init__(agent_type, device, logger, monitor)

    def predict(self, list_obs_data):
        obs_data = list_obs_data[0]
        action = self._epsilon_greedy(obs_data.feature, obs_data.legal_actions, self.epsilon, obs_data.raw_obs)
        return [ActData(act=action)]

    def exploit(self, env_obs):
        self._update_stuck_state(env_obs)
        obs_data = self.observation_process(env_obs)
        action = self._greedy_action(obs_data.feature, obs_data.legal_actions, env_obs=env_obs)
        self.last_pos = position_from_obs(env_obs)
        self.last_action = action
        return self.action_process(ActData(act=action))

    def _epsilon_greedy(self, feature, legal_actions=None, epsilon=0.1, env_obs=None):
        legal_actions = legal_actions or list(range(self.action_size))
        if np.random.random() < epsilon:
            return int(np.random.choice(legal_actions))
        return self._greedy_action(feature, legal_actions, env_obs=env_obs)

    def _greedy_action(self, feature, legal_actions=None, env_obs=None):
        legal_actions = legal_actions or list(range(self.action_size))
        self.model.eval()
        with torch.no_grad():
            feature_tensor = torch.as_tensor(feature, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.model(feature_tensor).squeeze(0).detach().cpu().numpy()
        illegal_actions = set(range(self.action_size)) - set(legal_actions)
        for action in illegal_actions:
            q_values[action] = -1e9

        if env_obs is not None:
            pos = position_from_obs(env_obs)
            blocked_actions = self.blocked_actions_by_pos.get(pos, set())
            for action in blocked_actions:
                if action in legal_actions:
                    q_values[action] = -1e9

            q_std = float(np.std(q_values[np.isfinite(q_values)]))
            if q_std > 1e-6:
                q_bonus = (q_values - np.mean(q_values[np.isfinite(q_values)])) / q_std
            else:
                q_bonus = np.zeros_like(q_values)
            heuristic_scores = heuristic_action_scores(env_obs, legal_actions, blocked_actions)
            combined_scores = heuristic_scores + 0.1 * np.clip(q_bonus, -3.0, 3.0)
            return int(np.argmax(combined_scores))

        return int(np.argmax(q_values))

    def learn(self, list_sample_data):
        self.model.train()
        return self.algorithm.learn(list_sample_data)

    def observation_process(self, raw_obs):
        legal_actions = legal_actions_from_obs(raw_obs)
        pos = position_from_obs(raw_obs)
        blocked_actions = self.blocked_actions_by_pos.get(pos, set())
        filtered_actions = [action for action in legal_actions if action not in blocked_actions]
        return ObsData(
            feature=extract_vector_feature(raw_obs),
            legal_actions=filtered_actions or legal_actions,
            raw_obs=raw_obs,
        )

    def action_process(self, act_data):
        return int(max(0, min(self.action_size - 1, act_data.act)))

    def decay_epsilon(self):
        self.total_env_steps += 1
        progress = min(1.0, self.total_env_steps / float(self.epsilon_decay_steps))
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * math.exp(-5.0 * progress)
        return self.epsilon

    def record_transition(self, prev_env_obs, action, next_env_obs):
        prev_pos = position_from_obs(prev_env_obs)
        next_pos = position_from_obs(next_env_obs)
        if prev_pos == next_pos and not next_env_obs.get("terminated", False):
            self.blocked_actions_by_pos.setdefault(prev_pos, set()).add(int(action))
        elif prev_pos in self.blocked_actions_by_pos:
            self.blocked_actions_by_pos[prev_pos].discard(int(action))

    def _update_stuck_state(self, env_obs):
        current_pos = position_from_obs(env_obs)
        if self.last_pos is not None and self.last_action is not None and current_pos == self.last_pos:
            self.blocked_actions_by_pos.setdefault(current_pos, set()).add(int(self.last_action))

    def save_model(self, path=None, id="1"):
        path = path or "ckpt"
        os.makedirs(path, exist_ok=True)
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        torch.save(
            {
                "model": self.model.state_dict(),
                "target_model": self.target_model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "total_env_steps": self.total_env_steps,
                "train_step": self.algorithm.train_step,
            },
            model_file_path,
        )
        if self.logger:
            self.logger.info(f"save model {model_file_path} successfully")

    def load_model(self, path=None, id="1"):
        path = path or "ckpt"
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        try:
            checkpoint = torch.load(model_file_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model"])
            self.target_model.load_state_dict(checkpoint.get("target_model", checkpoint["model"]))
            if "optimizer" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.epsilon = float(checkpoint.get("epsilon", self.epsilon_end))
            self.total_env_steps = int(checkpoint.get("total_env_steps", 0))
            self.algorithm.train_step = int(checkpoint.get("train_step", 0))
            if self.logger:
                self.logger.info(f"load model {model_file_path} successfully")
        except FileNotFoundError:
            if self.logger:
                self.logger.info(f"File {model_file_path} not found")
            raise
