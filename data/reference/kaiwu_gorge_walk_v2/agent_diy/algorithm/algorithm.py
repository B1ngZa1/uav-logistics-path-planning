#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import random
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def add(self, sample):
        self.buffer.append(
            {
                "state": np.asarray(sample["state"], dtype=np.float32),
                "action": int(sample["action"]),
                "reward": float(sample["reward"]),
                "next_state": np.asarray(sample["next_state"], dtype=np.float32),
                "done": float(sample.get("done", False)),
            }
        )

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)


class Algorithm:
    def __init__(
        self,
        model,
        target_model,
        optimizer,
        gamma,
        batch_size,
        replay_buffer_size,
        warmup_size,
        target_update_freq,
        grad_clip_norm,
        device,
        logger=None,
    ):
        self.model = model
        self.target_model = target_model
        self.optimizer = optimizer
        self.gamma = gamma
        self.batch_size = batch_size
        self.warmup_size = warmup_size
        self.target_update_freq = target_update_freq
        self.grad_clip_norm = grad_clip_norm
        self.device = device
        self.logger = logger

        self.replay_buffer = ReplayBuffer(replay_buffer_size)
        self.train_step = 0
        self.last_loss = 0.0

        self.update_target_model()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

    def learn(self, list_sample_data):
        for sample in list_sample_data:
            self.replay_buffer.add(sample)

        if len(self.replay_buffer) < self.warmup_size:
            return {
                "loss": self.last_loss,
                "buffer_size": len(self.replay_buffer),
                "train_step": self.train_step,
            }

        batch = self.replay_buffer.sample(min(self.batch_size, len(self.replay_buffer)))
        states = torch.as_tensor(np.stack([i["state"] for i in batch]), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor([i["action"] for i in batch], dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.as_tensor([i["reward"] for i in batch], dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(
            np.stack([i["next_state"] for i in batch]), dtype=torch.float32, device=self.device
        )
        dones = torch.as_tensor([i["done"] for i in batch], dtype=torch.float32, device=self.device)

        q_values = self.model(states).gather(1, actions).squeeze(1)

        with torch.no_grad():
            # Double DQN target: online net selects action, target net evaluates it.
            next_actions = self.model(next_states).argmax(dim=1, keepdim=True)
            next_q_values = self.target_model(next_states).gather(1, next_actions).squeeze(1)
            targets = rewards + (1.0 - dones) * self.gamma * next_q_values

        loss = F.smooth_l1_loss(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
        self.optimizer.step()

        self.train_step += 1
        self.last_loss = float(loss.detach().cpu().item())

        if self.train_step % self.target_update_freq == 0:
            self.update_target_model()

        return {
            "loss": self.last_loss,
            "buffer_size": len(self.replay_buffer),
            "train_step": self.train_step,
        }
