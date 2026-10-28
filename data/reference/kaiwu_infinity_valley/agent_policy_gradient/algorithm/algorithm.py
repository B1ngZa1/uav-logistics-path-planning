#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import torch
import os
import time
from torch.nn import functional as F
import numpy as np
from agent_policy_gradient.conf.conf import Config


class Algorithm:
    def __init__(self, model, optimizer, scheduler, device=None, logger=None, monitor=None):
        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.parameters = [p for param_group in self.optimizer.param_groups for p in param_group["params"]]
        self.train_step = 0

        self.last_report_monitor_time = 0
        self.logger = logger
        self.monitor = monitor

        self.vec_dim = np.prod(Config.FEATURE_VECTOR_SHAPE)

    def learn(self, list_sample_data):
        # Extract fields from SampleData array and convert to tensor
        # 从SampleData数组中提取字段并转换为tensor
        batch_obs = torch.stack([sample.obs for sample in list_sample_data]).to(self.device)
        batch_obs_legal = torch.stack([sample.obs_legal for sample in list_sample_data]).to(self.device)
        action = torch.stack([sample.action for sample in list_sample_data]).to(self.device).long()
        ret = torch.stack([sample.ret for sample in list_sample_data]).to(self.device)

        # Extract feature vectors and image features.
        # 提取特征向量和图像特征
        feature_vec = batch_obs[:, : self.vec_dim].reshape(-1, *Config.FEATURE_VECTOR_SHAPE)
        feature_img = batch_obs[:, self.vec_dim :].reshape(-1, *Config.FEATURE_IMAGE_SHAPE)

        obs_legal = batch_obs_legal.view(-1, *Config.ACTION_SHAPE)
        action = action.view(-1, 1)
        ret = ret.view(-1, 1)

        # Normalize accumulated returns
        # 对累积回报进行标准化
        ret_mean = ret.mean()
        ret_std = ret.std()
        ret = (ret - ret_mean) / (ret_std + 1e-8)

        self.model.train()
        self.optimizer.zero_grad()

        # Forward propagation
        # 前向传播
        act_logits = self.model(vec=feature_vec, img=feature_img)
        # Calculate the action distribution
        # 计算动作分布
        act_logits[obs_legal == 0] = -1e10
        act_dists = torch.distributions.Categorical(logits=act_logits)
        act_log_prob = act_dists.log_prob(action.squeeze(-1)).unsqueeze(-1)
        dist_entropy = act_dists.entropy().mean()
        loss = (-act_log_prob * ret).mean() - Config.ENTROPY_LOSS_COEFF * dist_entropy
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)

        # Gradient Descent
        # 梯度下降
        self.optimizer.step()
        self.train_step += 1

        # update the learning rate
        # 更新学习率
        self.scheduler.step(self.train_step)

        # Periodically report monitoring
        # 按照间隔上报监控
        now = time.time()
        if now - self.last_report_monitor_time >= 60:
            monitor_data = {
                "policy_loss": loss.detach(),
            }
            if self.monitor:
                self.monitor.put_data({os.getpid(): monitor_data})

            self.last_report_monitor_time = now
