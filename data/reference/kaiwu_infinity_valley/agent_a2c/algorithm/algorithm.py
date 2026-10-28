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
from agent_a2c.conf.conf import Config


class Algorithm:
    def __init__(self, model, optimizer, scheduler, device=None, logger=None, monitor=None):
        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.parameters = [p for param_group in self.optimizer.param_groups for p in param_group["params"]]
        self.train_step = 0

        self.logger = logger
        self.monitor = monitor

        self.vec_dim = np.prod(Config.FEATURE_VECTOR_SHAPE)
        self.last_report_monitor_time = 0

    def learn(self, list_sample_data):
        # Extract fields from SampleData array and convert to tensor
        # 从SampleData数组中提取字段并转换为tensor
        batch_obs = torch.stack([sample.obs for sample in list_sample_data]).to(self.device)
        batch_next_obs = torch.stack([sample.next_obs for sample in list_sample_data]).to(self.device)
        batch_obs_legal = torch.stack([sample.obs_legal for sample in list_sample_data]).to(self.device)
        action = torch.stack([sample.action for sample in list_sample_data]).to(self.device).long()
        val = torch.stack([sample.val for sample in list_sample_data]).to(self.device)
        reward = torch.stack([sample.reward for sample in list_sample_data]).to(self.device)
        ret = torch.stack([sample.ret for sample in list_sample_data]).to(self.device)
        done = torch.stack([sample.done for sample in list_sample_data]).to(self.device)

        # Extract feature vectors and image features.
        # 提取特征向量和图像特征
        feature_vec = batch_obs[:, : self.vec_dim].reshape(-1, *Config.FEATURE_VECTOR_SHAPE)
        feature_img = batch_obs[:, self.vec_dim :].reshape(-1, *Config.FEATURE_IMAGE_SHAPE)

        _feature_vec = batch_next_obs[:, : self.vec_dim].reshape(-1, *Config.FEATURE_VECTOR_SHAPE)
        _feature_img = batch_next_obs[:, self.vec_dim :].reshape(-1, *Config.FEATURE_IMAGE_SHAPE)

        obs_legal = batch_obs_legal.view(-1, *Config.ACTION_SHAPE)
        action = action.view(-1, 1)
        val = val.view(-1, 1)
        reward = reward.view(-1, 1)
        ret = ret.view(-1, 1)
        not_done = torch.where(
            done > 0.5, torch.tensor(0.0, device=self.device), torch.tensor(1.0, device=self.device)
        ).view(-1, 1)

        self.model.train()
        self.optimizer.zero_grad()

        # Forward propagation
        # 前向传播
        model_out = self.model(vec=feature_vec, img=feature_img)
        act_logits, pred_vals = model_out["act"], model_out["val"]

        # Calculate the value estimate for the next state
        # 计算下一个状态的价值估计
        with torch.no_grad():
            next_model_out = self.model(vec=_feature_vec, img=_feature_img)
            next_val = next_model_out["val"]

        # Calculate the action distribution
        # 计算动作分布
        act_logits[obs_legal == 0] = -1e10
        act_dists = torch.distributions.Categorical(logits=act_logits)
        act_log_prob = act_dists.log_prob(action.squeeze(-1)).unsqueeze(-1)
        dist_entropy = act_dists.entropy().mean()

        # Calculate the advantage function
        # 计算优势函数
        adv = reward + Config.GAMMA * next_val * not_done - val

        # Normalize advantage
        # 优势标准化
        adv_mean = adv.mean()
        adv_std = adv.std()
        adv = (adv - adv_mean) / (adv_std + 1e-8)

        # Calculate the target value
        # 计算目标值
        target_val = reward + Config.GAMMA * next_val * not_done

        # Calculate the loss
        # 计算损失
        policy_loss = (-act_log_prob * adv).mean() - Config.ENTROPY_LOSS_COEFF * dist_entropy

        # Calculate value loss using target values.
        # 使用目标值计算价值损失
        value_loss = F.mse_loss(target_val, pred_vals)

        loss = policy_loss + Config.VALUE_LOSS_COEFF * value_loss

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
                "policy_loss": policy_loss.detach(),
                "value_loss": value_loss.detach(),
            }
            if self.monitor:
                self.monitor.put_data({os.getpid(): monitor_data})

            self.last_report_monitor_time = now
