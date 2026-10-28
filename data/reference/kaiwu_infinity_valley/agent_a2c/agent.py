#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import os
import time
from agent_a2c.model.model import PolicyValueModel as Model
from agent_a2c.feature.definition import ActData, ObsData
from torch.nn import functional as F
import numpy as np
from kaiwudrl.interface.agent import BaseAgent
from agent_a2c.conf.conf import Config
from torch.optim.lr_scheduler import LambdaLR
from agent_a2c.algorithm.algorithm import Algorithm
from agent_a2c.feature.preprocessor import Preprocessor


def stable_softmax(x, dim=None):
    if dim is None:
        raise ValueError("dim must be specified")

    # Subtract the maximum value along the specified dimension for numerical stability
    # 沿指定维度减去最大值以实现数值稳定性
    max_vals, _ = torch.max(x, dim=dim, keepdim=True)
    x_exp = torch.exp(x - max_vals)
    # Compute the softmax
    # 计算 softmax
    softmax_vals = x_exp / torch.sum(x_exp, dim=dim, keepdim=True)
    return softmax_vals


class Agent(BaseAgent):
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None):
        super().__init__(agent_type, device, logger, monitor)
        self.lr = Config.START_LR
        self.target_lr = Config.TARGET_LR
        self.target_step = Config.TARGET_STEP

        self.device = device
        self.model = Model()
        self.model.to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=self.lr_lambda)

        self.predict_count = 0

        self.agent_type = agent_type
        self.logger = logger
        self.monitor = monitor

        self.vec_dim = np.prod(Config.FEATURE_VECTOR_SHAPE)

        self.algorithm = Algorithm(self.model, self.optimizer, self.scheduler, self.device, self.logger, self.monitor)
        self.preprocessor = Preprocessor()

    def lr_lambda(self, step):
        # Define learning rate decay function
        # 定义学习率衰减函数
        if step > self.target_step:
            return self.target_lr / self.lr
        else:
            return 1.0 - ((1.0 - self.target_lr / self.lr) * step / self.target_step)

    def __convert_to_tensor(self, data):
        if isinstance(data, list):
            return torch.tensor(
                np.array(data),
                dtype=torch.float32,
            ).to(self.device)
        else:
            return torch.tensor(
                data,
                dtype=torch.float32,
            ).to(self.device)

    def __predict_detail(self, list_obs_data, exploit_flag=False):
        batch = len(list_obs_data)
        feature_vec = self.__convert_to_tensor([obs_data.feature[: self.vec_dim] for obs_data in list_obs_data])
        feature_img = self.__convert_to_tensor([obs_data.feature[self.vec_dim :] for obs_data in list_obs_data]).view(
            batch, *Config.FEATURE_IMAGE_SHAPE
        )

        legal_act = [obs_data.legal_act for obs_data in list_obs_data]
        legal_act = torch.tensor(np.array(legal_act)).bool().to(self.device)

        model = self.model
        model.eval()

        list_act_data = list()
        with torch.no_grad():
            output_dict = self.model(vec=feature_vec, img=feature_img)
            output_val = output_dict["val"]
            output_act = output_dict["act"]
            output_act[legal_act == False] = -1e10
            action_dist = torch.distributions.Categorical(logits=output_act)

            if not exploit_flag:
                probs = action_dist.probs
                action = action_dist.sample()
                act = action.cpu().view(-1).tolist()
            else:
                probs = stable_softmax(output_act, dim=1)
                act = probs.argmax(dim=1).cpu().view(-1).tolist()

            # output_val shape: (batch_size, 1), squeeze 后变成 (batch_size,) 或标量
            # 转为列表后是 [val1, val2, ...] 而不是 [[val1], [val2], ...]
            list_act_data.append(ActData(act=act, val=output_val.squeeze(-1).tolist()))

        self.predict_count += 1
        return list_act_data

    def reset(self, env_obs):
        self.preprocessor = Preprocessor()

    def predict(self, list_obs_data):
        return self.__predict_detail(list_obs_data, exploit_flag=False)

    def exploit(self, env_obs):
        obs_data, remain_info = self.observation_process(env_obs)
        if not obs_data:
            return [[None, None, None]]
        act_data = self.__predict_detail([obs_data], exploit_flag=True)
        act, val = self.action_process(act_data[0])
        return act

    def learn(self, list_sample_data):
        self.algorithm.learn(list_sample_data)

    def save_model(self, path=None, id="1"):
        # To save the model, it can consist of multiple files,
        # and it is important to ensure that each filename includes the "model.ckpt-id" field.
        # 保存模型, 可以是多个文件, 需要确保每个文件名里包括了model.ckpt-id字段
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"

        # Copy the model's state dictionary to the CPU
        # 将模型的状态字典拷贝到CPU
        model_state_dict_cpu = {k: v.clone().cpu() for k, v in self.model.state_dict().items()}
        torch.save(model_state_dict_cpu, model_file_path)

        self.logger.info(f"save model {model_file_path} successfully")

    def load_model(self, path=None, id="1"):
        # When loading the model, you can load multiple files,
        # and it is important to ensure that each filename matches the one used during the save_model process.
        # 加载模型, 可以加载多个文件, 注意每个文件名需要和save_model时保持一致
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        self.model.load_state_dict(
            torch.load(model_file_path, map_location=self.device),
        )

        self.logger.info(f"load model {model_file_path} successfully")

    def observation_process(self, env_obs):
        """
        This function is an important feature processing function, mainly responsible for:
            - Parsing information in the raw data
            - Parsing preprocessed feature data
            - Processing the features and returning the processed feature vector
            - Concatenation of features
            - Annotation of legal actions
        Function inputs:
            - env_obs: Environment observation dict containing 'observation' and 'extra_info'
        Function outputs:
            - ObsData: Observation data for model inference
            - remain_info: Other data for reward calculation

        该函数是特征处理的重要函数, 主要负责：
            - 解析原始数据里的信息
            - 解析预处理后的特征数据
            - 对特征进行处理, 并返回处理后的特征向量
            - 特征的拼接
            - 合法动作的标注
        函数的输入：
            - env_obs: 环境观测字典，包含'observation'和'extra_info'
        函数的输出：
            - ObsData: 用于模型推理的观测数据
            - remain_info: 用于奖励计算的其他数据
        """
        obs = env_obs["observation"]
        extra_info = env_obs["extra_info"]

        (
            hero_info,
            treasure_infos,
            monster_infos,
            obstacle_map,
            memory_map,
            treasure_map,
            monster_map,
            legal_actions,
            legal_actions_feature,
            recent_position_map,
        ) = self.preprocessor.process(obs)

        frame_feature = env_obs["frame_no"] / 6100

        # Hero feature
        # 英雄特征
        hero_grid_pos = hero_info["grid_pos"]
        hero_pos = obs["frame_state"]["heroes"][0]["pos"]
        hero_feature = [0] * 2
        hero_feature[0] = hero_info["norm_pos"][0]
        hero_feature[1] = hero_info["norm_pos"][1]

        # Monster feature
        # 怪物特征
        monster_feature = []
        monster_feature_num = 2
        monster_feature_dim = 15
        monster_count = 0

        for monster_id in sorted(monster_infos.keys()):
            if monster_count >= monster_feature_num:
                break
            monster = monster_infos[monster_id]
            grid_distance = monster["grid_distance"]
            monster_feature.extend(
                [
                    monster["norm_pos"][0],
                    monster["norm_pos"][1],
                    grid_distance / (len(obs["map_info"]) * 1.414) if grid_distance != -1 else 1.0,
                    *(1.0 * ((monster["direction"] - 1) == i) for i in range(8)),
                    (max(monster["pos"]["x"] - hero_pos["x"], 0) / 128000),
                    (max(-monster["pos"]["x"] + hero_pos["x"], 0) / 128000),
                    (max(monster["pos"]["z"] - hero_pos["z"], 0) / 128000),
                    (max(-monster["pos"]["z"] + hero_pos["z"], 0) / 128000),
                ]
            )
            monster_count += 1
        while monster_count < monster_feature_num:
            monster_feature.extend([0] * monster_feature_dim)
            monster_count += 1

        # Treasure feature
        # 宝箱特征
        treasure_feature = []
        treasure_feature_num = 10
        treasure_feature_dim = 7
        treasure_count = 0

        min_grid_distance = float("inf")
        min_distance_treasure_count = -1
        min_l2_distance = float("inf")
        min_l2_distance_treasure_count = -1
        end_pos_direction = -1

        for treasure_id in sorted(treasure_infos.keys()):
            if treasure_infos[treasure_id]["status"] == 0:
                continue
            if treasure_count >= treasure_feature_num:
                break

            treasure = treasure_infos[treasure_id]
            grid_distance = treasure["grid_distance"]
            l2_distance = treasure["l2_distance"]

            if grid_distance != -1 and grid_distance < min_grid_distance:
                min_grid_distance = grid_distance
                min_distance_treasure_count = treasure_count
                end_pos_direction = treasure["direction"]

            if l2_distance < min_l2_distance:
                min_l2_distance = l2_distance
                min_l2_distance_treasure_count = treasure_count
                if grid_distance == -1:
                    end_pos_direction = treasure["direction"]

            treasure_feature.extend(
                [
                    treasure["norm_pos"][0],
                    treasure["norm_pos"][1],
                    grid_distance / (len(obs["map_info"]) * 1.414) if grid_distance != -1 else 1.0,
                    (max(treasure["pos"]["x"] - hero_pos["x"], 0) / 128000),
                    (max(-treasure["pos"]["x"] + hero_pos["x"], 0) / 128000),
                    (max(treasure["pos"]["z"] - hero_pos["z"], 0) / 128000),
                    (max(-treasure["pos"]["z"] + hero_pos["z"], 0) / 128000),
                ]
            )
            treasure_count += 1
        while treasure_count < treasure_feature_num:
            treasure_feature.extend([0] * treasure_feature_dim)
            treasure_count += 1

        # Target treasure feature
        # 目标宝箱特征
        end_pos_feature = [0] * (treasure_feature_num + 8)
        if min_distance_treasure_count != -1:
            end_pos_feature[min_distance_treasure_count] = 1
        elif min_l2_distance_treasure_count != -1:
            end_pos_feature[min_l2_distance_treasure_count] = 1
        if end_pos_direction != -1:
            direction_one_hot = [1.0 * ((end_pos_direction - 1) == i) for i in range(8)]
            end_pos_feature[-8:] = direction_one_hot

        # Feature vector
        # 特征向量
        feature_vec = (
            list([frame_feature])
            + list(hero_feature)
            + list(monster_feature)
            + list(treasure_feature)
            + list(end_pos_feature)
            + list(legal_actions_feature)
        )

        # Feature maps
        # 特征图
        feature_map = list(obstacle_map) + list(memory_map) + list(treasure_map) + list(monster_map)

        # Construct final feature
        # 构造最终特征
        feature = feature_vec + list(feature_map)

        # The remaining elements that are not required for prediction processing but are needed for reward calculation.
        # 其余不需要预测处理但是在计算reward时需要的
        remain_info = {
            "memory_map": memory_map,
            "monster_infos": monster_infos,
            "treasure_infos": treasure_infos,
            "recent_position_map": recent_position_map,
        }

        return ObsData(feature=feature, legal_act=legal_actions), remain_info

    def action_process(self, act_data):
        act, val = act_data.act, act_data.val
        return act[0], val[0]
