#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import torch
import numpy as np

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from kaiwudrl.interface.agent import BaseAgent
from agent_ppo.model.model import Model
from agent_ppo.feature.definition import *
from agent_ppo.conf.conf import Config
from agent_ppo.algorithm.algorithm import Algorithm
from agent_ppo.feature.preprocessor import Preprocessor


class Agent(BaseAgent):
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None):
        super().__init__(agent_type, device, logger, monitor)
        torch.manual_seed(0)
        self.device = device
        self.model = Model(device).to(self.device)
        initial_lr = Config.INIT_LEARNING_RATE_START
        parameters = self.model.parameters()
        self.optimizer = torch.optim.Adam(params=parameters, lr=initial_lr, betas=(0.9, 0.999), eps=1e-8)
        self.label_size_list = Config.LABEL_SIZE_LIST
        self.legal_action_size = Config.LEGAL_ACTION_SIZE_LIST
        self.logger = logger
        self.monitor = monitor
        self.algorithm = Algorithm(self.model, self.optimizer, self.device, self.logger, self.monitor)
        self.preprocessor = Preprocessor()

    def reset(self, env_obs):
        self.preprocessor = Preprocessor()

    def __predict_detail(self, list_obs_data, exploit_flag=False):
        feature = [obs_data.feature for obs_data in list_obs_data]
        legal_action = [obs_data.legal_action for obs_data in list_obs_data]
        self.model.set_eval_mode()

        s = torch.tensor(feature).view(1, Config.DIM_OF_OBSERVATION).float().to(self.device)
        with torch.no_grad():
            output_list = self.model(s, inference=True)

        np_output = []
        for idx, output in enumerate(output_list):
            np_out = output.cpu().numpy()
            np_output.append(np_out)

        logits, value = np_output[:2]
        list_act_data = list()
        for i in range(len(legal_action)):
            prob, action, d_action = self._sample_masked_action(logits[i], np.array(legal_action[i], dtype=np.float32))
            list_act_data.append(ActData(action=action, d_action=d_action, prob=prob, value=value))
        return list_act_data

    def predict(self, list_obs_data):
        return self.__predict_detail(list_obs_data, exploit_flag=False)

    def exploit(self, env_obs):
        obs_data, remain_info = self.observation_process(env_obs)
        if not obs_data:
            return 0
        act_data = self.__predict_detail([obs_data], exploit_flag=True)
        act = self.action_process(act_data[0], False)
        return act

    def learn(self, list_sample_data):
        return self.algorithm.learn(list_sample_data)

    def save_model(self, path=None, id="1"):
        # To save the model, it can consist of multiple files,
        # and it is important to ensure that each filename includes the "model.ckpt-id" field.
        # 保存模型, 可以是多个文件, 需要确保每个文件名里包括了model.ckpt-id字段
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"

        # Copy the model's state dictionary to the CPU
        # 将模型的状态字典拷贝到CPU
        model_state_dict = self.model.state_dict()
        model_state_dict_cpu = {k: v.clone().cpu() for k, v in self.model.state_dict().items()}
        torch.save(model_state_dict_cpu, model_file_path)

        self.logger.info(f"save model {model_file_path} successfully")

    def load_model(self, path=None, id="1"):
        # When loading the model, you can load multiple files,
        # and it is important to ensure that each filename matches the one used during the save_model process.
        # 加载模型, 可以加载多个文件, 注意每个文件名需要和save_model时保持一致
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        self.model.load_state_dict(torch.load(model_file_path, map_location=self.model.device))

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

        return (
            ObsData(
                feature=feature,
                legal_action=legal_actions,
                sub_action_mask=[
                    1,
                ]
                * Config.NUMB_HEAD,
            ),
            remain_info,
        )

    def action_process(self, act_data, is_stochastic=True):
        if is_stochastic:
            action = act_data.action
        else:
            action = act_data.d_action

        return int(action[0])

    # get final executable actions
    # 获取最终的可执行动作
    def _sample_masked_action(self, logits, legal_action):
        """
        Sample actions from predicted logits and legal actions
        return: probability, stochastic and deterministic actions with additional []
        """
        prob_list = []
        action_list = []
        d_action_list = []
        label_split_size = [sum(self.label_size_list[: index + 1]) for index in range(len(self.label_size_list))]
        legal_actions = np.split(legal_action, label_split_size[:-1])
        logits_split = np.split(logits, label_split_size[:-1])
        for index in range(0, len(self.label_size_list)):
            # Count non-zero elements, if all actions are illegal, then True
            # 统计非0元素数量，若全部动作都非法，则True
            if np.count_nonzero(legal_actions[index]) == 0:  # np.sum(~np.isnan(probs)) == 0:
                probs = [
                    0,
                ] * self.label_size_list[index]
                sample_action = 0
                d_action = 0
            else:
                probs = self._legal_soft_max(logits_split[index], legal_actions[index])
                sample_action = self._legal_sample(probs, use_max=False)
                d_action = self._legal_sample(probs, use_max=True)
            action_list.append(sample_action)
            d_action_list.append(d_action)
            prob_list += list(probs)

        return prob_list, action_list, d_action_list

    def _legal_soft_max(self, input_hidden, legal_action):
        _lsm_const_w, _lsm_const_e = 1e20, 1e-5
        _lsm_const_e = 0.00001

        tmp = input_hidden - _lsm_const_w * (1.0 - legal_action)
        tmp_max = np.max(tmp, keepdims=True)
        # Not necessary max clip 1
        # 不必将最大值裁剪到1
        tmp = np.clip(tmp - tmp_max, -_lsm_const_w, 1)
        tmp = (np.exp(tmp) + _lsm_const_e) * legal_action
        probs = tmp / (np.sum(tmp, keepdims=True) * 1.00001)
        return probs

    def _legal_sample(self, probs, legal_action=None, use_max=False):
        """
        Sample with probability, input probs should be 1D array
        按概率采样，输入概率应为一维数组
        """
        if use_max:
            return np.argmax(probs)

        return np.argmax(np.random.multinomial(1, probs, size=1))
