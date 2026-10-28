#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


from common_python.utils.common_func import create_cls
import numpy as np
from agent_policy_gradient.feature.preprocessor import bump, convert_pos_to_grid_pos
from agent_policy_gradient.conf.conf import Config

# The create_cls function is used to dynamically create a class. The first parameter of the function is the type name,
# and the remaining parameters are the attributes of the class, which should have a default value of None.
# create_cls函数用于动态创建一个类，函数第一个参数为类型名称，剩余参数为类的属性，属性默认值应设为None
ObsData = create_cls(
    "ObsData",
    feature=None,
    legal_act=None,
    frame_no=None,
)


ActData = create_cls(
    "ActData",
    act=None,
)


# Sample data structure for Policy Gradient training
# Policy Gradient训练样本数据结构
SampleData = create_cls(
    "SampleData",
    obs=int(np.prod(Config.FEATURE_VECTOR_SHAPE) + np.prod(Config.FEATURE_IMAGE_SHAPE)),
    next_obs=int(np.prod(Config.FEATURE_VECTOR_SHAPE) + np.prod(Config.FEATURE_IMAGE_SHAPE)),
    obs_legal=int(np.prod(Config.ACTION_SHAPE)),
    next_obs_legal=int(np.prod(Config.ACTION_SHAPE)),
    action=1,
    reward=1,
    ret=1,
    done=1,
    model_version=1,
)


def reward_shaping(frame_no, score, terminated, truncated, remain_info, _remain_info, obs, _obs):

    is_bump = False

    # Get the current position coordinates of the agent
    # 获取当前智能体的位置坐标
    pos = _obs["frame_state"]["heroes"][0]["pos"]
    curr_pos_x, curr_pos_z = pos["x"], pos["z"]

    # Get the agent's position from the previous frame
    # 获取智能体上一帧的位置
    prev_pos = obs["frame_state"]["heroes"][0]["pos"]
    prev_pos_x, prev_pos_z = prev_pos["x"], prev_pos["z"]

    # Rewards related to the treasure chest
    # 与宝箱相关的奖励
    reward_treasure_dist = 0
    reward_treasure = 0

    treasure_relative_infos = _remain_info.get("treasure_infos")
    prev_treasure_relative_infos = remain_info.get("treasure_infos")

    if prev_treasure_relative_infos and treasure_relative_infos:
        prev_end_treasure_id = min(
            prev_treasure_relative_infos,
            key=lambda k: (
                prev_treasure_relative_infos[k]["grid_distance"]
                if prev_treasure_relative_infos[k]["grid_distance"] != -1
                else float("inf")
            ),
        )
        end_treasure_relative_info = treasure_relative_infos.get(prev_end_treasure_id)
        prev_end_treasure_relative_info = prev_treasure_relative_infos.get(prev_end_treasure_id)

        if prev_end_treasure_relative_info and end_treasure_relative_info:
            if prev_end_treasure_relative_info["status"] == 1 and end_treasure_relative_info["status"] == 1:
                # Reward for getting closer to the treasure chest (only consider the nearest one)
                # 向宝箱靠近的奖励(仅考虑最近的宝箱)
                if (
                    prev_end_treasure_relative_info["grid_distance"] != -1
                    and end_treasure_relative_info["grid_distance"] != -1
                ):
                    reward_treasure_dist = (
                        2.0
                        * int(
                            end_treasure_relative_info["grid_distance"]
                            < prev_end_treasure_relative_info["grid_distance"]
                        )
                        - 1.0
                    )

            # Reward for getting the treasure chest
            # 获得宝箱的奖励
            if prev_end_treasure_relative_info["status"] == 1.0 and end_treasure_relative_info["status"] == 0.0:
                reward_treasure = 1

    # Penalty for bumping into the wall
    # 撞墙的惩罚
    reward_bump = 0
    is_bump = bump(curr_pos_x, curr_pos_z, prev_pos_x, prev_pos_z)

    # Determine whether it bumps into the wall
    # 判断是否撞墙
    if is_bump:
        # Give a relatively large penalty for bumping into the wall,
        # so that the agent can learn not to bump into the wall as soon as possible
        # 对撞墙给予一个比较大的惩罚，以便agent能够尽快学会不撞墙
        reward_bump = -1

    # Exploration Reward
    # 生存奖励
    reward_survival = int(not (terminated or truncated))

    # Exploration Reward
    # 探索奖励
    reward_exploration = 0
    recent_position_map = remain_info.get("recent_position_map")
    hero_grid_pos = convert_pos_to_grid_pos(curr_pos_x, curr_pos_z)

    if (hero_grid_pos[0], hero_grid_pos[1]) not in recent_position_map:
        reward_exploration = 1
    else:
        pass_times = recent_position_map[(hero_grid_pos[0], hero_grid_pos[1])]
        reward_exploration = max(-0.5 * pass_times, -10)

    # Reward for monster encounters
    # 怪物奖励
    reward_monster = 0

    if remain_info.get("monster_infos") and _remain_info.get("monster_infos"):
        monster_infos = _remain_info.get("monster_infos")
        monster_dists = list(map(lambda p: p["grid_distance"], monster_infos.values()))
        reachable_monster_dists = [dist for dist in monster_dists if dist != -1]
        if reachable_monster_dists and min(reachable_monster_dists) < 4:
            reward_monster -= 1

    """
    Concatenation of rewards: Here are 6 rewards provided, students can concatenate as needed,
    and can also add new rewards themselves
    奖励的拼接: 这里提供了6个奖励, 同学们按需自行拼接, 也可以自行添加新的奖励
    """
    reward_weight = {
        "reward_treasure_dist": 0.05,
        "reward_treasure": 10,
        "reward_bump": 0.5,
        "reward_survival": 0.01,
        "reward_monster": 0.5,
        "reward_exploration": 0.2,
    }
    reward_list = [
        reward_treasure_dist * reward_weight["reward_treasure_dist"],
        reward_treasure * reward_weight["reward_treasure"],
        reward_bump * reward_weight["reward_bump"],
        reward_survival * reward_weight["reward_survival"],
        reward_monster * reward_weight["reward_monster"],
        reward_exploration * reward_weight["reward_exploration"],
    ]
    reward_sum = round(sum(reward_list), 4)

    return reward_sum, reward_list, is_bump


class SampleManager:
    """
    Sample manager - for Policy Gradient distributed training sample collection and processing
    样本管理器 - 用于Policy Gradient分布式训练的样本收集和处理
    """

    def __init__(self):
        self.obs = []
        self._obs = []
        self.obs_legal = []
        self._obs_legal = []
        self.actions = []
        self.rewards = []
        self.rets = []
        self.dones = []
        self.model_versions = []
        self.count = 0

    def add(self, obs, _obs, obs_legal, _obs_legal, action, reward, ret, done, model_version=0):
        """
        Add a single sample
        添加单个样本
        """
        self.obs.append(obs)
        self._obs.append(_obs)
        self.obs_legal.append(obs_legal)
        self._obs_legal.append(_obs_legal)
        self.actions.append(action)
        self.rewards.append(reward)
        self.rets.append(ret)
        self.dones.append(done)
        self.model_versions.append(model_version)
        self.count += 1

    def process_last_frame(self):
        """
        Process the last frame: calculate return
        处理最后一帧: 计算return
        """
        if self.count == 0:
            return

        # Calculate return (cumulative reward)
        # 计算return (累计奖励)
        gamma = Config.GAMMA
        G_list = []

        for i in reversed(range(self.count)):
            reward = self.rewards[i]
            G_list.append(gamma * (G_list[-1] if len(G_list) != 0 else 0) + reward)
        G_list = list(reversed(G_list))

        for i in range(self.count):
            self.rets[i] = G_list[i]

    def _get_game_data(self):
        """
        Convert sample data to SampleData array format
        转换样本数据为SampleData数组格式
        """
        if self.count == 0:
            return []

        # Convert lists to numpy arrays
        # 将列表转为numpy数组
        obs = np.array(self.obs, dtype=np.float32)
        _obs = np.array(self._obs, dtype=np.float32)
        obs_legal = np.array(self.obs_legal, dtype=np.float32)
        _obs_legal = np.array(self._obs_legal, dtype=np.float32)
        actions = np.array(self.actions, dtype=np.float32)
        rewards = np.array(self.rewards, dtype=np.float32)
        rets = np.array(self.rets, dtype=np.float32)
        dones = np.array(self.dones, dtype=np.float32)
        model_versions = np.array(self.model_versions, dtype=np.float32)

        # Use list comprehension with numpy indexing
        # 使用列表推导式配合numpy索引
        return [
            SampleData(
                obs=obs[i],
                next_obs=_obs[i],
                obs_legal=obs_legal[i],
                next_obs_legal=_obs_legal[i],
                action=actions[i],
                reward=rewards[i],
                ret=rets[i],
                done=dones[i],
                model_version=model_versions[i],
            )
            for i in range(self.count)
        ]

    def get_game_data(self):
        """
        Get sample data and clear
        获取样本数据并清空
        """
        ret = self._get_game_data()
        self.clear()
        return ret

    def clear(self):
        """
        Clear all data
        清空所有数据
        """
        self.obs = []
        self._obs = []
        self.obs_legal = []
        self._obs_legal = []
        self.actions = []
        self.rewards = []
        self.rets = []
        self.dones = []
        self.model_versions = []
        self.count = 0
