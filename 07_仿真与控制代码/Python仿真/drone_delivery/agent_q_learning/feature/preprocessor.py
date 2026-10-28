#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
无人机配送 — Q-learning特征预处理模块
Drone Delivery — Q-learning Feature Preprocessing Module

职责 / Responsibility:
    将原始环境观测转换为Q-learning可用的状态特征和合法动作列表。
    Converts raw environment observations into state features and legal action lists
    usable by Q-learning.

参考 / Reference:
    kaiwu_gorge_walk_v2/agent_q_learning/feature/preprocessor.py

领域适配 / Domain Adaptation:
    Gorge Walk treasure collection → Drone pickup-delivery:
    - hero_pos → drone_pos (x,y grid)
    - treasure → delivery_order
    - treasure_state (10-bit binary) → order_status_mask (5-bit)
    - endpoint → depot at [30,10]
    - 4-direction (UP/DOWN/LEFT/RIGHT) → same
    - max_step=1000 → max_steps=500
    - treasure_score=100 → delivery_reward=+100
    - end_score=150 → safe_return_bonus=+500
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple

from agent_q_learning.conf.conf import Config
from agent_q_learning.feature.definition import ObsData, ActData


# ============================================================
# 动作空间常量
# Action space constants
# ============================================================

# 4方向动作映射: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
# 4-direction action mapping: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
ACTION_DELTAS: Dict[int, Tuple[int, int]] = {
    0: (0, 1),    # 上 / UP     (x, y+1)
    1: (0, -1),   # 下 / DOWN   (x, y-1)
    2: (-1, 0),   # 左 / LEFT   (x-1, y)
    3: (1, 0),    # 右 / RIGHT  (x+1, y)
}

# 动作名称 (用于日志) / Action names (for logging)
ACTION_NAMES: Dict[int, str] = {
    0: "UP",
    1: "DOWN",
    2: "LEFT",
    3: "RIGHT",
}


# ============================================================
# 观测提取辅助函数
# Observation extraction helper functions
# ============================================================

def _unpack_obs(env_obs: Dict[str, Any]) -> Dict[str, Any]:
    """
    解包环境观测，处理 {"observation": {...}} 包装格式。
    Unpack environment observation, handling {"observation": {...}} wrapper format.

    Args:
        env_obs: 环境观测 (可能包含包装层) / Environment observation (possibly wrapped)

    Returns:
        内部的观测字典 / Inner observation dict
    """
    if isinstance(env_obs, dict):
        return env_obs.get("observation", env_obs)
    return {}


def _extract_position(env_obs: Dict[str, Any]) -> Tuple[int, int]:
    """
    提取无人机位置 (x, y)。
    Extract drone position (x, y).

    Args:
        env_obs: 环境观测字典 / Environment observation dict

    Returns:
        (x, y) 整数坐标元组 / (x, y) integer coordinate tuple
    """
    obs = _unpack_obs(env_obs)
    pos = obs.get("position", np.array([0, 0]))
    if isinstance(pos, np.ndarray):
        return int(pos[0]), int(pos[1])
    if isinstance(pos, (list, tuple)):
        return int(pos[0]), int(pos[1])
    return 0, 0


def _order_status_bits(env_obs: Dict[str, Any]) -> int:
    """
    编码订单状态为整数比特掩码 (5-bit)。
    Encode order status as integer bitmask (5-bit).

    每位表示一个订单: 0=未完成, 1=已完成。
    Each bit represents one order: 0=incomplete, 1=completed.

    Args:
        env_obs: 环境观测字典 / Environment observation dict

    Returns:
        订单状态掩码 (0 ~ 31) / Order status mask (0 ~ 31)
    """
    obs = _unpack_obs(env_obs)
    status = obs.get("order_status", np.zeros(Config.NUM_ORDERS, dtype=np.int8))
    if isinstance(status, np.ndarray):
        bits = [str(int(b)) for b in status]
    elif isinstance(status, (list, tuple)):
        bits = [str(int(b)) for b in status]
    else:
        bits = ["0"] * Config.NUM_ORDERS
    return int("".join(bits), 2)


def _carrying_flag(env_obs: Dict[str, Any]) -> int:
    """
    提取携带状态标志。
    Extract carrying state flag.

    Args:
        env_obs: 环境观测字典 / Environment observation dict

    Returns:
        0=空手 (empty), 1=携带中 (carrying)
    """
    obs = _unpack_obs(env_obs)
    carrying = obs.get("carrying", 0)
    return int(carrying)


# ============================================================
# 状态编码 / State encoding
# ============================================================

def encode_state_key(env_obs: Dict[str, Any]) -> int:
    """
    将环境观测编码为唯一整数状态键。
    Encode environment observation into a unique integer state key.

    编码方式 / Encoding scheme::

        position_idx = x * GRID_SIZE + y        → 0 ~ 1599
        order_status = 5-bit mask                → 0 ~ 31
        carrying = 0 or 1                        → 0 ~ 1

        state_key = (position_idx << 6) | (order_status << 1) | carrying

    Args:
        env_obs: 环境观测字典 / Environment observation dict

    Returns:
        唯一整数状态键 / Unique integer state key
    """
    x, y = _extract_position(env_obs)
    pos_idx = int(x) * Config.GRID_SIZE + int(y)
    status_bits = _order_status_bits(env_obs)
    carrying = _carrying_flag(env_obs)

    # 位拼接: position(11位) + status(5位) + carrying(1位) = 17位
    # Bit concatenation: position(11 bits) + status(5 bits) + carrying(1 bit) = 17 bits
    return (pos_idx << 6) | (status_bits << 1) | carrying


def get_state_key(env_obs: Dict[str, Any]) -> int:
    """
    Q-learning使用的状态特征 — 状态键本身。
    Q-learning state feature — the state key itself.

    等同于 encode_state_key 的别名。
    Alias for encode_state_key, compatible with simpler naming conventions.

    Args:
        env_obs: 环境观测字典 / Environment observation dict

    Returns:
        整数状态键 / Integer state key
    """
    return encode_state_key(env_obs)


# ============================================================
# 障碍物与合法动作
# Obstacles and legal actions
# ============================================================

# 惰性初始化的障碍物掩码 / Lazily initialized obstacle mask
_OBSTACLE_MASK: Optional[np.ndarray] = None


def _build_obstacle_mask() -> np.ndarray:
    """
    构建障碍物掩码 — 与 Simple2DDeliveryEnv 中的 _build_obstacle_mask 保持一致。
    Build obstacle mask — consistent with _build_obstacle_mask in Simple2DDeliveryEnv.

    武汉学院Zone B地图的简化表示 / Simplified representation of Zone B map.

    Returns:
        布尔数组，True表示障碍物 / Boolean array, True = obstacle
    """
    mask = np.zeros((Config.GRID_SIZE, Config.GRID_SIZE), dtype=bool)

    # 教学楼群 / Teaching buildings
    mask[14:19, 14:21] = True
    # 宿舍楼群 / Dormitory buildings
    mask[26:31, 20:25] = True
    # 湖区水域 / Lake area
    mask[8:13, 4:9] = True
    # 电力线路由 / Power line route
    mask[22, 12:17] = True
    # 其他小建筑 / Other small buildings
    mask[6:8, 22:24] = True
    mask[19:21, 10:12] = True

    # 边界不可通行 / Boundaries are impassable
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True

    # 确保配送中心可行走 / Ensure depot is walkable
    depot_x, depot_y = Config.DEPOT
    mask[depot_x, depot_y] = False

    return mask


def _get_obstacle_mask() -> np.ndarray:
    """
    获取障碍物掩码 (惰性初始化)。
    Get obstacle mask (lazy initialization).
    """
    global _OBSTACLE_MASK
    if _OBSTACLE_MASK is None:
        _OBSTACLE_MASK = _build_obstacle_mask()
    return _OBSTACLE_MASK


def is_walkable(x: int, y: int) -> bool:
    """
    检查坐标是否可行走。
    Check if a coordinate is walkable.

    Args:
        x: x坐标 / x coordinate
        y: y坐标 / y coordinate

    Returns:
        True 表示可行走 / True means walkable
    """
    mask = _get_obstacle_mask()
    if not (0 <= x < Config.GRID_SIZE and 0 <= y < Config.GRID_SIZE):
        return False
    return not mask[x, y]


def compute_legal_actions(env_obs: Dict[str, Any]) -> List[int]:
    """
    从环境观测中计算合法动作列表。
    Compute legal action list from environment observation.

    合法动作 = 执行后不会进入障碍物或越界的动作。
    Legal actions = actions that won't result in entering an obstacle or going out of bounds.

    Args:
        env_obs: 环境观测字典 / Environment observation dict

    Returns:
        合法动作索引列表 / List of legal action indices: [0,1,2,3] subset
    """
    x, y = _extract_position(env_obs)
    legal: List[int] = []
    for action, (dx, dy) in ACTION_DELTAS.items():
        nx, ny = x + dx, y + dy
        if is_walkable(nx, ny):
            legal.append(action)

    # 安全回退: 如果所有方向都被阻挡，返回全部动作 (极少发生)
    # Safety fallback: if all directions are blocked, return all (rarely happens)
    return legal if legal else list(range(Config.ACTION_SIZE))


# ============================================================
# Preprocessor 类 — 特征处理主接口
# Preprocessor class — Feature processing main interface
# ============================================================

class Preprocessor:
    """
    特征预处理器 — 将原始环境观测转换为Q-learning所需的数据格式。
    Feature preprocessor — converts raw environment observations into data formats
    required by Q-learning.

    职责 / Responsibilities:
        1. observation_process: 编码状态特征 + 计算合法动作 → ObsData
        2. action_process: 提取可执行动作 → int

    Usage / 用法::

        preprocessor = Preprocessor()
        obs_data = preprocessor.observation_process(env_obs)
        action = preprocessor.action_process(act_data)
    """

    def __init__(self):
        """初始化预处理器 / Initialize preprocessor."""
        # 确保障碍物掩码已加载 / Ensure obstacle mask is loaded
        _get_obstacle_mask()

    def observation_process(self, env_obs: Dict[str, Any]) -> ObsData:
        """
        处理环境观测，生成观测数据。
        Process environment observation, produce observation data.

        步骤 / Steps:
            1. 编码状态键 / Encode state key (position + order_status + carrying)
            2. 计算合法动作 / Compute legal actions (avoid obstacles)
            3. 包装为 ObsData / Wrap into ObsData

        Args:
            env_obs: 原始环境观测 / Raw environment observation dict

        Returns:
            ObsData 对象，包含 feature (状态键) 和 legal_actions (合法动作列表)
            ObsData object with feature (state key) and legal_actions (legal action list)
        """
        # 编码状态键: 将position、order_status、carrying组合为唯一整数
        # Encode state key: combine position, order_status, carrying into unique int
        feature = encode_state_key(env_obs)

        # 计算合法动作: 检查4个方向中哪些不撞墙/不越界
        # Compute legal actions: check which of 4 directions avoid obstacles/OOB
        legal_actions = compute_legal_actions(env_obs)

        return ObsData(feature=feature, legal_actions=legal_actions)

    def action_process(self, act_data: ActData) -> int:
        """
        处理动作数据，提取可执行的动作索引。
        Process action data, extract executable action index.

        对动作进行合法性检查 (确保在 [0, ACTION_SIZE) 范围内)。
        Performs validity check (clamps to [0, ACTION_SIZE) range).

        Args:
            act_data: 动作数据对象 / Action data object (ActData)

        Returns:
            可执行的动作索引 (0=UP, 1=DOWN, 2=LEFT, 3=RIGHT)
            Executable action index
        """
        action = act_data.act
        if action is None:
            return 0
        # 钳位到有效范围 / Clamp to valid range
        return int(max(0, min(Config.ACTION_SIZE - 1, action)))
