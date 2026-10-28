#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
无人机配送 — SARSA特征预处理模块
Drone Delivery — SARSA Feature Preprocessing Module

职责 / Responsibility:
    将原始环境观测转换为SARSA可用的状态特征和合法动作列表。
    Converts raw environment observations into state features and legal action lists
    usable by SARSA.

SARSA的预处理与Q-learning完全相同 — 将观测编码为整数状态键。
SARSA's preprocessing is identical to Q-learning — encode observation as integer state key.

参考 / Reference:
    kaiwu_gorge_walk_v2/agent_sarsa/feature/preprocessor.py



领域适配 / Domain Adaptation:
    峡谷行走(Gorge Walk) — 64状态, 4动作
      → 无人机配送(Drone Delivery) — 102400状态, 4动作 + 订单状态
    Gorge Walk (64 states, 4 actions)
      → Drone Delivery (102400 states, 4 actions + order status)
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple

from agent_sarsa.conf.conf import Config


# ============================================================
# 动作空间常量
# Action Space Constants
# ============================================================

# 4方向动作映射: 0=UP(上), 1=DOWN(下), 2=LEFT(左), 3=RIGHT(右)
# 4-direction action mapping: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
ACTION_DELTAS = {
    0: (0, 1),    # 上 / UP: y + 1
    1: (0, -1),   # 下 / DOWN: y - 1
    2: (-1, 0),   # 左 / LEFT: x - 1
    3: (1, 0),    # 右 / RIGHT: x + 1
}

# 动作名称(调试用) / Action names (for debugging)
ACTION_NAMES = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}


# ============================================================
# 观测提取辅助函数
# Observation Extraction Helpers
# ============================================================

def _observation(env_obs: Dict[str, Any]) -> Dict[str, Any]:
    """
    从环境观测中提取observation子字典。
    Extract observation sub-dict from env observation.

    支持直接dict或 {"observation": {...}} 包装格式。
    Supports direct dict or {"observation": {...}} wrapper format.

    Args:
        env_obs: 环境观测 / Environment observation

    Returns:
        观测子字典 / Observation sub-dict
    """
    if isinstance(env_obs, dict):
        return env_obs.get("observation", env_obs)
    return {}


def drone_position(env_obs: Dict[str, Any]) -> Tuple[int, int]:
    """
    提取无人机位置 (x, y)。
    Extract drone position as (x, y).

    Args:
        env_obs: 环境观测字典 / Environment observation dict

    Returns:
        (x, y) 整数坐标 / (x, y) integer coordinates
    """
    obs = _observation(env_obs)
    pos = obs.get("position", np.array([0, 0]))
    if isinstance(pos, np.ndarray):
        return int(pos[0]), int(pos[1])
    if isinstance(pos, (list, tuple)):
        return int(pos[0]), int(pos[1])
    return 0, 0


def order_status_mask(env_obs: Dict[str, Any]) -> int:
    """
    编码订单状态为整数比特掩码。
    Encode order status as an integer bitmask.

    每位表示一个订单: 0=未完成, 1=已完成。
    Each bit represents one order: 0=incomplete, 1=completed.

    Args:
        env_obs: 环境观测字典 / Environment observation dict

    Returns:
        订单状态掩码 (0 ~ 31) / Order status mask (0 ~ 31)
    """
    obs = _observation(env_obs)
    status = obs.get("order_status", np.zeros(Config.NUM_ORDERS, dtype=np.int8))
    if isinstance(status, np.ndarray):
        bits = [str(int(b)) for b in status]
    elif isinstance(status, (list, tuple)):
        bits = [str(int(b)) for b in status]
    else:
        bits = ["0"] * Config.NUM_ORDERS
    return int("".join(bits), 2)


def carrying_flag(env_obs: Dict[str, Any]) -> int:
    """
    提取携带状态标志。
    Extract carrying state flag.

    Args:
        env_obs: 环境观测字典 / Environment observation dict

    Returns:
        0=空手(empty), 1=携带中(carrying)
    """
    obs = _observation(env_obs)
    carrying = obs.get("carrying", 0)
    if isinstance(carrying, (np.integer, int)):
        return int(carrying)
    return 0


# ============================================================
# 状态编码 (State Encoding)
# ============================================================

def encode_state_key(env_obs: Dict[str, Any]) -> int:
    """
    将环境观测编码为唯一整数状态键。
    Encode environment observation into a unique integer state key.

    编码方式 / Encoding scheme:
        position: x * GRID_SIZE + y            → 0 ~ 1599  (11 bits)
        order_status: 5-bit mask                → 0 ~ 31    (5 bits)
        carrying: 0 or 1                        → 0 ~ 1     (1 bit)

        state_key = (pos_idx << 6) | (status_bits << 1) | carrying

    这与 Simple2DDeliveryEnv.get_state_key() 方法产生的键完全相同。
    This produces the exact same key as Simple2DDeliveryEnv.get_state_key().

    Args:
        env_obs: 环境观测字典 / Environment observation dict

    Returns:
        唯一整数状态键 / Unique integer state key
    """
    x, y = drone_position(env_obs)
    pos_idx = int(x) * Config.GRID_SIZE + int(y)
    status_bits = order_status_mask(env_obs)
    carrying = carrying_flag(env_obs)
    # 位拼接 / Bit concatenation:
    # pos_idx 占高11位, status_bits占5位, carrying占1位
    return (pos_idx << 6) | (status_bits << 1) | carrying


def decode_state_key(state_key: int) -> Dict[str, Any]:
    """
    将状态键解码为可读组件。
    Decode state key into readable components.

    与 encode_state_key 互为逆操作。
    Inverse of encode_state_key.

    Args:
        state_key: 整数状态键 / Integer state key

    Returns:
        包含 position, order_status, carrying 的字典
        Dictionary with position, order_status, carrying
    """
    carrying = state_key & 1                     # 最低位 / LSB
    status_bits = (state_key >> 1) & 0b11111     # 中间5位 / Middle 5 bits
    pos_idx = state_key >> 6                     # 高11位 / Upper 11 bits

    x = pos_idx // Config.GRID_SIZE
    y = pos_idx % Config.GRID_SIZE

    # 解码订单状态为列表 / Decode order status to list
    order_status = []
    for i in range(Config.NUM_ORDERS):
        order_status.append((status_bits >> i) & 1)

    return {
        "position": (x, y),
        "order_status": order_status,
        "carrying": carrying,
    }


# ============================================================
# 合法动作过滤 (Legal Action Filtering)
# ============================================================

# 惰性初始化障碍物掩码 / Lazy-initialized obstacle mask
_OBSTACLE_MASK: Optional[np.ndarray] = None


def _build_obstacle_mask() -> np.ndarray:
    """
    构建障碍物掩码。
    Build obstacle mask.

    与 Simple2DDeliveryEnv._build_obstacle_mask 保持一致。
    Consistent with Simple2DDeliveryEnv._build_obstacle_mask.

    Returns:
        布尔数组，True=障碍物 / Boolean array, True=obstacle
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

    # 边界不可通行 / Boundaries impassable
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True

    # 确保配送中心可行走 / Ensure depot is walkable
    depot_x, depot_y = Config.DEPOT
    mask[depot_x, depot_y] = False

    return mask


def _get_obstacle_mask() -> np.ndarray:
    """获取障碍物掩码(惰性初始化) / Get obstacle mask (lazy init)."""
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
        True=可行走 / True=walkable
    """
    mask = _get_obstacle_mask()
    if not (0 <= x < Config.GRID_SIZE and 0 <= y < Config.GRID_SIZE):
        return False
    return not mask[x, y]


def legal_actions_from_position(x: int, y: int,
                                 blocked_actions: Optional[set] = None) -> List[int]:
    """
    根据位置计算合法动作列表。
    Compute legal action list from position.

    Args:
        x: x坐标 / x coordinate
        y: y坐标 / y coordinate
        blocked_actions: 已知被阻挡的动作集合 / Set of known blocked actions

    Returns:
        合法动作索引列表 / List of legal action indices
    """
    blocked = blocked_actions or set()
    legal = []
    for act, (dx, dy) in ACTION_DELTAS.items():
        if act in blocked:
            continue
        nx, ny = x + dx, y + dy
        if is_walkable(nx, ny):
            legal.append(act)
    # 若全部被阻挡(不应发生), 返回全部动作
    # If all blocked (should not happen), return all actions
    return legal if legal else list(range(Config.ACTION_SIZE))


def legal_actions_from_obs(env_obs: Dict[str, Any]) -> List[int]:
    """
    从环境观测中提取合法动作列表。
    Extract legal action list from environment observation.

    Args:
        env_obs: 环境观测字典 / Environment observation dict

    Returns:
        合法动作索引列表 / List of legal action indices
    """
    x, y = drone_position(env_obs)
    return legal_actions_from_position(x, y)


# ============================================================
# Preprocessor 类 (Preprocessor Class)
# ============================================================

class Preprocessor:
    """
    SARSA特征预处理器 — 将原始观测转换为特征向量。
    SARSA feature preprocessor — converts raw observations to feature vectors.

    提供统一的 observation_process() 和 action_process() 接口，
    与Kaiwu框架的BaseAgent模式保持一致。
    Provides unified observation_process() and action_process() interfaces,
    consistent with Kaiwu framework's BaseAgent pattern.
    """

    def __init__(self):
        """初始化预处理器 / Initialize preprocessor."""
        self.obstacle_mask = _get_obstacle_mask()

    def observation_process(self, env_obs: Dict[str, Any]) -> Dict[str, Any]:
        """
        将原始环境观测处理为SARSA可用的特征。
        Process raw environment observation into SARSA-usable features.

        提取并编码:
        - 无人机位置 → 位置索引 pos_idx (0~1599)
        - 订单状态 → 5位掩码 status_bits (0~31)
        - 携带状态 → 标志位 carrying (0/1)
        - 合法动作列表

        Extract and encode:
        - drone position → position index pos_idx (0~1599)
        - order status → 5-bit mask status_bits (0~31)
        - carrying state → flag carrying (0/1)
        - legal action list

        Args:
            env_obs: 环境观测字典 / Environment observation dict

        Returns:
            特征字典, 包含 feature(状态键), legal_actions 等
            Feature dict with feature (state key), legal_actions, etc.
        """
        # 编码状态键 / Encode state key
        feature = encode_state_key(env_obs)

        # 计算合法动作 / Compute legal actions
        x, y = drone_position(env_obs)
        legal_actions = legal_actions_from_position(x, y)

        return {
            "feature": feature,
            "legal_actions": legal_actions,
            "position": (x, y),
            "raw_obs": env_obs,
        }

    def action_process(self, act_data) -> int:
        """
        将动作数据处理为可执行的动作整数。
        Process action data into executable action integer.

        支持ActData对象和原始整数。
        Supports ActData objects and raw integers.

        Args:
            act_data: ActData对象或整数, 或列表 / ActData object, int, or list

        Returns:
            可执行的动作索引(0~3) / Executable action index (0~3)
        """
        # 处理列表(来自predict的返回值) / Handle list (return value from predict)
        if isinstance(act_data, list):
            act_data = act_data[0]

        # 从ActData对象提取动作 / Extract action from ActData object
        if hasattr(act_data, "act"):
            act = act_data.act
        else:
            act = act_data

        # 裁剪到有效范围 / Clamp to valid range
        return int(max(0, min(Config.ACTION_SIZE - 1, act)))

    def get_state_key(self, env_obs: Dict[str, Any]) -> int:
        """
        便捷方法: 直接从观测获取状态键。
        Convenience method: get state key directly from observation.

        尝试优先使用环境自带的 get_state_key() 方法，
        如果不可用则回退到 encode_state_key()。
        Tries to use environment's get_state_key() method first,
        falls back to encode_state_key() if unavailable.

        Args:
            env_obs: 环境观测字典(simple_delivery_env返回的格式)
                     Environment observation dict (format returned by simple_delivery_env)

        Returns:
            整数状态键 / Integer state key
        """
        # 注意: env_obs 可能是 get_state_key 方法的拥有者
        # Note: env_obs might be the object that owns get_state_key method
        return encode_state_key(env_obs)
