#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright (c) 2026 Drone Delivery RL Project. All Rights Reserved.
###########################################################################
"""
Monte Carlo Feature Preprocessor for Drone Food Delivery.

蒙特卡洛特征预处理器，用于无人机外卖配送任务。
参考实现：Kaiwu Gorge Walk v2 agent_monte_carlo/feature/preprocessor.py

职责 / Responsibility:
    将原始环境观测转换为MC可用的状态特征和合法动作列表。
    Converts raw environment observations into state features and legal action
    lists usable by Monte Carlo learning.

蒙特卡洛特性 / Monte Carlo Characteristics:
    - 状态编码与Q-learning完全一致（32位整数状态键）
      State encoding is identical to Q-learning (32-bit integer state key)
    - MC不区分在线/离线策略，所以预处理逻辑与Q-learning相同
      MC does not distinguish on/off-policy, so preprocessing logic is same as Q-learning
    - legal_actions过滤对MC尤其重要（减少无效状态-动作空间）
      legal_actions filtering is especially important for MC (reduces invalid state-action space)

领域适配 / Domain Adaptation:
    Gorge Walk v2 → Drone Delivery:
        grid_size=128 → grid_size=40    (武汉学院Zone B / WHU Zone B)
        max_step=1000 → max_steps=500   (最大步数)
        4-direction → same 4-direction  (相同4方向: UP/DOWN/LEFT/RIGHT)

状态编码方案 / State encoding scheme:
    state_key = (pos_idx << 6) | (order_status_bits << 1) | carrying_flag

    Where:
        pos_idx            = x * GRID_SIZE + y    → 0-1599   (11 bits, shifted 6)
        order_status_bits  = 5-bit binary mask    → 0-31     (5 bits, shifted 1)
        carrying_flag      = 0 or 1               → 0-1      (1 bit, LSB)

    Total state space: 1600 * 32 * 2 = 102,400 states
"""


import numpy as np
from typing import Dict, List, Any, Optional, Tuple

from agent_monte_carlo.conf.conf import Config


# ============================================================================
# 动作空间常量
# Action space constants
# ============================================================================

# 4方向动作映射: 0=UP(上), 1=DOWN(下), 2=LEFT(左), 3=RIGHT(右)
# 4-direction action mapping: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
ACTION_DELTAS = {
    0: (0, 1),    # UP    / 上
    1: (0, -1),   # DOWN  / 下
    2: (-1, 0),   # LEFT  / 左
    3: (1, 0),    # RIGHT / 右
}

# 动作名称映射 (用于调试和日志)
# Action name mapping (for debugging and logging)
ACTION_NAMES = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}


# ============================================================================
# 观测提取辅助函数
# Observation extraction helper functions
# ============================================================================

def _observation(env_obs):
    """
    从环境观测中提取observation子字典。
    Extract the observation sub-dict from env observation.

    支持两种格式:
        - 直接dict: {"position": [...], "order_status": [...], ...}
        - 包装格式: {"observation": {"position": [...], ...}}

    Supports two formats:
        - Direct dict
        - Wrapped format with "observation" key

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
    Extract drone position (x, y).

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

    5个订单，每个1位: 0=未完成(incomplete), 1=已完成(completed)
    5 orders, each 1 bit

    Examples:
        All incomplete → 00000 = 0
        Order 0 done    → 00001 = 1
        All done        → 11111 = 31

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
        0=空手(empty), 1=携带中(carrying a package)
    """
    obs = _observation(env_obs)
    carrying = obs.get("carrying", 0)
    if isinstance(carrying, (np.integer, int)):
        return int(carrying)
    return 0


def step_count(env_obs: Dict[str, Any]) -> int:
    """
    提取当前步数。
    Extract current step count.

    Args:
        env_obs: 环境观测字典 / Environment observation dict

    Returns:
        当前步数 / Current step count
    """
    obs = _observation(env_obs)
    try:
        return int(obs.get("step_no", obs.get("steps", 0)))
    except (TypeError, ValueError):
        return 0


# ============================================================================
# 状态编码 (State encoding)
# ============================================================================

def encode_state_key(env_obs: Dict[str, Any]) -> int:
    """
    将环境观测编码为唯一整数状态键。
    Encode environment observation into a unique integer state key.

    Encoding scheme (编码方案):
        position: x * GRID_SIZE + y            → 0~1599  (11 bits, shifted left 6)
        order_status: 5-bit mask                → 0~31    (5 bits, shifted left 1)
        carrying: 0 or 1                        → 0~1     (1 bit, at LSB)

        state_key = pos_idx * 64 + status_bits * 2 + carrying
                  = (pos_idx << 6) | (status_bits << 1) | carrying

    Args:
        env_obs: 环境观测字典 / Environment observation dict

    Returns:
        唯一整数状态键 / Unique integer state key
    """
    x, y = drone_position(env_obs)
    pos_idx = int(x) * Config.GRID_SIZE + int(y)
    status_bits = order_status_mask(env_obs)
    carrying = carrying_flag(env_obs)
    # 位拼接: pos(11b) | status(5b) | carrying(1b)
    # Bit concatenation: pos(11b) | status(5b) | carrying(1b)
    return (pos_idx << 6) | (status_bits << 1) | carrying


def decode_state_key(state_key: int) -> Dict[str, Any]:
    """
    将状态键解码为可读的组件。
    Decode state key into readable components.

    用于调试和可视化。For debugging and visualization.

    Args:
        state_key: 整数状态键 / Integer state key

    Returns:
        包含 position, order_status, carrying 的字典
        Dict with position (tuple), order_status (list), carrying (int)
    """
    carrying = state_key & 1
    status_bits = (state_key >> 1) & 0b11111  # 5 bits
    pos_idx = state_key >> 6

    x = pos_idx // Config.GRID_SIZE
    y = pos_idx % Config.GRID_SIZE

    # 解码订单状态: 从最低位到最高位
    # Decode order status: LSB to MSB
    order_status = []
    for i in range(Config.NUM_ORDERS):
        order_status.append((status_bits >> i) & 1)

    return {
        "position": (x, y),
        "order_status": order_status,
        "carrying": carrying,
        "pos_idx": pos_idx,
    }


def monte_carlo_feature(env_obs: Dict[str, Any]) -> int:
    """
    MC使用的状态特征 — 即状态键本身 (表格型方法).
    MC state feature — the state key itself (tabular method).

    与Q-learning/SARSA完全相同，因为MC也是表格型方法。
    Identical to Q-learning/SARSA since MC is also a tabular method.

    Args:
        env_obs: 环境观测字典 / Environment observation dict

    Returns:
        整数状态键 / Integer state key
    """
    return encode_state_key(env_obs)


# ============================================================================
# 合法动作过滤 (Legal action filtering)
# ============================================================================

# 预定义障碍物掩码 (与 simple_delivery_env 中的布局一致)
# Predefined obstacle mask (consistent with simple_delivery_env layout)
_OBSTACLE_MASK: Optional[np.ndarray] = None


def _build_obstacle_mask() -> np.ndarray:
    """
    构建障碍物掩码。
    Build obstacle mask.

    与 Simple2DDeliveryEnv 中的 _build_obstacle_mask 方法保持一致。
    Consistent with _build_obstacle_mask in Simple2DDeliveryEnv.

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
        True表示可行走 / True means walkable
    """
    mask = _get_obstacle_mask()
    if not (0 <= x < Config.GRID_SIZE and 0 <= y < Config.GRID_SIZE):
        return False
    return not mask[x, y]


def legal_actions_from_obs(env_obs: Dict[str, Any]) -> List[int]:
    """
    从环境观测中提取合法动作列表。
    Extract legal action list from environment observation.

    合法动作 = 执行后不会进入障碍物或越界的动作。
    Legal actions = actions that won't go into obstacles or out of bounds.

    MC尤其需要合法动作过滤——不需要浪费计算在不可能的动作上。
    MC especially needs legal action filtering — no need to waste
    computation on impossible actions.

    Args:
        env_obs: 环境观测字典 / Environment observation dict

    Returns:
        合法动作索引列表 / List of legal action indices
    """
    x, y = drone_position(env_obs)
    legal = []
    for action, (dx, dy) in ACTION_DELTAS.items():
        nx, ny = x + dx, y + dy
        if is_walkable(nx, ny):
            legal.append(action)

    # 如果所有方向都不可行走，返回全部动作 (不应该发生)
    # If all directions blocked, return all actions (should not happen)
    return legal if legal else list(range(Config.ACTION_SIZE))


# ============================================================================
# 主预处理接口 — Monte Carlo版本
# Main preprocessing interface — Monte Carlo version
# ============================================================================

def observation_process(env_obs: Dict[str, Any]) -> Any:
    """
    处理原始环境观测，生成MC可用的ObsData。

    Process raw environment observation into ObsData usable by MC agent.

    这是MC agent在predict/exploit中调用的主要接口。
    This is the main interface called by the MC agent in predict/exploit.

    处理步骤 (Processing steps):
        1. 编码状态键 (encode state key)
        2. 计算合法动作 (compute legal actions)
        3. 创建ObsData对象 (create ObsData object)

    Args:
        env_obs: 原始环境观测 / Raw environment observation dict

    Returns:
        ObsData object with feature (int state key) and legal_actions (list)
        包含feature（整数状态键）和legal_actions（列表）的ObsData对象
    """
    from agent_monte_carlo.feature.definition import ObsData

    # Step 1: Encode state key from observation
    # 步骤1：从观测编码状态键
    feature = encode_state_key(env_obs)

    # Step 2: Compute legal actions
    # 步骤2：计算合法动作
    legal_actions = legal_actions_from_obs(env_obs)

    # Step 3: Create ObsData
    # 步骤3：创建ObsData
    return ObsData(feature=feature, legal_actions=legal_actions, raw_obs=env_obs)


def action_process(act_data: Any) -> int:
    """
    将动作数据处理为可执行的整型动作。

    Process action data into an executable integer action.

    将动作限制在有效范围[0, ACTION_SIZE-1]内。
    Clamps the action to the valid range [0, ACTION_SIZE-1].

    Args:
        act_data: 动作数据对象 (ActData) / Action data object (ActData)

    Returns:
        可执行的动作索引 / Executable action index
    """
    action = act_data.act if hasattr(act_data, "act") else act_data
    return int(max(0, min(Config.ACTION_SIZE - 1, action)))


# ============================================================================
# 启发式动作评分 (用于平局打破 / for tie-breaking)
# ============================================================================

def heuristic_action_scores(
    env_obs: Dict[str, Any],
    legal_actions: Optional[List[int]] = None,
    blocked_actions: Optional[set] = None,
) -> np.ndarray:
    """
    为每个动作计算启发式评分，用于Q值相同时的平局打破。
    Compute heuristic scores for each action, used for tie-breaking
    when Q-values are equal.

    对MC尤其重要：在训练初期Q值全为0时，启发式提供有意义的动作选择。
    Especially important for MC: when Q-values are all 0 in early training,
    heuristics provide meaningful action selection.

    优先级 / Priority:
        1. 倾向接近未完成目标的方向 / Prefer direction toward unfinished target
        2. 避免撞墙 / Avoid wall collisions
        3. 避免已标记为阻塞的动作 / Avoid actions marked as blocked

    Args:
        env_obs: 环境观测字典 / Environment observation dict
        legal_actions: 合法动作列表 / List of legal actions
        blocked_actions: 已阻塞的动作集合 / Set of blocked actions

    Returns:
        长度为 ACTION_SIZE 的评分数组 / Score array of length ACTION_SIZE
    """
    if legal_actions is None:
        legal_actions = list(range(Config.ACTION_SIZE))
    if blocked_actions is None:
        blocked_actions = set()

    x, y = drone_position(env_obs)
    scores = np.full(Config.ACTION_SIZE, -1e9, dtype=np.float32)

    # 确定目标方向 (简化: 曼哈顿距离引导)
    # Determine target direction (simplified: Manhattan distance guidance)
    obs = _observation(env_obs)
    carrying = int(obs.get("carrying", 0) if isinstance(obs, dict) else 0)
    order_status = obs.get("order_status", np.zeros(Config.NUM_ORDERS, dtype=np.int8))

    if hasattr(order_status, 'tolist'):
        order_status = order_status.tolist()
    elif isinstance(order_status, np.ndarray):
        order_status = order_status.tolist()

    # 固定商家和配送点 / Fixed merchants and delivery points
    merchants = [
        (16, 17), (22, 15), (14, 12), (25, 18), (18, 24)
    ]
    delivery_points = [
        (20, 25), (28, 22), (10, 20), (32, 12), (16, 17)
    ]
    depot = tuple(Config.DEPOT)

    # 找到当前目标 / Find current target
    target = None
    if carrying == 0:
        # 空手 → 最近未完成商家 / Empty → nearest unfinished merchant
        best_dist = float("inf")
        for k in range(Config.NUM_ORDERS):
            if k < len(order_status) and order_status[k] == 0:
                mx, my = merchants[k]
                dist = abs(mx - x) + abs(my - y)
                if dist < best_dist:
                    best_dist = dist
                    target = (mx, my)
    if target is None and carrying == 0:
        best_dist = float("inf")
        for k in range(Config.NUM_ORDERS):
            if k < len(order_status) and order_status[k] == 0:
                dx_p, dy_p = delivery_points[k]
                dist = abs(dx_p - x) + abs(dy_p - y)
                if dist < best_dist:
                    best_dist = dist
                    target = (dx_p, dy_p)
    if target is None and carrying != 0:
        # 携带中 → 最近未完成配送点 / Carrying → nearest unfinished delivery
        best_dist = float("inf")
        for k in range(Config.NUM_ORDERS):
            if k < len(order_status) and order_status[k] == 0:
                dx_p, dy_p = delivery_points[k]
                dist = abs(dx_p - x) + abs(dy_p - y)
                if dist < best_dist:
                    best_dist = dist
                    target = (dx_p, dy_p)
    if target is None:
        # 全部完成 → 回配送中心 / All done → return to depot
        target = depot

    tx, ty = target if target else depot

    for action in legal_actions:
        if action in blocked_actions:
            scores[action] = -50.0
            continue

        dx, dy = ACTION_DELTAS.get(action, (0, 0))
        nx, ny = x + dx, y + dy

        # 越界或障碍物 → 低分
        # Out of bounds or obstacle → low score
        if not is_walkable(nx, ny):
            scores[action] = -100.0
            continue

        # 曼哈顿距离减少量 → 高分
        # Manhattan distance reduction → high score
        prev_dist = abs(tx - x) + abs(ty - y)
        new_dist = abs(tx - nx) + abs(ty - ny)
        scores[action] = float(prev_dist - new_dist)

    return scores
