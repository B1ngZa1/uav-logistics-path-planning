"""
问题层级一：2D固定商家配送环境

基于武汉学院Zone B的简化2D网格世界。
4方向离散动作，固定商家和配送点。
适合Q-learning、SARSA、Monte Carlo和A*算法。

使用示例:
    from environment.simple_delivery_env import Simple2DDeliveryEnv

    env = Simple2DDeliveryEnv()
    obs, info = env.reset(seed=42)

    for _ in range(500):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Dict, Any, Tuple


class Simple2DDeliveryEnv(gym.Env):
    """问题层级一：2D固定商家配送环境

    外卖骑手基础模型：在2D网格地图上完成固定订单的取送货任务。

    ## 状态空间
    - position: (2,) int32 — 无人机网格坐标 [x, y]
    - order_status: (5,) int8 — 每个订单的状态 (0=未完成, 1=已完成)
    - carrying: int — 当前是否携带货物 (0=空手, 1=携带中)

    ## 动作空间
    Discrete(4) — 0=上, 1=下, 2=左, 3=右

    ## 奖励
    - +100: 完成一单配送
    - +500: 全部订单完成并返回配送中心
    - -1: 每步时间惩罚
    - -200: 碰撞障碍物（终止）
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    # 动作映射：4方向
    ACTION_DELTAS = {
        0: np.array([0, 1]),    # 上
        1: np.array([0, -1]),   # 下
        2: np.array([-1, 0]),   # 左
        3: np.array([1, 0]),    # 右
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None, render_mode: Optional[str] = None):
        super().__init__()
        config = config or {}

        # 地图参数
        self.grid_size = config.get("grid_size", 40)
        self.num_orders = config.get("num_orders", 5)
        self.max_steps = config.get("max_steps", 500)
        self.render_mode = render_mode

        # 动作与观测空间
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Dict({
            "position": spaces.Box(0, self.grid_size - 1, shape=(2,), dtype=np.int32),
            "order_status": spaces.MultiBinary(self.num_orders),
            "carrying": spaces.Discrete(3),  # 0=空手, 1=携带中(去送货), 2=已取货未定
        })

        # 固定地图元素（基于武汉学院Zone B布局，缩放到40×40网格）
        self.depot = np.array(config.get("depot", [30, 10]))
        self.merchants = np.array(config.get("merchants", [
            [16, 17], [22, 15], [14, 12], [25, 18], [18, 24]
        ]))
        self.delivery_points = np.array(config.get("delivery_points", [
            [20, 25], [28, 22], [10, 20], [32, 12], [16, 17]
        ]))

        # 障碍物掩码
        self.obstacles = self._build_obstacle_mask()

        # 运行时状态
        self.position = self.depot.copy()
        self.order_status = np.zeros(self.num_orders, dtype=np.int8)
        self.carrying = 0
        self.current_order = -1
        self.steps = 0

    def _build_obstacle_mask(self) -> np.ndarray:
        """构建障碍物掩码（基于武汉学院建筑布局简化）"""
        mask = np.zeros((self.grid_size, self.grid_size), dtype=bool)

        # 教学楼群
        mask[14:19, 14:21] = True
        # 宿舍楼群
        mask[26:31, 20:25] = True
        # 湖区水域
        mask[8:13, 4:9] = True
        # 电力线路由
        mask[22, 12:17] = True
        # 其他小建筑
        mask[6:8, 22:24] = True
        mask[19:21, 10:12] = True

        # 边界不可通行
        mask[0, :] = True
        mask[-1, :] = True
        mask[:, 0] = True
        mask[:, -1] = True

        # 确保配送中心、商家、配送点在可行走区域
        for point in [self.depot] + list(self.merchants) + list(self.delivery_points):
            mask[point[0], point[1]] = False

        return mask

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None
              ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """重置环境到初始状态"""
        super().reset(seed=seed)

        self.position = self.depot.copy()
        self.order_status = np.zeros(self.num_orders, dtype=np.int8)
        self.carrying = 0
        self.current_order = -1
        self.steps = 0

        return self._get_obs(), self._get_info()

    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        """执行动作并返回 (obs, reward, terminated, truncated, info)"""
        # 解析动作
        dx, dy = self.ACTION_DELTAS.get(action, (0, 0))
        new_pos = self.position + np.array([dx, dy])

        # 移动与碰撞
        if self._is_valid(new_pos):
            self.position = new_pos

        # 奖励初始化
        reward = -1.0  # 每步时间惩罚
        terminated = False
        self.steps += 1

        # 配送逻辑
        delivery_event = self._check_delivery()
        if delivery_event:
            reward += 100.0

        # 全部完成检查
        if self.order_status.all():
            if np.array_equal(self.position, self.depot):
                reward += 500.0
                terminated = True

        # 碰撞终止
        if self._check_collision():
            reward -= 200.0
            terminated = True

        # 截断
        truncated = self.steps >= self.max_steps

        return self._get_obs(), float(reward), terminated, truncated, self._get_info()

    def _is_valid(self, pos: np.ndarray) -> bool:
        """检查坐标是否可行走"""
        x, y = int(pos[0]), int(pos[1])
        return (0 <= x < self.grid_size and 0 <= y < self.grid_size
                and not self.obstacles[x, y])

    def _check_delivery(self) -> bool:
        """检查并处理配送事件，返回是否完成了一单配送"""
        for k in range(self.num_orders):
            if self.order_status[k] == 1:
                continue  # 已完成的跳过

            if self.carrying == 0:
                # 空手 → 前往商家取货
                if np.array_equal(self.position, self.merchants[k]):
                    self.carrying = 1
                    self.current_order = k
                    return False  # 取货不算完成配送
            else:
                # 携带货物 → 前往配送点送货
                if (self.current_order == k and
                        np.array_equal(self.position, self.delivery_points[k])):
                    self.order_status[k] = 1
                    self.carrying = 0
                    self.current_order = -1
                    return True  # 完成一单

        return False

    def _check_collision(self) -> bool:
        """检查是否碰撞障碍物"""
        return not self._is_valid(self.position)

    def _get_obs(self) -> Dict[str, np.ndarray]:
        """构建观测字典"""
        return {
            "position": self.position.copy(),
            "order_status": self.order_status.copy(),
            "carrying": self.carrying,
        }

    def _get_info(self) -> Dict[str, Any]:
        """构建信息字典"""
        return {
            "position": self.position.tolist(),
            "completed": int(self.order_status.sum()),
            "carrying": self.carrying,
            "current_order": self.current_order,
            "steps": self.steps,
            "depot_reached": bool(np.array_equal(self.position, self.depot)),
        }

    def get_state_key(self) -> int:
        """将当前状态编码为唯一整数键（用于表格型RL）

        编码方式：
        - position: x*40 + y → 0-1599 (11 bits)
        - order_status: 5 bits → 0-31
        - carrying: 1 bit → 0-1
        合并：(pos << 6) | (status << 1) | carrying
        """
        pos_idx = int(self.position[0]) * self.grid_size + int(self.position[1])
        status_bits = int("".join(str(b) for b in self.order_status), 2)
        return (pos_idx << 6) | (status_bits << 1) | self.carrying

    def render(self) -> Optional[np.ndarray]:
        """渲染当前状态（ASCII网格）"""
        if self.render_mode is None:
            return None

        grid = [["·" for _ in range(self.grid_size)] for _ in range(self.grid_size)]

        # 障碍物
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if self.obstacles[x, y]:
                    grid[x][y] = "█"

        # 配送中心
        grid[self.depot[0]][self.depot[1]] = "D"

        # 商家
        for i, m in enumerate(self.merchants):
            if not self.order_status[i]:
                grid[m[0]][m[1]] = f"M{i+1}"

        # 配送点
        for i, d in enumerate(self.delivery_points):
            if not self.order_status[i]:
                grid[d[0]][d[1]] = f"T{i+1}"

        # 无人机
        px, py = int(self.position[0]), int(self.position[1])
        grid[px][py] = "●" if self.carrying == 0 else "◉"

        # 打印
        lines = []
        for row in grid:
            lines.append(" ".join(row))

        result = "\n".join(lines)
        if self.render_mode == "human":
            print(result)
            print(f"Steps: {self.steps}, Completed: {int(self.order_status.sum())}/{self.num_orders}")
            print(f"Carrying: {self.carrying}, Current Order: {self.current_order}")

        return None


# 注册环境
gym.envs.registration.register(
    id="Simple2DDelivery-v0",
    entry_point="environment.simple_delivery_env:Simple2DDeliveryEnv",
    max_episode_steps=500,
)
