"""
问题层级一：经典强化学习算法实现

包含：Q-learning、SARSA、Monte Carlo、A* 搜索

用于2D固定商家配送环境的算法对比实验。
"""
import heapq
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any


# ============================================================
# A* 路径搜索（传统基线）
# ============================================================

def a_star_2d(start: Tuple[int, int], goal: Tuple[int, int],
              obstacle_mask: np.ndarray) -> Optional[List[Tuple[int, int]]]:
    """2D网格A*最短路径搜索

    Args:
        start: 起点 (x, y)
        goal: 终点 (x, y)
        obstacle_mask: 布尔数组，True表示障碍物

    Returns:
        路径坐标列表，或 None（无可行路径）
    """
    def heuristic(a, b):
        # 曼哈顿距离（4方向移动）
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    open_set = [(0, start)]
    came_from = {}
    g_score = {start: 0}
    visited = set()

    while open_set:
        _, current = heapq.heappop(open_set)
        if current in visited:
            continue
        visited.add(current)

        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]

        for dx, dy in [(0, 1), (0, -1), (-1, 0), (1, 0)]:
            neighbor = (current[0] + dx, current[1] + dy)
            if (0 <= neighbor[0] < obstacle_mask.shape[0] and
                    0 <= neighbor[1] < obstacle_mask.shape[1] and
                    not obstacle_mask[neighbor]):
                tentative_g = g_score[current] + 1
                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score, neighbor))

    return None


def a_star_tsp_order(start: Tuple[int, int],
                     merchants: List[Tuple[int, int]],
                     deliveries: List[Tuple[int, int]],
                     depot: Tuple[int, int],
                     obstacle_mask: np.ndarray) -> Tuple[List[int], int]:
    """使用A*和贪心排列求解取送货顺序问题

    贪心策略：每次选择距离当前位置最近且路径最短的未完成订单。

    Returns:
        (访问顺序, 总距离)
    """
    pos = start
    remaining = list(range(len(merchants)))
    order_sequence = []
    total_dist = 0

    while remaining:
        # 找到最近的商家
        best_dist = float('inf')
        best_k = -1
        for k in remaining:
            path_to_merchant = a_star_2d(pos, merchants[k], obstacle_mask)
            if path_to_merchant is None:
                continue
            path_to_delivery = a_star_2d(merchants[k], deliveries[k], obstacle_mask)
            if path_to_delivery is None:
                continue
            total = len(path_to_merchant) + len(path_to_delivery) - 2
            if total < best_dist:
                best_dist = total
                best_k = k

        if best_k == -1:
            break

        order_sequence.append(best_k)
        total_dist += best_dist
        pos = deliveries[best_k]
        remaining.remove(best_k)

    # 返回配送中心
    path_back = a_star_2d(pos, depot, obstacle_mask)
    if path_back:
        total_dist += len(path_back) - 1

    return order_sequence, total_dist


# ============================================================
# Q-learning
# ============================================================

class QLearningAgent:
    """表格型Q-learning智能体

    Q(s, a) ← Q(s, a) + α [r + γ max_a' Q(s', a') - Q(s, a)]
    """

    def __init__(self, action_dim: int = 4, lr: float = 0.1, gamma: float = 0.99,
                 epsilon: float = 1.0, epsilon_min: float = 0.05,
                 epsilon_decay: float = 0.9995):
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.q_table: Dict[int, np.ndarray] = {}
        self.train_steps = 0

    def get_action(self, state_key: int, training: bool = True) -> int:
        """ε-greedy动作选择"""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)

        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_dim)

        q_values = self.q_table[state_key]
        max_q = np.max(q_values)
        # 随机打破平局
        best_actions = np.where(q_values == max_q)[0]
        return int(np.random.choice(best_actions))

    def update(self, state_key: int, action: int, reward: float,
               next_state_key: int, done: bool) -> float:
        """Q-learning更新，返回TD误差"""
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_dim)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.action_dim)

        target = reward
        if not done:
            target += self.gamma * np.max(self.q_table[next_state_key])

        td_error = target - self.q_table[state_key][action]
        self.q_table[state_key][action] += self.lr * td_error

        # ε衰减
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.train_steps += 1

        return td_error

    def get_q_values(self, state_key: int) -> np.ndarray:
        """获取状态的所有Q值"""
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_dim)
        return self.q_table[state_key].copy()

    def get_stats(self) -> Dict[str, Any]:
        return {
            "q_table_size": len(self.q_table),
            "epsilon": self.epsilon,
            "train_steps": self.train_steps,
        }


# ============================================================
# SARSA
# ============================================================

class SARSAAgent:
    """SARSA（在线策略TD学习）

    Q(s, a) ← Q(s, a) + α [r + γ Q(s', a') - Q(s, a)]

    与Q-learning的关键区别：使用实际执行的下一动作a'，
    而不是max_a' Q(s', a')。这使SARSA对探索噪声更保守，
    在有障碍物的环境中安全边际更大。
    """

    def __init__(self, action_dim: int = 4, lr: float = 0.1, gamma: float = 0.99,
                 epsilon: float = 1.0, epsilon_min: float = 0.05,
                 epsilon_decay: float = 0.9995):
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.q_table: Dict[int, np.ndarray] = {}
        self.train_steps = 0

    def get_action(self, state_key: int, training: bool = True) -> int:
        """ε-greedy动作选择"""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)

        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_dim)

        q_values = self.q_table[state_key]
        max_q = np.max(q_values)
        best_actions = np.where(q_values == max_q)[0]
        return int(np.random.choice(best_actions))

    def update(self, state_key: int, action: int, reward: float,
               next_state_key: int, next_action: int, done: bool) -> float:
        """SARSA更新（需要next_action）"""
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_dim)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.action_dim)

        target = reward
        if not done:
            target += self.gamma * self.q_table[next_state_key][next_action]

        td_error = target - self.q_table[state_key][action]
        self.q_table[state_key][action] += self.lr * td_error

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.train_steps += 1

        return td_error

    def get_stats(self) -> Dict[str, Any]:
        return {
            "q_table_size": len(self.q_table),
            "epsilon": self.epsilon,
            "train_steps": self.train_steps,
        }


# ============================================================
# Monte Carlo（首次访问）
# ============================================================

class MonteCarloAgent:
    """首次访问Monte Carlo方法

    V(s_t) ← V(s_t) + α [G_t - V(s_t)]

    完整回合后才更新，使用从t到结束的实际累积回报。
    """

    def __init__(self, action_dim: int = 4, lr: float = 0.01, gamma: float = 0.99,
                 epsilon: float = 1.0, epsilon_min: float = 0.05,
                 epsilon_decay: float = 0.999):
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.q_table: Dict[int, np.ndarray] = {}
        # 每回合的轨迹
        self.episode_history: List[Tuple[int, int, float]] = []  # (state, action, reward)
        self.train_episodes = 0

    def get_action(self, state_key: int, training: bool = True) -> int:
        """ε-greedy动作选择"""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)

        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_dim)

        q_values = self.q_table[state_key]
        max_q = np.max(q_values)
        best_actions = np.where(q_values == max_q)[0]
        return int(np.random.choice(best_actions))

    def store_transition(self, state_key: int, action: int, reward: float):
        """存储一步转移"""
        self.episode_history.append((state_key, action, reward))

    def finish_episode(self):
        """回合结束，计算回报并更新Q值"""
        # 计算每步的回报
        G = 0.0
        visited = set()

        for t in reversed(range(len(self.episode_history))):
            state_key, action, reward = self.episode_history[t]
            G = reward + self.gamma * G

            # 首次访问Monte Carlo
            visit_key = (state_key, action)
            if visit_key not in visited:
                visited.add(visit_key)

                if state_key not in self.q_table:
                    self.q_table[state_key] = np.zeros(self.action_dim)

                # 增量更新
                old_q = self.q_table[state_key][action]
                self.q_table[state_key][action] += self.lr * (G - old_q)

        # 衰减探索率
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.train_episodes += 1

        # 清空轨迹
        self.episode_history = []

        return G  # 返回回合总回报

    def get_stats(self) -> Dict[str, Any]:
        return {
            "q_table_size": len(self.q_table),
            "epsilon": self.epsilon,
            "train_episodes": self.train_episodes,
        }


# ============================================================
# 算法训练的辅助函数
# ============================================================

def train_qlearning(env, agent: QLearningAgent, num_episodes: int,
                    eval_interval: int = 100) -> Dict[str, List[float]]:
    """训练Q-learning并记录指标"""
    rewards_history = []
    delivery_history = []
    steps_history = []

    for ep in range(num_episodes):
        obs, _ = env.reset()
        state_key = env.get_state_key()
        ep_reward = 0.0
        done = False

        while not done:
            action = agent.get_action(state_key, training=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            next_state_key = env.get_state_key()
            done = terminated or truncated

            agent.update(state_key, action, reward, next_state_key, terminated)
            state_key = next_state_key
            ep_reward += reward

        rewards_history.append(ep_reward)
        delivery_history.append(int(obs["order_status"].sum()))
        steps_history.append(env.steps)

        if (ep + 1) % eval_interval == 0:
            avg_r = np.mean(rewards_history[-eval_interval:])
            avg_d = np.mean(delivery_history[-eval_interval:])
            print(f"Ep {ep+1:5d} | Avg Reward: {avg_r:8.1f} | "
                  f"Avg Deliveries: {avg_d:.1f} | ε: {agent.epsilon:.3f}")

    return {
        "rewards": rewards_history,
        "deliveries": delivery_history,
        "steps": steps_history,
    }


def train_sarsa(env, agent: SARSAAgent, num_episodes: int,
                eval_interval: int = 100) -> Dict[str, List[float]]:
    """训练SARSA并记录指标"""
    rewards_history = []
    delivery_history = []

    for ep in range(num_episodes):
        obs, _ = env.reset()
        state_key = env.get_state_key()
        action = agent.get_action(state_key, training=True)
        ep_reward = 0.0
        done = False

        while not done:
            obs, reward, terminated, truncated, _ = env.step(action)
            next_state_key = env.get_state_key()
            done = terminated or truncated
            next_action = agent.get_action(next_state_key, training=not done)

            agent.update(state_key, action, reward, next_state_key, next_action, terminated)
            state_key = next_state_key
            action = next_action
            ep_reward += reward

        rewards_history.append(ep_reward)
        delivery_history.append(int(obs["order_status"].sum()))

        if (ep + 1) % eval_interval == 0:
            avg_r = np.mean(rewards_history[-eval_interval:])
            avg_d = np.mean(delivery_history[-eval_interval:])
            print(f"Ep {ep+1:5d} | Avg Reward: {avg_r:8.1f} | "
                  f"Avg Deliveries: {avg_d:.1f} | ε: {agent.epsilon:.3f}")

    return {"rewards": rewards_history, "deliveries": delivery_history}


def train_monte_carlo(env, agent: MonteCarloAgent, num_episodes: int,
                      eval_interval: int = 100) -> Dict[str, List[float]]:
    """训练Monte Carlo并记录指标"""
    rewards_history = []
    delivery_history = []

    for ep in range(num_episodes):
        obs, _ = env.reset()
        state_key = env.get_state_key()
        ep_reward = 0.0
        done = False

        while not done:
            action = agent.get_action(state_key, training=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            next_state_key = env.get_state_key()
            done = terminated or truncated

            agent.store_transition(state_key, action, reward)
            state_key = next_state_key
            ep_reward += reward

        agent.finish_episode()
        rewards_history.append(ep_reward)
        delivery_history.append(int(obs["order_status"].sum()))

        if (ep + 1) % eval_interval == 0:
            avg_r = np.mean(rewards_history[-eval_interval:])
            avg_d = np.mean(delivery_history[-eval_interval:])
            print(f"Ep {ep+1:5d} | Avg Reward: {avg_r:8.1f} | "
                  f"Avg Deliveries: {avg_d:.1f} | ε: {agent.epsilon:.3f}")

    return {"rewards": rewards_history, "deliveries": delivery_history}


def evaluate_agent(env, agent, num_episodes: int = 10) -> Dict[str, float]:
    """确定性评估智能体性能

    Returns:
        包含 avg_reward, avg_deliveries, avg_steps, delivery_rate 的字典
    """
    total_reward = 0.0
    total_deliveries = 0
    total_steps = 0

    for _ in range(num_episodes):
        obs, _ = env.reset(seed=42 + _)
        state_key = env.get_state_key()
        done = False

        while not done:
            if hasattr(agent, 'get_action'):
                action = agent.get_action(state_key, training=False)
            else:
                action = env.action_space.sample()

            obs, reward, terminated, truncated, _ = env.step(action)
            state_key = env.get_state_key() if hasattr(env, 'get_state_key') else 0
            total_reward += reward
            done = terminated or truncated

        total_deliveries += int(obs["order_status"].sum())
        total_steps += env.steps

    return {
        "avg_reward": total_reward / num_episodes,
        "avg_deliveries": total_deliveries / num_episodes,
        "avg_steps": total_steps / num_episodes,
        "delivery_rate": total_deliveries / (env.num_orders * num_episodes),
    }
