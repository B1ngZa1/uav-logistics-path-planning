"""
全层级实验运行脚本 / Full-Level Experiment Runner

职责 / Responsibility: 统一的L1-L4训练实验入口，支持6种算法、检查点恢复、
    CSV日志持久化和异常隔离。
参考 / Reference: run_level1_experiments.py, run_wuhan_training.py
领域适配 / Domain Adaptation: 从单层级脚本→全层级统一框架，新增断点恢复、
    结构化日志、算法隔离等生产级特性。

使用方式 / Usage:
    # L1 全算法训练
    python run_full_experiments.py --level L1 --episodes 2000

    # L2 指定算法
    python run_full_experiments.py --level L2 --algos q_learning,sarsa --episodes 1000

    # 断点恢复
    python run_full_experiments.py --level L1 --resume

    # L3/L4 武汉场景
    python run_full_experiments.py --level L3 --episodes 500 --zone B
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import pickle
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ============================================================
# 路径设置 / Path setup
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # 无人机项目设计/
sys.path.insert(0, str(SCRIPT_DIR))

# ============================================================
# 全局常量 / Global constants
# ============================================================

# 六种算法标识 / Six algorithm identifiers
ALL_ALGOS = ["q_learning", "sarsa", "monte_carlo", "dqn", "a2c", "ppo"]

# 层级→算法映射 / Level → algorithm mapping
LEVEL_ALGO_MAP = {
    "L1": ["q_learning", "sarsa", "monte_carlo"],         # 离散表格型 / Discrete tabular
    "L2": ["dqn", "a2c", "ppo"],                                  # 函数近似 / Function approx
    "L3": ["a2c", "ppo"],                                  # 连续控制 / Continuous control
    "L4": ["ppo"],                                         # CTDE多机 / CTDE multi-drone
}

# 层级→环境映射 / Level → environment mapping
LEVEL_ENV_MAP = {
    "L1": "simple_delivery",    # Simple2DDeliveryEnv
    "L2": "random_delivery",    # Random2DDeliveryEnv
    "L3": "wuhan_3d",           # WuhanDroneDeliveryEnv (Zone B, 单机)
    "L4": "wuhan_3d_multi",     # WuhanDroneDeliveryEnv (Zone A, 多机)
}

# 检查点保存间隔（默认） / Default checkpoint interval
DEFAULT_CHECKPOINT_INTERVAL = 500

# CSV列名 / CSV column names
CSV_COLUMNS = [
    "level", "algo", "episode", "episode_reward",
    "completion_rate", "collision_count", "steps",
    "epsilon", "q_table_size", "elapsed_time",
]


# ============================================================
# 日志工具 / Logging utilities
# ============================================================

def setup_logging(log_dir: Path, level: str, algo: str) -> logging.Logger:
    """配置双输出日志（stdout + 文件）。

    / Configure dual-output logging (stdout + file).

    Args:
        log_dir: 日志目录 / Log directory
        level: 层级标识 / Level identifier (L1-L4)
        algo: 算法标识 / Algorithm identifier

    Returns:
        配置好的Logger实例 / Configured Logger instance
    """
    logger_name = f"experiment.{level}.{algo}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    # 防止重复添加handler / Prevent duplicate handlers
    if logger.handlers:
        return logger

    # 格式 / Format
    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-7s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # stdout handler
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    # 文件handler / File handler
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{level}_{algo}_{datetime.now():%Y%m%d_%H%M%S}.log"
    fh = logging.FileHandler(str(log_file), encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.info("日志文件 / Log file: %s", log_file)
    return logger


# ============================================================
# CSV记录器 / CSV recorder
# ============================================================

class CSVRecorder:
    """训练指标CSV持久化记录器。

    / CSV persistence recorder for training metrics.

    特性 / Features:
    - 实时追加写入，不缓存整个数组 / Real-time append, no in-memory buffer
    - 自动创建表头 / Auto-create header
    - 支持断点恢复后续写 / Support resume continuation
    """

    def __init__(self, csv_path: Path):
        self.csv_path = csv_path
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = None
        self._writer = None
        self._init_file()

    def _init_file(self):
        """初始化CSV文件（不存在则创建表头）。

        / Initialize CSV file (create header if not exists).
        """
        file_exists = self.csv_path.exists() and self.csv_path.stat().st_size > 0
        self._file = open(str(self.csv_path), "a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=CSV_COLUMNS)
        if not file_exists:
            self._writer.writeheader()

    def record(self, row: Dict[str, Any]):
        """追加一行记录。/ Append one row."""
        self._writer.writerow(row)
        self._file.flush()

    def close(self):
        """关闭文件。/ Close file."""
        if self._file and not self._file.closed:
            self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def get_last_episode(self) -> int:
        """读取CSV中最后一行的episode编号，用于断点恢复。

        / Read last episode number from CSV for checkpoint resume.
        """
        if not self.csv_path.exists():
            return 0
        last_ep = 0
        with open(str(self.csv_path), "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    last_ep = max(last_ep, int(row.get("episode", 0)))
                except (ValueError, TypeError):
                    pass
        return last_ep


# ============================================================
# 检查点管理 / Checkpoint management
# ============================================================

class CheckpointManager:
    """检查点保存与恢复管理器。

    / Checkpoint save and resume manager.

    支持表格型（pickle序列化Q-table）和网络型（torch.save）两种检查点格式。
    """

    def __init__(self, checkpoint_dir: Path, level: str, algo: str):
        self.checkpoint_dir = checkpoint_dir / level / algo
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.level = level
        self.algo = algo

    def _ckpt_path(self, episode: int) -> Path:
        return self.checkpoint_dir / f"ckpt_ep{episode:06d}.pkl"

    def _latest_path(self) -> Path:
        return self.checkpoint_dir / "latest_checkpoint.json"

    def save(self, episode: int, agent: Any, extra_data: Optional[Dict] = None):
        """保存检查点。

        / Save checkpoint.

        Args:
            episode: 当前episode编号 / Current episode number
            agent: 智能体对象 / Agent object
            extra_data: 额外保存的数据 / Additional data to save
        """
        ckpt_path = self._ckpt_path(episode)
        data = {
            "episode": episode,
            "level": self.level,
            "algo": self.algo,
            "timestamp": datetime.now().isoformat(),
        }

        # 根据智能体类型选择序列化方式
        # / Choose serialization based on agent type
        if hasattr(agent, "q_table"):
            # 表格型智能体 / Tabular agent
            data["agent_type"] = "tabular"
            data["q_table"] = dict(agent.q_table)
            data["epsilon"] = agent.epsilon
            data["train_steps"] = agent.train_steps
        elif hasattr(agent, "actor"):
            # 网络型智能体（PPO/A2C/DQN） / Network agent
            data["agent_type"] = "network"
            # 保存模型权重到单独文件 / Save weights to separate file
            model_path = self.checkpoint_dir / f"model_ep{episode:06d}.pt"
            agent.save(str(model_path))
            data["model_path"] = str(model_path)
        else:
            data["agent_type"] = "unknown"

        if extra_data:
            data["extra"] = extra_data

        # 保存检查点元数据 / Save checkpoint metadata
        with open(str(ckpt_path), "wb") as f:
            pickle.dump(data, f)

        # 更新最新检查点指针 / Update latest checkpoint pointer
        latest = self._latest_path()
        with open(str(latest), "w", encoding="utf-8") as f:
            json.dump({
                "episode": episode,
                "path": str(ckpt_path),
                "timestamp": data["timestamp"],
            }, f, indent=2)

    def load_latest(self) -> Optional[Dict]:
        """加载最新检查点。/ Load latest checkpoint.

        Returns:
            检查点数据字典，无检查点则返回None / Checkpoint data dict, None if no checkpoint
        """
        latest = self._latest_path()
        if not latest.exists():
            return None

        with open(str(latest), "r", encoding="utf-8") as f:
            meta = json.load(f)

        ckpt_path = Path(meta["path"])
        if not ckpt_path.exists():
            return None

        with open(str(ckpt_path), "rb") as f:
            return pickle.load(f)

    def restore_agent(self, agent: Any, ckpt_data: Dict) -> int:
        """将检查点数据恢复到智能体。

        / Restore checkpoint data into agent.

        Args:
            agent: 智能体对象 / Agent object
            ckpt_data: 检查点数据 / Checkpoint data

        Returns:
            恢复的episode编号 / Restored episode number
        """
        if ckpt_data.get("agent_type") == "tabular" and hasattr(agent, "q_table"):
            agent.q_table = ckpt_data.get("q_table", {})
            agent.epsilon = ckpt_data.get("epsilon", agent.epsilon)
            agent.train_steps = ckpt_data.get("train_steps", 0)
        elif ckpt_data.get("agent_type") == "network" and hasattr(agent, "load"):
            model_path = ckpt_data.get("model_path")
            if model_path and Path(model_path).exists():
                agent.load(model_path)
        return ckpt_data.get("episode", 0)


# ============================================================
# 智能体工厂 / Agent factory
# ============================================================

def create_agent(algo: str, action_dim: int = 4, state_dim: int = 50):
    """根据算法名创建对应智能体。

    / Create agent based on algorithm name.

    Args:
        algo: 算法标识 / Algorithm identifier
        action_dim: 动作维度 / Action dimension
        state_dim: 状态维度（连续空间用） / State dimension (for continuous spaces)

    Returns:
        创建的智能体实例 / Created agent instance

    Raises:
        ImportError: 模块不存在 / Module not found
        ValueError: 不支持的算法 / Unsupported algorithm
    """
    if algo == "q_learning":
        from algorithm.classical_rl import QLearningAgent
        return QLearningAgent(
            action_dim=action_dim, lr=0.1, gamma=0.99,
            epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.9995,
        )
    elif algo == "sarsa":
        from algorithm.classical_rl import SARSAAgent
        return SARSAAgent(
            action_dim=action_dim, lr=0.1, gamma=0.99,
            epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.9995,
        )
    elif algo == "monte_carlo":
        from algorithm.classical_rl import MonteCarloAgent
        return MonteCarloAgent(
            action_dim=action_dim, lr=0.01, gamma=0.99,
            epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.999,
        )
    elif algo == "dqn":
        # DQN: 复用PPOAgent的Critic结构做Q网络
        # / DQN: reuse PPOAgent's Critic structure as Q-network
        raise NotImplementedError(
            "DQN算法尚未实现独立Agent类，"
            "请在 algorithm/ 下添加 dqn_agent.py 后注册。"
            "/ DQN agent not yet implemented. "
            "Add dqn_agent.py under algorithm/ and register here."
        )
    elif algo == "a2c":
        # A2C: 可复用PPOAgent（去掉clip即为A2C）
        # / A2C: can reuse PPOAgent (without clip = A2C)
        from algorithm.ppo_agent import PPOAgent
        return PPOAgent({
            "state_dim": state_dim,
            "action_dim": action_dim,
            "hidden_dim": 256,
            "lr": 7e-4,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_epsilon": 1.0,  # clip=1.0 等价于无裁剪（A2C）/ no clipping (A2C)
            "entropy_coef": 0.01,
            "value_coef": 0.5,
            "max_grad_norm": 0.5,
            "update_epochs": 1,
            "batch_size": 64,
            "action_limit": 1.0,
        })
    elif algo == "ppo":
        from algorithm.ppo_agent import PPOAgent
        from config import PPO_CONFIG
        return PPOAgent({
            "state_dim": state_dim,
            "action_dim": action_dim,
            "hidden_dim": PPO_CONFIG["hidden_dim"],
            "lr": PPO_CONFIG["learning_rate"],
            "gamma": PPO_CONFIG["gamma"],
            "gae_lambda": PPO_CONFIG["gae_lambda"],
            "clip_epsilon": PPO_CONFIG["clip_epsilon"],
            "entropy_coef": PPO_CONFIG["entropy_coef"],
            "value_coef": PPO_CONFIG["value_coef"],
            "max_grad_norm": PPO_CONFIG["max_grad_norm"],
            "update_epochs": PPO_CONFIG["update_epochs"],
            "batch_size": PPO_CONFIG["batch_size"],
            "action_limit": 1.0,
        })
    else:
        raise ValueError(f"不支持的算法 / Unsupported algorithm: {algo}")


# ============================================================
# 环境工厂 / Environment factory
# ============================================================

def create_env(level: str, zone: str = "B", seed: int = 42):
    """根据层级创建对应环境。

    / Create environment based on level.

    Args:
        level: 层级标识 / Level identifier (L1-L4)
        zone: 武汉场景分区 / Wuhan scene zone (A-D)
        seed: 随机种子 / Random seed

    Returns:
        (env, env_info) 元组 / Tuple of (env, env_info dict)
    """
    if level == "L1":
        from environment.simple_delivery_env import Simple2DDeliveryEnv
        env = Simple2DDeliveryEnv()
        env_info = {
            "type": "discrete",
            "action_dim": 4,
            "state_dim": None,  # 表格型不需要 / Tabular doesn't need
            "max_steps": env.max_steps,
            "num_orders": env.num_orders,
        }
        return env, env_info

    elif level == "L2":
        from environment.random_delivery_env import Random2DDeliveryEnv
        env = Random2DDeliveryEnv()
        env_info = {
            "type": "discrete",
            "action_dim": 8,
            "state_dim": 50,  # vectorized observation dim
            "max_steps": env.max_steps,
            "num_orders": None,  # 随机生成 / Randomly generated
        }
        return env, env_info

    elif level in ("L3", "L4"):
        try:
            from environment.wuhan_drone_env import WuhanDroneDeliveryEnv
            from config import WUHAN_ENV_CONFIG
        except ImportError:
            raise ImportError(
                "L3/L4环境依赖 wuhan_drone_env 和 config，请确认文件存在。"
                "/ L3/L4 env requires wuhan_drone_env and config."
            )

        env_config = dict(WUHAN_ENV_CONFIG)
        env_config["zone_id"] = zone
        env_config["seed"] = seed
        if level == "L4":
            env_config["num_drones"] = 3  # 多机 / Multi-drone
        else:
            env_config["num_drones"] = 1  # 单机 / Single drone

        env = WuhanDroneDeliveryEnv(env_config)
        obs, _ = env.reset(seed=seed)
        state_dim = env.flatten_observation(obs).shape[0]

        env_info = {
            "type": "continuous",
            "action_dim": 4,
            "state_dim": state_dim,
            "max_steps": int(
                env_config["max_episode_time_s"] / env_config["control_dt_s"]
            ),
            "num_drones": env_config["num_drones"],
            "zone": zone,
        }
        return env, env_info

    else:
        raise ValueError(f"不支持的层级 / Unsupported level: {level}")


# ============================================================
# L1训练循环 / L1 training loop
# ============================================================

def train_l1_single(
    algo: str,
    num_episodes: int,
    eval_interval: int,
    checkpoint_interval: int,
    recorder: CSVRecorder,
    ckpt_mgr: CheckpointManager,
    logger: logging.Logger,
    resume_from: int = 0,
    seed: int = 42,
) -> Dict[str, Any]:
    """L1层级单算法训练。

    / L1 level single algorithm training.

    对于表格型算法（Q-learning, SARSA, Monte Carlo），直接驱动训练循环。
    """
    from environment.simple_delivery_env import Simple2DDeliveryEnv

    np.random.seed(seed + hash(algo) % 10000)
    env = Simple2DDeliveryEnv()
    action_dim = 4
    agent = create_agent(algo, action_dim=action_dim)

    # 断点恢复 / Resume from checkpoint
    start_ep = 0
    if resume_from > 0:
        ckpt_data = ckpt_mgr.load_latest()
        if ckpt_data:
            start_ep = ckpt_mgr.restore_agent(agent, ckpt_data)
            logger.info(
                "从检查点恢复 / Resumed from checkpoint: episode %d", start_ep
            )

    results = {"rewards": [], "deliveries": [], "steps": [], "collisions": []}
    t_start = time.time()

    for ep in range(start_ep, num_episodes):
        ep_start = time.time()
        obs, _ = env.reset(seed=seed + ep)
        state_key = env.get_state_key()
        ep_reward = 0.0
        ep_collisions = 0
        done = False

        # SARSA需要预选动作 / SARSA needs pre-selected action
        if algo == "sarsa":
            action = agent.get_action(state_key, training=True)

        while not done:
            if algo != "sarsa":
                action = agent.get_action(state_key, training=True)

            obs, reward, terminated, truncated, info = env.step(action)
            next_state_key = env.get_state_key()
            done = terminated or truncated

            # 检测碰撞 / Detect collision
            if reward <= -100:  # 碰撞惩罚阈值 / Collision penalty threshold
                ep_collisions += 1

            # 根据算法类型执行更新 / Update based on algorithm type
            if algo == "q_learning":
                agent.update(state_key, action, reward, next_state_key, terminated)
            elif algo == "sarsa":
                next_action = agent.get_action(next_state_key, training=not done)
                agent.update(
                    state_key, action, reward,
                    next_state_key, next_action, terminated,
                )
                action = next_action
            elif algo == "monte_carlo":
                agent.store_transition(state_key, action, reward)

            state_key = next_state_key
            ep_reward += reward

        # Monte Carlo回合结束更新 / MC end-of-episode update
        if algo == "monte_carlo":
            agent.finish_episode()

        # 计算完成率 / Compute completion rate
        completed = int(obs["order_status"].sum()) if "order_status" in obs else 0
        total_orders = env.num_orders
        completion_rate = completed / total_orders if total_orders > 0 else 0.0

        ep_time = time.time() - ep_start
        results["rewards"].append(ep_reward)
        results["deliveries"].append(completed)
        results["steps"].append(env.steps)
        results["collisions"].append(ep_collisions)

        # 获取智能体统计 / Get agent stats
        epsilon = getattr(agent, "epsilon", 0.0)
        q_table_size = len(getattr(agent, "q_table", {}))

        # 写入CSV / Write to CSV
        recorder.record({
            "level": "L1",
            "algo": algo,
            "episode": ep + 1,
            "episode_reward": f"{ep_reward:.2f}",
            "completion_rate": f"{completion_rate:.4f}",
            "collision_count": ep_collisions,
            "steps": env.steps,
            "epsilon": f"{epsilon:.4f}",
            "q_table_size": q_table_size,
            "elapsed_time": f"{ep_time:.4f}",
        })

        # 定期日志 / Periodic logging
        if (ep + 1) % eval_interval == 0:
            recent = min(eval_interval, len(results["rewards"]))
            avg_r = np.mean(results["rewards"][-recent:])
            avg_c = np.mean(results["deliveries"][-recent:])
            avg_col = np.mean(results["collisions"][-recent:])
            logger.info(
                "Ep %5d | AvgR: %8.1f | AvgDel: %.1f/%d | "
                "AvgCol: %.2f | eps: %.3f | Q-size: %d",
                ep + 1, avg_r, avg_c, total_orders,
                avg_col, epsilon, q_table_size,
            )

        # 保存检查点 / Save checkpoint
        if (ep + 1) % checkpoint_interval == 0:
            ckpt_mgr.save(ep + 1, agent)
            logger.info("检查点已保存 / Checkpoint saved: episode %d", ep + 1)

    # 最终检查点 / Final checkpoint
    ckpt_mgr.save(num_episodes, agent)
    total_time = time.time() - t_start

    return {
        "algo": algo,
        "total_episodes": num_episodes,
        "total_time": total_time,
        "final_avg_reward": float(np.mean(results["rewards"][-100:])),
        "final_avg_deliveries": float(np.mean(results["deliveries"][-100:])),
        "final_avg_collisions": float(np.mean(results["collisions"][-100:])),
    }


# ============================================================
# L2训练循环 / L2 training loop
# ============================================================

def train_l2_single(
    algo: str,
    num_episodes: int,
    eval_interval: int,
    checkpoint_interval: int,
    recorder: CSVRecorder,
    ckpt_mgr: CheckpointManager,
    logger: logging.Logger,
    resume_from: int = 0,
    seed: int = 42,
) -> Dict[str, Any]:
    """L2层级单算法训练（DQN/A2C/PPO，函数近似型）。

    / L2 level single algorithm training (DQN/A2C/PPO, function approximation).

    L2使用Random2DDeliveryEnv（8方向离散动作、随机订单、时间窗口）。
    算法通过PPOAgent接口与环境交互，连续动作输出被离散化到8个方向。
    """
    from environment.random_delivery_env import Random2DDeliveryEnv

    np.random.seed(seed + hash(algo) % 10000)
    env = Random2DDeliveryEnv()
    action_dim = 8
    state_dim = env.state_dim  # 从环境获取向量化观测维度 / Get vectorized obs dim from env

    agent = create_agent(
        algo,
        action_dim=action_dim,
        state_dim=state_dim,
    )

    # 断点恢复 / Resume
    start_ep = 0
    if resume_from > 0:
        ckpt_data = ckpt_mgr.load_latest()
        if ckpt_data:
            start_ep = ckpt_mgr.restore_agent(agent, ckpt_data)
            logger.info("从检查点恢复 / Resumed: episode %d", start_ep)

    results = {"rewards": [], "deliveries": [], "steps": [], "collisions": []}
    t_start = time.time()

    for ep in range(start_ep, num_episodes):
        ep_start = time.time()
        obs, info = env.reset(seed=seed + ep)
        ep_reward = 0.0
        ep_collisions = 0
        done = False

        # 提取向量化观测 / Extract vectorized observation
        state = obs["vector"].astype(np.float32)

        while not done:
            # 通过PPOAgent接口选择动作 / Select action via PPOAgent interface
            action_out, log_prob, value = agent.select_action(state)
            # 连续动作→离散方向映射 / Continuous action → discrete direction mapping
            # argmax选择最大分量，取模保证在[0, action_dim)范围
            discrete_action = int(np.argmax(action_out[:action_dim])) % action_dim

            obs, reward, terminated, truncated, info = env.step(discrete_action)
            done = terminated or truncated

            # 碰撞检测：terminated且不在合法位置 → 碰撞终止
            # / Collision detection: terminated and not at valid position → collision
            if terminated and not env._is_valid(env.position):
                ep_collisions += 1

            # 提取下一状态 / Extract next state
            next_state = obs["vector"].astype(np.float32)

            # 存储经验并更新 / Store transition and update
            agent.store_transition(
                state, action_out, reward * 0.01,
                bool(terminated), log_prob, value,
            )

            state = next_state
            ep_reward += reward

        # 回合结束更新网络 / End-of-episode network update
        next_value = 0.0 if terminated else agent.estimate_value(state)
        agent.update(next_value)

        # 从info获取完成率 / Get completion rate from info
        completed = info.get("completed", 0)
        total_orders = info.get("total_orders", len(getattr(env, "orders", [])))
        completion_rate = completed / total_orders if total_orders > 0 else 0.0

        ep_time = time.time() - ep_start
        results["rewards"].append(ep_reward)
        results["deliveries"].append(completed)
        results["steps"].append(getattr(env, "steps", 0))
        results["collisions"].append(ep_collisions)

        recorder.record({
            "level": "L2",
            "algo": algo,
            "episode": ep + 1,
            "episode_reward": f"{ep_reward:.2f}",
            "completion_rate": f"{completion_rate:.4f}",
            "collision_count": ep_collisions,
            "steps": getattr(env, "steps", 0),
            "epsilon": "0.0",
            "q_table_size": 0,
            "elapsed_time": f"{ep_time:.4f}",
        })

        if (ep + 1) % eval_interval == 0:
            recent = min(eval_interval, len(results["rewards"]))
            avg_r = np.mean(results["rewards"][-recent:])
            avg_c = np.mean(results["deliveries"][-recent:])
            logger.info(
                "Ep %5d | AvgR: %8.1f | AvgDel: %.1f | Col: %.2f",
                ep + 1, avg_r, avg_c,
                np.mean(results["collisions"][-recent:]),
            )

        if (ep + 1) % checkpoint_interval == 0:
            ckpt_mgr.save(ep + 1, agent)
            logger.info("检查点已保存 / Checkpoint saved: episode %d", ep + 1)

    ckpt_mgr.save(num_episodes, agent)
    total_time = time.time() - t_start

    return {
        "algo": algo,
        "total_episodes": num_episodes,
        "total_time": total_time,
        "final_avg_reward": float(np.mean(results["rewards"][-100:])),
        "final_avg_deliveries": float(np.mean(results["deliveries"][-100:])),
        "final_avg_collisions": float(np.mean(results["collisions"][-100:])),
    }


# ============================================================
# L3/L4训练循环 / L3/L4 training loop (Wuhan 3D)
# ============================================================

def train_wuhan_single(
    algo: str,
    level: str,
    num_episodes: int,
    eval_interval: int,
    checkpoint_interval: int,
    recorder: CSVRecorder,
    ckpt_mgr: CheckpointManager,
    logger: logging.Logger,
    zone: str = "B",
    resume_from: int = 0,
    seed: int = 42,
) -> Dict[str, Any]:
    """L3/L4层级单算法训练（武汉3D场景）。

    / L3/L4 single algorithm training (Wuhan 3D scene).

    复用 run_wuhan_training.py 的 run_episode 模式。
    """
    try:
        import torch
        from environment.wuhan_drone_env import WuhanDroneDeliveryEnv
        from config import WUHAN_ENV_CONFIG, PPO_CONFIG
        from algorithm.delivery_strategy import ScoreFirstDeliveryPlanner
    except ImportError as e:
        raise ImportError(
            f"L3/L4依赖缺失 / L3/L4 dependency missing: {e}"
        )

    np.random.seed(seed)
    torch.manual_seed(seed)

    env_config = dict(WUHAN_ENV_CONFIG)
    env_config["zone_id"] = zone
    env_config["seed"] = seed
    num_drones = 3 if level == "L4" else 1
    env_config["num_drones"] = num_drones

    env = WuhanDroneDeliveryEnv(env_config)
    obs, _ = env.reset(seed=seed)
    state_dim = env.flatten_observation(obs).shape[0]

    agent = create_agent(algo, action_dim=4, state_dim=state_dim)
    planner = ScoreFirstDeliveryPlanner()

    # 断点恢复 / Resume
    start_ep = 0
    if resume_from > 0:
        ckpt_data = ckpt_mgr.load_latest()
        if ckpt_data:
            start_ep = ckpt_mgr.restore_agent(agent, ckpt_data)
            logger.info("从检查点恢复 / Resumed: episode %d", start_ep)

    max_steps = int(
        np.ceil(env_config["max_episode_time_s"] / env_config["control_dt_s"])
    )
    results = {"rewards": [], "deliveries": [], "steps": [], "collisions": []}
    t_start = time.time()

    for ep in range(start_ep, num_episodes):
        ep_start = time.time()
        observation, _ = env.reset(seed=seed + ep)
        state = env.flatten_observation(observation)
        ep_reward = 0.0
        ep_collisions = 0
        terminated = False
        truncated = False
        info = {}

        for _ in range(max_steps):
            physical_action, log_prob, value = agent.select_action(
                state, deterministic=False,
            )
            action = planner.compose_action(env, physical_action)
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_state = env.flatten_observation(next_obs)
            ep_reward += reward

            if reward <= -100:
                ep_collisions += 1

            agent.store_transition(
                state, physical_action, reward * 0.01,
                bool(terminated), log_prob, value,
            )

            state = next_state
            if terminated or truncated:
                break

        # 更新 / Update
        next_value = 0.0 if terminated else agent.estimate_value(state)
        agent.update(next_value)

        # 指标 / Metrics
        served = len(info.get("served_task_ids", []))
        total_tasks = env_config.get("num_drones", 1) * 3  # 粗估
        completion_rate = served / total_tasks if total_tasks > 0 else 0.0
        ep_steps = int(info.get("step", max_steps))

        ep_time = time.time() - ep_start
        results["rewards"].append(ep_reward)
        results["deliveries"].append(served)
        results["steps"].append(ep_steps)
        results["collisions"].append(ep_collisions)

        recorder.record({
            "level": level,
            "algo": algo,
            "episode": ep + 1,
            "episode_reward": f"{ep_reward:.2f}",
            "completion_rate": f"{completion_rate:.4f}",
            "collision_count": ep_collisions,
            "steps": ep_steps,
            "epsilon": "0.0",
            "q_table_size": 0,
            "elapsed_time": f"{ep_time:.4f}",
        })

        if (ep + 1) % eval_interval == 0:
            recent = min(eval_interval, len(results["rewards"]))
            logger.info(
                "Ep %5d | AvgR: %8.1f | AvgServed: %.1f | Col: %.2f",
                ep + 1,
                np.mean(results["rewards"][-recent:]),
                np.mean(results["deliveries"][-recent:]),
                np.mean(results["collisions"][-recent:]),
            )

        if (ep + 1) % checkpoint_interval == 0:
            ckpt_mgr.save(ep + 1, agent)
            logger.info("检查点已保存 / Checkpoint saved: episode %d", ep + 1)

    ckpt_mgr.save(num_episodes, agent)
    total_time = time.time() - t_start
    env.close()

    return {
        "algo": algo,
        "total_episodes": num_episodes,
        "total_time": total_time,
        "final_avg_reward": float(np.mean(results["rewards"][-100:])),
        "final_avg_deliveries": float(np.mean(results["deliveries"][-100:])),
        "final_avg_collisions": float(np.mean(results["collisions"][-100:])),
    }


# ============================================================
# 主调度器 / Main dispatcher
# ============================================================

def run_experiment(
    level: str,
    algos: List[str],
    num_episodes: int,
    eval_interval: int,
    checkpoint_interval: int,
    zone: str,
    seed: int,
    resume: bool,
    output_dir: Path,
    log_dir: Path,
):
    """运行一组实验（单层级、多算法）。

    / Run a set of experiments (single level, multiple algorithms).

    特性 / Features:
    - 算法隔离：单个算法崩溃不影响其他算法
    - 完整CSV记录
    - 检查点保存与恢复
    - 最终汇总输出
    """
    # 创建CSV记录器 / Create CSV recorder
    csv_path = output_dir / f"training_log_{level}_{datetime.now():%Y%m%d_%H%M%S}.csv"
    recorder = CSVRecorder(csv_path)

    results = []
    failed_algos = []

    for algo in algos:
        print(f"\n{'='*70}")
        print(f"  Level: {level} | Algorithm: {algo}")
        print(f"  Episodes: {num_episodes} | Zone: {zone}")
        print(f"{'='*70}\n")

        logger = setup_logging(log_dir, level, algo)
        ckpt_dir = output_dir / "checkpoints"
        ckpt_mgr = CheckpointManager(ckpt_dir, level, algo)

        # 断点恢复检测 / Detect resume point
        resume_from = 0
        if resume:
            resume_from = recorder.get_last_episode()
            if resume_from > 0:
                logger.info(
                    "断点恢复：从episode %d继续 / Resuming from episode %d",
                    resume_from, resume_from,
                )

        try:
            if level == "L1":
                result = train_l1_single(
                    algo=algo,
                    num_episodes=num_episodes,
                    eval_interval=eval_interval,
                    checkpoint_interval=checkpoint_interval,
                    recorder=recorder,
                    ckpt_mgr=ckpt_mgr,
                    logger=logger,
                    resume_from=resume_from,
                    seed=seed,
                )
            elif level == "L2":
                result = train_l2_single(
                    algo=algo,
                    num_episodes=num_episodes,
                    eval_interval=eval_interval,
                    checkpoint_interval=checkpoint_interval,
                    recorder=recorder,
                    ckpt_mgr=ckpt_mgr,
                    logger=logger,
                    resume_from=resume_from,
                    seed=seed,
                )
            elif level in ("L3", "L4"):
                result = train_wuhan_single(
                    algo=algo,
                    level=level,
                    num_episodes=num_episodes,
                    eval_interval=eval_interval,
                    checkpoint_interval=checkpoint_interval,
                    recorder=recorder,
                    ckpt_mgr=ckpt_mgr,
                    logger=logger,
                    zone=zone,
                    resume_from=resume_from,
                    seed=seed,
                )
            else:
                raise ValueError(f"不支持的层级 / Unsupported level: {level}")

            results.append(result)
            logger.info(
                "算法 %s 完成 / Algorithm %s finished: "
                "time=%.1fs, avg_reward=%.1f, avg_del=%.1f",
                algo, algo, result["total_time"],
                result["final_avg_reward"], result["final_avg_deliveries"],
            )

        except NotImplementedError as e:
            logger.warning(
                "算法 %s 跳过（未实现）/ Skipped (not implemented): %s",
                algo, e,
            )
            failed_algos.append((algo, str(e)))

        except Exception as e:
            logger.error(
                "算法 %s 失败 / Algorithm %s FAILED: %s",
                algo, algo, e,
            )
            logger.debug(traceback.format_exc())
            failed_algos.append((algo, str(e)))

    recorder.close()

    # 汇总输出 / Summary output
    print(f"\n{'='*70}")
    print(f"  EXPERIMENT SUMMARY — {level}")
    print(f"{'='*70}")
    print(f"{'Algorithm':<15} {'Time(s)':>10} {'Avg Reward':>12} "
          f"{'Avg Deliveries':>16} {'Avg Collisions':>16}")
    print("-" * 70)
    for r in results:
        print(f"{r['algo']:<15} {r['total_time']:>10.1f} "
              f"{r['final_avg_reward']:>12.1f} "
              f"{r['final_avg_deliveries']:>16.1f} "
              f"{r['final_avg_collisions']:>16.2f}")
    if failed_algos:
        print(f"\n失败算法 / Failed algorithms:")
        for fname, ferr in failed_algos:
            print(f"  - {fname}: {ferr}")

    print(f"\n训练数据CSV / Training data CSV: {csv_path}")

    # 保存JSON汇总 / Save JSON summary
    summary_path = output_dir / f"summary_{level}_{datetime.now():%Y%m%d_%H%M%S}.json"
    with open(str(summary_path), "w", encoding="utf-8") as f:
        json.dump({
            "level": level,
            "episodes": num_episodes,
            "zone": zone,
            "seed": seed,
            "results": results,
            "failed": [{"algo": a, "error": e} for a, e in failed_algos],
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2, ensure_ascii=False)
    print(f"汇总JSON / Summary JSON: {summary_path}")


# ============================================================
# 命令行入口 / CLI entry point
# ============================================================

def parse_args():
    """解析命令行参数。/ Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="全层级无人机配送实验运行脚本 / "
                    "Full-level drone delivery experiment runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例 / Examples:
  # L1全算法
  python run_full_experiments.py --level L1 --episodes 2000

  # L2指定算法
  python run_full_experiments.py --level L2 --algos q_learning,sarsa --episodes 1000

  # L3武汉场景Zone B
  python run_full_experiments.py --level L3 --zone B --episodes 500

  # 断点恢复
  python run_full_experiments.py --level L1 --resume
        """,
    )
    parser.add_argument(
        "--level", type=str, required=True, choices=["L1", "L2", "L3", "L4"],
        help="问题层级 / Problem level (L1-L4)",
    )
    parser.add_argument(
        "--algos", type=str, default=None,
        help="算法列表（逗号分隔），默认使用该层级所有可用算法 / "
             "Algorithm list (comma-separated), default: all for level",
    )
    parser.add_argument(
        "--episodes", type=int, default=1000,
        help="每个算法的训练episode数 / Training episodes per algorithm (default: 1000)",
    )
    parser.add_argument(
        "--eval-interval", type=int, default=100,
        help="评估间隔 / Evaluation interval (default: 100)",
    )
    parser.add_argument(
        "--checkpoint-interval", type=int, default=DEFAULT_CHECKPOINT_INTERVAL,
        help=f"检查点保存间隔 / Checkpoint interval (default: {DEFAULT_CHECKPOINT_INTERVAL})",
    )
    parser.add_argument(
        "--zone", type=str, default="B", choices=["A", "B", "C", "D"],
        help="武汉场景分区（仅L3/L4）/ Wuhan scene zone (L3/L4 only, default: B)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="随机种子 / Random seed (default: 42)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="从最新检查点恢复 / Resume from latest checkpoint",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="输出目录（默认: 02_数据保存/experiment_results/）/ "
             "Output directory (default: 02_数据保存/experiment_results/)",
    )
    parser.add_argument(
        "--log-dir", type=str, default=None,
        help="日志目录（默认: 输出目录/logs/）/ "
             "Log directory (default: output_dir/logs/)",
    )
    return parser.parse_args()


def main():
    """主入口。/ Main entry point."""
    args = parse_args()

    # 确定算法列表 / Determine algorithm list
    if args.algos:
        algos = [a.strip() for a in args.algos.split(",")]
        # 验证算法名 / Validate algorithm names
        for a in algos:
            if a not in ALL_ALGOS:
                print(f"错误：不支持的算法 '{a}' / Error: unsupported algorithm '{a}'")
                print(f"支持的算法 / Supported: {ALL_ALGOS}")
                sys.exit(1)
    else:
        algos = LEVEL_ALGO_MAP[args.level]

    # 确定输出目录 / Determine output directories
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = PROJECT_ROOT / "02_数据保存" / "experiment_results"

    log_dir = Path(args.log_dir) if args.log_dir else output_dir / "logs"

    # 打印实验配置 / Print experiment config
    print(f"\n{'#'*70}")
    print(f"  无人机配送全层级实验 / Drone Delivery Full-Level Experiment")
    print(f"{'#'*70}")
    print(f"  层级 / Level:    {args.level}")
    print(f"  算法 / Algos:    {', '.join(algos)}")
    print(f"  Episodes:        {args.episodes}")
    print(f"  Eval interval:   {args.eval_interval}")
    print(f"  Ckpt interval:   {args.checkpoint_interval}")
    print(f"  Zone:            {args.zone}")
    print(f"  Seed:            {args.seed}")
    print(f"  Resume:          {args.resume}")
    print(f"  Output:          {output_dir}")
    print(f"{'#'*70}\n")

    run_experiment(
        level=args.level,
        algos=algos,
        num_episodes=args.episodes,
        eval_interval=args.eval_interval,
        checkpoint_interval=args.checkpoint_interval,
        zone=args.zone,
        seed=args.seed,
        resume=args.resume,
        output_dir=output_dir,
        log_dir=log_dir,
    )

    print("\n实验完成! / Experiments complete!")


if __name__ == "__main__":
    main()
