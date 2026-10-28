"""武汉参考场景：路线规划器 + PPO 连续控制训练入口。

示例：

    python run_wuhan_training.py --zone B --episodes 1000
    python run_wuhan_training.py --zone A --episodes 1 --max-steps 2 --no-save
    python run_wuhan_training.py --resume outputs/models/wuhan_ppo.checkpoints/latest.pt --episodes 100

规划器负责“尽可能多配送且每个候选都保留返航预算”，PPO 仅学习三轴
加速度和偏航角速度。这与开悟峡谷漫步增强策略的“精确路线兜底 +
学习局部动作”结构一致，同时避免把离散目标评分误算进连续策略概率。

训练产物：

* 周期检查点保存在 ``<output-stem>.checkpoints``，并包含优化器、训练进度、
  随机数状态和可复现实验配置；
* ``output`` 是最终的同格式检查点，既可以作为模型使用，也可以继续训练；
* JSONL 日志按追加模式写入，便于在一次文件中保留新运行和恢复运行的完整轨迹。
"""

from __future__ import annotations

import argparse
import json
import random
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from algorithm.delivery_strategy import ScoreFirstDeliveryPlanner
from algorithm.ppo_agent import PPOAgent
from config import PPO_CONFIG, WUHAN_ENV_CONFIG
from environment.wuhan_drone_env import WuhanDroneDeliveryEnv


CHECKPOINT_VERSION = 1
DEFAULT_SEED = 42
REWARD_SCALE = 0.01


def _json_safe(value: Any) -> Any:
    """Convert common NumPy/PyTorch/path values to JSON-compatible values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class JsonlLogger:
    """Append one self-contained, structured record per line."""

    def __init__(self, path: Path, run_id: str, seed: int):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.seed = seed
        self._stream = self.path.open("a", encoding="utf-8")

    def write(self, event: str, **payload: Any) -> None:
        record = {
            "schema_version": CHECKPOINT_VERSION,
            "timestamp_utc": _utc_timestamp(),
            "run_id": self.run_id,
            "seed": self.seed,
            "event": event,
            **payload,
        }
        self._stream.write(
            json.dumps(_json_safe(record), ensure_ascii=False) + "\n"
        )
        self._stream.flush()

    def close(self) -> None:
        self._stream.close()


def build_agent_config(state_dim: int) -> Dict[str, Any]:
    return {
        "state_dim": state_dim,
        "action_dim": 4,
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
    }


def build_agent(state_dim: int) -> PPOAgent:
    return PPOAgent(build_agent_config(state_dim))


def run_episode(
    env: WuhanDroneDeliveryEnv,
    agent: PPOAgent,
    planner: ScoreFirstDeliveryPlanner,
    *,
    training: bool,
    seed: int,
    max_steps: Optional[int] = None,
    reward_scale: float = REWARD_SCALE,
):
    observation, _ = env.reset(seed=seed)
    state = env.flatten_observation(observation)
    episode_reward = 0.0
    terminated = False
    truncated = False
    info = {}
    limit = max_steps or int(
        np.ceil(env.config.max_episode_time_s / env.config.control_dt_s)
    )

    for _ in range(limit):
        physical_action, log_prob, value = agent.select_action(
            state, deterministic=not training
        )
        action = planner.compose_action(env, physical_action)
        (
            next_observation,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)
        next_state = env.flatten_observation(next_observation)
        episode_reward += reward

        if training:
            # terminated 是真实终局，不进行 bootstrap；TimeLimit 截断仍
            # 对 next_state 估值，避免把时间边界误当作价值为零。
            agent.store_transition(
                state,
                physical_action,
                reward * reward_scale,
                bool(terminated),
                log_prob,
                value,
            )
        state = next_state
        if terminated or truncated:
            break
    else:
        truncated = True

    metrics = {}
    if training:
        next_value = 0.0 if terminated else agent.estimate_value(state)
        agent.update(next_value)
        metrics = dict(agent.last_update_metrics)

    return {
        "reward": float(episode_reward),
        "steps": int(info.get("step", limit)),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "termination_reason": info.get("termination_reason"),
        "mission_phase": info.get("mission_phase"),
        "served_task_ids": info.get("served_task_ids", []),
        "mission_score": float(info.get("mission_score", 0.0)),
        "objective_rank": info.get("objective_rank", []),
        "battery_soc": info.get("battery_soc", []),
        "update": metrics,
    }


def _capture_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Optional[Dict[str, Any]]) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(tuple(state["numpy"]))
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("torch_cuda"):
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _load_torch_payload(path: Path) -> Dict[str, Any]:
    """Load a full checkpoint on CPU across supported PyTorch versions."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch < 2.6 has no weights_only argument.
        return torch.load(path, map_location="cpu")


def _restore_agent(agent: PPOAgent, checkpoint: Dict[str, Any]) -> None:
    """Restore both policy weights and optimizer state from old/new files."""
    agent.actor.load_state_dict(checkpoint["actor"])
    agent.critic.load_state_dict(checkpoint["critic"])
    if "actor_optimizer" in checkpoint:
        agent.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
    if "critic_optimizer" in checkpoint:
        agent.critic_optimizer.load_state_dict(
            checkpoint["critic_optimizer"]
        )


def _atomic_torch_save(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(path)


def save_checkpoint(
    agent: PPOAgent,
    path: Path,
    *,
    completed_episodes: int,
    run_id: str,
    seed: int,
    run_config: Dict[str, Any],
) -> None:
    """Save a resumable checkpoint in the same top-level format PPOAgent.load expects."""
    payload = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "completed_episodes": completed_episodes,
        "run_id": run_id,
        "seed": seed,
        "run_config": run_config,
        "actor": agent.actor.state_dict(),
        "critic": agent.critic.state_dict(),
        "actor_optimizer": agent.actor_optimizer.state_dict(),
        "critic_optimizer": agent.critic_optimizer.state_dict(),
        "config": agent.config,
        "rng_state": _capture_rng_state(),
    }
    _atomic_torch_save(payload, Path(path))


def _new_run_id() -> str:
    return (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "-"
        + uuid.uuid4().hex[:8]
    )


def _evaluation_summary(results):
    if not results:
        return {
            "episode_count": 0,
            "reward_mean": None,
            "reward_std": None,
            "steps_mean": None,
            "mission_score_mean": None,
            "mission_score_max": None,
            "served_task_count_mean": None,
            "mission_complete_rate": None,
            "safe_return_rate": None,
            "termination_reasons": {},
        }

    rewards = np.asarray([item["reward"] for item in results], dtype=float)
    steps = np.asarray([item["steps"] for item in results], dtype=float)
    scores = np.asarray(
        [item["mission_score"] for item in results], dtype=float
    )
    served = np.asarray(
        [len(item["served_task_ids"]) for item in results], dtype=float
    )
    reasons = {}
    for item in results:
        reason = item.get("termination_reason") or "unknown"
        reasons[reason] = reasons.get(reason, 0) + 1
    count = len(results)
    mission_complete_count = reasons.get("mission_complete", 0)
    safe_return_count = sum(
        reasons.get(reason, 0)
        for reason in ("mission_complete", "safe_return_partial")
    )
    return {
        "episode_count": count,
        "reward_mean": float(rewards.mean()),
        "reward_std": float(rewards.std()),
        "steps_mean": float(steps.mean()),
        "mission_score_mean": float(scores.mean()),
        "mission_score_max": float(scores.max()),
        "served_task_count_mean": float(served.mean()),
        "mission_complete_rate": float(mission_complete_count / count),
        "safe_return_rate": float(safe_return_count / count),
        "termination_reasons": reasons,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="训练武汉多目标配送—安全返航 PPO"
    )
    parser.add_argument(
        "--zone", choices=("A", "B", "C", "D"), default=None
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1000,
        help="本次运行训练的回合数；恢复时是在检查点之后新增的回合数",
    )
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="新运行的基础种子；恢复运行使用检查点中的种子",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="仅用于调试；正式训练建议由 max_episode_time_s 控制",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="最终检查点路径，默认 outputs/models/wuhan_ppo.pt",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="周期检查点目录，默认 <output-stem>.checkpoints",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=100,
        help="每多少个累计训练回合保存一次检查点",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="JSONL 日志路径，默认与 output 同目录同名",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="从完整检查点恢复；--episodes 表示本次额外训练回合数",
    )
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.episodes < 1 or args.eval_episodes < 0:
        raise ValueError("episodes 至少为 1，eval-episodes 不能为负")
    if args.checkpoint_interval < 1:
        raise ValueError("checkpoint-interval 至少为 1")
    if args.max_steps is not None and args.max_steps < 1:
        raise ValueError("max-steps 至少为 1")

    resume_payload = None
    resume_config = {}
    if args.resume is not None:
        resume_payload = _load_torch_payload(args.resume)
        resume_config = resume_payload.get("run_config", {})

    zone = args.zone or resume_config.get("zone", "A")
    max_steps = (
        args.max_steps
        if args.max_steps is not None
        else resume_config.get("max_steps")
    )
    default_output = Path("outputs/models/wuhan_ppo.pt")
    output = Path(args.output or resume_config.get("output", default_output))
    checkpoint_dir = Path(
        args.checkpoint_dir
        or resume_config.get(
            "checkpoint_dir",
            output.parent / f"{output.stem}.checkpoints",
        )
    )
    log_file = Path(
        args.log_file
        or resume_config.get("log_file", output.with_suffix(".jsonl"))
    )

    checkpoint_seed = (
        resume_payload.get("seed") if resume_payload is not None else None
    )
    if checkpoint_seed is not None and args.seed is not None:
        if int(args.seed) != int(checkpoint_seed):
            raise ValueError(
                "恢复运行不能更改检查点中的 seed；请省略 --seed"
            )
    seed = int(
        checkpoint_seed
        if checkpoint_seed is not None
        else (args.seed if args.seed is not None else DEFAULT_SEED)
    )

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env_config = dict(WUHAN_ENV_CONFIG)
    env_config.update(
        {
            "zone_id": zone,
            "num_drones": 1,
            "seed": seed,
        }
    )
    if max_steps is not None:
        env_config["max_episode_time_s"] = (
            max_steps * env_config["control_dt_s"]
        )

    env = WuhanDroneDeliveryEnv(env_config)
    logger = None
    try:
        initial_observation, _ = env.reset(seed=seed)
        state_dim = env.flatten_observation(initial_observation).shape[0]
        agent = build_agent(state_dim)
        planner = ScoreFirstDeliveryPlanner()

        completed_episodes = 0
        run_id = _new_run_id()
        legacy_resume = False
        if resume_payload is not None:
            _restore_agent(agent, resume_payload)
            completed_episodes = int(
                resume_payload.get("completed_episodes", 0)
            )
            run_id = resume_payload.get("run_id") or run_id
            legacy_resume = "checkpoint_version" not in resume_payload
            # Restore after constructing/loading the agent: construction itself
            # consumes random numbers and must not overwrite saved RNG state.
            _restore_rng_state(resume_payload.get("rng_state"))

        run_config = {
            "zone": zone,
            "episodes_per_invocation": args.episodes,
            "eval_episodes": args.eval_episodes,
            "seed": seed,
            "max_steps": max_steps,
            "checkpoint_interval": args.checkpoint_interval,
            "reward_scale": REWARD_SCALE,
            "output": str(output.resolve()),
            "checkpoint_dir": str(checkpoint_dir.resolve()),
            "log_file": str(log_file.resolve()),
            "env": env_config,
            "ppo": agent.config,
        }
        logger = JsonlLogger(log_file, run_id, seed)
        logger.write(
            "run_start",
            seed=seed,
            resumed_from=(
                str(args.resume.resolve()) if args.resume is not None else None
            ),
            legacy_resume=legacy_resume,
            completed_episodes=completed_episodes,
            config=run_config,
        )

        if args.resume is not None:
            print(
                f"恢复运行 {run_id}: 已完成 {completed_episodes} 回合，"
                f"本次再训练 {args.episodes} 回合",
                flush=True,
            )

        def persist_checkpoint(episode_number: int, event: str) -> None:
            if args.no_save:
                return
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = checkpoint_dir / (
                f"checkpoint_ep{episode_number:06d}.pt"
            )
            save_checkpoint(
                agent,
                checkpoint_path,
                completed_episodes=episode_number,
                run_id=run_id,
                seed=seed,
                run_config=run_config,
            )
            latest_path = checkpoint_dir / "latest.pt"
            if latest_path != checkpoint_path:
                save_checkpoint(
                    agent,
                    latest_path,
                    completed_episodes=episode_number,
                    run_id=run_id,
                    seed=seed,
                    run_config=run_config,
                )
            logger.write(
                event,
                episode=episode_number,
                path=str(checkpoint_path.resolve()),
                latest_path=str(latest_path.resolve()),
            )
            print(
                f"检查点已保存：{checkpoint_path.resolve()}", flush=True
            )

        for local_episode in range(args.episodes):
            episode_number = completed_episodes + local_episode + 1
            episode_seed = seed + episode_number - 1
            result = run_episode(
                env,
                agent,
                planner,
                training=True,
                seed=episode_seed,
                max_steps=max_steps,
            )
            logger.write(
                "train_episode",
                episode=episode_number,
                seed=episode_seed,
                metrics=result,
            )
            if (
                local_episode == 0
                or (episode_number % 10 == 0)
                or local_episode + 1 == args.episodes
            ):
                print(
                    json.dumps(
                        {
                            "mode": "train",
                            "run_id": run_id,
                            "episode": episode_number,
                            "seed": episode_seed,
                            **result,
                        },
                        ensure_ascii=False,
                        default=lambda value: _json_safe(value),
                    ),
                    flush=True,
                )
            if episode_number % args.checkpoint_interval == 0:
                persist_checkpoint(episode_number, "checkpoint_saved")

        completed_episodes += args.episodes
        if not args.no_save:
            save_checkpoint(
                agent,
                output,
                completed_episodes=completed_episodes,
                run_id=run_id,
                seed=seed,
                run_config=run_config,
            )
            latest_path = checkpoint_dir / "latest.pt"
            if latest_path != output:
                save_checkpoint(
                    agent,
                    latest_path,
                    completed_episodes=completed_episodes,
                    run_id=run_id,
                    seed=seed,
                    run_config=run_config,
                )
            logger.write(
                "final_checkpoint_saved",
                episode=completed_episodes,
                path=str(output.resolve()),
                latest_path=str(latest_path.resolve()),
            )
            print(f"模型已保存：{output.resolve()}", flush=True)

        evaluation_results = []
        for episode in range(args.eval_episodes):
            evaluation_seed = seed + 100000 + episode
            result = run_episode(
                env,
                agent,
                planner,
                training=False,
                seed=evaluation_seed,
                max_steps=max_steps,
            )
            evaluation_results.append(result)
            logger.write(
                "evaluation_episode",
                episode=episode + 1,
                seed=evaluation_seed,
                metrics=result,
            )
            print(
                json.dumps(
                    {
                        "mode": "eval",
                        "run_id": run_id,
                        "episode": episode + 1,
                        "seed": evaluation_seed,
                        **result,
                    },
                    ensure_ascii=False,
                    default=lambda value: _json_safe(value),
                ),
                flush=True,
            )

        evaluation_metrics = _evaluation_summary(evaluation_results)
        logger.write(
            "evaluation_summary",
            seed=seed + 100000,
            metrics=evaluation_metrics,
        )
        logger.write(
            "run_end",
            seed=seed,
            completed_episodes=completed_episodes,
            evaluation_metrics=evaluation_metrics,
        )
        return {
            "run_id": run_id,
            "completed_episodes": completed_episodes,
            "evaluation_metrics": evaluation_metrics,
        }
    finally:
        env.close()
        if logger is not None:
            logger.close()


if __name__ == "__main__":
    main()
