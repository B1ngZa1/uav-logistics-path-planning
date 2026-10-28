"""Evaluate the authoritative multi-drone :class:`AirspaceEnv` scenarios.

The policy consumes the flattened observation exposed by ``AirspaceEnv`` and
emits a flattened action which is reshaped back to the environment's action
space before every step.  Keeping that conversion here is important for S1:
the three-drone scenario has a 33-dimensional observation and a 9-dimensional
policy action (the environment action itself is shaped ``(3, 3)``).
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithm.ppo_agent import PPOAgent
from config import ENV_CONFIG, PPO_CONFIG
from environment.airspace_env import AirspaceEnv


SCENARIOS: Dict[str, Dict[str, int]] = {
    "S1": {"num_uavs": 3, "num_targets": 5, "num_obstacles": 0},
    "S2": {"num_uavs": 3, "num_targets": 5, "num_obstacles": 5},
    "S3": {"num_uavs": 3, "num_targets": 10, "num_obstacles": 10},
    "S4": {"num_uavs": 5, "num_targets": 15, "num_obstacles": 15},
    "S5": {"num_uavs": 6, "num_targets": 20, "num_obstacles": 20},
    "S6": {"num_uavs": 5, "num_targets": 15, "num_obstacles": 15},
}


def scenario_config(
    scenario_name: str, *, max_steps: Optional[int] = None
) -> Dict[str, object]:
    """Return a validated copy of the environment config for a scenario."""

    name = scenario_name.upper()
    if name not in SCENARIOS:
        valid = ", ".join(SCENARIOS)
        raise ValueError(
            f"unknown scenario {scenario_name!r}; expected one of {valid}"
        )
    if max_steps is not None and max_steps < 1:
        raise ValueError("max_steps must be at least 1")

    config = dict(ENV_CONFIG)
    config.update(SCENARIOS[name])
    if max_steps is not None:
        config["max_steps"] = max_steps
    return config


def build_scenario_env(
    scenario_name: str, *, max_steps: Optional[int] = None
) -> AirspaceEnv:
    """Construct the single source of truth used by evaluation and tests."""

    return AirspaceEnv(scenario_config(scenario_name, max_steps=max_steps))


def _space_size(space) -> int:
    """Return the number of scalar values in a Gym space shape."""

    if space.shape is None:
        raise ValueError("scenario evaluation requires shaped Box spaces")
    return int(np.prod(space.shape, dtype=np.int64))


def build_agent_for_env(
    env: AirspaceEnv, *, hidden_dim: Optional[int] = None
) -> PPOAgent:
    """Build a PPO policy whose dimensions are derived from ``env``.

    This prevents the historical single-drone ``11``/``3`` defaults from
    being used with S1's flattened 33/9 contract.
    """

    state_dim = _space_size(env.observation_space)
    action_dim = _space_size(env.action_space)
    action_limit = np.asarray(env.action_space.high, dtype=np.float32).reshape(
        -1
    )
    if hidden_dim is None:
        hidden_dim = PPO_CONFIG["hidden_dim"]
    if hidden_dim < 4 or hidden_dim % 4:
        raise ValueError("hidden_dim must be a positive multiple of 4")
    return PPOAgent(
        {
            "state_dim": state_dim,
            "action_dim": action_dim,
            "hidden_dim": hidden_dim,
            "lr": PPO_CONFIG["learning_rate"],
            "gamma": PPO_CONFIG["gamma"],
            "gae_lambda": PPO_CONFIG["gae_lambda"],
            "clip_epsilon": PPO_CONFIG["clip_epsilon"],
            "entropy_coef": PPO_CONFIG["entropy_coef"],
            "value_coef": PPO_CONFIG["value_coef"],
            "max_grad_norm": PPO_CONFIG["max_grad_norm"],
            "update_epochs": PPO_CONFIG["update_epochs"],
            "batch_size": PPO_CONFIG["batch_size"],
            "action_limit": action_limit,
        }
    )


def _checkpoint_config(checkpoint: Path) -> Dict[str, object]:
    """Read checkpoint metadata before constructing the matching network."""

    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"checkpoint does not exist or is not a file: {checkpoint}"
        )
    try:
        try:
            payload = torch.load(
                checkpoint, map_location="cpu", weights_only=False
            )
        except TypeError:  # pragma: no cover - older PyTorch compatibility
            payload = torch.load(checkpoint, map_location="cpu")
    except Exception as exc:
        raise ValueError(f"unable to read checkpoint {checkpoint}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint {checkpoint} must contain a dictionary")
    config = payload.get("config", {})
    return config if isinstance(config, dict) else {}


def _load_checkpoint(agent: PPOAgent, checkpoint: Path) -> None:
    """Load an evaluation checkpoint with a useful missing-file error."""

    try:
        agent.load(checkpoint, load_optimizers=False)
    except (KeyError, RuntimeError, ValueError) as exc:
        raise ValueError(
            f"checkpoint {checkpoint} is incompatible with the scenario "
            f"(expected state_dim={agent.state_dim}, "
            f"action_dim={agent.action_dim})"
        ) from exc


def _evaluate_episode(
    env: AirspaceEnv,
    agent: PPOAgent,
    *,
    episode: int,
    seed: int,
    deterministic: bool,
) -> Dict[str, object]:
    """Run one seeded episode and return its terminal metrics."""

    env.seed(seed)
    state = np.asarray(env.reset(), dtype=np.float32)
    expected_state_shape = env.observation_space.shape
    if state.shape != expected_state_shape:
        raise ValueError(
            f"environment observation shape {state.shape} does not match "
            f"observation_space {expected_state_shape}"
        )

    total_reward = 0.0
    done = False
    info: Dict[str, object] = {}
    while not done:
        action, _, _ = agent.select_action(
            state, deterministic=deterministic
        )
        expected_action_shape = env.action_space.shape
        flat_action_shape = (int(np.prod(expected_action_shape)),)
        if action.shape != flat_action_shape:
            raise ValueError(
                f"policy action shape {action.shape} does not match the "
                f"flattened action_space {flat_action_shape}"
            )
        actions = action.reshape(expected_action_shape)
        if not env.action_space.contains(actions.astype(np.float32)):
            raise ValueError(
                f"reshaped policy action is outside action_space "
                f"{expected_action_shape}"
            )
        state, reward, done, info = env.step(actions)
        state = np.asarray(state, dtype=np.float32)
        total_reward += float(reward)

    return {
        "episode": episode,
        "seed": seed,
        "reward": float(total_reward),
        "delivered": int(info["delivered"]),
        "collisions": int(info["collisions"]),
        "steps": int(info["step"]),
    }


def run_scenario(
    scenario_name: str,
    num_episodes: int = 100,
    *,
    checkpoint: Optional[os.PathLike[str] | str] = None,
    seed: int = 42,
    deterministic: bool = True,
    max_steps: Optional[int] = None,
    verbose: bool = True,
) -> List[Dict[str, object]]:
    """Evaluate a scenario using a seeded, optionally loaded PPO policy.

    Evaluation is deterministic by default: each episode gets ``seed + ep``
    and the policy uses its mean action.  Pass ``deterministic=False`` only
    when stochastic policy sampling is intentionally desired.
    """

    if num_episodes < 1:
        raise ValueError("num_episodes must be at least 1")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    name = scenario_name.upper()
    env = build_scenario_env(name, max_steps=max_steps)
    checkpoint_path = Path(checkpoint) if checkpoint is not None else None
    checkpoint_metadata = (
        _checkpoint_config(checkpoint_path)
        if checkpoint_path is not None
        else {}
    )
    saved_hidden_dim = checkpoint_metadata.get("hidden_dim")
    hidden_dim = (
        int(saved_hidden_dim)
        if saved_hidden_dim is not None
        else None
    )
    agent = build_agent_for_env(env, hidden_dim=hidden_dim)
    if checkpoint_path is not None:
        _load_checkpoint(agent, checkpoint_path)

    if verbose:
        print(
            f"Running {name}: {env.num_uavs} drones, "
            f"{env.num_targets} targets, {env.num_obstacles} obstacles"
        )
        print(
            f"Policy dimensions: state={agent.state_dim}, "
            f"action={agent.action_dim}; "
            f"env action shape={env.action_space.shape}; "
            f"mode={'deterministic' if deterministic else 'stochastic'}"
        )
        if checkpoint_path is not None:
            print(f"Loaded checkpoint: {checkpoint_path.resolve()}")

    results = [
        _evaluate_episode(
            env,
            agent,
            episode=episode,
            seed=seed + episode,
            deterministic=deterministic,
        )
        for episode in range(num_episodes)
    ]

    rewards = [float(result["reward"]) for result in results]
    delivered = [int(result["delivered"]) for result in results]
    collisions = [int(result["collisions"]) for result in results]
    if verbose:
        print(f"\n{name} evaluation:")
        print(f"  reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
        print(
            f"  delivered: {np.mean(delivered):.2f}/{env.num_targets}"
        )
        print(f"  collisions: {np.mean(collisions):.2f}")
        print(
            f"  completion rate: "
            f"{np.mean(np.asarray(delivered) / env.num_targets) * 100:.1f}%"
        )
    return results


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Deterministically evaluate the authoritative AirspaceEnv scenarios."
    )
    parser.add_argument(
        "--scenario",
        choices=tuple(SCENARIOS),
        default="S1",
        help="scenario to evaluate (default: S1)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="number of evaluation episodes (default: 100)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="optional PPO checkpoint produced by main_train.py",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="base seed; episode N uses seed + N (default: 42)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="optional episode step cap, useful for smoke tests",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--deterministic",
        dest="deterministic",
        action="store_true",
        help="use policy mean actions (default)",
    )
    mode.add_argument(
        "--stochastic",
        dest="deterministic",
        action="store_false",
        help="sample policy actions instead of using the mean",
    )
    parser.set_defaults(deterministic=True)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    run_scenario(
        args.scenario,
        args.episodes,
        checkpoint=args.checkpoint,
        seed=args.seed,
        deterministic=args.deterministic,
        max_steps=args.max_steps,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
