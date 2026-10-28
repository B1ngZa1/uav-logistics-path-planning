"""
无人机外卖配送 — 冒烟测试入口
Drone Food Delivery — Smoke Test Entry Point

用于验证代码正确性，在提交到平台前运行。
Run before submitting to the platform to verify code correctness.

使用方式 / Usage:
    python train_test.py --algo q_learning --episodes 5
"""
import argparse
import os
import sys

# 确保项目路径在 sys.path 中 / Ensure project path is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def smoke_test_q_learning(episodes: int = 5):
    """Q-learning冒烟测试 / Q-learning Smoke Test"""
    from environment.simple_delivery_env import Simple2DDeliveryEnv
    from agent_q_learning.agent import Agent
    from agent_q_learning.feature.definition import SampleManager

    print("=" * 50)
    print("Q-learning Smoke Test")
    print("=" * 50)

    env = Simple2DDeliveryEnv()
    agent = Agent()

    for ep in range(episodes):
        env_obs, _ = env.reset(seed=42 + ep)
        agent.reset(env_obs)
        obs_data = agent.observation_process(env_obs)
        collector = SampleManager()
        done = False

        while not done:
            act_data = agent.predict([obs_data])
            act = agent.action_process(act_data)
            _, reward, terminated, truncated, env_obs = env.step(act)
            next_obs_data = agent.observation_process(env_obs)
            shaped_reward = agent.reward_shaping(reward)
            done = terminated or truncated

            collector.add(
                obs=obs_data, _obs=next_obs_data,
                act=act, rew=shaped_reward, done=done, ret=0.0
            )
            obs_data = next_obs_data

        collector.process_last_frame()
        train_data = collector.get_game_data()
        if train_data:
            agent.learn(train_data)

        print(f"  Episode {ep+1}/{episodes} — "
              f"steps={env.steps}, deliveries={int(env_obs.get('order_status', [0]).sum())}")

    print("  Q-learning smoke test PASSED\n")


def smoke_test_sarsa(episodes: int = 5):
    """SARSA冒烟测试 / SARSA Smoke Test"""
    from environment.simple_delivery_env import Simple2DDeliveryEnv
    from agent_sarsa.agent import Agent
    from agent_sarsa.feature.definition import SampleManager

    print("=" * 50)
    print("SARSA Smoke Test")
    print("=" * 50)

    env = Simple2DDeliveryEnv()
    agent = Agent()

    for ep in range(episodes):
        env_obs, _ = env.reset(seed=42 + ep)
        agent.reset(env_obs)
        obs_data = agent.observation_process(env_obs)
        collector = SampleManager()

        # SARSA需要预选下一个动作 / SARSA needs pre-selected next action
        act_data = agent.predict([obs_data])
        act = agent.action_process(act_data)
        done = False

        while not done:
            _, reward, terminated, truncated, env_obs = env.step(act)
            next_obs_data = agent.observation_process(env_obs)
            next_act_data = agent.predict([next_obs_data])
            next_act = agent.action_process(next_act_data)
            shaped_reward = agent.reward_shaping(reward)
            done = terminated or truncated

            collector.add(
                obs=obs_data, _obs=next_obs_data,
                act=act, rew=shaped_reward, done=done,
                ret=0.0, next_act=next_act
            )
            obs_data = next_obs_data
            act = next_act

        collector.process_last_frame()
        train_data = collector.get_game_data()
        if train_data:
            agent.learn(train_data)

        print(f"  Episode {ep+1}/{episodes} — "
              f"steps={env.steps}, deliveries={int(env_obs.get('order_status', [0]).sum())}")

    print("  SARSA smoke test PASSED\n")


def smoke_test_monte_carlo(episodes: int = 5):
    """Monte Carlo冒烟测试 / Monte Carlo Smoke Test"""
    from environment.simple_delivery_env import Simple2DDeliveryEnv
    from agent_monte_carlo.agent import Agent
    from agent_monte_carlo.feature.definition import SampleManager

    print("=" * 50)
    print("Monte Carlo Smoke Test")
    print("=" * 50)

    env = Simple2DDeliveryEnv()
    agent = Agent()

    for ep in range(episodes):
        env_obs, _ = env.reset(seed=42 + ep)
        agent.reset(env_obs)
        obs_data = agent.observation_process(env_obs)
        collector = SampleManager()
        done = False

        while not done:
            act_data = agent.predict([obs_data])
            act = agent.action_process(act_data)
            _, reward, terminated, truncated, env_obs = env.step(act)
            next_obs_data = agent.observation_process(env_obs)
            shaped_reward = agent.reward_shaping(reward)
            done = terminated or truncated

            collector.add(
                obs=obs_data, _obs=next_obs_data,
                act=act, rew=shaped_reward, done=done, ret=0.0
            )
            obs_data = next_obs_data

        collector.process_last_frame()
        train_data = collector.get_game_data()
        if train_data:
            agent.learn(train_data)

        print(f"  Episode {ep+1}/{episodes} — "
              f"steps={env.steps}, deliveries={int(env_obs.get('order_status', [0]).sum())}")

    print("  Monte Carlo smoke test PASSED\n")


def smoke_test_dqn(episodes: int = 5):
    """DQN冒烟测试 / DQN Smoke Test"""
    from environment.random_delivery_env import Random2DDeliveryEnv
    from agent_dqn.agent import Agent
    from agent_dqn.feature.definition import SampleManager

    print("=" * 50)
    print("DQN Smoke Test")
    print("=" * 50)

    env = Random2DDeliveryEnv()
    agent = Agent()

    for ep in range(episodes):
        env_obs, _ = env.reset(seed=42 + ep)
        agent.reset(env_obs)
        obs_data = agent.observation_process(env_obs)
        collector = SampleManager()
        done = False

        while not done:
            act_data = agent.predict([obs_data])
            act = agent.action_process(act_data)
            _, reward, terminated, truncated, env_obs = env.step(act)
            next_obs_data = agent.observation_process(env_obs)
            shaped_reward = agent.reward_shaping(reward)
            done = terminated or truncated

            collector.add(
                obs=obs_data, _obs=next_obs_data,
                act=act, rew=shaped_reward, done=done, ret=0.0
            )
            obs_data = next_obs_data

        collector.process_last_frame()
        train_data = collector.get_game_data()
        if train_data:
            agent.learn(train_data)

        print(f"  Episode {ep+1}/{episodes} — "
              f"steps={env.steps}, completed={env_obs.get('order_status', [0]).sum() if hasattr(env_obs, 'get') else '?'}")

    print("  DQN smoke test PASSED\n")


def smoke_test_a2c(episodes: int = 5):
    """A2C冒烟测试 / A2C Smoke Test"""
    from environment.random_delivery_env import Random2DDeliveryEnv
    from agent_a2c.agent import Agent
    from agent_a2c.feature.definition import SampleManager

    print("=" * 50)
    print("A2C Smoke Test")
    print("=" * 50)

    env = Random2DDeliveryEnv()
    agent = Agent()

    for ep in range(episodes):
        env_obs, _ = env.reset(seed=42 + ep)
        agent.reset(env_obs)
        obs_data = agent.observation_process(env_obs)
        collector = SampleManager()
        done = False

        while not done:
            act_data = agent.predict([obs_data])
            act = agent.action_process(act_data)
            _, reward, terminated, truncated, env_obs = env.step(act)
            next_obs_data = agent.observation_process(env_obs)
            shaped_reward = agent.reward_shaping(reward)
            done = terminated or truncated

            collector.add(
                obs=obs_data, _obs=next_obs_data,
                act=act, rew=shaped_reward, done=done, ret=0.0
            )
            obs_data = next_obs_data

        collector.process_last_frame()
        train_data = collector.get_game_data()
        if train_data:
            agent.learn(train_data)

        print(f"  Episode {ep+1}/{episodes} — "
              f"steps={env.steps}, completed={env.info.get('completed', '?')}")

    print("  A2C smoke test PASSED\n")


def main():
    parser = argparse.ArgumentParser(description="无人机外卖配送 — 冒烟测试 / Drone Delivery Smoke Test")
    parser.add_argument("--algo", type=str, default="all",
                        choices=["all", "q_learning", "sarsa", "monte_carlo", "dqn", "a2c"],
                        help="算法名称 / Algorithm name")
    parser.add_argument("--episodes", type=int, default=5,
                        help="测试回合数 / Number of test episodes")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  无人机外卖配送 — 冒烟测试")
    print("  Drone Food Delivery — Smoke Test")
    print("=" * 60 + "\n")

    if args.algo in ("all", "q_learning"):
        try:
            smoke_test_q_learning(args.episodes)
        except Exception as e:
            print(f"  Q-learning FAILED: {e}\n")

    if args.algo in ("all", "sarsa"):
        try:
            smoke_test_sarsa(args.episodes)
        except Exception as e:
            print(f"  SARSA FAILED: {e}\n")

    if args.algo in ("all", "monte_carlo"):
        try:
            smoke_test_monte_carlo(args.episodes)
        except Exception as e:
            print(f"  Monte Carlo FAILED: {e}\n")

    if args.algo in ("all", "dqn"):
        try:
            smoke_test_dqn(args.episodes)
        except Exception as e:
            print(f"  DQN FAILED: {e}\n")

    if args.algo in ("all", "a2c"):
        try:
            smoke_test_a2c(args.episodes)
        except Exception as e:
            print(f"  A2C FAILED: {e}\n")

    print("=" * 60)
    print("  冒烟测试完成 / Smoke Test Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
