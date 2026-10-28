"""
无人机配送 — Monte Carlo训练工作流
Drone Delivery — Monte Carlo Training Workflow

职责 / Responsibility:
    单机MC训练。关键差异：完整回合结束后才学习，用累积回报G_t更新。
    MC updates only at episode end using full cumulative returns G_t.

参考 / Reference:
    kaiwu_gorge_walk_v2/agent_monte_carlo/workflow/train_workflow.py
"""
import os, sys, time
import numpy as np
from agent_monte_carlo.conf.conf import Config
from agent_monte_carlo.feature.definition import SampleManager

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


class EpisodeRunner:
    """MC回合运行器 — 先收集完整回合再学习 / Collect full episode then learn"""

    def __init__(self, env, agent, logger=None, monitor=None):
        self.env, self.agent = env, agent
        self.logger, self.monitor = logger, monitor
        self.episode_rewards, self.episode_deliveries, self.episode_steps = [], [], []
        self.best_avg_reward = -float('inf')

    def run_episode(self, seed=None):
        env_obs, _ = self.env.reset(seed=seed) if seed else self.env.reset()
        self.agent.reset(env_obs)
        obs_data = self.agent.observation_process(env_obs)
        collector, ep_r, ep_steps, done = SampleManager(), 0.0, 0, False

        # 收集阶段：执行完整回合，不学习 / Collect phase: execute full episode, no learning
        while not done:
            act_data = self.agent.predict([obs_data])
            act = self.agent.action_process(act_data)
            env_obs, reward, terminated, truncated, _ = self.env.step(act)
            shaped_reward = self.agent.reward_shaping(reward)
            next_obs_data = self.agent.observation_process(env_obs)
            done = terminated or truncated
            collector.add(obs=obs_data, _obs=next_obs_data, act=act, rew=shaped_reward, done=done, ret=0.0)
            obs_data = next_obs_data
            ep_r += reward; ep_steps += 1

        collector.process_last_frame()
        # 学习阶段：回合结束后统一学习 / Learn all at once after episode
        train_data = collector.get_game_data()
        if train_data and len(train_data) > 0:
            self.agent.learn(train_data)

        deliveries = int(env_obs.get("order_status", np.zeros(Config.NUM_ORDERS)).sum())
        self.episode_rewards.append(ep_r)
        self.episode_deliveries.append(deliveries)
        self.episode_steps.append(ep_steps)
        return {"reward": ep_r, "deliveries": deliveries, "steps": ep_steps, "epsilon": self.agent.epsilon}

    def evaluate(self, num_episodes=10):
        tr, td, ts = 0.0, 0, 0
        for ep in range(num_episodes):
            env_obs, _ = self.env.reset(seed=30000 + ep)
            done = False
            while not done:
                act = self.agent.action_process(self.agent.exploit(env_obs))
                env_obs, reward, terminated, truncated, _ = self.env.step(act)
                tr += reward; ts += 1; done = terminated or truncated
            td += int(env_obs.get("order_status", np.zeros(Config.NUM_ORDERS)).sum())
        return {"avg_reward": tr / num_episodes, "avg_deliveries": td / num_episodes,
                "avg_steps": ts / num_episodes, "delivery_rate": td / (Config.NUM_ORDERS * num_episodes)}

    def get_recent_avg_reward(self, window=100):
        if not self.episode_rewards: return 0.0
        return np.mean(self.episode_rewards[-window:])


def workflow(envs, agents, logger=None, monitor=None):
    if len(envs) == 0 or len(agents) == 0:
        raise ValueError("envs and agents must not be empty")
    runner = EpisodeRunner(envs[0], agents[0], logger, monitor)
    num_ep, ev_int, sv_int = getattr(Config, 'NUM_EPISODES', 10000), getattr(Config, 'EVAL_INTERVAL', 100), 500
    start_time = time.time()
    for ep in range(num_ep):
        stats = runner.run_episode()
        if (ep + 1) % ev_int == 0:
            avg_r = runner.get_recent_avg_reward(ev_int)
            avg_d = np.mean(runner.episode_deliveries[-ev_int:]) if runner.episode_deliveries else 0
            ev = runner.evaluate(5)
            print(f"Ep {ep+1:5d} | AvgR: {avg_r:8.1f} | AvgD: {avg_d:.1f}/5 | ε: {agents[0].epsilon:.4f} | EvalR: {ev['avg_reward']:.1f}")
            if avg_r > runner.best_avg_reward:
                runner.best_avg_reward = avg_r; agents[0].save_model(id="best")
        if (ep + 1) % sv_int == 0: agents[0].save_model(id=f"ep{ep+1}")
        yield stats
    agents[0].save_model(id="latest")
    fe = runner.evaluate(100)
    print(f"Final: AvgR={fe['avg_reward']:.1f}, AvgD={fe['avg_deliveries']:.1f}/5, Rate={fe['delivery_rate']:.1%}")
