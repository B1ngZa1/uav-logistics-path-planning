"""
无人机配送 — SARSA训练工作流
Drone Delivery — SARSA Training Workflow

职责 / Responsibility:
    单机SARSA训练循环。与Q-learning的区别：需要预选下一动作a'
    SARSA使用在线策略更新：Q(s,a) += lr*(r + gamma*Q(s',a') - Q(s,a))

参考 / Reference:
    kaiwu_gorge_walk_v2/agent_sarsa/workflow/train_workflow.py

领域适配 / Domain Adaptation:
    峡谷宝箱收集 → 无人机取送货
"""
import os, sys, time
import numpy as np
from agent_sarsa.conf.conf import Config
from agent_sarsa.feature.definition import SampleManager

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


class EpisodeRunner:
    """SARSA回合运行器 / SARSA Episode Runner"""

    def __init__(self, env, agent, logger=None, monitor=None):
        self.env, self.agent = env, agent
        self.logger, self.monitor = logger, monitor
        self.episode_rewards, self.episode_deliveries, self.episode_steps = [], [], []
        self.best_avg_reward = -float('inf')

    def run_episode(self, seed=None):
        """SARSA训练回合 — 需要预选初始动作"""
        env_obs, _ = self.env.reset(seed=seed) if seed else self.env.reset()
        self.agent.reset(env_obs)
        obs_data = self.agent.observation_process(env_obs)
        # SARSA关键差异：循环前先选动作
        act_data = self.agent.predict([obs_data])
        act = self.agent.action_process(act_data)
        collector, ep_r, ep_steps, done = SampleManager(), 0.0, 0, False

        while not done:
            env_obs, reward, terminated, truncated, _ = self.env.step(act)
            shaped_reward = self.agent.reward_shaping(reward)
            next_obs_data = self.agent.observation_process(env_obs)
            done = terminated or truncated
            next_act = self.agent.action_process(self.agent.predict([next_obs_data])) if not done else -1

            collector.add(obs=obs_data, _obs=next_obs_data, act=act, rew=shaped_reward,
                          done=done, ret=0.0, next_act=next_act)
            obs_data, act = next_obs_data, (next_act if not done else -1)
            ep_r += reward; ep_steps += 1

        collector.process_last_frame()
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
            env_obs, _ = self.env.reset(seed=20000 + ep)
            done = False
            while not done:
                act_data = self.agent.exploit(env_obs)
                act = self.agent.action_process(act_data)
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
