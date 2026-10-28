"""
PPO智能体
项目：基于深度强化学习的多无人机协同物流配送路径规划
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from pathlib import Path


class ActorNetwork(nn.Module):
    """Actor策略网络"""

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU()
        )
        self.mean_head = nn.Linear(hidden_dim // 4, action_dim)
        self.log_std_head = nn.Linear(hidden_dim // 4, action_dim)

    def forward(self, state):
        x = self.network(state)
        mean = self.mean_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, -20, 2)
        return mean, log_std


class CriticNetwork(nn.Module):
    """Critic价值网络"""

    def __init__(self, state_dim, hidden_dim=256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.Tanh(),
            nn.Linear(hidden_dim // 4, 1)
        )

    def forward(self, state):
        return self.network(state)


class PPOAgent:
    """PPO智能体"""

    def __init__(self, config):
        self.config = dict(config)
        self.state_dim = config.get('state_dim', 11)
        self.action_dim = config.get('action_dim', 3)
        self.hidden_dim = config.get('hidden_dim', 256)
        self.lr = config.get('lr', 3e-4)
        self.gamma = config.get('gamma', 0.99)
        self.gae_lambda = config.get('gae_lambda', 0.95)
        self.clip_epsilon = config.get('clip_epsilon', 0.2)
        self.entropy_coef = config.get('entropy_coef', 0.01)
        self.value_coef = config.get('value_coef', 0.5)
        self.max_grad_norm = config.get('max_grad_norm', 0.5)
        # update_epochs 表示一次 rollout 后 PPO 重复优化的轮数。
        # num_epochs 仅作为旧配置兼容项，不能再与训练回合数混用。
        self.update_epochs = config.get(
            'update_epochs', config.get('num_epochs', 10)
        )
        self.batch_size = config.get('batch_size', 256)
        action_limit = np.asarray(
            config.get('action_limit', 3.0), dtype=np.float32
        )
        if action_limit.ndim == 0:
            action_limit = np.full(
                self.action_dim, float(action_limit), dtype=np.float32
            )
        if action_limit.shape != (self.action_dim,):
            raise ValueError(
                f"action_limit 期望形状 {(self.action_dim,)}，"
                f"实际为 {action_limit.shape}"
            )
        if not np.isfinite(action_limit).all() or np.any(action_limit <= 0):
            raise ValueError("action_limit 必须为有限正数")

        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.action_limit = torch.as_tensor(
            action_limit, dtype=torch.float32, device=self.device
        )

        # 网络
        self.actor = ActorNetwork(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)
        self.critic = CriticNetwork(self.state_dim, self.hidden_dim).to(self.device)

        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr)

        # 经验缓冲区
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        self.last_update_metrics = {}

    def select_action(self, state, deterministic=False):
        """选择动作。

        训练模式从高斯分布重参数化采样；评估模式使用均值动作。
        动作通过 tanh 映射到环境范围，log_prob 包含 Jacobian 修正。
        """
        state_array = np.asarray(state, dtype=np.float32)
        expected_shape = (self.state_dim,)
        if state_array.shape != expected_shape:
            raise ValueError(
                f"state 期望形状 {expected_shape}，实际为 "
                f"{state_array.shape}"
            )
        if not np.isfinite(state_array).all():
            raise ValueError("state 必须只包含有限数值")
        state_tensor = torch.as_tensor(
            state_array, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            mean, log_std = self.actor(state_tensor)
            std = log_std.exp()
            dist = Normal(mean, std)
            raw_action = mean if deterministic else dist.rsample()
            squashed_action = torch.tanh(raw_action)
            action = squashed_action * self.action_limit
            correction = torch.log(
                self.action_limit
                * (1.0 - squashed_action.pow(2))
                + 1e-6
            )
            log_prob = (
                dist.log_prob(raw_action) - correction
            ).sum(dim=-1)
            value = self.critic(state_tensor)

        return action.cpu().numpy()[0], log_prob.cpu().item(), value.cpu().item()

    def store_transition(self, state, action, reward, done, log_prob, value):
        """存储经验"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def estimate_value(self, state):
        """只估计状态价值，不采样动作。"""
        state_tensor = torch.as_tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            value = self.critic(state_tensor)
        return float(value.item())

    def compute_gae(self, next_value):
        """计算广义优势估计(GAE)"""
        values = self.values + [next_value]
        gae = 0
        advantages = []
        returns = []

        for t in reversed(range(len(self.rewards))):
            delta = self.rewards[t] + self.gamma * values[t + 1] * (1 - self.dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - self.dones[t]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])

        return advantages, returns

    def update(self, next_value):
        """更新网络"""
        if not self.states:
            raise ValueError("经验缓冲区为空，无法执行 PPO 更新")
        advantages, returns = self.compute_gae(next_value)

        # 转换为tensor
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.FloatTensor(np.array(self.actions)).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)

        # 归一化优势
        advantages = (
            advantages - advantages.mean()
        ) / (advantages.std(unbiased=False) + 1e-8)

        # 多轮更新
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0
        total_approx_kl = 0
        total_clip_fraction = 0
        update_count = 0

        for _ in range(self.update_epochs):
            # 随机打乱
            indices = np.random.permutation(len(self.states))

            for start in range(0, len(self.states), self.batch_size):
                end = start + self.batch_size
                batch_idx = indices[start:end]

                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]

                # 计算新的log概率
                mean, log_std = self.actor(batch_states)
                std = log_std.exp()
                dist = Normal(mean, std)
                normalized_actions = torch.clamp(
                    batch_actions / self.action_limit,
                    -0.999999,
                    0.999999,
                )
                raw_actions = 0.5 * (
                    torch.log1p(normalized_actions)
                    - torch.log1p(-normalized_actions)
                )
                correction = torch.log(
                    self.action_limit
                    * (1.0 - normalized_actions.pow(2))
                    + 1e-6
                )
                new_log_probs = (
                    dist.log_prob(raw_actions) - correction
                ).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()

                # PPO裁剪目标
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy

                # 价值损失
                values = self.critic(batch_states).squeeze(-1)
                critic_loss = self.value_coef * (batch_returns - values).pow(2).mean()

                # 更新Actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                # 更新Critic
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                with torch.no_grad():
                    approx_kl = (
                        batch_old_log_probs - new_log_probs
                    ).mean()
                    clip_fraction = (
                        torch.abs(ratio - 1.0)
                        > self.clip_epsilon
                    ).float().mean()
                total_entropy += entropy.item()
                total_approx_kl += approx_kl.item()
                total_clip_fraction += clip_fraction.item()
                update_count += 1

        # 清空缓冲区
        self.clear_buffer()

        divisor = max(update_count, 1)
        self.last_update_metrics = {
            'policy_loss': total_actor_loss / divisor,
            'value_loss': total_critic_loss / divisor,
            'entropy': total_entropy / divisor,
            'approx_kl': total_approx_kl / divisor,
            'clip_fraction': total_clip_fraction / divisor,
            'updates': update_count,
        }
        return (
            self.last_update_metrics['policy_loss'],
            self.last_update_metrics['value_loss'],
        )

    def clear_buffer(self):
        """清空经验缓冲区"""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()

    def save(self, path):
        """保存模型"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'config': self.config,
        }, str(path))

    def load(self, path, load_optimizers=True):
        """加载模型，并可选恢复优化器状态。"""
        # PPOAgent.save stores the training config as well as tensors.  Newer
        # PyTorch releases default to weights_only=True, which cannot restore
        # the numpy action-limit array in this trusted, project-local format.
        try:
            checkpoint = torch.load(
                path, map_location=self.device, weights_only=False
            )
        except TypeError:  # pragma: no cover - older PyTorch compatibility
            checkpoint = torch.load(path, map_location=self.device)
        if not isinstance(checkpoint, dict):
            raise ValueError("checkpoint 必须是包含模型状态的字典")
        saved_config = checkpoint.get('config', {})
        if isinstance(saved_config, dict):
            saved_state_dim = saved_config.get('state_dim')
            saved_action_dim = saved_config.get('action_dim')
            if saved_state_dim is not None and int(saved_state_dim) != self.state_dim:
                raise ValueError(
                    f"checkpoint state_dim={saved_state_dim}，"
                    f"当前 agent state_dim={self.state_dim}"
                )
            if saved_action_dim is not None and int(saved_action_dim) != self.action_dim:
                raise ValueError(
                    f"checkpoint action_dim={saved_action_dim}，"
                    f"当前 agent action_dim={self.action_dim}"
                )
        if 'actor' not in checkpoint or 'critic' not in checkpoint:
            raise ValueError("checkpoint 缺少 actor 或 critic 状态")
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        if load_optimizers and 'actor_optimizer' in checkpoint:
            self.actor_optimizer.load_state_dict(
                checkpoint['actor_optimizer']
            )
        if load_optimizers and 'critic_optimizer' in checkpoint:
            self.critic_optimizer.load_state_dict(
                checkpoint['critic_optimizer']
            )
