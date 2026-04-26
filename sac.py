
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from collections import deque
import random

# ─────────────────────────── Replay Buffer ─────────────────────────── #

class ReplayBuffer:
    """Simple FIFO experience replay buffer."""

    def __init__(self, capacity: int = 1_000_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)),
            torch.FloatTensor(np.array(actions)),
            torch.FloatTensor(np.array(rewards)).unsqueeze(1),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(np.array(dones)).unsqueeze(1),
        )

    def __len__(self):
        return len(self.buffer)


# ─────────────────────────── Networks ──────────────────────────────── #

class MLP(nn.Module):
    """Shared MLP backbone."""

    def __init__(self, input_dim, hidden_dims, output_dim, activation=nn.ReLU):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), activation()]
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class GaussianPolicy(nn.Module):
    """
    Stochastic actor that outputs a squashed Gaussian (tanh) action.
    Uses the reparameterisation trick for differentiable sampling.
    """

    LOG_STD_MIN = -20
    LOG_STD_MAX = 2

    def __init__(self, state_dim, action_dim, hidden_dims=(256, 256)):
        super().__init__()
        layers = []
        prev = state_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        self.trunk = nn.Sequential(*layers)
        self.mean_head = nn.Linear(prev, action_dim)
        self.log_std_head = nn.Linear(prev, action_dim)

    def forward(self, state):
        x = self.trunk(state)
        mean = self.mean_head(x)
        log_std = self.log_std_head(x).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mean, std)
        # reparameterisation trick
        x_t = dist.rsample()
        action = torch.tanh(x_t)
        # log-prob with tanh squashing correction
        log_prob = dist.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob, mean

    def deterministic(self, state):
        mean, _ = self.forward(state)
        return torch.tanh(mean)


class TwinQNetwork(nn.Module):
    """Twin Q-networks (clipped double-Q trick)."""

    def __init__(self, state_dim, action_dim, hidden_dims=(256, 256)):
        super().__init__()
        self.q1 = MLP(state_dim + action_dim, hidden_dims, 1)
        self.q2 = MLP(state_dim + action_dim, hidden_dims, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa), self.q2(sa)


# ─────────────────────────── SAC Agent ─────────────────────────────── #

class SACAgent:
    """
    Soft Actor-Critic with automatic entropy (alpha) tuning.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: tuple = (256, 256),
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha_lr: float = 3e-4,
        init_alpha: float = 0.2,
        buffer_capacity: int = 1_000_000,
        batch_size: int = 256,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.action_dim = action_dim

        # ── Networks ──
        self.policy = GaussianPolicy(state_dim, action_dim, hidden_dims).to(self.device)
        self.critic = TwinQNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.critic_target = TwinQNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # ── Optimisers ──
        self.policy_optim = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # ── Automatic entropy tuning ──
        self.target_entropy = -action_dim  # heuristic: -dim(A)
        self.log_alpha = torch.tensor(np.log(init_alpha), requires_grad=True, device=self.device)
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

        # ── Replay buffer ──
        self.buffer = ReplayBuffer(buffer_capacity)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    # ── Action selection ── #
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if deterministic:
                action = self.policy.deterministic(state_t)
            else:
                action, _, _ = self.policy.sample(state_t)
        return action.cpu().numpy().flatten()

    # ── Soft update of target network ── #
    def _soft_update(self):
        for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

    # ── Single gradient step ── #
    def update(self):
        if len(self.buffer) < self.batch_size:
            return {}

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # ── Critic loss ──
        with torch.no_grad():
            next_actions, next_log_probs, _ = self.policy.sample(next_states)
            q1_next, q2_next = self.critic_target(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next) - self.alpha.detach() * next_log_probs
            q_target = rewards + (1 - dones) * self.gamma * q_next

        q1, q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # ── Policy loss ──
        new_actions, log_probs, _ = self.policy.sample(states)
        q1_new, q2_new = self.critic(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        policy_loss = (self.alpha.detach() * log_probs - q_new).mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        # ── Alpha (entropy coefficient) loss ──
        alpha_loss = -(self.log_alpha * (log_probs.detach() + self.target_entropy)).mean()

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        # ── Target network soft update ──
        self._soft_update()

        return {
            "critic_loss": critic_loss.item(),
            "policy_loss": policy_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "alpha": self.alpha.item(),
        }

    # ── Save / Load ── #
    def save(self, path: str):
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "policy": self.policy.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "log_alpha": self.log_alpha,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt["policy"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        self.log_alpha = ckpt["log_alpha"]
