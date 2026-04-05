# ppo_agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(ActorCritic, self).__init__()

        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, state):
        features = self.shared(state)
        action_logits = self.actor(features)
        action_probs = F.softmax(action_logits, dim=-1)
        action_probs = torch.clamp(action_probs, min=1e-8)
        action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True)
        state_value = self.critic(features)
        return action_probs, state_value


class PPOAgent:
    def __init__(self, state_dim, action_dim, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.policy = ActorCritic(state_dim, action_dim).to(self.device)
        self.old_policy = ActorCritic(state_dim, action_dim).to(self.device)
        self.old_policy.load_state_dict(self.policy.state_dict())

        self.optimizer = optim.Adam(self.policy.parameters(), lr=config.learning_rate,
                                    eps=1e-5)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=200, gamma=0.5)

        self.entropy_coeff = getattr(config, 'entropy_coeff', 0.01)

        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True

    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_probs, state_value = self.old_policy(state_tensor)
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob.item(), state_value.item()

    def select_greedy_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_probs, _ = self.policy(state_tensor)
        return torch.argmax(action_probs, dim=-1).item()

    def update(self, states, actions, rewards, log_probs, values, done):
        states_arr = np.array(states, dtype=np.float32)
        states_t = torch.as_tensor(states_arr, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        old_log_probs_t = torch.as_tensor(log_probs, dtype=torch.float32, device=self.device)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        values_t = torch.as_tensor(values, dtype=torch.float32, device=self.device)

        returns = self._compute_returns(rewards_t, done)
        advantages = returns - values_t
        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_loss = 0.0
        for _ in range(self.config.ppo_epochs):
            new_probs, new_values = self.policy(states_t)
            dist = Categorical(new_probs)
            new_log_probs = dist.log_prob(actions_t)
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_log_probs - old_log_probs_t)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio,
                                1 - self.config.epsilon,
                                1 + self.config.epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = F.mse_loss(new_values.squeeze(), returns)

            # PPO loss with entropy bonus for exploration
            loss = actor_loss + 0.5 * critic_loss - self.entropy_coeff * entropy
            total_loss += loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            self.optimizer.step()

        self.old_policy.load_state_dict(self.policy.state_dict())
        self.scheduler.step()
        return total_loss / self.config.ppo_epochs

    def _compute_returns(self, rewards, done):
        returns = []
        R = 0.0
        for r in reversed(rewards.cpu().numpy()):
            R = r + self.config.gamma * R
            returns.insert(0, R)
        return torch.FloatTensor(returns).to(self.device)

    def save(self, path):
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.__dict__
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        key = 'policy_state_dict' if 'policy_state_dict' in checkpoint else 'agent_state_dict'
        self.policy.load_state_dict(checkpoint[key])
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.old_policy.load_state_dict(self.policy.state_dict())

    def state_dict(self):
        return self.policy.state_dict()

    def optimizer_state_dict(self):
        return self.optimizer.state_dict()

