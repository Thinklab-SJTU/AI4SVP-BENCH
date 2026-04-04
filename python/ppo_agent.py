# ppo_agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
# ppo_agent.py 中的 ActorCritic 类修改
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(ActorCritic, self).__init__()
        
        # Use more stable initialization
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)  # Use smaller gain
                nn.init.constant_(m.bias, 0.0)
        
        # Don't use LayerNorm, use simpler normalization
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Apply initialization
        self.apply(init_weights)
    
    def forward(self, state):
        # Simple input normalization
        state_mean = state.mean(dim=-1, keepdim=True)
        state_std = state.std(dim=-1, keepdim=True) + 1e-8
        state = (state - state_mean) / state_std
        
        features = self.shared(state)
        
        # Actor with stable softmax
        action_logits = self.actor(features)
        
        # Numerically stable softmax
        action_logits = action_logits - action_logits.max(dim=-1, keepdim=True)[0]
        action_probs = F.softmax(action_logits, dim=-1)
        
        # Ensure probabilities are valid
        action_probs = torch.clamp(action_probs, min=1e-10, max=1.0)
        action_probs = action_probs / (action_probs.sum(dim=-1, keepdim=True) + 1e-10)
        
        # Critic
        state_value = self.critic(features)
        
        return action_probs, state_value

# Modified PPOAgent class in ppo_agent.py
class PPOAgent:
    def __init__(self, state_dim, action_dim, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Policy network - directly placed on GPU
        self.policy = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=config.learning_rate)
        
        # Old policy
        self.old_policy = ActorCritic(state_dim, action_dim).to(self.device)
        self.old_policy.load_state_dict(self.policy.state_dict())
        
        # Experience buffer
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
        
        # If using GPU, set benchmark to accelerate
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
    
    def select_action(self, state):
        """Select action (with exploration) - GPU version"""
        # Move state to GPU
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_probs, state_value = self.old_policy(state_tensor)
        
        # Create distribution on GPU
        dist = Categorical(action_probs)
        
        # Sample on GPU
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        # Transfer to CPU and convert to Python types
        action = action.item()
        log_prob = log_prob.item()
        state_value = state_value.item()
        
        return action, log_prob, state_value
    
    def select_greedy_action(self, state):
        """Select greedy action (for evaluation) - GPU version"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_probs, _ = self.policy(state_tensor)
        
        # Find action with maximum probability on GPU
        action = torch.argmax(action_probs, dim=-1).item()
        
        return action
    
    def update(self, states, actions, rewards, log_probs, values, done):
        """PPO update - GPU optimized version"""
        # Create tensors directly on GPU to reduce memory transfer
        if isinstance(states, list):
            states_array = np.array(states, dtype=np.float32)
            states = torch.as_tensor(states_array, dtype=torch.float32, device=self.device)
        else:
            states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        
        # Create other tensors directly on GPU
        actions = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        old_log_probs = torch.as_tensor(log_probs, dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        values = torch.as_tensor(values, dtype=torch.float32, device=self.device)
        
        # Calculate returns and advantages
        returns = self.compute_returns_gpu(rewards, done)
        advantages = returns - values
        
        # Normalize advantages
        if advantages.std() > 0:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        total_loss = 0.0
        
        # Use GPU parallel computation
        for _ in range(self.config.ppo_epochs):
            # Forward propagation
            new_action_probs, new_values = self.policy(states)
            
            # Create distribution
            dist = Categorical(new_action_probs)
            new_log_probs = dist.log_prob(actions)
            
            # Calculate ratio
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # PPO objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.config.epsilon, 1 + self.config.epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Critic loss
            critic_loss = F.mse_loss(new_values.squeeze(), returns)
            
            # Total loss
            loss = actor_loss + 0.5 * critic_loss
            total_loss += loss.item()
            
            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            
            self.optimizer.step()
        
        # Update old policy
        self.old_policy.load_state_dict(self.policy.state_dict())
        
        # Clear buffer
        self.clear_buffer()
        
        return total_loss / self.config.ppo_epochs
    
    def compute_returns_gpu(self, rewards, done):
        """GPU version of return calculation"""
        returns = []
        R = 0
        
        # Calculate sequence on CPU
        rewards_cpu = rewards.cpu().numpy()
        for r in reversed(rewards_cpu):
            R = r + self.config.gamma * R
            returns.insert(0, R)
        
        # Transfer back to GPU
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        return returns_tensor
    
    def compute_returns(self, rewards, done):
        """Calculate discounted returns"""
        returns = []
        R = 0
        
        # Calculate backwards
        for r in reversed(rewards):
            R = r + self.config.gamma * R
            returns.insert(0, R)
        
        returns = torch.FloatTensor(returns).to(self.device)
        return returns
    
    def store_transition(self, state, action, reward, log_prob, value, done):
        """Store experience"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)
    def clear_buffer(self):
        """Clear experience buffer"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
    
    def save(self, path):
        """Save model"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.__dict__
        }, path)
    
    def load(self, path):
        """Load model"""
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.old_policy.load_state_dict(self.policy.state_dict())
    
    def state_dict(self):
        """Get agent state dict"""
        return self.policy.state_dict()
    
    def optimizer_state_dict(self):
        """Get optimizer state dict"""
        return self.optimizer.state_dict()