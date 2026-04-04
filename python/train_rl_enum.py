# train_rl_enum.py
import os
import torch
import numpy as np
import sys
import matplotlib.pyplot as plt
from collections import deque
from enum_environment import EnumEnvironment
from ppo_agent import ActorCritic,PPOAgent
# Try to import C++ lattice environment
sys.path.append('../lib')
try:
    import lattice_env
    CPP_ENV_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import lattice_env: {e}")
    CPP_ENV_AVAILABLE = False
#

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"Number of GPUs: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
else:
    print("Warning: CUDA not available, will use CPU")
class RLEnumTrainer:
    def __init__(self, config):
        self.config = config
        
        # Create C++ lattice object
        self.lattice = lattice_env.create_lattice_int(config.dimension, config.dimension)
        #self.lattice.setSVPChallenge(config.dimension, config.seed)
        self.lattice.setRandom(config.dimension, config.dimension, 100, 1000)
        self.lattice.computeGSO()
        
        # Create RL environment wrapper
        self.env = EnumEnvironment(self.lattice, config)
        
        # Create PPO agent
        state_dim = self.env.state_space_dim
        #print(self.env.lattice)
        action_dim = self.env.action_space_dim
        self.agent = PPOAgent(state_dim, action_dim, config)
        
        # Training records
        self.episode_rewards = []
        self.best_norms = []
        self.training_losses = []
    
    def train(self, num_episodes=1000):
        """Main training loop with GPU optimization"""
        print(f"Starting RL-ENUM training on {self.agent.device}, "
              f"dimension={self.config.dimension}, {num_episodes} episodes")
        
        # 
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        for episode in range(num_episodes):
            # Reset environment
            state = self.env.reset(radius=self.config.radius)
            
            # 
            state = state.astype(np.float32)
            
            episode_reward = 0
            episode_steps = 0
            done = False
            
            # Collect data for this episode
            states, actions, rewards, log_probs, values = [], [], [], [], []
            
            while not done and episode_steps < self.config.max_steps:
                # Select action
                action, log_prob, value = self.agent.select_action(state)
                
                # Take one step
                next_state, reward, done, info = self.env.step(action)
                next_state = next_state.astype(np.float32)
                
                # Store experience
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                log_probs.append(log_prob)
                values.append(value)
                
                # Update state
                state = next_state
                episode_reward += reward
                episode_steps += 1
            
            # Record episode results
            self.episode_rewards.append(episode_reward)
            if 'best_norm' in info:
                self.best_norms.append(info['best_norm'])
            
            # Train agent with batch data
            if len(states) > 0:
                loss = self.agent.update(
                    states, actions, rewards, log_probs, values, done
                )
                self.training_losses.append(loss)
            else:
                self.training_losses.append(0.0)
            
            # Regular output and saving
            if episode % 10 == 0:
                #
                if torch.cuda.is_available():
                    memory_allocated = torch.cuda.memory_allocated() / 1024**2
                    memory_cached = torch.cuda.memory_reserved() / 1024**2
                    print(f"Episode {episode}: Reward={episode_reward:.2f}, Steps={episode_steps}")
                    print(f"  GPU Memory: {memory_allocated:.1f}MB allocated, "
                          f"{memory_cached:.1f}MB cached")
                else:
                    print(f"Episode {episode}: Reward={episode_reward:.2f}, Steps={episode_steps}")
                
                if len(self.best_norms) > 0:
                    print(f"  Best norm: {self.best_norms[-1]:.2f}")
                
                # Save checkpoint
                if episode % 50 == 0:
                    self.save_checkpoint(episode)
                    
                    #
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        
        print("Training completed!")
        return self.episode_rewards, self.best_norms
    
    def evaluate(self, test_dimensions=[100]):
        """Evaluate trained agent"""
        results = {}
        
        for dim in test_dimensions:
            print(f"\nEvaluating dimension {dim}:")
            
            # Create test lattice
            test_lattice = lattice_env.create_lattice_int(dim, dim)
            test_lattice.setSVPChallenge(dim, 0)
            test_lattice.computeGSO()
            test_env = EnumEnvironment(test_lattice, self.config)
            print("hello")
            # Use agent-guided ENUM
            state = test_env.reset(radius=10000 * (dim ** 2))
            done = False
            steps = 0
            
            while not done and steps < 1000:
                action = self.agent.select_greedy_action(state)
                state, reward, done, info = test_env.step(action)
                steps += 1
            
            # Record results
            if 'best_norm' in info:
                results[dim] = {
                    'norm': info['best_norm'],
                    'steps': steps,
                    'solved': info.get('solved', False)
                }
                print(f"  Norm: {info['best_norm']:.2f}, Steps: {steps}")
        
        return results
    
    def save_checkpoint(self, episode):
        """Save model checkpoint"""
        checkpoint = {
            'episode': episode,
            'agent_state_dict': self.agent.state_dict(),
            'optimizer_state_dict': self.agent.optimizer_state_dict(),
            'episode_rewards': self.episode_rewards,
            'best_norms': self.best_norms,
            'config': self.config.__dict__
        }
        
        torch.save(checkpoint, f"checkpoints/rl_enum_ep{episode}.pt")
        print(f"Checkpoint saved to checkpoints/rl_enum_ep{episode}.pt")
    
    def plot_training_progress(self):
        """Plot training progress"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Reward curve
        axes[0, 0].plot(self.episode_rewards)
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Reward')
        axes[0, 0].grid(True)
        
        # Moving average reward
        window = 10
        if len(self.episode_rewards) >= window:
            moving_avg = np.convolve(self.episode_rewards, np.ones(window)/window, mode='valid')
            axes[0, 1].plot(moving_avg)
            axes[0, 1].set_title(f'Moving Average Reward (window={window})')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].grid(True)
        
        # Best norm
        if self.best_norms:
            axes[1, 0].plot(self.best_norms)
            axes[1, 0].set_title('Best Norm Found')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Norm')
            axes[1, 0].grid(True)
        
        # Training loss
        if self.training_losses:
            axes[1, 1].plot(self.training_losses)
            axes[1, 1].set_title('Training Loss')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('training_progress.png', dpi=150)
        plt.show()

# Configuration class
class Config:
    def __init__(self):
        self.dimension = 20  # Training dimension
        self.seed = 9
        self.radius = 100000
        self.max_steps = 100000
        self.gamma = 0.99  # Discount factor
        self.epsilon = 0.2  # PPO clip parameter
        self.learning_rate = 0.003
        self.batch_size = 64
        self.ppo_epochs = 4  # Add missing parameter

if __name__ == "__main__":
    # Create configuration
    config = Config()
    
    # Create trainer
    trainer = RLEnumTrainer(config)
    trainer.agent.load('./checkpoints/rl_enum_ep450.pt')
    # Create necessary directories
    #os.makedirs("checkpoints", exist_ok=True)
    #os.makedirs("models", exist_ok=True)
    
    # Start training
    #rewards, norms = trainer.train(num_episodes=500)
    
    # Evaluate
    results = trainer.evaluate()
    
    # Plot results
    trainer.plot_training_progress()
    
    # Save final model
    #trainer.agent.save("models/rl_enum_final.pt")