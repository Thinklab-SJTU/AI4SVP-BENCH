# Supplementary code for train_rl_enum.py
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

class RLEnumTrainer:
    def __init__(self, config):
        self.config = config
        
        # Create C++ lattice object
        self.lattice = lattice_env.create_lattice_int(config.dimension, config.dimension)
        self.lattice.setSVPChallenge(config.dimension, config.seed)
        self.lattice.computeGSO()
        
        # Create RL environment wrapper
        self.env = EnumEnvironment(self.lattice, config)
        
        # Create PPO agent
        state_dim = self.env.state_space_dim
        action_dim = self.env.action_space_dim
        self.agent = PPOAgent(state_dim, action_dim, config)
        
        # Training records
        self.episode_rewards = []
        self.best_norms = []
        self.training_losses = []
    
    def train(self, num_episodes=1000):
        """Main training loop"""
        print(f"Starting RL-ENUM training, dimension={self.config.dimension}, {num_episodes} episodes")
        
        for episode in range(num_episodes):
            # Reset environment
            state = self.env.reset(radius=self.config.radius)
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
            
            # Train agent
            loss = self.agent.update(
                states, actions, rewards, log_probs, values, done
            )
            self.training_losses.append(loss)
            
            # Regular output and saving
            if episode % 10 == 0:
                print(f"Episode {episode}: Reward={episode_reward:.2f}, Steps={episode_steps}")
                if len(self.best_norms) > 0:
                    print(f"  Best norm: {self.best_norms[-1]:.2f}")
                
                # Save checkpoint
                if episode % 50 == 0:
                    self.save_checkpoint(episode)
        
        print("Training completed!")
        return self.episode_rewards, self.best_norms
    
    def evaluate(self, test_dimensions=[20, 30, 40]):
        """Evaluate trained agent"""
        results = {}
        
        for dim in test_dimensions:
            print(f"\nEvaluating dimension {dim}:")
            
            # Create test lattice
            test_lattice = lattice_env.create_lattice_int(dim, dim)
            #test_lattice.setSVPChallenge(dim, 0)
            test_lattice.setRandom(dim, dim, 100, 1000)
            test_lattice.computeGSO()
            test_env = EnumEnvironment(test_lattice, self.config)
            
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
        self.dimension = 40  # Training dimension
        self.seed = 9
        self.radius = 1000000
        self.max_steps = 500
        self.gamma = 0.99  # Discount factor
        self.epsilon = 0.2  # PPO clip parameter
        self.learning_rate = 0.0003
        self.batch_size = 64
        self.ppo_epochs = 4  # Add missing parameter
def evaluate_trained_model(model_path, config, test_cases=None):
    """
    Load and evaluate a trained model
    
    Args:
        model_path: Path to model weights file
        config: Configuration object
        test_cases: List of test cases, each element is (dimension, radius)
    """
    
    if test_cases is None:
        # Default test cases
        test_cases = [
            (20, 50000),   # Low dimension, small radius
            (30, 200000),  # Medium dimension
            (40, 500000),  # Higher dimension
            (50, 1000000), # High dimension
        ]
    
    print(f"Loading model: {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    
    # Create new trainer (for evaluation only)
    trainer = RLEnumTrainer(config)
    
    # Load model weights to agent
    trainer.agent.load(checkpoint['agent_state_dict'])
    trainer.agent.optimizer.load(checkpoint['optimizer_state_dict'])
    
    print(f"Model loaded successfully, training episodes: {checkpoint.get('episode', 'Unknown')}")
    
    # Evaluate each test case
    results = []
    
    for dim, radius in test_cases:
        print(f"\n{'='*60}")
        print(f"Evaluating dimension {dim}, radius {radius}")
        print('='*60)
        
        # Create test lattice
        test_lattice = lattice_env.create_lattice_int(dim, dim)
        test_lattice.setSVPChallenge(dim, config.seed)
        test_lattice.computeGSO()
        
        # Create test environment
        test_env = EnumEnvironment(test_lattice, config)
        
        # Reset environment
        state = test_env.reset(radius=radius)
        done = False
        steps = 0
        episode_reward = 0
        best_norm = float('inf')
        
        # Execute RL-ENUM search
        start_time = time.time()
        
        while not done and steps < config.max_steps * 2:  # Allow more steps during evaluation
            # Use greedy policy (no exploration)
            action = trainer.agent.select_greedy_action(state)
            
            # Execute one step
            next_state, reward, done, info = test_env.step(action)
            
            # Update state
            state = next_state
            episode_reward += reward
            steps += 1
            
            # Record best norm
            if 'best_norm' in info and info['best_norm'] < best_norm:
                best_norm = info['best_norm']
            
            # Output progress every 50 steps
            if steps % 50 == 0:
                print(f"  Step {steps}: Current rho={info.get('current_rho', 0):.2f}, "
                      f"Reward={reward:.2f}, Best norm={best_norm:.2f}")
        
        elapsed_time = time.time() - start_time
        
        # Get final results
        final_state = test_env.get_state()
        final_best_norm = final_state.get('best_norm', float('inf'))
        solved = final_state.get('solved', False)
        
        # Compare with traditional ENUM
        print("\nComparing with traditional ENUM algorithm:")
        trad_start = time.time()
        trad_coeff = test_lattice.ENUM(radius)
        trad_vector = test_lattice.mulVecBasis(trad_coeff)
        trad_norm = vector_norm(trad_vector)
        trad_time = time.time() - trad_start
        
        print(f"Traditional ENUM: Norm={trad_norm:.2f}, Time={trad_time:.2f} seconds")
        print(f"RL-ENUM:          Norm={final_best_norm:.2f}, Time={elapsed_time:.2f} seconds, "
              f"Steps={steps}, Total reward={episode_reward:.2f}")
        
        if trad_norm > 0:
            improvement = (trad_norm - final_best_norm) / trad_norm * 100
            print(f"Improvement: {improvement:.1f}%")
        
        # Record results
        results.append({
            'dimension': dim,
            'radius': radius,
            'rl_norm': final_best_norm,
            'rl_time': elapsed_time,
            'rl_steps': steps,
            'rl_reward': episode_reward,
            'rl_solved': solved,
            'traditional_norm': trad_norm,
            'traditional_time': trad_time,
            'speedup': trad_time / elapsed_time if elapsed_time > 0 else 0,
            'quality_improvement': ((trad_norm - final_best_norm) / trad_norm if trad_norm > 0 else 0)
        })
    
    return results

def vector_norm(v):
    """Compute Euclidean norm of a vector"""
    return np.sqrt(np.sum(np.array(v, dtype=np.float64)**2))

def plot_evaluation_results(results):
    """Plot evaluation results"""
    if not results:
        print("No results to plot")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Extract data
    dimensions = [r['dimension'] for r in results]
    rl_times = [r['rl_time'] for r in results]
    trad_times = [r['traditional_time'] for r in results]
    rl_norms = [r['rl_norm'] for r in results]
    trad_norms = [r['traditional_norm'] for r in results]
    speedups = [r['speedup'] for r in results]
    improvements = [r['quality_improvement'] * 100 for r in results]  # Convert to percentage
    
    # Plot 1: Time comparison
    x = np.arange(len(dimensions))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, trad_times, width, label='Traditional ENUM', color='lightcoral', alpha=0.8)
    axes[0, 0].bar(x + width/2, rl_times, width, label='RL-ENUM', color='lightgreen', alpha=0.8)
    axes[0, 0].set_xlabel('Dimension')
    axes[0, 0].set_ylabel('Time (seconds)')
    axes[0, 0].set_title('Time Comparison')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(dimensions)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add time value labels
    for i, (trad, rl) in enumerate(zip(trad_times, rl_times)):
        axes[0, 0].text(i - width/2, trad + 0.1, f'{trad:.1f}', ha='center', va='bottom', fontsize=9)
        axes[0, 0].text(i + width/2, rl + 0.1, f'{rl:.1f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Norm comparison
    axes[0, 1].bar(x - width/2, trad_norms, width, label='Traditional ENUM', color='lightcoral', alpha=0.8)
    axes[0, 1].bar(x + width/2, rl_norms, width, label='RL-ENUM', color='lightgreen', alpha=0.8)
    axes[0, 1].set_xlabel('Dimension')
    axes[0, 1].set_ylabel('Vector Norm')
    axes[0, 1].set_title('Solution Quality Comparison')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(dimensions)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Add norm value labels
    for i, (trad, rl) in enumerate(zip(trad_norms, rl_norms)):
        axes[0, 1].text(i - width/2, trad + max(trad_norms)*0.02, f'{trad:.1f}', ha='center', va='bottom', fontsize=9)
        axes[0, 1].text(i + width/2, rl + max(trad_norms)*0.02, f'{rl:.1f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 3: Speedup ratio
    axes[0, 2].plot(dimensions, speedups, 'bo-', linewidth=2, markersize=8)
    axes[0, 2].set_xlabel('Dimension')
    axes[0, 2].set_ylabel('Speedup Ratio (Traditional/RL)')
    axes[0, 2].set_title('Speedup Effect')
    axes[0, 2].axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Baseline')
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].legend()
    
    # Add speedup value labels
    for i, s in enumerate(speedups):
        axes[0, 2].text(dimensions[i], s + max(speedups)*0.02, f'{s:.2f}x', ha='center', va='bottom')
    
    # Plot 4: Quality improvement
    axes[1, 0].plot(dimensions, improvements, 'ro-', linewidth=2, markersize=8)
    axes[1, 0].set_xlabel('Dimension')
    axes[1, 0].set_ylabel('Quality Improvement (%)')
    axes[1, 0].set_title('Solution Quality Improvement')
    axes[1, 0].axhline(y=0.0, color='r', linestyle='--', alpha=0.5, label='Baseline')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # Add improvement value labels
    for i, imp in enumerate(improvements):
        axes[1, 0].text(dimensions[i], imp + max(improvements)*0.02, f'{imp:.1f}%', ha='center', va='bottom')
    
    # Plot 5: RL steps
    rl_steps = [r['rl_steps'] for r in results]
    axes[1, 1].bar(dimensions, rl_steps, color='skyblue', alpha=0.8)
    axes[1, 1].set_xlabel('Dimension')
    axes[1, 1].set_ylabel('RL-ENUM Steps')
    axes[1, 1].set_title('RL-ENUM Search Steps')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add step value labels
    for i, steps in enumerate(rl_steps):
        axes[1, 1].text(dimensions[i], steps + max(rl_steps)*0.02, f'{steps}', ha='center', va='bottom')
    
    # Plot 6: Cumulative reward
    rl_rewards = [r['rl_reward'] for r in results]
    axes[1, 2].bar(dimensions, rl_rewards, color='gold', alpha=0.8)
    axes[1, 2].set_xlabel('Dimension')
    axes[1, 2].set_ylabel('Cumulative Reward')
    axes[1, 2].set_title('RL-ENUM Cumulative Reward')
    axes[1, 2].grid(True, alpha=0.3)
    
    # Add reward value labels
    for i, reward in enumerate(rl_rewards):
        axes[1, 2].text(dimensions[i], reward + max(rl_rewards)*0.02, f'{reward:.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('evaluation_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print summary table
    print("\n" + "="*80)
    print("Evaluation Results Summary:")
    print("="*80)
    print(f"{'Dimension':<8} {'Traditional ENUM':<12} {'RL-ENUM':<12} {'Speedup':<8} {'Quality Impr.':<10}")
    print("-"*80)
    
    for result in results:
        print(f"{result['dimension']:<8} "
              f"{result['traditional_norm']:<8.2f}({result['traditional_time']:.1f}s) "
              f"{result['rl_norm']:<8.2f}({result['rl_time']:.1f}s) "
              f"{result['speedup']:<8.2f}x "
              f"{result['quality_improvement']*100:<8.1f}%")

# Add evaluation option to main function
def main():
    import argparse
    import time
    
    parser = argparse.ArgumentParser(description='RL-ENUM Training and Evaluation')
    parser.add_argument('--mode', type=str, default='train', 
                       choices=['train', 'evaluate', 'both'],
                       help='Running mode: train, evaluate, both (train then evaluate)')
    parser.add_argument('--model', type=str, default='checkpoints/rl_enum_ep450.pt',
                       help='Path to model to load (for evaluation mode)')
    parser.add_argument('--episodes', type=int, default=500,
                       help='Number of training episodes')
    parser.add_argument('--test_dims', type=str, default='20,30,40,50',
                       help='Test dimensions, comma-separated')
    
    args = parser.parse_args()
    
    # Create configuration
    config = Config()
    
    # Create necessary directories
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    if args.mode in ['train', 'both']:
        print("="*60)
        print("Starting RL-ENUM Training")
        print("="*60)
        
        # Create trainer
        trainer = RLEnumTrainer(config)
        
        # Start training
        start_time = time.time()
        rewards, norms = trainer.train(num_episodes=args.episodes)
        training_time = time.time() - start_time
        
        print(f"Training completed, time elapsed: {training_time:.1f} seconds")
        
        # Save final model
        trainer.agent.save("models/rl_enum_final.pt")
        
        # Plot training progress
        trainer.plot_training_progress()
    
    if args.mode in ['evaluate', 'both']:
        print("\n" + "="*60)
        print("Starting Model Evaluation")
        print("="*60)
        
        # Parse test dimensions
        test_dimensions = [int(d) for d in args.test_dims.split(',')]
        test_cases = [(dim, 10000 * (dim ** 2)) for dim in test_dimensions]
        
        # Evaluate trained model
        results = evaluate_trained_model(args.model, config, test_cases)
        
        # Plot evaluation results
        plot_evaluation_results(results)
        
        # Save evaluation results to file
        import json
        with open('evaluation_results.json', 'w') as f:
            # Convert numpy types to Python basic types
            serializable_results = []
            for r in results:
                serializable_r = {}
                for key, value in r.items():
                    if isinstance(value, (np.integer, np.floating)):
                        serializable_r[key] = value.item()
                    else:
                        serializable_r[key] = value
                serializable_results.append(serializable_r)
            
            json.dump(serializable_results, f, indent=2)
        
        print(f"\nEvaluation results saved to: evaluation_results.json")
        print(f"Evaluation charts saved to: evaluation_results.png")

if __name__ == "__main__":
    main()