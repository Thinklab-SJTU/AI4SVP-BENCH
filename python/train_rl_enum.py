# train_rl_enum.py
import os
import torch
import numpy as np
import sys
import matplotlib.pyplot as plt
import time

from enum_environment import EnumEnvironment, STATE_DIM, ACTION_DIM
from ppo_agent import PPOAgent

sys.path.append('../lib')
try:
    import lattice_env
    CPP_ENV_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import lattice_env: {e}")
    CPP_ENV_AVAILABLE = False

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}"
              f"  ({torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB)")


class Config:
    dimension     = 40       # use dim=40 SVP challenges (smallest available)
    num_seeds     = 5        # seeds 0-4 for training variety
    # Curriculum: start with large radius (easy), decay to target over training
    radius_start  = 3.6e8    # ~ 100 * b1^2 — easy for random offsets
    radius_target = 4.0e6    # target ENUM benchmark radius
    radius_decay_episodes = 400  # linearly decay from start to target over N episodes
    max_steps     = 3000     # max steps per episode
    gamma         = 0.99
    epsilon       = 0.2
    learning_rate = 3e-4
    batch_size    = 64
    ppo_epochs    = 4
    entropy_coeff = 0.01


class RLEnumTrainer:
    def __init__(self, config):
        self.config = config
        self.agent = PPOAgent(STATE_DIM, ACTION_DIM, config)

        # Pre-create lattice objects for each training seed (SVP challenge + LLL)
        self.lattices = []
        for s in range(config.num_seeds):
            lat = lattice_env.create_lattice_int(config.dimension, config.dimension)
            lat.setSVPChallenge(config.dimension, s)
            lat.LLL(0.99)        # LLL-reduce: gives ENUM a reasonable GS profile
            lat.computeGSO()
            b1 = lat.b1Norm()
            self.lattices.append(lat)
            print(f"  Train lattice seed={s}: b1(post-LLL)={b1:.2f}, "
                  f"radius: {config.radius_start:.2e} -> {config.radius_target:.2e}")

        self.env = EnumEnvironment(self.lattices[0], config)

        self.episode_rewards = []
        self.best_norms = []
        self.training_losses = []
        os.makedirs("checkpoints", exist_ok=True)

    def _switch_lattice(self, seed_idx):
        self.env.lattice = self.lattices[seed_idx]
        self.env.wrapper = lattice_env.RL_ENUM_Wrapper(self.lattices[seed_idx])

    def _curriculum_radius(self, episode, total_episodes):
        """Linearly decay radius from start to target over decay_episodes."""
        t = min(episode / self.config.radius_decay_episodes, 1.0)
        return self.config.radius_start + t * (self.config.radius_target - self.config.radius_start)

    def train(self, num_episodes=500):
        print(f"\nStarting RL-ENUM training on {self.agent.device}, "
              f"dim={self.config.dimension}, episodes={num_episodes}")

        for episode in range(num_episodes):
            seed_idx = episode % self.config.num_seeds
            self._switch_lattice(seed_idx)
            radius = self._curriculum_radius(episode, num_episodes)

            state = self.env.reset(radius=radius)
            state = state.astype(np.float32)

            episode_reward = 0
            episode_steps = 0
            done = False
            states, actions, rewards, log_probs, values = [], [], [], [], []

            while not done and episode_steps < self.config.max_steps:
                action, log_prob, value = self.agent.select_action(state)
                next_state, reward, done, info = self.env.step(action)
                next_state = next_state.astype(np.float32)

                states.append(state)
                actions.append(action)
                rewards.append(reward)
                log_probs.append(log_prob)
                values.append(value)

                state = next_state
                episode_reward += reward
                episode_steps += 1

            self.episode_rewards.append(episode_reward)
            norm = info.get('best_norm', float('inf'))
            self.best_norms.append(norm)

            if len(states) > 0:
                loss = self.agent.update(
                    states, actions, rewards, log_probs, values, done)
                self.training_losses.append(loss)
            else:
                self.training_losses.append(0.0)

            if episode % 10 == 0:
                recent_norms = [n for n in self.best_norms[-20:] if n < 1e14]
                avg_norm = np.mean(recent_norms) if recent_norms else float('inf')
                print(f"Episode {episode:4d} | R={radius:.2e} | steps={episode_steps:5d} "
                      f"| reward={episode_reward:8.2f} | norm={norm:.2f} "
                      f"| avg_norm(20)={avg_norm:.2f}")

            if episode % 50 == 0 and episode > 0:
                self._save_checkpoint(episode)

        print("Training complete.")
        self._save_checkpoint(num_episodes)
        return self.episode_rewards, self.best_norms

    def evaluate(self, test_dimensions=None, seeds=None, num_trials=1):
        """Evaluate trained agent on SVP challenge lattices (min dim=40 available)."""
        if test_dimensions is None:
            # Evaluate on dim=40 SVP challenges (smallest available)
            test_dimensions = [40]
        if seeds is None:
            seeds = [0, 1, 2]

        results = {}
        for dim in test_dimensions:
            results[dim] = {}
            print(f"\n{'='*50}")
            print(f"Evaluating dim={dim}")
            for seed in seeds:
                norms, steps_list = [], []
                for trial in range(num_trials):
                    lat = lattice_env.create_lattice_int(dim, dim)
                    lat.setSVPChallenge(dim, seed)
                    lat.LLL(0.99)
                    lat.computeGSO()
                    b1 = lat.b1Norm()
                    radius = self.config.radius_target

                    env = EnumEnvironment(lat, self.config)
                    state = env.reset(radius=radius)
                    done = False
                    steps = 0

                    while not done and steps < self.config.max_steps:
                        action = self.agent.select_greedy_action(state)
                        state, _, done, info = env.step(action)
                        steps += 1

                    norm = info.get('best_norm', float('inf'))
                    norms.append(norm)
                    steps_list.append(steps)

                avg_norm = np.mean([n for n in norms if n < 1e14]) if any(n < 1e14 for n in norms) else float('inf')
                results[dim][seed] = {'norms': norms, 'avg_norm': avg_norm,
                                      'avg_steps': np.mean(steps_list), 'b1': b1}
                print(f"  seed={seed}: avg_norm={avg_norm:.2f}, b1={b1:.2f}, "
                      f"avg_steps={np.mean(steps_list):.0f}")
        return results

    def _save_checkpoint(self, episode):
        path = f"checkpoints/rl_enum_ep{episode}.pt"
        torch.save({
            'episode': episode,
            'policy_state_dict': self.agent.state_dict(),
            'optimizer_state_dict': self.agent.optimizer_state_dict(),
            'episode_rewards': self.episode_rewards,
            'best_norms': self.best_norms,
            'config': self.config.__dict__
        }, path)
        print(f"  Checkpoint saved: {path}")

    def plot_training_progress(self):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        axes[0].plot(self.episode_rewards)
        axes[0].set_title('Episode Reward')
        axes[0].set_xlabel('Episode')
        axes[0].grid(True)

        window = 20
        if len(self.episode_rewards) >= window:
            ma = np.convolve(self.episode_rewards,
                             np.ones(window) / window, mode='valid')
            axes[1].plot(ma)
            axes[1].set_title(f'Reward Moving Avg (w={window})')
            axes[1].set_xlabel('Episode')
            axes[1].grid(True)

        valid_norms = [n if n < 1e14 else np.nan for n in self.best_norms]
        axes[2].plot(valid_norms, alpha=0.5)
        axes[2].set_title('Best Norm per Episode')
        axes[2].set_xlabel('Episode')
        axes[2].grid(True)

        plt.tight_layout()
        plt.savefig('training_progress.png', dpi=150)
        plt.show()
        print("Plot saved to training_progress.png")


if __name__ == "__main__":
    config = Config()

    trainer = RLEnumTrainer(config)

    # Train from scratch
    rewards, norms = trainer.train(num_episodes=500)

    # Evaluate on training dimension
    results = trainer.evaluate(test_dimensions=[config.dimension], seeds=[0, 1, 2])

    trainer.plot_training_progress()
