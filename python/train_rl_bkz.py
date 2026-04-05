# train_rl_bkz.py
"""
Train a PPO agent to select adaptive BKZ block sizes.

The agent observes the GS-norm profile of an LLL-reduced lattice and
sequentially chooses β ∈ {10,15,20,25,30,35,40} for each BKZ tour.

Reward = log(b1_prev / b1_curr) * 10 - time_penalty * elapsed
This is dense (non-zero every tour), solving the sparse-reward issue
that plagued the RL-ENUM approach.

Multi-dimension training: episodes alternate between dim=40 and dim=50,
with the dimension encoded in the state (feature[12] = dim/100).
"""
import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

from bkz_environment import BKZEnvironment, STATE_DIM_BKZ, ACTION_DIM_BKZ, BETA_CHOICES
from ppo_agent import PPOAgent


# ── Configuration ─────────────────────────────────────────────────────────────
class BKZConfig:
    dimensions    = [40, 50]   # jointly trained on both dimensions
    num_seeds     = 5          # SVP challenge seeds 0..4
    max_tours     = 20         # max BKZ tours per episode
    stagnation_tol = 1e-3      # stop early if improvement < 0.1%
    time_penalty  = 0.5        # per-second reward penalty

    # PPO
    gamma         = 0.99
    epsilon       = 0.2
    learning_rate = 3e-4
    batch_size    = 64
    ppo_epochs    = 4
    entropy_coeff = 0.02       # slightly higher than ENUM (denser reward signal)


# ── Trainer ───────────────────────────────────────────────────────────────────
class RLBKZTrainer:
    def __init__(self, config: BKZConfig):
        self.config = config
        self.env = BKZEnvironment(
            max_tours=config.max_tours,
            stagnation_tol=config.stagnation_tol,
            time_penalty_coeff=config.time_penalty,
        )
        self.agent = PPOAgent(STATE_DIM_BKZ, ACTION_DIM_BKZ, config)

        self.episode_rewards   = []
        self.final_norms       = []
        self.initial_norms     = []
        self.beta_counts       = {b: 0 for b in BETA_CHOICES}

        os.makedirs("checkpoints_bkz", exist_ok=True)

    def train(self, num_episodes: int = 500):
        print(f"\n{'='*65}")
        print(f"RL-BKZ Training  |  dims={self.config.dimensions}"
              f"  |  seeds 0-{self.config.num_seeds-1}"
              f"  |  episodes={num_episodes}")
        print(f"Action space: β ∈ {BETA_CHOICES}")
        print(f"Device: {self.agent.device}")
        print(f"{'='*65}\n")

        for episode in range(num_episodes):
            # Cycle: dim alternates 40/50, seed cycles 0-4
            dim  = self.config.dimensions[episode % len(self.config.dimensions)]
            seed = (episode // len(self.config.dimensions)) % self.config.num_seeds

            state = self.env.reset(dim, seed)
            state = state.astype(np.float32)

            ep_reward  = 0.0
            done       = False
            states, actions, rewards, log_probs, values = [], [], [], [], []

            while not done:
                action, log_prob, value = self.agent.select_action(state)
                next_state, reward, done, info = self.env.step(action)
                next_state = next_state.astype(np.float32)

                states.append(state)
                actions.append(action)
                rewards.append(reward)
                log_probs.append(log_prob)
                values.append(value)

                self.beta_counts[info['beta']] += 1
                state = next_state
                ep_reward += reward

            self.episode_rewards.append(ep_reward)
            self.final_norms.append(self.env.b1_norm)
            self.initial_norms.append(info['b1_initial'])

            if len(states) > 0:
                self.agent.update(states, actions, rewards, log_probs, values, done)

            if episode % 10 == 0:
                recent = self.final_norms[-20:]
                avg_norm    = np.mean(recent)
                avg_initial = np.mean(self.initial_norms[-20:])
                avg_reward  = np.mean(self.episode_rewards[-20:])
                reduction   = 1.0 - avg_norm / avg_initial
                print(f"ep {episode:4d}  dim={dim}  seed={seed}"
                      f"  tours={info['tour']:2d}"
                      f"  b1: {info['b1_initial']:.1f}->{self.env.b1_norm:.1f}"
                      f"  avg_reward(20)={avg_reward:.3f}"
                      f"  avg_reduction(20)={reduction*100:.1f}%")

            if episode % 50 == 0 and episode > 0:
                self._save_checkpoint(episode)

        self._save_checkpoint(num_episodes)
        print("\nTraining complete.")
        print("β usage:", {b: self.beta_counts[b] for b in BETA_CHOICES})
        return self.episode_rewards, self.final_norms

    def _save_checkpoint(self, episode: int):
        path = f"checkpoints_bkz/rl_bkz_ep{episode}.pt"
        torch.save({
            'episode': episode,
            'policy_state_dict': self.agent.state_dict(),
            'optimizer_state_dict': self.agent.optimizer_state_dict(),
            'episode_rewards': self.episode_rewards,
            'final_norms': self.final_norms,
            'beta_counts': self.beta_counts,
            'config': self.config.__dict__,
        }, path)
        print(f"  Checkpoint: {path}")

    def plot(self):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Reward
        axes[0].plot(self.episode_rewards, alpha=0.4, label='raw')
        w = 20
        if len(self.episode_rewards) >= w:
            ma = np.convolve(self.episode_rewards, np.ones(w)/w, mode='valid')
            axes[0].plot(range(w-1, len(self.episode_rewards)), ma, label=f'MA-{w}')
        axes[0].set_title('Episode Reward')
        axes[0].legend(); axes[0].grid(True)

        # Norm reduction %
        reductions = [(1 - f/i)*100 for f, i in zip(self.final_norms, self.initial_norms)]
        axes[1].plot(reductions, alpha=0.4)
        if len(reductions) >= w:
            ma = np.convolve(reductions, np.ones(w)/w, mode='valid')
            axes[1].plot(range(w-1, len(reductions)), ma)
        axes[1].set_title('b1 Reduction %')
        axes[1].set_ylabel('%'); axes[1].grid(True)

        # β usage histogram
        betas = list(self.beta_counts.keys())
        counts = [self.beta_counts[b] for b in betas]
        axes[2].bar([str(b) for b in betas], counts)
        axes[2].set_title('β Usage (all episodes)')
        axes[2].set_xlabel('block size β'); axes[2].grid(True, axis='y')

        plt.tight_layout()
        plt.savefig('training_progress_bkz.png', dpi=150)
        print("Plot saved: training_progress_bkz.png")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=500)
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Resume from checkpoint')
    args = parser.parse_args()

    cfg = BKZConfig()
    trainer = RLBKZTrainer(cfg)

    if args.checkpoint:
        trainer.agent.load(args.checkpoint)
        print(f"Resumed from {args.checkpoint}")

    rewards, norms = trainer.train(num_episodes=args.episodes)
    trainer.plot()
