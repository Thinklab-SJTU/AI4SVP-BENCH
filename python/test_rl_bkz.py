# test_rl_bkz.py
"""
Evaluate a trained RL-BKZ agent against multiple baselines.

Baselines compared:
  1. RL agent (greedy policy)
  2. Fixed β=20 for N tours
  3. Progressive β schedule: [10, 20, 30, 40, ...]
  4. Best single β (exhaustive scan, oracle-like)

Usage:
  python test_rl_bkz.py --checkpoint checkpoints_bkz/rl_bkz_ep500.pt \
                         --dims 40 50 --seeds 0 1 2 3 4 --max_tours 20
"""
import os, sys, time, argparse
import numpy as np
import torch

from fpylll import IntegerMatrix, LLL, BKZ, GSO
from fpylll.algorithms.bkz2 import BKZReduction as BKZ2

from bkz_environment import (
    BKZEnvironment, STATE_DIM_BKZ, ACTION_DIM_BKZ, BETA_CHOICES,
    _load_svp_challenge,
)
from ppo_agent import PPOAgent


# ── Config (must match training) ─────────────────────────────────────────────
class BKZConfig:
    dimensions = [40, 50]
    num_seeds  = 5
    max_tours  = 20
    stagnation_tol  = 1e-3
    time_penalty    = 0.5
    gamma           = 0.99
    epsilon         = 0.2
    learning_rate   = 3e-4
    batch_size      = 64
    ppo_epochs      = 4
    entropy_coeff   = 0.02


# ── Helper: run one BKZ schedule ─────────────────────────────────────────────
def _run_schedule(dim, seed, beta_schedule, max_tours=20, stagnation_tol=1e-3):
    """
    Run a fixed β schedule (list or callable schedule).
    Returns (final_b1, total_time, beta_sequence_used).
    """
    A = _load_svp_challenge(dim, seed)
    M = GSO.Mat(A); LLL.Reduction(M)(); M.update_gso()
    bkz = BKZ2(M)

    b1_prev = M.get_r(0, 0) ** 0.5
    total_t = 0.0
    betas_used = []
    stagnation = 0

    for tour in range(max_tours):
        if callable(beta_schedule):
            beta = beta_schedule(tour, b1_prev, M)
        else:
            beta = beta_schedule[tour % len(beta_schedule)]

        t0 = time.time()
        bkz(BKZ.Param(block_size=beta, max_loops=1))
        total_t += time.time() - t0
        betas_used.append(beta)

        b1_curr = M.get_r(0, 0) ** 0.5
        if (b1_prev - b1_curr) / (b1_prev + 1e-10) < stagnation_tol:
            stagnation += 1
            if stagnation >= 2:
                break
        else:
            stagnation = 0
        b1_prev = b1_curr

    return M.get_r(0, 0) ** 0.5, total_t, betas_used


def _run_rl(agent, dim, seed, max_tours=20, stagnation_tol=1e-3, time_penalty=0.5):
    """Run RL greedy policy. Returns (final_b1, total_time, betas_used)."""
    env = BKZEnvironment(max_tours=max_tours, stagnation_tol=stagnation_tol,
                         time_penalty_coeff=time_penalty)
    state = env.reset(dim, seed)
    done = False
    betas_used = []
    total_t = 0.0

    while not done:
        action = agent.select_greedy_action(state)
        state, _, done, info = env.step(action)
        betas_used.append(info['beta'])
        total_t += info['elapsed']

    return env.b1_norm, total_t, betas_used


# ── Main evaluation ───────────────────────────────────────────────────────────
def evaluate(checkpoint_path, dimensions, seeds, max_tours, stagnation_tol=1e-3):
    cfg = BKZConfig()
    cfg.max_tours = max_tours

    agent = PPOAgent(STATE_DIM_BKZ, ACTION_DIM_BKZ, cfg)
    agent.load(checkpoint_path)
    agent.policy.eval()

    # Progressive β schedule: increasing from small to large
    progressive = [10, 15, 20, 25, 30, 35, 40]

    header = (f"{'dim':>5} {'seed':>5} {'b1_LLL':>8}"
              f"  {'RL norm':>9} {'RL t':>6} {'RL betas':<22}"
              f"  {'fix20':>9} {'t':>6}"
              f"  {'prog':>9} {'t':>6}"
              f"  {'b40':>9} {'t':>6}")
    sep = '-' * len(header)
    print(f"\n{sep}\n{header}\n{sep}")

    all_results = []
    for dim in dimensions:
        for seed in seeds:
            # Initial b1 after LLL
            A0 = _load_svp_challenge(dim, seed)
            M0 = GSO.Mat(A0); LLL.Reduction(M0)(); M0.update_gso()
            b1_lll = M0.get_r(0,0)**0.5

            # RL agent
            rl_norm, rl_t, rl_betas = _run_rl(
                agent, dim, seed, max_tours, stagnation_tol)

            # Fixed β=20
            f20_norm, f20_t, _ = _run_schedule(
                dim, seed, [20], max_tours, stagnation_tol)

            # Progressive schedule
            prog_norm, prog_t, _ = _run_schedule(
                dim, seed, progressive, max_tours, stagnation_tol)

            # Fixed β=40 (strong single beta)
            b40_norm, b40_t, _ = _run_schedule(
                dim, seed, [40], max_tours, stagnation_tol)

            rl_beta_str = str(rl_betas[:6]) + ('...' if len(rl_betas) > 6 else '')
            print(f"{dim:>5} {seed:>5} {b1_lll:>8.1f}"
                  f"  {rl_norm:>9.2f} {rl_t:>5.3f}s {rl_beta_str:<22}"
                  f"  {f20_norm:>9.2f} {f20_t:>5.3f}s"
                  f"  {prog_norm:>9.2f} {prog_t:>5.3f}s"
                  f"  {b40_norm:>9.2f} {b40_t:>5.3f}s")

            all_results.append(dict(
                dim=dim, seed=seed, b1_lll=b1_lll,
                rl_norm=rl_norm, rl_t=rl_t, rl_betas=rl_betas,
                f20_norm=f20_norm, f20_t=f20_t,
                prog_norm=prog_norm, prog_t=prog_t,
                b40_norm=b40_norm, b40_t=b40_t,
            ))

    print(sep)
    _print_summary(all_results)
    return all_results


def _print_summary(results):
    print("\nSummary (lower norm = better, lower time = better):")

    for dim in sorted(set(r['dim'] for r in results)):
        sub = [r for r in results if r['dim'] == dim]
        def avg(key): return np.mean([r[key] for r in sub])
        print(f"\n  dim={dim}  ({len(sub)} seeds):")
        print(f"    RL         : norm={avg('rl_norm'):.2f}  t={avg('rl_t'):.4f}s")
        print(f"    Fixed β=20 : norm={avg('f20_norm'):.2f}  t={avg('f20_t'):.4f}s")
        print(f"    Progressive: norm={avg('prog_norm'):.2f}  t={avg('prog_t'):.4f}s")
        print(f"    Fixed β=40 : norm={avg('b40_norm'):.2f}  t={avg('b40_t'):.4f}s")

        # How often RL beats fixed β=20
        rl_wins = sum(1 for r in sub if r['rl_norm'] < r['f20_norm'] - 0.5)
        rl_ties = sum(1 for r in sub if abs(r['rl_norm'] - r['f20_norm']) <= 0.5)
        print(f"    RL vs β=20 : wins={rl_wins}, ties={rl_ties}, losses={len(sub)-rl_wins-rl_ties}")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RL-BKZ model")
    parser.add_argument("--checkpoint", default="checkpoints_bkz/rl_bkz_ep500.pt")
    parser.add_argument("--dims", nargs="+", type=int, default=[40, 50])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--max_tours", type=int, default=20)
    args = parser.parse_args()

    evaluate(
        checkpoint_path=args.checkpoint,
        dimensions=args.dims,
        seeds=args.seeds,
        max_tours=args.max_tours,
    )
