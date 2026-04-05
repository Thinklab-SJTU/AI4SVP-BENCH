# test_rl_model.py
"""
Standalone evaluation script for trained RL-ENUM model.
Tests a checkpoint across multiple SVP challenge dimensions and compares:
  - RL agent (greedy policy)
  - Baseline: always offset=0 (Schnorr-Euchner / Babai rounding)
  - Reference: standard C++ ENUM (exhaustive, non-RL)
"""
import sys, os, time, argparse
import numpy as np
import torch

sys.path.append('../lib')
try:
    import lattice_env
except ImportError as e:
    print(f"Error: {e}"); sys.exit(1)

from enum_environment import EnumEnvironment, STATE_DIM, ACTION_DIM
from ppo_agent import PPOAgent


# ── Config ──────────────────────────────────────────────────────────────────
class Config:
    dimension     = 40
    num_seeds     = 5
    radius_start  = 3.6e8
    radius_target = 4.0e6
    radius_decay_episodes = 400
    max_steps     = 3000
    gamma         = 0.99
    epsilon       = 0.2
    learning_rate = 3e-4
    batch_size    = 64
    ppo_epochs    = 4
    entropy_coeff = 0.01


# ── Helpers ──────────────────────────────────────────────────────────────────
def make_lattice(dim, seed, lll=True):
    lat = lattice_env.create_lattice_int(dim, dim)
    lat.setSVPChallenge(dim, seed)
    if lll:
        lat.LLL(0.99)
    lat.computeGSO()
    return lat


def run_rl_agent(agent, lat, radius, max_steps, greedy=True):
    """Run RL agent on a lattice. Returns (norm, steps, elapsed)."""
    cfg = Config()
    cfg.max_steps = max_steps
    env = EnumEnvironment(lat, cfg)
    state = env.reset(radius=radius)
    done = False; steps = 0; t0 = time.time()
    while not done and steps < max_steps:
        if greedy:
            action = agent.select_greedy_action(state)
        else:
            action, _, _ = agent.select_action(state)
        state, _, done, info = env.step(action)
        steps += 1
    elapsed = time.time() - t0
    norm = info['best_norm'] if info['best_norm'] < 1e14 else float('inf')
    return norm, steps, elapsed


def run_baseline_offset0(lat, radius, max_steps):
    """Baseline: always pick offset=0 (Babai rounding / Schnorr-Euchner first step)."""
    cfg = Config()
    cfg.max_steps = max_steps
    env = EnumEnvironment(lat, cfg)
    state = env.reset(radius=radius)
    done = False; steps = 0; t0 = time.time()
    while not done and steps < max_steps:
        state, _, done, info = env.step(5)  # action=5 → offset=0
        steps += 1
    elapsed = time.time() - t0
    norm = info['best_norm'] if info['best_norm'] < 1e14 else float('inf')
    return norm, steps, elapsed


def run_cpp_enum(lat, radius, timeout_sec=10.0):
    """Reference: exhaustive C++ ENUM. Skipped if dim > 40 (too slow)."""
    import multiprocessing as mp

    def _worker(q):
        try:
            import sys; sys.path.append('../lib')
            import lattice_env as le, numpy as np, time
            t0 = time.time()
            coeff = lat.ENUM(radius)
            elapsed = time.time() - t0
            v = lat.mulVecBasis(coeff)
            q.put(('ok', float(np.linalg.norm(v)), elapsed))
        except Exception as e:
            q.put(('err', str(e), 0.0))

    q = mp.Queue()
    p = mp.Process(target=_worker, args=(q,), daemon=True)
    p.start()
    p.join(timeout=timeout_sec)
    if p.is_alive():
        p.terminate(); p.join()
        return float('inf'), timeout_sec  # timed out
    if q.empty():
        return float('inf'), 0.0
    status, val, t = q.get()
    return (val, t) if status == 'ok' else (float('inf'), t)


# ── Main evaluation ───────────────────────────────────────────────────────────
def evaluate(checkpoint_path, dimensions, seeds, radius, max_steps, lll=True, enum_timeout=10.0):
    # Load agent
    cfg = Config()
    agent = PPOAgent(STATE_DIM, ACTION_DIM, cfg)
    agent.load(checkpoint_path)
    agent.policy.eval()
    print(f"\nCheckpoint: {checkpoint_path}")
    print(f"Radius: {radius:.2e}  |  max_steps: {max_steps}  |  LLL pre-process: {lll}")

    header = f"{'dim':>5} {'seed':>5} {'b1_norm':>10} {'RL norm':>12} {'RL steps':>9} {'RL time':>9} "  \
             f"{'Base norm':>12} {'Base steps':>10} {'ENUM norm':>12} {'ENUM time':>9}"
    sep = "-" * len(header)
    print(f"\n{sep}\n{header}\n{sep}")

    all_results = []
    for dim in dimensions:
        for seed in seeds:
            lat_rl   = make_lattice(dim, seed, lll=lll)
            lat_base = make_lattice(dim, seed, lll=lll)
            lat_cpp  = make_lattice(dim, seed, lll=lll)
            b1 = lat_rl.b1Norm()

            rl_norm,   rl_steps,   rl_t   = run_rl_agent(agent, lat_rl, radius, max_steps)
            base_norm, base_steps, base_t = run_baseline_offset0(lat_base, radius, max_steps)
            try:
                cpp_norm, cpp_t = run_cpp_enum(lat_cpp, radius, timeout_sec=enum_timeout)
                cpp_note = "(timeout)" if cpp_norm == float('inf') else ""
            except Exception:
                cpp_norm, cpp_t, cpp_note = float('inf'), 0.0, "(error)"

            def fmt(v): return f"{v:.2f}" if v < 1e14 else "  inf"
            cpp_str = f"{fmt(cpp_norm):>12} {cpp_t:>7.2f}s{cpp_note}"
            print(f"{dim:>5} {seed:>5} {b1:>10.2f} {fmt(rl_norm):>12} {rl_steps:>9} {rl_t:>8.3f}s "
                  f"{fmt(base_norm):>12} {base_steps:>10} {cpp_str}")

            all_results.append(dict(
                dim=dim, seed=seed, b1=b1,
                rl_norm=rl_norm, rl_steps=rl_steps, rl_time=rl_t,
                base_norm=base_norm, base_steps=base_steps,
                cpp_norm=cpp_norm, cpp_time=cpp_t,
            ))

    print(sep)

    # Summary: win rate of RL vs baseline
    solved_rl   = sum(1 for r in all_results if r['rl_norm']   < 1e14)
    solved_base = sum(1 for r in all_results if r['base_norm'] < 1e14)
    solved_cpp  = sum(1 for r in all_results if r['cpp_norm']  < 1e14)
    total = len(all_results)
    rl_better = sum(1 for r in all_results
                    if r['rl_norm'] < r['base_norm'] - 1.0)
    base_better = sum(1 for r in all_results
                      if r['base_norm'] < r['rl_norm'] - 1.0)

    print(f"\nSummary ({total} tests):")
    print(f"  Solved  — RL: {solved_rl}/{total}  Baseline(offset=0): {solved_base}/{total}"
          f"  C++ENUM: {solved_cpp}/{total}")
    print(f"  RL better than baseline: {rl_better}  |  baseline better than RL: {base_better}")

    return all_results


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RL-ENUM model")
    parser.add_argument("--checkpoint", default="checkpoints/rl_enum_ep500.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--dims", nargs="+", type=int, default=[40, 50],
                        help="Dimensions to test (available: 40 50 60 70 80 90 100 ...)")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2],
                        help="SVP challenge seeds to test")
    parser.add_argument("--radius", type=float, default=4e6,
                        help="ENUM search radius")
    parser.add_argument("--max_steps", type=int, default=3000,
                        help="Max RL steps per episode")
    parser.add_argument("--no_lll", action="store_true",
                        help="Skip LLL pre-processing")
    parser.add_argument("--enum_timeout", type=float, default=5.0,
                        help="Timeout (sec) for C++ ENUM reference; -1 to skip")
    args = parser.parse_args()

    evaluate(
        checkpoint_path=args.checkpoint,
        dimensions=args.dims,
        seeds=args.seeds,
        radius=args.radius,
        max_steps=args.max_steps,
        lll=not args.no_lll,
        enum_timeout=args.enum_timeout,
    )
