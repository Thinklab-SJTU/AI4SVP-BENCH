# bkz_environment.py
"""
RL environment for learning adaptive BKZ block-size selection.

The agent observes the current GS-norm profile of a lattice and chooses
a BKZ block size β for the next tour. The goal is to minimise b1_norm
as efficiently as possible (norm quality vs. wall-clock time trade-off).

Key implementation note:
  We keep the SAME M (fpylll GSO.Mat) and bkz (BKZ2Reduction) objects
  across tours within a single episode. Re-creating BKZ2 would reset
  internal BKZ state and degrade performance.
"""
import numpy as np
import os
import sys

from fpylll import IntegerMatrix, LLL, BKZ, GSO
from fpylll.algorithms.bkz2 import BKZReduction as BKZ2

# ── Action space ─────────────────────────────────────────────────────────────
BETA_CHOICES = [10, 15, 20, 25, 30, 35, 40]
ACTION_DIM_BKZ = len(BETA_CHOICES)   # 7

# ── State space ───────────────────────────────────────────────────────────────
# [0-9]  : 10-point GS-norm profile  B[i] / B[0]  (evenly spaced, log-scaled)
# [10]   : b1_norm / b1_initial      (reduction progress, 1.0 → <1.0)
# [11]   : tour_count / max_tours    (time pressure)
# [12]   : dim / 100                 (dimension signal for generalisation)
STATE_DIM_BKZ = 13


# ── SVP challenge loader (no dependency on svp_hyperopt.py) ──────────────────
_CHALLENGE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', 'svp_challenge_list'
)

def _load_svp_challenge(dim: int, seed: int) -> IntegerMatrix:
    """Load an SVP-challenge lattice from the text file into fpylll."""
    path = os.path.join(_CHALLENGE_DIR, f'svp_challenge_{dim}_{seed}.txt')
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip()]
    rows = []
    for line in lines:
        row = [int(x) for x in line.split()]
        if row:
            rows.append(row)
        if len(rows) == dim:
            break
    while len(rows) < dim:
        rows.append([0] * dim)
    A = IntegerMatrix(dim, dim)
    for i in range(dim):
        for j in range(dim):
            A[i, j] = rows[i][j]
    return A


class BKZEnvironment:
    """
    Gym-style environment wrapping fpylll BKZ2.0 for RL training.

    Episode lifecycle:
      env.reset(dim, seed)   →  state (np.ndarray[13])
      env.step(action)       →  (state, reward, done, info)  × max_tours

    The same M / BKZ2 objects are preserved within an episode so that
    BKZ2.0's internal continuation state is maintained across tours.
    """

    def __init__(self, max_tours: int = 20, stagnation_tol: float = 1e-3,
                 time_penalty_coeff: float = 0.5):
        """
        Args:
            max_tours:            maximum BKZ tours per episode
            stagnation_tol:       episode ends if relative b1 improvement
                                  is below this for 2 consecutive tours
            time_penalty_coeff:   per-second penalty in reward
        """
        self.max_tours = max_tours
        self.stagnation_tol = stagnation_tol
        self.time_penalty_coeff = time_penalty_coeff

        # Per-episode state (set in reset)
        self._M = None
        self._bkz = None
        self._b1_initial = None
        self._b1_prev = None
        self._tour_count = 0
        self._stagnation_count = 0
        self._dim = None

    # ── Public API ────────────────────────────────────────────────────────────

    def reset(self, dim: int, seed: int) -> np.ndarray:
        """Load lattice, apply LLL, initialise BKZ2 object. Returns state."""
        A = _load_svp_challenge(dim, seed)
        self._M = GSO.Mat(A)
        lll = LLL.Reduction(self._M)
        lll()
        self._M.update_gso()

        self._bkz = BKZ2(self._M)
        self._b1_initial = self._M.get_r(0, 0) ** 0.5
        self._b1_prev = self._b1_initial
        self._tour_count = 0
        self._stagnation_count = 0
        self._dim = dim

        return self._get_state()

    def step(self, action: int):
        """
        Execute one BKZ tour with block size BETA_CHOICES[action].

        Returns:
            state   : np.ndarray[13]
            reward  : float
            done    : bool
            info    : dict
        """
        import time
        assert 0 <= action < ACTION_DIM_BKZ, f"Invalid action {action}"

        beta = BETA_CHOICES[action]
        params = BKZ.Param(block_size=beta, max_loops=1)

        t0 = time.time()
        self._bkz(params)
        elapsed = time.time() - t0

        b1_curr = self._M.get_r(0, 0) ** 0.5
        self._tour_count += 1

        # ── Reward ────────────────────────────────────────────────────────────
        # Log-scale improvement minus time cost.
        # log(b1_prev / b1_curr) ≥ 0 when norm decreases; 0 if no change.
        improvement = max(np.log(self._b1_prev / (b1_curr + 1e-10)), 0.0)
        reward = improvement * 10.0 - self.time_penalty_coeff * elapsed

        # ── Stagnation check ─────────────────────────────────────────────────
        rel_change = (self._b1_prev - b1_curr) / (self._b1_prev + 1e-10)
        if rel_change < self.stagnation_tol:
            self._stagnation_count += 1
        else:
            self._stagnation_count = 0

        # ── Update ───────────────────────────────────────────────────────────
        self._b1_prev = b1_curr

        done = (self._tour_count >= self.max_tours
                or self._stagnation_count >= 2)

        info = {
            'b1_norm': b1_curr,
            'b1_initial': self._b1_initial,
            'beta': beta,
            'tour': self._tour_count,
            'elapsed': elapsed,
            'improvement': improvement,
        }
        return self._get_state(), reward, done, info

    @property
    def b1_norm(self) -> float:
        return self._M.get_r(0, 0) ** 0.5 if self._M is not None else float('inf')

    # ── Feature extraction ───────────────────────────────────────────────────

    def _get_state(self) -> np.ndarray:
        dim = self._dim
        b1_sq = max(self._M.get_r(0, 0), 1e-10)

        # 10 evenly-spaced GS-norm samples, normalised by B[0]
        indices = [int(round(i * (dim - 1) / 9)) for i in range(10)]
        profile = []
        for idx in indices:
            b_sq = max(self._M.get_r(idx, idx), 1e-10)
            # log-ratio, clipped to [-2, 0] (GS norms decrease along the profile)
            ratio = np.clip(np.log(b_sq / b1_sq) / 5.0, -2.0, 0.5)
            profile.append(float(ratio))

        features = profile + [
            self._b1_prev / self._b1_initial,      # [10] reduction progress
            self._tour_count / self.max_tours,      # [11] time budget used
            self._dim / 100.0,                      # [12] dimension signal
        ]
        arr = np.array(features, dtype=np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.5, neginf=-2.0)
        return arr
