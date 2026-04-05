# enum_environment.py
import numpy as np
import sys
sys.path.append('../lib')
try:
    import lattice_env
    CPP_ENV_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import lattice_env: {e}")
    CPP_ENV_AVAILABLE = False

# State dimension: 20 meaningful features capturing ENUM search state
STATE_DIM = 20
# Action space: offsets -5..+5 (11 actions)
ACTION_DIM = 11

class EnumEnvironment:
    def __init__(self, lattice, config):
        self.lattice = lattice
        self.config = config
        self.wrapper = lattice_env.RL_ENUM_Wrapper(lattice)
        self.state_space_dim = STATE_DIM
        self.action_space_dim = ACTION_DIM
        self.step_count = 0

    def reset(self, radius=None):
        if radius is None:
            radius = self.config.radius
        self.wrapper.reset(radius)
        self.step_count = 0
        state_obj = self.wrapper.get_state()
        return self._extract_features(state_obj, radius)

    def step(self, action):
        self.step_count += 1
        offset_action = action - 5  # map [0,10] -> [-5,+5]
        reward, done, info_str = self.wrapper.step(offset_action)
        state_obj = self.wrapper.get_state()
        next_state = self._extract_features(state_obj, state_obj.radius)
        info = {
            'best_norm': state_obj.best_norm,
            'current_k': state_obj.current_k,
            'current_rho': state_obj.current_rho,
            'solved': state_obj.found_solution,
            'total_steps': self.step_count,
            'info_str': info_str
        }
        return next_state, reward, done, info

    def _extract_features(self, s, radius):
        """
        20 informative features for the ENUM search state.

        The ENUM algorithm descends a tree from level k=n-1 down to k=0.
        At each level k the agent picks coefficient c_k = round(center_k) + offset.
        Features capture: depth, budget usage, local geometry, global GS profile.

        Feature layout (indices 0-19):
          [0]     k / n                       — current normalised depth
          [1]     rho_parent / R              — budget used above this level
          [2]     remaining = 1 - [1]         — fraction of budget still available
          [3]     B_k / (B_0 + eps)           — relative GS-norm at k (clipped, log)
          [4]     |center_k - round(center_k)| — distance to nearest integer (0..0.5)
          [5]     sign of (center_k - round(center_k)) — direction bias
          [6]     sqrt(max(R - rho_parent, 0)) / (sqrt(B_k) + eps)
                  — effective radius at k (how many coefficients could fit)
          [7]     has_solution flag
          [8]     best_norm / (sqrt(R) + eps) — quality of best solution found
          [9]     (temp_vec[k] - center_k) / 5 — current tried offset (clipped)
          [10-19] GS norm profile: B[i*n//10] / (B[0] + eps) for i=0..9
                  — global lattice shape seen by the agent
        """
        n = int(s.num_rows) if s.num_rows > 0 else 1
        k = int(s.current_k)
        k = max(0, min(k, n - 1))

        R = float(s.radius) if s.radius > 1e-8 else 1.0

        # GS norms (m_B values)
        gs = []
        if hasattr(s, 'gs_norms') and len(s.gs_norms) >= n:
            gs = [max(float(x), 1e-8) for x in s.gs_norms[:n]]
        else:
            gs = [1.0] * n
        B0 = max(gs[0], 1e-8)
        Bk = max(gs[k], 1e-8)

        # rho at parent level (rho[k+1]) — budget already committed
        rho_array = []
        if hasattr(s, 'rho_array') and len(s.rho_array) >= n + 1:
            rho_array = [float(x) for x in s.rho_array[:n + 1]]
        else:
            rho_array = [0.0] * (n + 1)

        rho_parent = max(rho_array[k + 1], 0.0) if k + 1 <= n else 0.0
        rho_parent_ratio = min(rho_parent / R, 2.0)

        # Center at k
        center_k = float(s.current_center) if hasattr(s, 'current_center') else 0.0
        if hasattr(s, 'center_array') and len(s.center_array) > k:
            center_k = float(s.center_array[k])
        frac = center_k - round(center_k)   # signed fractional part in (-0.5, 0.5]
        abs_frac = abs(frac)
        sign_frac = 1.0 if frac >= 0 else -1.0

        # Current coefficient
        temp_vec_k = 0.0
        if hasattr(s, 'current_coeffs') and len(s.current_coeffs) > k:
            temp_vec_k = float(s.current_coeffs[k])
        coeff_offset = np.clip((temp_vec_k - center_k) / 5.0, -2.0, 2.0)

        # Effective radius at k
        budget_left = max(R - rho_parent, 0.0)
        eff_radius = np.sqrt(budget_left) / (np.sqrt(Bk) + 1e-8)
        eff_radius = np.clip(eff_radius, 0.0, 20.0)

        # Solution quality
        best_norm = float(s.best_norm) if s.best_norm < 1e15 else R
        best_ratio = np.clip(best_norm / (np.sqrt(R) + 1e-8), 0.0, 10.0)

        # GS norm profile (10 evenly-spaced samples across [0, n-1])
        indices = [int(round(i * (n - 1) / 9)) for i in range(10)]
        gs_profile = [np.clip(gs[idx] / B0, 0.0, 10.0) for idx in indices]

        features = [
            float(k) / n,                                # [0] depth
            rho_parent_ratio,                            # [1] budget used
            max(1.0 - rho_parent_ratio, 0.0),           # [2] budget remaining
            np.clip(np.log(Bk / B0 + 1e-8) / 5.0, -2.0, 2.0),  # [3] log relative GS norm
            abs_frac,                                    # [4] |frac(center)|
            sign_frac * abs_frac,                        # [5] signed frac
            eff_radius / 10.0,                           # [6] effective radius (scaled)
            1.0 if s.has_solution else 0.0,              # [7] has_solution
            best_ratio / 10.0,                           # [8] best norm (scaled)
            coeff_offset,                                # [9] tried offset
        ] + gs_profile                                   # [10-19] GS profile

        features_arr = np.array(features, dtype=np.float32)
        # Safety: replace any NaN/Inf
        features_arr = np.nan_to_num(features_arr, nan=0.0, posinf=1.0, neginf=-1.0)
        return features_arr

    def get_best_norm(self):
        return self.wrapper.get_state().best_norm

    def get_best_vector(self):
        return self.wrapper.get_best_vector()

    def is_terminated(self):
        s = self.wrapper.get_state()
        return s.terminated or self.wrapper.is_terminated()
