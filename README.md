# From Classical to AI-Augmented: A Benchmark for Evolving Shortest Vector Problem Solvers

The Shortest Vector Problem (SVP) is a foundational computational challenge in lattice theory and a core basis for Post-Quantum Cryptography (PQC) security. With Artificial Intelligence (AI) advancement, researchers have started exploring new approaches to solving SVP via Machine Learning (ML) and Deep Learning (DL). Though promising, the emerging field of “AI for SVP” (AI4SVP) faces challenges in its early stages, especially the absence of generic methodological taxonomy and high-quality benchmark datasets, with inconsistencies in experimental setup and evaluation protocols. To address this, we introduce the first comprehensive benchmarking framework for AI4SVP. We systematically categorize 20 SVP solvers into three classical paradigms: sieving, enumeration, and lattice reduction. Building on these, we design **AI4SVP-Bench** (this repo), a modular framework with three AI-Augmented task interfaces: **AI4Enum**, **AI4BKZ**, and **AI4Sieve**. Evaluated on lattice instances spanning 80 dimensions, our AI components show significant gains: AI4Enum reduces node visits by 33.6% and time by 28.6% on 60-dimensional instances; AI4BKZ cuts SVP oracle calls by 25.3% and runtime by 58.4%; AI4Sieve reduces list operations by 18.0%. Further, We conduct extensive hyperparameter optimization on existing SVP solvers across multiple test scenarios in different dimensions, empirically verifying whether such observed optimal parameterization exhibits instance-agnostic universality. Overall, our study demonstrates empirical feasibility, draws desired principles, and aims to promote coordinated innovations for future development in the novel interdisciplinary AI4SVP realm.

**Note**: The paper is currently under single-blind review for KDD'26 AI for Sciences Track.

---

## Reproduction Guide

### Directory Structure

```
ai4svp/
├── build/                    # CMake build output (generated)
├── include/                  # C++ headers (Lattice class)
├── lib/                      # Compiled shared library (lattice_env.*.so)
├── pybind/                   # C++ pybind11 wrappers
├── src/                      # C++ algorithm implementations
├── svp_challenge_list/       # SVP challenge files (dim 40–200, seeds 0–9)
├── python/                   # All Python experiment scripts
│   ├── test_plug.py          # Benchmark classical reduction algorithms + plots
│   ├── test_enum.py          # ENUM algorithm test (configurable dim/seed/radius)
│   ├── svp_hyperopt.py       # Hyperparameter optimisation (HEBO)
│   ├── train_rl_enum.py      # AI4Enum: RL-ENUM training
│   ├── test_rl_model.py      # AI4Enum: RL-ENUM evaluation
│   ├── bkz_environment.py    # AI4BKZ: RL-BKZ environment (fpylll BKZ2.0)
│   ├── train_rl_bkz.py       # AI4BKZ: RL-BKZ training
│   ├── test_rl_bkz.py        # AI4BKZ: RL-BKZ evaluation
│   └── checkpoints*/         # Saved model weights
├── sieve/                    # AI4Sieve: sieve-based solvers
│   ├── kg_sieve.py           # Gauss sieve + Double sieve
│   ├── nv_sieve.py           # Nguyen-Vidick sieve
│   ├── main.py               # AI-enhanced NV sieve pipeline
│   └── config.py             # Sieve configuration
├── requirements.txt          # Python dependencies (SVP conda env)
└── README.md                 # This file
```

---

## System Requirements

### Hardware
- **GPU**: NVIDIA A100-SXM4-80GB (CUDA compute capability 8.0) | Driver: 535.183.01
- **VRAM**: 80 GB
- **CPU**: Intel Xeon Platinum 8378A @ 3.00GHz | 128 cores (2 sockets × 32 cores × 2 threads) | 96 MB L3 cache
- **RAM**: 1.0 TB (1024 GiB)
- **Storage**: 300 GB overlay filesystem

### Software & Versions
| Component | Version | Notes |
|-----------|---------|-------|
| CUDA | 11.8+ | Bundled with torch==2.6.0 (cu118) |
| cuDNN | 8.x | Auto-managed by PyTorch |
| Python | 3.11 | Exact version in requirements.txt |
| PyTorch | 2.6.0+cu118 | GPU acceleration for RL policy network |
| fpylll | 0.6.4 | Lattice reduction backend; must be installed correctly with cysignals |
| CMake | 3.10+ | For C++ pybind11 compilation |

---

## 1. Environment Setup

### 1.1 Create Conda Environment

```bash
conda create -n SVP python=3.11
# Activate (adjust path to your Anaconda installation):
source /path/to/anaconda3/envs/SVP/bin/activate SVP
```

### 1.2 Install Python Dependencies

```bash
pip install -r requirements.txt
```

Key packages:

| Package | Version | Purpose |
|---------|---------|---------|
| torch | 2.1.0+cu118 | RL policy network (GPU) |
| fpylll | 0.6.4 | BKZ2.0 lattice reduction |
| cysignals | 1.12.5 | fpylll dependency |
| HEBO | 0.3.6 | Bayesian hyperparameter search |
| numpy | 1.24.4 | Numerical computation |
| pandas | 1.5.3 | HEBO data handling |
| pymoo | 0.6.0 | Multi-objective optimisation |
| cma | 3.2.2 | CMA-ES optimiser |
| matplotlib | 3.10.x | Plotting |
| scikit-learn | 1.8.0 | AI sieve model |

### 1.3 Compile the C++ Lattice Library

```bash
cd ai4svp
mkdir -p build && cd build
cmake ..
make -j4
# Produces: lib/lattice_env.cpython-311-x86_64-linux-gnu.so
```

> **Important:** All Python scripts must be run from the `python/` directory
> because they use `sys.path.append('../lib')` to locate the compiled library.

```bash
cd ai4svp/python   # run all python/ commands from here
```

---

## 2. Classical Reduction Algorithms

### 2.1 Benchmark All Algorithms with Plots

```bash
python test_plug.py
```

Tests LLL, BKZ, DeepLLL, PotLLL, DualLLL, DualBKZ, DeepBKZ, PotBKZ, etc.
across configurable dimensions and saves comparison plots to `results/`.

To change test dimensions, edit line 332 of `test_plug.py`:
```python
test_dimensions = list(range(40, 121, 10))   # [40, 50, ..., 120]
```

---

## 3. ENUM Algorithm

### 3.1 Run ENUM (configurable dimension / seed / radius)

```bash
# Default: dim=40, seed=0, radius=4e6
python test_enum.py

# Custom parameters
python test_enum.py --dim 40 --seed 0 --radius 4000000
python test_enum.py --dim 50 --seed 2 --radius 8000000
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--dim` | 40 | Lattice dimension (SVP challenge files: 40–200) |
| `--seed` | 0 | SVP challenge seed (0–9) |
| `--radius` | 4000000 | ENUM search radius R |

**Example output:**
```
SVP Challenge: dim=40, seed=0, ENUM radius=4.00e+06
--------------------------------------------------
Initial b1 norm: 1881.0149
ENUM found norm:  1709.1735
Time:             0.3784s
```

---

## 4. Hyperparameter Optimisation

Uses HEBO (Hierarchical Expected-improvement Bayesian Optimisation) to tune
algorithm parameters automatically.

```bash
# Edit the __main__ block in svp_hyperopt.py, then run:
python svp_hyperopt.py
```

Or call programmatically:

```python
from svp_hyperopt import SVPHyperOptimizer

optimizer = SVPHyperOptimizer(
    dim=40,
    seed=0,
    max_evaluations=20,
    algorithm='fplll_BKZ2.0',        # see table below
    obj_weight={'time': 0.3, 'norm': 0.7},
    timeout_seconds=120,
)
best_params, best_obj = optimizer.optimize(n_suggestions=4)
```

**Supported algorithms and optimised parameters:**

| Algorithm | Optimised Parameters |
|-----------|---------------------|
| `LLL`, `deepLLL`, `potLLL`, `dualLLL`, `dualDeepLLL`, `dualPotLLL`, `HKZ` | `delta` ∈ [0.5, 1] |
| `BKZ`, `deepBKZ`, `dualBKZ`, `dualDeepBKZ` | `beta` ∈ [2, dim], `delta` ∈ [0.5, 1] |
| `potBKZ` | `beta` ∈ [2, dim], `delta` ∈ [0.981, 1] |
| `L2` | `delta` ∈ [0.5, 1], `eta` ∈ [0.51, 1] |
| `ENUM` | `log_R` ∈ [log(3e6), log(5e6)] |
| `fplll_BKZ2.0` | `beta` ∈ [2, dim] |
| `fplll_self_dual_BKZ` | `beta` ∈ [2, dim] |

Results saved to `opt_results/<algorithm>_dim<d>_<timestamp>/`.

---

## 5. AI4BKZ — RL-guided BKZ

A PPO agent adaptively selects the BKZ block size β ∈ {10,15,20,25,30,35,40}
at each tour based on the current GS-norm profile. Uses fpylll BKZ2.0 as the
underlying executor. Trained jointly on dim=40 and dim=50; generalises to
dim=60 without retraining.

### 5.1 Train

```bash
python train_rl_bkz.py --episodes 500
# Saves checkpoints to checkpoints_bkz/rl_bkz_ep*.pt
# Saves training plot to training_progress_bkz.png
```

Resume from a checkpoint:
```bash
python train_rl_bkz.py --episodes 500 --checkpoint checkpoints_bkz/rl_bkz_ep250.pt
```

### 5.2 Evaluate

```bash
python test_rl_bkz.py \
    --checkpoint checkpoints_bkz/rl_bkz_ep500.pt \
    --dims 40 50 60 \
    --seeds 0 1 2 3 4 \
    --max_tours 20
```

**Generalisation results (dim=60, never seen during training):**

| Method | Avg norm ↓ | Avg time | RL wins |
|--------|-----------|----------|---------|
| **RL agent** | **2177.78** | 1.57s | **5/5** |
| Fixed β=20 | 2277.13 | 0.10s | — |
| Progressive β=[10,15,20,25,30,35,40] | 2427.27 | 1.31s | — |
| Fixed β=40 (strong baseline) | 2082.03 | 16.87s | — |

RL achieves norm quality close to β=40 while being **~10× faster**.

---

## 6. AI4Enum — RL-guided ENUM

A PPO agent learns to guide the ENUM coefficient search. Uses curriculum
learning (search radius decays from 3.6×10⁸ → 4×10⁶ over 400 episodes) to
provide a dense reward signal.

### 6.1 Train

```bash
python train_rl_enum.py
# Trains 500 episodes on dim=40 SVP challenges (seeds 0–4)
# Saves checkpoints to checkpoints/rl_enum_ep*.pt
```

### 6.2 Evaluate

```bash
python test_rl_model.py \
    --checkpoint checkpoints/rl_enum_ep500.pt \
    --dims 40 \
    --seeds 0 1 2 3 4 \
    --radius 4e6 \
    --max_steps 3000 \
    --enum_timeout 10
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--checkpoint` | `checkpoints/rl_enum_ep500.pt` | Model checkpoint |
| `--dims` | `[40]` | Dimensions to evaluate |
| `--seeds` | `[0,1,2]` | SVP challenge seeds |
| `--radius` | `4e6` | ENUM search radius |
| `--max_steps` | `3000` | Max RL steps per episode |
| `--enum_timeout` | `10` | Timeout (s) for exhaustive C++ ENUM reference |

---

## 7. AI4Sieve — Sieve Algorithms

All sieve scripts live in `sieve/` and must be run from that directory:

```bash
cd ai4svp/sieve
```

### 7.1 Gauss Sieve

```python
import numpy as np
from kg_sieve import gauss_sieve_direct

basis = np.loadtxt('../svp_challenge_list/svp_challenge_40_0.txt')[:40]
shortest, stats = gauss_sieve_direct(basis, c=1000, verbose=True)
print(f”Shortest vector norm: {np.linalg.norm(shortest):.4f}”)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `basis` | — | dim × dim numpy array (lattice basis) |
| `c` | 1000 | Collision count before stopping |
| `verbose` | True | Print progress |

### 7.2 Double Sieve

```python
from kg_sieve import double_sieve_direct

shortest, stats = double_sieve_direct(
    basis,
    gamma=0.99,
    minkowski_bound=None,   # auto-computed if None
    max_iterations=30,
    verbose=True,
)
```

### 7.3 Nguyen-Vidick (NV) Sieve

```python
from nv_sieve import nguyen_vidick_sieve_direct

shortest, stats = nguyen_vidick_sieve_direct(
    basis,
    gamma=0.99,
    max_iterations=50,
    verbose=True,
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `gamma` | 0.99 | Reduction factor ∈ (0, 1) — smaller = more aggressive |
| `max_iterations` | 50 | Maximum sieve iterations |

### 7.4 AI-Enhanced NV Sieve (`main.py`)

Full pipeline: data collection → neural network training → AI-guided sieve.

```bash
cd ai4svp/sieve

# Full pipeline
python main.py --mode all --dim 60 --gamma 0.85 --top_k 10

# Individual stages
python main.py --mode collect --dim 60           # Collect training samples
python main.py --mode train   --dim 60           # Train center-match model
python main.py --mode test    --dim 60           # Test AI-enhanced sieve
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--mode` | `all` | `collect` / `train` / `test` / `all` |
| `--dim` | 100 | Lattice dimension |
| `--gamma` | 0.8 | Sieve reduction parameter |
| `--top_k` | 10 | Centers checked per vector (AI model) |

**Outputs:**
- `training_data/training_data.pkl` — collected feature/label pairs
- `center_match_model.pth` — trained neural network
- `svp_result/` — shortest vector results

---

## 8. SVP Challenge Files

Pre-generated instances in `svp_challenge_list/svp_challenge_{dim}_{seed}.txt`:

| Dimensions | Seeds |
|-----------|-------|
| 40, 50, 60, 70, 80, 90 | 0–9 |
| 100, 110, 120, …, 200 (step 10) | 0–9 |

---

## 9. Experimental Configuration

### Overview
Each AI-augmented solver has a `Config` class that controls training/evaluation behavior. This section documents the main hyperparameters and how to modify them.

### AI4BKZ Configuration
Located in `python/train_rl_bkz.py` (lines 29–42):

```python
class BKZConfig:
    dimensions    = [40, 50]   # Train on both; generalizes to dim=60
    num_seeds     = 5          # SVP challenge seed indices 0–4
    max_tours     = 20         # Max BKZ reduction tours per episode
    stagnation_tol = 1e-3      # Early stopping: halt if b1 improvement < 0.1%
    time_penalty  = 0.5        # Reward penalty per elapsed second
    
    # PPO hyperparameters
    gamma         = 0.99       # Discount factor
    epsilon       = 0.2        # PPO clipping range
    learning_rate = 3e-4       # Optimizer learning rate
    batch_size    = 64         # Trajectories per PPO update
    ppo_epochs    = 4          # PPO update epochs per batch
    entropy_coeff = 0.02       # Entropy regularization (encourages exploration)
```

**Customization**:
```bash
# To train on only dim=40 (faster):
# Edit train_rl_bkz.py line 30: dimensions = [40]

# To use 100 episodes instead of 500:
python train_rl_bkz.py --episodes 100

# To use larger exploration:
# Edit train_rl_bkz.py line 42: entropy_coeff = 0.05
```

**Action Space**: β ∈ {10, 15, 20, 25, 30, 35, 40} (7 discrete actions per tour)

**Reward**: `(log(b1_prev / b1_curr)) * 10 - time_elapsed * time_penalty`
- Dense signal: every tour produces a reward (no sparse-reward problem)
- Dimensionless: log-scale norm change is comparable across dimensions

---

### AI4Enum Configuration
Located in `python/train_rl_enum.py` (lines 30–43):

```python
class EnumConfig:
    dimensions     = [40]              # Train on dim=40 only
    num_seeds      = 5                 # SVP seeds 0–4
    max_steps      = 3000              # RL steps per episode
    search_radius_init = 3.6e8         # Curriculum: start large radius
    search_radius_final = 4e6          # Curriculum: end small radius
    curriculum_episodes = 400          # Episodes to decay R over
    
    gamma          = 0.99              # Discount factor
    epsilon        = 0.2               # PPO clip range
    learning_rate  = 1e-3              # Actor-critic learning rate
    batch_size     = 32                # Trajectories per update
    ppo_epochs     = 5                 # PPO epochs
    entropy_coeff  = 0.01              # Lower than BKZ (sparser reward)
```

**Curriculum Learning**:
- ENUM has sparse rewards (only when shortest vector found)
- Radius R decays from 3.6×10⁸ → 4×10⁶ over 400 episodes
- Early episodes: easy (large R) → dense rewards → quick learning
- Late episodes: hard (small R) → challenge agent to refine

**Evaluation**:
```bash
python test_rl_model.py \
    --checkpoint checkpoints/rl_enum_ep500.pt \
    --dims 40 \
    --seeds 0 1 2 3 4 \
    --radius 4e6 \
    --max_steps 3000
```

---

### Hyperparameter Optimization (HEBO)
Located in `python/svp_hyperopt.py` (lines ~150–175):

```python
SVPHyperOptimizer(
    dim=40,
    seed=0,
    max_evaluations=20,         # Number of BO iterations
    algorithm='fplll_BKZ2.0',   # Algorithm to tune
    obj_weight={'time': 0.3, 'norm': 0.7},  # Multi-objective: favor norm over time
    timeout_seconds=120,        # Wall-clock timeout per evaluation
)
```

**Supported Algorithms & Search Spaces**:

| Algorithm | Tuned Parameter | Search Range |
|-----------|-----------------|--------------|
| LLL variants | `delta` | [0.5, 1.0] |
| BKZ variants | `beta`, `delta` | β ∈ [2, dim], δ ∈ [0.5, 1.0] |
| ENUM | `log_R` | [log(3e6), log(5e6)] |
| fplll_BKZ2.0 | `beta` | [2, dim] |

**Multi-objective Tuning**:
- `obj_weight={'time': 0.3, 'norm': 0.7}` → prioritize norm quality (70%) over speed (30%)
- Useful for PQC applications where vector quality matters more than time

---

### AI4Sieve Configuration
Located in `sieve/config.py`:

```python
config = {
    'gamma': 0.99,              # Reduction factor per iteration
    'max_iterations': 50,       # Sieve loop iterations
    'top_k': 10,                # Centers checked by AI model per vector (AI4Sieve)
    'batch_size': 128,          # Mini-batch size for neural network inference
}
```

**Modes**:
- `--mode collect`: Generate training data (collect features & labels from NV sieve runs)
- `--mode train`: Train center-matching neural network (outputs which centers to check)
- `--mode test`: Run AI-enhanced sieve with trained model
- `--mode all`: Full pipeline (collect → train → test)

---

### Known Limitations & Recommendations

| Issue | Recommendation |
|-------|-----------------| 
| **Sparse ENUM rewards** | Use curriculum learning (R decay); increasing `entropy_coeff` can help exploration |
| **Sieve memory usage** | Large dims (>100) require significant RAM; monitor with `free -h` |
| **fpylll installation** | If build fails, ensure cysignals is installed: `pip install cysignals==1.12.5` |
| **C++ compilation** | If CMake fails, check Python version matches 3.11: `python --version` |

---

## 10. Quick-Start Commands

```bash
# ── Setup ──────────────────────────────────────────────────────────────
source /path/to/anaconda3/envs/SVP/bin/activate SVP
cd ai4svp/build && make -j4 && cd ../python

# ── Classical benchmark ────────────────────────────────────────────────
python test_plug.py

# ── ENUM: dim=50, seed=1 ───────────────────────────────────────────────
python test_enum.py --dim 50 --seed 1 --radius 8000000

# ── Hyperparameter search (edit __main__ first) ────────────────────────
python svp_hyperopt.py

# ── AI4BKZ: train → evaluate ──────────────────────────────────────────
python train_rl_bkz.py --episodes 500
python test_rl_bkz.py --dims 40 50 60 --seeds 0 1 2 3 4

# ── AI4Enum: train → evaluate ─────────────────────────────────────────
python train_rl_enum.py
python test_rl_model.py --dims 40 --seeds 0 1 2 3 4

# ── AI4Sieve ──────────────────────────────────────────────────────────
cd ../sieve
python main.py --mode all --dim 60 --gamma 0.85
```

