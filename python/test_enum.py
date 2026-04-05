import sys
import argparse
import numpy as np
import time

sys.path.append('../lib')
try:
    import lattice_env
except ImportError as e:
    print(f"Error: Could not import lattice_env: {e}")
    sys.exit(1)

parser = argparse.ArgumentParser(
    description="Test ENUM algorithm on an SVP Challenge lattice",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument('--dim',    type=int,   default=40,      help='Lattice dimension (challenge files: 40-200)')
parser.add_argument('--seed',   type=int,   default=0,       help='SVP challenge seed')
parser.add_argument('--radius', type=float, default=4000000, help='ENUM search radius R')
args = parser.parse_args()

dim         = args.dim
seed        = args.seed
enum_radius = args.radius

print(f"SVP Challenge: dim={dim}, seed={seed}, ENUM radius={enum_radius:.2e}")
print("-" * 50)

lat = lattice_env.create_lattice_int(dim, dim)
lat.setSVPChallenge(dim, seed)
lat.computeGSO()

initial_norm = lat.b1Norm()
print(f"Initial b1 norm: {initial_norm:.4f}")

start_time = time.time()
coeff_vector = lat.ENUM(enum_radius)
elapsed = time.time() - start_time

v = lat.mulVecBasis(coeff_vector)
found_norm = np.linalg.norm(v)

print(f"ENUM found norm:  {found_norm:.4f}")
print(f"Time:             {elapsed:.4f}s")
print(f"Coeff vector:     {list(coeff_vector)}")
