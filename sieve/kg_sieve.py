import numpy as np
from numpy.linalg import norm
import os
import sys
import time
from multiprocessing import Pool
from itertools import combinations
from math import comb
from random import randint, sample
from utils import remove_zeros, par_remove_zeros
def gauss_sieve_direct(basis, c=1000, verbose=True):
    """
    Gauss sieve directly on basis vectors with statistics
    
    Parameters:
        basis:      Basis for the lattice
        c:          Number of collisions before stopping
        verbose:    Whether to print details
    
    Returns:
        v:          Shortest vector found
        stats:      Statistics dictionary
    """
    # Initialize lists and statistics
    L = []  # List of vectors
    S = []  # Stack
    K = 0   # Collision counter
    
    # Statistics
    stats = {
        'collisions': 0,
        'distance_checks': 0,
        'norm_comparisons': 0,
        'vector_operations': 0,
        'vectors_added_to_L': 0,
        'vectors_added_to_S': 0,
        'vectors_popped_from_S': 0
    }
    
    # Initialize with basis vectors
    for i in range(len(basis)):
        S.append(basis[i].copy())
    
    if verbose:
        print(f"Gauss sieve started...")
        print(f"Lattice dimension: {len(basis)}")
        print(f"Initial vectors: {len(S)}")
    
    try:
        while K < c:
            # Draw vector from stack or use zero if stack is empty
            if len(S) > 0:
                v_new = S.pop()
                stats['vectors_popped_from_S'] += 1
            else:
                # If stack is empty, we can't continue
                break
            
            # Reduce the vector
            v_new, iter_stats = gauss_reduce_with_stats(v_new, L, S)
            stats['distance_checks'] += iter_stats['distance_checks']
            stats['norm_comparisons'] += iter_stats['norm_comparisons']
            stats['vector_operations'] += iter_stats['vector_operations']
            
            # Check if vector is zero (collision)
            if np.count_nonzero(v_new) == 0:
                K += 1
                stats['collisions'] += 1
                if verbose and K % 100 == 0:
                    print(f"\rCollisions: {K}/{c}", end="")
            else:
                # Add to list
                L.append(v_new)
                stats['vectors_added_to_L'] += 1
                
                # Find shortest in L
                if len(L) > 0 and verbose:
                    min_norm = norm(min(L, key=lambda v: norm(v)))
                    print(f"\rVectors in L: {len(L)}, Min norm: {min_norm:.4f}, Collisions: {K}/{c}", end="")
    
    except KeyboardInterrupt:
        if verbose:
            print("\nInterrupted by user")
    
    # Find shortest vector
    if len(L) > 0:
        shortest = min(L, key=lambda v: norm(v))
    elif len(S) > 0:
        shortest = min(S, key=lambda v: norm(v))
    else:
        shortest = None
    
    if verbose:
        print(f"\n\nGauss sieve completed")
        print(f"Collisions reached: {K}/{c}")
        print(f"Final vectors in L: {len(L)}")
        if shortest is not None:
            print(f"Shortest vector norm: {norm(shortest):.4f}")
    
    return shortest, stats

def gauss_reduce_with_stats(p, L, S):
    """
    Reduce vector p using vectors in L with statistics
    
    Parameters:
        p:  Vector to reduce
        L:  List of vectors
        S:  Stack of vectors
    
    Returns:
        p:  Reduced vector
        stats: Statistics dictionary
    """
    stats = {
        'distance_checks': 0,
        'norm_comparisons': 0,
        'vector_operations': 0
    }
    
    # Check for shorter vectors in L
    shorter_found = True
    while shorter_found:
        shorter_found = False
        for v in L:
            if np.count_nonzero(v) > 0:
                stats['norm_comparisons'] += 2  # Comparing norm(v) <= norm(p) and norm(p-v) <= norm(p)
                stats['distance_checks'] += 1    # Computing norm(p-v)
                
                if (norm(v) <= norm(p)) and (norm(p - v) <= norm(p)):
                    p = p - v
                    shorter_found = True
                    stats['vector_operations'] += 1
                    break
    
    # Check vectors in L that can be reduced by p
    indices_to_remove = []
    vectors_to_add = []
    
    for i, v in enumerate(L):
        stats['norm_comparisons'] += 2  # Comparing norm(v) > norm(p) and norm(v-p) <= norm(v)
        stats['distance_checks'] += 1    # Computing norm(v-p)
        
        if (norm(v) > norm(p)) and (norm(v - p) <= norm(v)):
            indices_to_remove.append(i)
            vectors_to_add.append(v - p)
            stats['vector_operations'] += 1
    
    # Remove and add vectors (reverse to maintain indices)
    for idx in reversed(indices_to_remove):
        L.pop(idx)
    
    for vec in vectors_to_add:
        S.append(vec)
    
    return p, stats

def double_sieve_direct(basis, gamma=0.99, minkowski_bound=None, max_iterations=30, verbose=True):
    """
    Double sieve for SVP Challenge
    """

    d = len(basis)
    

    if minkowski_bound is None:

        vol_est = np.linalg.det(basis)
        minkowski_bound = (d ** 0.5) * (vol_est ** (1/d))
    

    S = [basis[i].copy() for i in range(d)]
    N = len(S)
    
    if verbose:
        print(f"Double Sieve for SVP Challenge - Dim: {d}")
        print(f"Initial vectors: {N} (basis vectors)")
        print(f"Gamma: {gamma}, Minkowski bound: {minkowski_bound:.4f}")
    

    total_stats = {
        'iterations': 0,
        'vectors_processed': 0,
        'distance_checks': 0,
        'successful_matches': 0,
        'time': 0
    }
    
    S_original = S.copy()
    iteration = 0
    
    import time
    start_time = time.time()
    
    while len(S) > 0 and iteration < max_iterations:

        S_next, marked, iter_stats = lattice_sieve_two_with_stats(S, gamma)
        

        iteration += 1
        total_stats['iterations'] = iteration
        total_stats['vectors_processed'] += len(S)
        total_stats['distance_checks'] += iter_stats['distance_checks']
        total_stats['successful_matches'] += iter_stats['successful_matches']
        
        if len(S_next) > 0:

            if len(S_next) > N:
                from random import sample
                S_next = sample(S_next, N)
            

            min_norm = norm(min(S_next, key=lambda v: norm(v)))
            
            if verbose:
                print(f"Iter {iteration}: vectors {len(S)}->{len(S_next)}, "
                      f"min norm: {min_norm:.4f}, checks: {iter_stats['distance_checks']}")
            

            if min_norm < minkowski_bound:
                if verbose:
                    print(f"Found vector below Minkowski bound!")
                S = S_next
                break
        else:
            if verbose:
                print(f"Set empty at iteration {iteration}")
            break
        
        S = S_next
    
    total_stats['time'] = time.time() - start_time
    

    if len(S) > 0:
        shortest = min(S, key=lambda v: norm(v))
    else:
        shortest = min(S_original, key=lambda v: norm(v))
    
    return shortest, total_stats

def lattice_sieve_two_with_stats(S, gamma):
    """

    """
    if not S or len(S) < 2:
        return [], 0, {'distance_checks': 0, 'successful_matches': 0}
    
    S_p = []
    R = sum(norm(v) for v in S) / len(S)
    gR = gamma * R
    

    distance_checks = 0
    successful_matches = 0
    

    for i in range(len(S)):
        for j in range(i + 1, len(S)):
            v, w = S[i], S[j]

            distance_checks += 1
            if norm(v - w) <= gR:
                S_p.append(v - w)
                successful_matches += 1

            else:
                distance_checks += 1
                if norm(v + w) <= gR:
                    S_p.append(v + w)
                    successful_matches += 1
    
    return S_p, successful_matches, {
        'distance_checks': distance_checks,
        'successful_matches': successful_matches
    }
    
def print_statistics(stats):
    """
    Print statistics in a readable format
    """
    print("\n=== Statistics ===")
    if 'iterations' in stats:
        print(f"Iterations: {stats['iterations']}")
    
    if 'vectors_processed' in stats:
        print(f"Total vectors processed: {stats['vectors_processed']}")
    
    if 'distance_checks' in stats:
        print(f"Total distance checks: {stats['distance_checks']:,}")
    
    if 'norm_comparisons' in stats:
        print(f"Total norm comparisons: {stats['norm_comparisons']:,}")
    
    if 'vector_operations' in stats:
        print(f"Total vector operations: {stats['vector_operations']:,}")
    
    if 'successful_matches' in stats:
        print(f"Successful matches: {stats['successful_matches']:,}")
    
    if 'collisions' in stats:
        print(f"Collisions: {stats['collisions']}")
    
    if 'vectors_added_to_L' in stats:
        print(f"Vectors added to L: {stats['vectors_added_to_L']}")
    
    if 'reduced_pairs_checked' in stats:
        print(f"Reduced pairs checked: {stats['reduced_pairs_checked']:,}")
    
    if 'unique_pairs_generated' in stats:
        print(f"Unique pairs generated: {stats['unique_pairs_generated']:,}")
    
    # Calculate averages
    if 'vectors_processed' in stats and stats['vectors_processed'] > 0:
        if 'distance_checks' in stats:
            avg_checks = stats['distance_checks'] / stats['vectors_processed']
            print(f"Average distance checks per vector: {avg_checks:.2f}")
        
        if 'successful_matches' in stats:
            match_rate = stats['successful_matches'] / stats['vectors_processed'] * 100
            print(f"Match rate: {match_rate:.2f}%")

def read_svp_challenge_direct(dim: int, seed: int = 0) -> np.ndarray:
    """
    Directly read SVP Challenge format file
    
    Note: SVP Challenge files usually have shape (2*dim, dim),
    we need to extract the first dim rows as lattice basis
    """
    file_name = f"../svp_challenge_list/svp_challenge_{dim}_{seed}.txt"
    
    if not os.path.exists(file_name):
        # Try alternative path
        alt_path = f"svp_challenge_list/svp_challenge_{dim}_{seed}.txt"
        if os.path.exists(alt_path):
            file_name = alt_path
        else:
            print(f"Error: File {file_name} does not exist")
            return None
    
    print(f"Reading SVP Challenge file: {file_name}")
    
    # Read file
    data = []
    with open(file_name, 'r') as f:
        for line in f:
            if line.strip():
                row = [float(x) for x in line.split()]
                data.append(row)
    
    # Convert to numpy array
    basis_full = np.array(data)
    
    print(f"Original file shape: {basis_full.shape}")
    
    # SVP Challenge files are usually (2*dim, dim) shape
    # We need to extract the first dim rows as lattice basis
    if basis_full.shape[1] == dim and basis_full.shape[0] >= dim:
        # Extract first dim rows
        basis = basis_full[:dim, :]
        print(f"Extracted basis shape: {basis.shape}")
        
        # Verify basis quality
        rank = np.linalg.matrix_rank(basis)
        print(f"Basis rank: {rank}/{dim}")
        
        if rank < dim:
            print("Warning: Basis not full rank!")
            # Add small perturbation to make it full rank
            np.random.seed(42)
            noise = np.random.randn(dim, dim) * 1e-8
            basis = basis + noise
        
        return basis
    elif basis_full.shape[0] == dim and basis_full.shape[1] == dim:
        # Already correct shape
        print(f"Basis shape correct: {basis_full.shape}")
        return basis_full
    else:
        print(f"Warning: Abnormal file shape {basis_full.shape}, expected (>={dim}, {dim})")
        return None
        
def run_all_sieves(dim, seed=0, gamma=0.99, max_iterations=50, collisions=1000):
    """
    Run all sieve algorithms on the same SVP Challenge instance
    
    Parameters:
        dim:            Dimension
        seed:           SVP Challenge seed
        gamma:          Reduction factor for list sieves
        max_iterations: Maximum iterations for list sieves
        collisions:     Collision count for Gauss sieve
    
    Returns:
        results:        Dictionary with results from all sieves
    """
    # Read SVP Challenge basis
    basis = read_svp_challenge_direct(dim, seed)
    if basis is None:
        print(f"Failed to read SVP Challenge basis for dim={dim}")
        return None
    
    print(f"\n{'='*60}")
    print(f"Running sieves on SVP Challenge dim={dim}, seed={seed}")
    print(f"Basis shape: {basis.shape}")
    print(f"{'='*60}\n")
    
    results = {}
    
    # Run Gauss Sieve
    print(f"\n[1] Running Gauss Sieve...")
    start_time = time.time()
    gauss_result, gauss_stats = gauss_sieve_direct(basis, c=collisions, verbose=True)
    gauss_time = time.time() - start_time
    gauss_stats['time'] = gauss_time
    
    if gauss_result is not None:
        gauss_norm = norm(gauss_result)
        print(f"Gauss sieve completed in {gauss_time:.2f}s")
        print(f"Shortest vector norm: {gauss_norm:.4f}")
        results['gauss'] = {'vector': gauss_result, 'norm': gauss_norm, 'stats': gauss_stats}
    
    # Run Double Sieve
    print(f"\n[2] Running Double Sieve...")
    start_time = time.time()
    double_result, double_stats = double_sieve_direct(basis, gamma=gamma, max_iterations=max_iterations, verbose=True)
    double_time = time.time() - start_time
    double_stats['time'] = double_time
    
    if double_result is not None:
        double_norm = norm(double_result)
        print(f"Double sieve completed in {double_time:.2f}s")
        print(f"Shortest vector norm: {double_norm:.4f}")
        results['double'] = {'vector': double_result, 'norm': double_norm, 'stats': double_stats}
    
    # Compare results
    print(f"\n{'='*60}")
    print(f"SUMMARY - Dimension {dim}")
    print(f"{'='*60}")
    
    if 'gauss' in results and 'double' in results:
        print(f"Gauss Sieve:    {results['gauss']['norm']:.4f} (time: {results['gauss']['stats']['time']:.2f}s)")
        print(f"Double Sieve:   {results['double']['norm']:.4f} (time: {results['double']['stats']['time']:.2f}s)")
        
        # Compare distance checks (main operation)
        gauss_checks = results['gauss']['stats']['distance_checks']
        double_checks = results['double']['stats']['distance_checks']
        
        print(f"\nDistance checks comparison:")
        print(f"  Gauss Sieve:  {gauss_checks:,}")
        print(f"  Double Sieve: {double_checks:,}")
        
        if gauss_checks > 0 and double_checks > 0:
            ratio = gauss_checks / double_checks
            print(f"  Ratio (Gauss/Double): {ratio:.2f}x")
    
    return results

if __name__ == "__main__":
    import time
    
    # Test with a specific dimension
    for seed in range(2, 9, 2):
        for dim in range(40, 101, 10):
            results = run_all_sieves(dim, seed, gamma=0.99, max_iterations=30, collisions=500)