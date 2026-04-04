#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SVP Challenge Solver - Using Nguyen-Vidick Sieve
Directly solve shortest vector from SVP Challenge lattice basis
"""

import numpy as np
from numpy.linalg import norm
import os
import sys
import time

# Utility functions
def remove_zeros(S):
    """Remove zero vectors"""
    return [v for v in S if norm(v) > 1e-10]

def exists_close_center(C, v, gR):
    """
    Given a vector v, check if any center in list C can reduce v
    
    Parameters:
        C:      Center list
        v:      Vector to reduce
        gR:     Reduction threshold (gamma * R)
    
    Returns:
        True, c: if center c is found
        False, 0: if not found
    """
    for c in C:
        if norm(v - c) <= gR:
            return (True, c)
    return (False, 0)

def lattice_sieve(S, gamma, verbose=False):
    """
    
    Parameters:
        S:      Current sieve set
        gamma:  Norm reduction factor
        verbose: Whether to output details
    
    Returns:
        S_p:    Set for next step
        stats:  Statistics dictionary
    """
    if not S:
        return [], {}
    
    # Calculate average norm as R
    R = sum(norm(v) for v in S) / len(S)
    gR = gamma * R
    
    # Lists for centers and next set
    C = []
    S_p = []
    
    # Statistics counters
    stats = {
        'total_vectors': len(S),
        'direct_pass': 0,
        'matched': 0,
        'new_centers': 0,
        'distance_checks': 0,
        'norm_comparisons': 0,
        'vector_subtractions': 0
    }
    
    for v in S:
        v_norm = norm(v)
        stats['norm_comparisons'] += 1
        
        # If vector is small enough, add to next set
        if v_norm <= gR:
            S_p.append(v)
            stats['direct_pass'] += 1
        else:
            # Check if there is a close center
            found, c, checks = exists_close_center_with_stats(C, v, gR)
            stats['distance_checks'] += checks
            
            if found:
                # Reduce vector
                S_p.append(v - c)
                stats['matched'] += 1
                stats['vector_subtractions'] += 1
            else:
                # Add as new center
                C.append(v)
                stats['new_centers'] += 1
    
    if verbose:
        print(f"  Total vectors: {stats['total_vectors']}")
        print(f"  Direct pass: {stats['direct_pass']}, Matched: {stats['matched']}, New centers: {stats['new_centers']}")
        print(f"  Distance checks: {stats['distance_checks']}")
        print(f"  Norm comparisons: {stats['norm_comparisons']}")
        print(f"  Vector subtractions: {stats['vector_subtractions']}")
    
    return S_p, stats

def exists_close_center_with_stats(C, v, gR):
    """
    Check centers list to see if we can reduce vector v
    
    Parameters:
        C:      List of centers
        v:      Vector to reduce
        gR:     Reduction threshold (gamma * R)
    
    Returns:
        found:  Boolean whether center found
        center: The center if found
        checks: Number of distance checks performed
    """
    checks = 0
    for c in C:
        checks += 1
        if norm(v - c) <= gR:
            return (True, c, checks)
    return (False, None, checks)

def nguyen_vidick_sieve_direct(basis, gamma=0.99, max_iterations=50, verbose=True):
    """
    Run Nguyen-Vidick sieve directly on basis vectors
    
    Parameters:
        basis:          Lattice basis matrix (dim x dim)
        gamma:          Reduction factor
        max_iterations: Maximum iterations
        verbose:        Whether to print details
    
    Returns:
        shortest_vector: Found shortest vector
        total_stats:     Total statistics dictionary
    """
    # Use basis vectors directly as initial set
    S = [basis[i].copy() for i in range(len(basis))]
    
    if verbose:
        print(f"Direct NV sieve started...")
        print(f"Lattice dimension: {len(basis)}")
        print(f"Initial vectors: {len(S)} (basis vectors)")
        
        # Initial statistics
        norms = [norm(v) for v in S]
        print(f"Initial norm range: {min(norms):.4f} ~ {max(norms):.4f}")
        print(f"Initial average norm: {sum(norms)/len(norms):.4f}")
        print(f"Gamma: {gamma}")
    
    S_current = S
    S_best = S.copy()
    iteration = 0
    
    # Total statistics
    total_stats = {
        'iterations': 0,
        'total_vectors_processed': 0,
        'total_direct_pass': 0,
        'total_matched': 0,
        'total_new_centers': 0,
        'total_distance_checks': 0,
        'total_norm_comparisons': 0,
        'total_vector_subtractions': 0
    }
    
    while len(S_current) > 0 and iteration < max_iterations:
        if verbose:
            print(f"\nIteration {iteration+1}/{max_iterations}:")
        
        S_next, iter_stats = lattice_sieve(S_current, gamma, verbose=verbose)
        S_next = remove_zeros(S_next)
        
        iteration += 1
        
        # Accumulate statistics
        total_stats['iterations'] = iteration
        total_stats['total_vectors_processed'] += iter_stats['total_vectors']
        total_stats['total_direct_pass'] += iter_stats['direct_pass']
        total_stats['total_matched'] += iter_stats['matched']
        total_stats['total_new_centers'] += iter_stats['new_centers']
        total_stats['total_distance_checks'] += iter_stats['distance_checks']
        total_stats['total_norm_comparisons'] += iter_stats['norm_comparisons']
        total_stats['total_vector_subtractions'] += iter_stats['vector_subtractions']
        
        if len(S_next) > 0:
            # Find minimum norm in current iteration
            min_norm = norm(min(S_next, key=lambda v: norm(v)))
            
            # Update best set
            current_best_norm = norm(min(S_best, key=lambda v: norm(v)))
            if min_norm < current_best_norm:
                S_best = S_next.copy()
                if verbose:
                    print(f"  Updated best vector, norm: {min_norm:.4f}")
        else:
            if verbose:
                print(f"  Set empty, stopping iteration")
            break
        
        S_current = S_next
    
    if verbose:
        print(f"\nSieve completed, total iterations: {iteration}")
        print("\n=== Statistics Summary ===")
        print(f"Total iterations: {total_stats['iterations']}")
        print(f"Total vectors processed: {total_stats['total_vectors_processed']}")
        print(f"Total direct passes: {total_stats['total_direct_pass']}")
        print(f"Total matches found: {total_stats['total_matched']}")
        print(f"Total new centers: {total_stats['total_new_centers']}")
        print(f"Total distance checks: {total_stats['total_distance_checks']}")
        print(f"Total norm comparisons: {total_stats['total_norm_comparisons']}")
        print(f"Total vector subtractions: {total_stats['total_vector_subtractions']}")
        
        # Calculate averages
        avg_checks_per_vector = total_stats['total_distance_checks'] / total_stats['total_vectors_processed'] if total_stats['total_vectors_processed'] > 0 else 0
        avg_matches_per_iteration = total_stats['total_matched'] / total_stats['iterations'] if total_stats['iterations'] > 0 else 0
        print(f"\nAverage distance checks per vector: {avg_checks_per_vector:.2f}")
        print(f"Average matches per iteration: {avg_matches_per_iteration:.2f}")
    
    # Find shortest vector from best set
    if len(S_best) > 0:
        shortest_vector = min(S_best, key=lambda v: norm(v))
        if verbose:
            print(f"Found shortest vector, norm: {norm(shortest_vector):.4f}")
        return shortest_vector, total_stats
    else:
        # If no vectors left, return shortest from original set
        if len(S) > 0:
            shortest_vector = min(S, key=lambda v: norm(v))
            if verbose:
                print(f"Returning original shortest vector, norm: {norm(shortest_vector):.4f}")
            return shortest_vector, total_stats
        else:
            print("Error: No vectors")
            return None, total_stats

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

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="SVP Challenge Solver - Direct NV Sieve")
    parser.add_argument('--dim', type=int, default=40, help='Dimension')
    parser.add_argument('--seed', type=int, default=0, help='SVP Challenge seed')
    parser.add_argument('--gamma', type=float, default=0.99, help='Sieve parameter gamma')
    parser.add_argument('--max_iter', type=int, default=50, help='Maximum iterations')
    
    args = parser.parse_args()
    
    dim = args.dim
    seed = args.seed
    gamma = args.gamma
    max_iter = args.max_iter
    
    print(f"========== SVP Challenge Solver ==========")
    print(f"Dimension: {dim}")
    print(f"Seed: {seed}")
    print(f"gamma: {gamma}")
    print(f"Maximum iterations: {max_iter}")
    print("=" * 40)
    
    # Read SVP Challenge lattice basis
    start_time = time.time()
    basis = read_svp_challenge_direct(dim, seed)
    
    if basis is None:
        print("Cannot read lattice basis, exiting")
        return
    
    load_time = time.time() - start_time
    print(f"Basis loading time: {load_time:.4f}s")
    print()
    
    # Run direct NV sieve
    start_time = time.time()
    result, iterations = nguyen_vidick_sieve_direct(
        basis, 
        gamma=gamma, 
        max_iterations=max_iter,
        verbose=True
    )
    sieve_time = time.time() - start_time
    
    if result is not None:
        print(f"\n========== Results ==========")
        print(f"Shortest vector norm: {norm(result):.4f}")
        print(f"Iterations: {iterations}")
        print(f"Sieve time: {sieve_time:.4f}s")
        print(f"Total time: {load_time + sieve_time:.4f}s")
        
        # Verify if vector is in lattice (rough verification)
        # Note: Actually need to verify if vector is indeed integer linear combination of basis
        print(f"\nVector first 10 components: {result[:10]}")
        
        # Save result
        result_file = f"svp_result_dim{dim}_seed{seed}.txt"
        np.savetxt(result_file, result)
        print(f"Result saved to: {result_file}")
    else:
        print("No solution found")

if __name__ == "__main__":
    main()