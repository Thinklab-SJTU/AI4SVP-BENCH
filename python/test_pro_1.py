# debug.py
import os
import torch
import numpy as np
import sys
import matplotlib.pyplot as plt
import time
from collections import deque
import warnings
warnings.filterwarnings('ignore')

# Try to import C++ lattice environment
sys.path.append('../lib')
try:
    import lattice_env
    CPP_ENV_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import lattice_env: {e}")
    CPP_ENV_AVAILABLE = False

# Try to import fpylll
try:
    from fpylll import IntegerMatrix, LLL, BKZ, GSO
    from fpylll.algorithms.bkz2 import BKZReduction as BKZ2
    from fpylll import Enumeration, EnumerationError
    from fpylll.tools.bkz_simulator import simulate
    FPLLL_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import fpylll: {e}")
    FPLLL_AVAILABLE = False
    
# Modify fplll lattice creation function to read only first dim rows from SVP challenge file
def create_fplll_lattice_from_svp_challenge(dim, seed=5778):
    """Create a lattice using fpylll from SVP challenge file (read first dim rows)"""
    if not FPLLL_AVAILABLE:
        raise ImportError("fpylll is not available")
    
    try:
        # Read SVP challenge file
        file_name = f"../svp_challenge_list/svp_challenge_{dim}_{seed}.txt"
        
        with open(file_name, 'r') as f:
            # Read all lines
            lines = f.readlines()
        
        # Parse matrix data, only take first dim rows
        matrix_data = []
        row_count = 0
        for line in lines:
            if row_count >= dim:  # Only take first dim rows
                break
            # Remove whitespace and split numbers
            row = [int(x.strip()) for x in line.strip().split()]
            if row:  # Only add non-empty rows
                matrix_data.append(row)
                row_count += 1
        
        # Check if enough rows were read
        if len(matrix_data) < dim:
            print(f"    Warning: SVP challenge file has only {len(matrix_data)} rows, need {dim}")
            # If not enough rows, pad with zeros
            while len(matrix_data) < dim:
                matrix_data.append([0] * dim)
        
        # Create IntegerMatrix
        A = IntegerMatrix(dim, dim)
        
        # Fill matrix
        for i in range(dim):
            for j in range(dim):
                A[i, j] = int(matrix_data[i][j])
        
        print(f"    Created fplll lattice from SVP challenge file, using first {dim} rows")
        return A
        
    except FileNotFoundError:
        print(f"    Error: SVP challenge file {file_name} not found")
        print(f"    Falling back to random lattice for fplll")
        return IntegerMatrix.random(dim, "uniform", bits=30)
    except Exception as e:
        print(f"    Error reading SVP challenge file: {e}")
        return IntegerMatrix.random(dim, "uniform", bits=30)

# Modify fplll test function to use lattice created from SVP challenge file
# Alternative solution using GSO to compute norms
# Modify BKZ 2.0 and Self-dual BKZ test functions
def test_fplll_bkz20(dim, beta=20, seed=5778):
    """Test BKZ 2.0 from fpylll using SVP challenge lattice"""
    if not FPLLL_AVAILABLE:
        return float('inf'), float('inf'), float('inf')
    
    try:
        # Create lattice (from SVP challenge file)
        A = create_fplll_lattice_from_svp_challenge(dim, seed)
        
        # LLL reduce first
        M = GSO.Mat(A)
        L = LLL.Reduction(M)
        L()
        
        # Get initial norm
        initial_norm = np.linalg.norm([float(x) for x in A[0]])
        
        # Start timing
        start_time = time.time()
        
        # Create BKZ 2.0 object and call it
        bkz = BKZ2(A)
        
        # BKZ 2.0 parameters - according to bkz2.py implementation
        params = BKZ.Param(
            block_size=beta,
            strategies=BKZ.DEFAULT_STRATEGY,
            flags=BKZ.VERBOSE,
            max_loops=5,
            rerandomization_density=0  # Rerandomization density
        )
        
        # Directly call BKZReduction object, which will execute BKZ algorithm
        # According to bkz2.py implementation, BKZReduction object is callable
        bkz(params)
        
        end_time = time.time()
        
        # Get final norm
        final_norm = np.linalg.norm([float(x) for x in A[0]])
        
        return end_time - start_time, final_norm, initial_norm
        
    except Exception as e:
        print(f"    Error in fplll BKZ 2.0: {e}")
        import traceback
        traceback.print_exc()
        return float('inf'), float('inf'), float('inf')

def test_fplll_self_dual_bkz(dim, beta=20, seed=5778):
    """Test Self-Dual BKZ from fpylll using SVP challenge lattice"""
    if not FPLLL_AVAILABLE:
        return float('inf'), float('inf'), float('inf')
    
    try:
        # Create lattice (from SVP challenge file)
        A = create_fplll_lattice_from_svp_challenge(dim, seed)
        
        # LLL reduce first
        M = GSO.Mat(A)
        L = LLL.Reduction(M)
        L()
        
        # Get initial norm
        initial_norm = np.linalg.norm([float(x) for x in A[0]])
        
        # Start timing
        start_time = time.time()
        
        # For Self-Dual BKZ, use fpylll's BKZ class with SD flag
        # According to fpylll documentation, Self-Dual BKZ uses BKZ.SD flag
        params = BKZ.Param(
            block_size=beta,
            strategies=BKZ.DEFAULT_STRATEGY,
            flags=BKZ.VERBOSE | BKZ.SD,  # Use SD flag for Self-Dual BKZ
            max_loops=5,
            auto_abort=True
        )
        
        # Use BKZ.reduction method to execute Self-Dual BKZ
        BKZ.reduction(A, params)
        
        end_time = time.time()
        
        # Get final norm
        final_norm = np.linalg.norm([float(x) for x in A[0]])
        
        return end_time - start_time, final_norm, initial_norm
        
    except Exception as e:
        print(f"    Error in fplll self-dual BKZ: {e}")
        import traceback
        traceback.print_exc()
        
        # If BKZ.SD is not available, try alternative method
        # Some versions of fpylll may use different flags
        print(f"    Trying alternative method for self-dual BKZ...")
        try:
            # Try using GH_BND flag as alternative
            A = create_fplll_lattice_from_svp_challenge(dim, seed)
            M = GSO.Mat(A)
            L = LLL.Reduction(M)
            L()
            initial_norm = np.linalg.norm([float(x) for x in A[0]])
            
            start_time = time.time()
            
            # Use BKZ class high-level interface
            from fpylll import BKZ
            param = BKZ.EasyParam(
                block_size=beta,
                flags=BKZ.GH_BND,  # Use GH_BND flag
                max_loops=5
            )
            
            # Execute BKZ reduction
            BKZ.reduction(A, param)
            
            end_time = time.time()
            final_norm = np.linalg.norm([float(x) for x in A[0]])
            
            return end_time - start_time, final_norm, initial_norm
            
        except Exception as e2:
            print(f"    Alternative method also failed: {e2}")
            return float('inf'), float('inf'), float('inf')

# In test_single_dimension function, modify fplll algorithm beta parameters to match other algorithms
def test_single_dimension(dim, seed=5778, beta=5, enum_radius=4000000):
    """
    Test performance of all algorithms on a single dimension
    Returns: (running_time_dict, b1_norm_dict)
    """
    if not CPP_ENV_AVAILABLE:
        raise ImportError("C++ lattice environment not available")
    
    # Define algorithms and their parameters
    algorithms = {
        'LLL': ('LLL', {'delta': 0.752}),
        'BKZ': ('BKZ', {'beta': 26, 'delta': 0.98458}),
        'L2': ('L2', {'delta': 0.981, 'eta': 0.746}),
        'deepLLL': ('deepLLL', {'delta': 0.99}),
        'potLLL': ('potLLL', {'delta': 0.99}),
        'dualLLL': ('dualLLL', {'delta': 0.970}),
        'dualDeepLLL': ('dualDeepLLL', {'delta': 0.99}),
        'dualPotLLL': ('dualPotLLL', {'delta': 0.99}),
        'dualBKZ': ('dualBKZ', {'beta': 17, 'delta': 0.685}),
        'deepBKZ': ('deepBKZ', {'beta': 2, 'delta': 0.99}),
        'dualDeepBKZ': ('dualDeepBKZ', {'beta': 2, 'delta': 0.99}),
        'potBKZ': ('potBKZ', {'beta': 13, 'delta': 0.99253}),
    }
    
    # Store results
    times_dict = {}
    norms_dict = {}
    
    # Test each reduction algorithm
    for algo_name, (func_name, kwargs) in algorithms.items():
        print(f"  Testing {algo_name} on dimension {dim}...")
        
        # Create new lattice instance
        lat = lattice_env.create_lattice_int(dim, dim)
        lat.setSVPChallenge(dim, seed);
        lat.computeGSO()
        
        # Record initial norm
        initial_norm = lat.b1Norm()
        
        # Execute algorithm and time it
        start_time = time.time()
        
        # Dynamically call algorithm
        func = getattr(lat, func_name)
        try:
            # Pass different parameters based on algorithm
            if func_name in ['BKZ', 'dualBKZ', 'deepBKZ', 'dualDeepBKZ', 'potBKZ']:
                result = func(kwargs['beta'], kwargs['delta'])
            elif func_name == 'L2':
                result = func(kwargs['delta'], kwargs['eta'])
            else:
                result = func(kwargs['delta'])
        except Exception as e:
            print(f"    Error running {algo_name}: {e}")
            times_dict[algo_name] = float('inf')
            norms_dict[algo_name] = float('inf')
            continue
        
        end_time = time.time()
        
        # Get norm after reduction
        final_norm = lat.b1Norm()
        
        # Store results
        times_dict[algo_name] = end_time - start_time
        norms_dict[algo_name] = final_norm
        
        print(f"    Time: {times_dict[algo_name]:.4f}s, b1_norm: {initial_norm:.2f} -> {final_norm:.2f}")
    
    # Test fplll algorithms
    if FPLLL_AVAILABLE:
        # Test BKZ 2.0 - use same beta as other algorithms
        fplll_beta = beta  # Use passed beta parameter
        
        print(f"  Testing fplll_BKZ2.0 on dimension {dim}...")
        try:
            time_taken, final_norm, initial_norm = test_fplll_bkz20(dim, fplll_beta, seed)
            times_dict['fplll_BKZ2.0'] = time_taken
            norms_dict['fplll_BKZ2.0'] = final_norm
            print(f"    Time: {time_taken:.4f}s, b1_norm: {initial_norm:.2f} -> {final_norm:.2f}")
        except Exception as e:
            print(f"    Error testing fplll_BKZ2.0: {e}")
            times_dict['fplll_BKZ2.0'] = float('inf')
            norms_dict['fplll_BKZ2.0'] = float('inf')
        
        # Test self-dual BKZ
        print(f"  Testing fplll_self_dual_BKZ on dimension {dim}...")
        try:
            time_taken, final_norm, initial_norm = test_fplll_self_dual_bkz(dim, fplll_beta, seed)
            times_dict['fplll_self_dual_BKZ'] = time_taken
            norms_dict['fplll_self_dual_BKZ'] = final_norm
            print(f"    Time: {time_taken:.4f}s, b1_norm: {initial_norm:.2f} -> {final_norm:.2f}")
        except Exception as e:
            print(f"    Error testing fplll_self_dual_BKZ: {e}")
            times_dict['fplll_self_dual_BKZ'] = float('inf')
            norms_dict['fplll_self_dual_BKZ'] = float('inf')
    
    return times_dict, norms_dict


def run_comprehensive_test(dimensions, seed=5778, beta=5, enum_radius=4000000):
    """
    Run comprehensive test covering multiple dimensions and all algorithms
    """
    if not CPP_ENV_AVAILABLE:
        print("Error: C++ lattice environment not available")
        return None, None
    
    # Define all algorithm names (including ENUM)
    all_algorithms = [
        'LLL', 'BKZ', 'L2', 'deepLLL', 'potLLL',
        'dualLLL', 'dualDeepLLL', 'dualPotLLL', 'dualBKZ',
        'deepBKZ', 'dualDeepBKZ', 'potBKZ'#, 'ENUM', 'HKZ'
        #'L2'
    ]
    
    # Add fplll algorithms if available
    if FPLLL_AVAILABLE:
        all_algorithms.extend(['fplll_BKZ2.0', 'fplll_self_dual_BKZ'])
    
    # Initialize result storage structures
    all_times = {algo: [] for algo in all_algorithms}
    all_norms = {algo: [] for algo in all_algorithms}
    dimensions_list = []
    
    # Test each dimension
    for dim in dimensions:
        print(f"\n{'='*60}")
        print(f"Testing dimension: {dim}")
        print(f"{'='*60}")
        
        try:
            times_dict, norms_dict = test_single_dimension(dim, seed, beta, enum_radius)
            
            # Collect results
            dimensions_list.append(dim)
            for algo in all_algorithms:
                all_times[algo].append(times_dict.get(algo, float('inf')))
                all_norms[algo].append(norms_dict.get(algo, float('inf')))
                
        except Exception as e:
            print(f"Error testing dimension {dim}: {e}")
            continue
    
    return dimensions_list, all_times, all_norms

def plot_results(dimensions, all_times, all_norms, output_dir="./results"):
    """
    Plot result graphs
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Define algorithm groups (for distinguishing different curves)
    algorithm_groups = {
        'LLL Variants': ['LLL', 'deepLLL', 'potLLL'],
        'Dual LLL Variants': ['dualLLL', 'dualDeepLLL', 'dualPotLLL'],
        'BKZ Variants': ['BKZ', 'deepBKZ', 'potBKZ'],
        'Dual BKZ Variants': ['dualBKZ', 'dualDeepBKZ'],
        'Other': ['L2']#, 'HKZ','ENUM']
    }
    
    # Add fplll algorithms to a separate group
    if 'fplll_BKZ2.0' in all_times:
        algorithm_groups['fplll Algorithms'] = ['fplll_BKZ2.0', 'fplll_self_dual_BKZ']
    
    # Define colors and line styles for different groups
    colors = plt.cm.tab20(np.linspace(0, 1, len(algorithm_groups)))
    line_styles = ['-', '--', '-.', ':', '-', '--']
    
    # Figure 1: Running time comparison
    plt.figure(figsize=(16, 10))
    
    for i, (group_name, algorithms) in enumerate(algorithm_groups.items()):
        for j, algo in enumerate(algorithms):
            if algo in all_times and len(all_times[algo]) == len(dimensions):
                # Filter out infinite values
                valid_times = [t if t != float('inf') else None for t in all_times[algo]]
                valid_dims = [d for d, t in zip(dimensions, valid_times) if t is not None]
                valid_times = [t for t in valid_times if t is not None]
                
                if valid_times:
                    # Use same color but different line style for algorithms in same group
                    line_style = line_styles[j % len(line_styles)]
                    marker = 'o' if 'fplll' not in algo else 's'
                    linestyle = line_style if 'fplll' not in algo else '--'
                    linewidth = 2 if 'fplll' not in algo else 3
                    
                    plt.plot(valid_dims, valid_times, 
                            label=f'{algo} ({group_name})',
                            color=colors[i],
                            linestyle=linestyle,
                            linewidth=linewidth,
                            marker=marker)
    
    plt.xlabel('Dimension', fontsize=14)
    plt.ylabel('Time (seconds)', fontsize=14)
    plt.title('Running Time Comparison of Lattice Reduction Algorithms', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # Use log scale because time differences can be large
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'time_comparison.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'time_comparison.pdf'), bbox_inches='tight')
    plt.show()
    
    # Figure 2: b1_norm comparison
    plt.figure(figsize=(16, 10))
    
    for i, (group_name, algorithms) in enumerate(algorithm_groups.items()):
        for j, algo in enumerate(algorithms):
            if algo in all_norms and len(all_norms[algo]) == len(dimensions):
                # Filter out infinite values
                valid_norms = [n if n != float('inf') else None for n in all_norms[algo]]
                valid_dims = [d for d, n in zip(dimensions, valid_norms) if n is not None]
                valid_norms = [n for n in valid_norms if n is not None]
                
                if valid_norms:
                    line_style = line_styles[j % len(line_styles)]
                    marker = 'o' if 'fplll' not in algo else 's'
                    linestyle = line_style if 'fplll' not in algo else '--'
                    linewidth = 2 if 'fplll' not in algo else 3
                    
                    plt.plot(valid_dims, valid_norms,
                            label=f'{algo} ({group_name})',
                            color=colors[i],
                            linestyle=linestyle,
                            linewidth=linewidth,
                            marker=marker)
    
    plt.xlabel('Dimension', fontsize=14)
    plt.ylabel('b1_norm (log scale)', fontsize=14)
    plt.title('Reduction Quality Comparison (b1_norm)', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # Use log scale
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'norm_comparison.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'norm_comparison.pdf'), bbox_inches='tight')
    plt.show()
    
    # Figure 3: Show only best algorithms (quality-wise - smallest b1_norm)
    plt.figure(figsize=(14, 8))
    
    # Find algorithm with smallest b1_norm for each dimension
    best_norms = []
    best_algorithms = []
    
    for idx, dim in enumerate(dimensions):
        min_norm = float('inf')
        best_algo = None
        
        for algo in all_norms:
            if idx < len(all_norms[algo]) and all_norms[algo][idx] < min_norm:
                min_norm = all_norms[algo][idx]
                best_algo = algo
        
        if best_algo and min_norm != float('inf'):
            best_norms.append(min_norm)
            best_algorithms.append(best_algo)
        else:
            best_norms.append(None)
            best_algorithms.append(None)
    
    # Plot best algorithm norms
    valid_indices = [i for i, n in enumerate(best_norms) if n is not None]
    valid_dims = [dimensions[i] for i in valid_indices]
    valid_norms = [best_norms[i] for i in valid_indices]
    valid_algos = [best_algorithms[i] for i in valid_indices]
    
    plt.plot(valid_dims, valid_norms, 'g-', linewidth=2, marker='o', label='Best Algorithm (Quality)')
    
    # Add algorithm name labels
    for dim, norm_val, algo in zip(valid_dims, valid_norms, valid_algos):
        plt.annotate(algo, (dim, norm_val), 
                    textcoords="offset points", 
                    xytext=(0,10), 
                    ha='center',
                    fontsize=8)
    
    plt.xlabel('Dimension', fontsize=14)
    plt.ylabel('b1_norm (log scale)', fontsize=14)
    plt.title('Best Performing Algorithm by Dimension (Quality - Smallest b1_norm)', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'best_algorithm_norm.png'), dpi=300)
    plt.show()
    
    # Figure 4: Comparison between custom BKZ and fplll BKZ variants
    if 'fplll_BKZ2.0' in all_times and 'BKZ' in all_times:
        plt.figure(figsize=(12, 8))
        
        # Plot BKZ variants
        bkz_variants = ['BKZ', 'deepBKZ', 'potBKZ', 'fplll_BKZ2.0', 'fplll_self_dual_BKZ']
        colors_bkz = plt.cm.Set1(np.linspace(0, 1, len(bkz_variants)))
        
        for idx, algo in enumerate(bkz_variants):
            if algo in all_times and len(all_times[algo]) == len(dimensions):
                valid_times = [t if t != float('inf') else None for t in all_times[algo]]
                valid_dims = [d for d, t in zip(dimensions, valid_times) if t is not None]
                valid_times = [t for t in valid_times if t is not None]
                
                if valid_times:
                    marker = 'o' if 'fplll' not in algo else 's'
                    linestyle = '-' if 'fplll' not in algo else '--'
                    label = f'fplll: {algo}' if 'fplll' in algo else algo
                    
                    plt.plot(valid_dims, valid_times,
                            label=label,
                            color=colors_bkz[idx],
                            linestyle=linestyle,
                            linewidth=2,
                            marker=marker)
        
        plt.xlabel('Dimension', fontsize=14)
        plt.ylabel('Time (seconds, log scale)', fontsize=14)
        plt.title('BKZ Variants Comparison (Custom vs fplll)', fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'bkz_variants_comparison.png'), dpi=300)
        plt.show()
    
    # Save data to file
    import json
    data = {
        'dimensions': dimensions,
        'times': all_times,
        'norms': all_norms
    }
    
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(data, f, indent=2, default=str)
    
    print(f"\nResults saved to {output_dir}/")

def main():
    """Main function"""
    if not CPP_ENV_AVAILABLE:
        print("C++ lattice environment not available. Exiting.")
        return
    
    # Print library availability
    print("="*70)
    print("LIBRARY AVAILABILITY CHECK")
    print("="*70)
    print(f"C++ lattice_env: {'? Available' if CPP_ENV_AVAILABLE else '? Not Available'}")
    print(f"fpylll: {'? Available' if FPLLL_AVAILABLE else '? Not Available'}")
    print("="*70)
    
    # Set test parameters
    beta = 40  # Use smaller beta for testing
    seed = 8
    enum_radius = 4000000
    
    # Define dimensions to test
    # Note: Start with lower dimensions
    test_dimensions = list(range(40, 101, 10))  # [40, 60, 80]
    
    # If want to test more dimensions, adjust accordingly
    # test_dimensions = [30, 40, 50, 60, 70, 80]
    
    print("\n" + "="*70)
    print("COMPREHENSIVE LATTICE REDUCTION ALGORITHM COMPARISON")
    print("="*70)
    print(f"Parameters: beta={beta}, seed={seed}")
    print(f"All algorithms using SAME SVP challenge lattice")
    print(f"Dimensions to test: {test_dimensions}")
    print("="*70)
    
    # Run tests
    dimensions, all_times, all_norms = run_comprehensive_test(
        test_dimensions, seed, beta, enum_radius
    )
    
    if dimensions:
        # Plot results
        plot_results(dimensions, all_times, all_norms)
        
        # Print summary table
        print("\n" + "="*70)
        print("SUMMARY RESULTS")
        print("="*70)
        
        # Create summary table
        headers = ["Dim"] + list(all_times.keys())
        
        # Print time table
        print("\nRunning Times (seconds):")
        print("-" * (12 * len(headers)))
        print("".join(f"{h:>12}" for h in headers))
        print("-" * (12 * len(headers)))
        
        for idx, dim in enumerate(dimensions):
            row = [f"{dim}"]
            for algo in all_times.keys():
                if idx < len(all_times[algo]):
                    if all_times[algo][idx] == float('inf'):
                        row.append("INF")
                    else:
                        row.append(f"{all_times[algo][idx]:.4f}")
                else:
                    row.append("N/A")
            print("".join(f"{cell:>12}" for cell in row))
        
        # Print b1_norm table
        print("\n\nb1_norms:")
        print("-" * (12 * len(headers)))
        print("".join(f"{h:>12}" for h in headers))
        print("-" * (12 * len(headers)))
        
        for idx, dim in enumerate(dimensions):
            row = [f"{dim}"]
            for algo in all_norms.keys():
                if idx < len(all_norms[algo]):
                    if all_norms[algo][idx] == float('inf'):
                        row.append("INF")
                    else:
                        row.append(f"{all_norms[algo][idx]:.2f}")
                else:
                    row.append("N/A")
            print("".join(f"{cell:>12}" for cell in row))
    else:
        print("No valid test results obtained.")

if __name__ == "__main__":
    main()