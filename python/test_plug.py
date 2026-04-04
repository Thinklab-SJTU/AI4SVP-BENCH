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

def test_single_dimension(dim, seed=5778, beta=5, enum_radius=4000000):
    """
    Test performance of all algorithms on a single dimension
    Returns: (running_time_dict, b1_norm_dict)
    """
    if not CPP_ENV_AVAILABLE:
        raise ImportError("C++ lattice environment not available")
    
    # Define algorithms to test and their parameters
    algorithms = {
        'LLL': ('LLL', {'delta': 0.752}),
        'BKZ': ('BKZ', {'beta': 26, 'delta': 0.98458}),
        #'HKZ': ('HKZ', {'delta': 0.99}),
        'L2': ('L2', {'delta': 0.981, 'eta': 0.746}),
        'deepLLL': ('deepLLL', {'delta': 0.99}),
        'potLLL': ('potLLL', {'delta': 0.99}),
        'dualLLL': ('dualLLL', {'delta': 0.970}),
        'dualDeepLLL': ('dualDeepLLL', {'delta': 0.99}),
        'dualPotLLL': ('dualPotLLL', {'delta': 0.99}),
        'dualBKZ': ('dualBKZ', {'beta': 17, 'delta': 0.685}),
        'deepBKZ': ('deepBKZ', {'beta': beta, 'delta': 0.99}),
        'dualDeepBKZ': ('dualDeepBKZ', {'beta': beta, 'delta': 0.99}),
        'potBKZ': ('potBKZ', {'beta': 13, 'delta': 0.99253}),
    }
    
    # ENUM algorithm handled separately (does not modify lattice)
    enum_algorithms = {
        'ENUM': ('ENUM', {'R': enum_radius})
    }
    
    # Store results
    times_dict = {}
    norms_dict = {}
    
    # Test each reduction algorithm (modifies lattice)
    for algo_name, (func_name, kwargs) in algorithms.items():
        print(f"  Testing {algo_name} on dimension {dim}...")
        
        # Create new lattice instance
        lat = lattice_env.create_lattice_int(dim, dim)
        lat.setSVPChallenge(dim, seed);
        #lat.setRandom(dim, dim, 100000, 1000000)
        lat.computeGSO()
        
        # Record initial b1_norm
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
        
        # Get b1_norm after reduction
        final_norm = lat.b1Norm()
        
        # Store results
        times_dict[algo_name] = end_time - start_time
        norms_dict[algo_name] = final_norm
        
        print(f"    Time: {times_dict[algo_name]:.4f}s, b1_norm: {initial_norm:.2f} -> {final_norm:.2f}")
    
    # Test ENUM algorithm (does not modify lattice, needs separate handling)
    '''for algo_name, (func_name, kwargs) in enum_algorithms.items():
        print(f"  Testing {algo_name} on dimension {dim}...")
        
        # Create new lattice instance
        lat = lattice_env.create_lattice_int(dim, dim)
        lat.setSVPChallenge(dim, seed)
        #lat.setRandom(dim, dim, 100000, 1000000)
        lat.computeGSO()
        
        # Record initial b1_norm
        initial_norm = lat.b1Norm()
        
        # Execute ENUM algorithm and time it
        start_time = time.time()
        try:
            coeff_vector = lat.ENUM(kwargs['R'])
            v = lat.mulVecBasis(coeff_vector)
            final_norm = np.linalg.norm(v)
        except Exception as e:
            print(f"    Error running {algo_name}: {e}")
            times_dict[algo_name] = float('inf')
            norms_dict[algo_name] = float('inf')
            continue
        
        end_time = time.time()
        
        # Store results
        times_dict[algo_name] = end_time - start_time
        norms_dict[algo_name] = final_norm
        
        print(f"    Time: {times_dict[algo_name]:.4f}s, Found norm: {final_norm:.2f} (initial b1: {initial_norm:.2f})")'''
    
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
    ]
    
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
    
    # Define colors and line styles for different groups
    colors = plt.cm.tab20(np.linspace(0, 1, len(algorithm_groups)))
    line_styles = ['-', '--', '-.', ':', '-']
    
    # Figure 1: Running time comparison
    plt.figure(figsize=(14, 8))
    
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
                    plt.plot(valid_dims, valid_times, 
                            label=f'{algo} ({group_name})',
                            color=colors[i],
                            linestyle=line_style,
                            linewidth=2,
                            marker='o' if j == 0 else 's')
    
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
    plt.figure(figsize=(14, 8))
    
    for i, (group_name, algorithms) in enumerate(algorithm_groups.items()):
        for j, algo in enumerate(algorithms):
            if algo in all_norms and len(all_norms[algo]) == len(dimensions):
                # Filter out infinite values
                valid_norms = [n if n != float('inf') else None for n in all_norms[algo]]
                valid_dims = [d for d, n in zip(dimensions, valid_norms) if n is not None]
                valid_norms = [n for n in valid_norms if n is not None]
                
                if valid_norms:
                    line_style = line_styles[j % len(line_styles)]
                    plt.plot(valid_dims, valid_norms,
                            label=f'{algo} ({group_name})',
                            color=colors[i],
                            linestyle=line_style,
                            linewidth=2,
                            marker='o' if j == 0 else 's')
    
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
    
    # Figure 3: Show only best algorithms (time-wise)
    plt.figure(figsize=(12, 6))
    
    # Find algorithm with shortest time for each dimension
    best_times = []
    best_algorithms = []
    
    for idx, dim in enumerate(dimensions):
        min_time = float('inf')
        best_algo = None
        
        for algo in all_times:
            if idx < len(all_times[algo]) and all_times[algo][idx] < min_time:
                min_time = all_times[algo][idx]
                best_algo = algo
        
        if best_algo and min_time != float('inf'):
            best_times.append(min_time)
            best_algorithms.append(best_algo)
        else:
            best_times.append(None)
            best_algorithms.append(None)
    
    # Plot best algorithm times
    valid_indices = [i for i, t in enumerate(best_times) if t is not None]
    valid_dims = [dimensions[i] for i in valid_indices]
    valid_times = [best_times[i] for i in valid_indices]
    valid_algos = [best_algorithms[i] for i in valid_indices]
    
    plt.plot(valid_dims, valid_times, 'b-', linewidth=2, marker='o', label='Best Algorithm')
    
    # Add algorithm name labels
    for dim, time_val, algo in zip(valid_dims, valid_times, valid_algos):
        plt.annotate(algo, (dim, time_val), 
                    textcoords="offset points", 
                    xytext=(0,10), 
                    ha='center',
                    fontsize=8)
    
    plt.xlabel('Dimension', fontsize=14)
    plt.ylabel('Time (seconds, log scale)', fontsize=14)
    plt.title('Best Performing Algorithm by Dimension (Time)', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'best_algorithm_time.png'), dpi=300)
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
    
    # Set test parameters
    beta = 2
    seed = 0
    enum_radius = 4000000
    
    # Define dimensions to test
    # Note: ENUM algorithm is very slow at high dimensions, recommend starting with lower dimensions
    test_dimensions = list(range(120, 121, 10))  # [10, 15, 20, 25, 30, 35, 40]
    
    # If want to test more dimensions, adjust accordingly
    # test_dimensions = [10, 20, 30, 40]
    
    print("="*70)
    print("COMPREHENSIVE LATTICE REDUCTION ALGORITHM COMPARISON")
    print("="*70)
    print(f"Parameters: beta={beta}, seed={seed}, ENUM radius={enum_radius}")
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
        print("-" * (10 * len(headers)))
        print("".join(f"{h:>10}" for h in headers))
        print("-" * (10 * len(headers)))
        
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
            print("".join(f"{cell:>10}" for cell in row))
        
        # Print b1_norm table
        print("\n\nb1_norms:")
        print("-" * (10 * len(headers)))
        print("".join(f"{h:>10}" for h in headers))
        print("-" * (10 * len(headers)))
        
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
            print("".join(f"{cell:>10}" for cell in row))
    else:
        print("No valid test results obtained.")

if __name__ == "__main__":
    main()