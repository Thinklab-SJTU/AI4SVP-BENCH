import numpy as np
import matplotlib.pyplot as plt
import os
plt.rcParams.update({
    'font.size': 40,           # overall
    'axes.titlesize': 40,      # toplable
    'axes.labelsize': 40,      # lable dim time
    'legend.fontsize': 40,     # parameter
    'xtick.labelsize': 18,     # X
    'ytick.labelsize': 18      # Y
})
def plot_from_table_data():
    """
    Plot runtime and output vector norm charts based on table data
    """
    # Dimension list
    dimensions = [40, 50, 60, 70, 80, 90, 100, 110, 120]
    
    # Data extracted from table
    # Format: {algorithm_name: {'norm': [norm values per dimension], 'time': [time values per dimension]}}
    data = {
        'LLL': {
            'norm': [1881.01, 2075.02, 3331.46, 4358.57, 4571.58, 5885.45, 6816.38, 9949.12, 11334.7],
            'time': [0.0013, 0.0024, 0.0039, 0.0061, 0.0086, 0.0121, 0.0164, 0.0206, 0.0254]
        },
        'deepLLL': {
            'norm': [1709.17, 1908.04, 2147.38, 2240.45, 2606.00, 2875.76, 3074.66, 3268.57, 3588.86],
            'time': [0.0174, 0.1279, 0.9277, 5.3400, 16.044, 64.576, 271.26, 1350.9, 5902.5]
        },
        'potLLL': {
            'norm': [1709.17, 2009.50, 2409.29, 2721.72, 3042.10, 3785.94, 4804.52, 5337.36, 5354.19],
            'time': [0.0030, 0.0157, 0.0421, 0.1270, 0.2351, 0.3874, 0.8370, 1.3471, 2.0565]
        },
        'dualLLL': {
            'norm': [1881.01, 2075.02, 3331.46, 4358.57, 4571.58, 5885.45, 6816.38, 9949.12, 11334.7],
            'time': [0.0017, 0.0030, 0.0051, 0.0079, 0.0118, 0.0167, 0.0222, 0.0287, 0.0315]
        },
        'dualDeepLLL': {
            'norm': [1745.16, 2110.14, 2177.29, 2537.94, 3002.95, 2980.71, 3816.65, 3780.47, 4551.68],
            'time': [0.0711, 0.8272, 3.1091, 31.368, 91.623, 335.53, 1898.6, 6362.1, 17134]
        },
        'dualPotLLL': {
            'norm': [1745.16, 2075.02, 2652.77, 2741.96, 3451.04, 4237.74, 4602.23, 5352.31, 6070.34],
            'time': [0.0124, 0.0431, 0.1485, 0.4424, 0.5716, 1.1706, 2.4013, 3.7827, 5.7534]
        },
        'L2': {
            'norm': [1881.01, 2075.02, 3331.46, 4358.57, 4571.58, 5885.45, 6816.38, 9949.12, 11334.7],
            'time': [0.0008, 0.0014, 0.0031, 0.0032, 0.0052, 0.0460, 0.0140, 0.0900, 0.0584]
        },
        'HKZ': {
            'norm': [1709.17, 1893.17, 2018.35, 2235.41, 2478.92, None, None, None, None],
            'time': [2.2412, 1109.7, 28437, 146897, 462733, None, None, None, None]
        },
        'BKZ': {
            'norm': [1709.17, 1931.44, 2061.37, 2392.98, 2673.44, 3079.18, 3430.77, 3691.31, 4273.73],
            'time': [0.1253, 1.0679, 5.7524, 17.023, 82.258, 164.66, 135.60, 1203.1, 1054.2]
        },
        'BKZ 2.0': {
            'norm': [1709.17, 1908.04, 2149.89, 2235.24, 2765.82, 3189.11, 3573.98, 3702.21, 4203.52],
            'time': [0.0826, 0.1290, 0.3034, 0.4942, 0.7598, 1.0863, 1.5957, 2.1996, 2.9410]
        },
        'deepBKZ': {
            'norm': [1709.17, 1908.04, 2147.38, 2240.45, 2606.00, 2875.76, 3074.66, 3268.57, 3588.86],
            'time': [0.0174, 0.1266, 0.9412, 5.3416, 15.669, 64.066, 270.79, 1334.3, 5968.2]
        },
        'potBKZ': {
            'norm': [1709.17, 2009.51, 2556.34, 2847.97, 3406.28, 3770.36, 4110.43, 5055.98, 5689.92],
            'time': [0.0176, 0.0354, 0.0637, 0.1836, 0.2441, 0.4797, 0.7952, 1.5238, 2.0573]
        },
        'dualBKZ': {
            'norm': [1881.01, 2075.02, 3331.46, 4358.57, 4571.58, 5885.45, 6816.38, 9949.12, 11334.7],
            'time': [0.0018, 0.0032, 0.0056, 0.0086, 0.0128, 0.0175, 0.0251, 0.0900, 0.0158]
        },
        'selfdualBKZ': {
            'norm': [1709.17, 1893.17, 2171.24, 2516.03, 2811.44, 3085.02, 3955.59, 4739.25, 5672.78],
            'time': [0.0631, 0.1164, 0.1720, 0.2484, 0.3260, 0.4669, 0.5773, 0.7578, 0.9614]
        },
        'dualDeepBKZ': {
            'norm': [1745.16, 2110.14, 2177.29, 2537.94, 3002.95, 2980.71, 3816.65, 3780.47, 4551.68],
            'time': [0.0715, 0.8308, 3.1677, 31.458, 91.962, 336.03, 1863.68, 6368.2, 15438]
        },
        'Enum': {
            'norm': [1709.17, 1893.17, 1995.80, 2208.15, 2445.00, None, None, None, None],
            'time': [0.3659, 331.59, 9682, 75691, 285232, None, None, None, None]
        },
        'Nguyen-Vidick sieve': {
            'norm': [1745.16, 2075.02, 2600.59, 3417.64, 4116.71, 5123.12, 6366.40, 8270.25, 10387.8],
            'time': [0.0031, 0.0066, 0.0052, 0.0063, 0.0076, 0.0095, 0.0632, 0.0206, 0.0253]
        },
        'Double Sieve': {
            'norm': [1881.01, 2075.02, 3078.39, 4358.57, 4435.80, 5188.24, 7226.05, 8672.47, 11020.1],
            'time': [0.0050, 0.0107, 0.0183, 0.01964, 0.0223, 0.0247, 0.0339, 0.0356, 0.0425]
        },
        'Gauss sieve': {
            'norm': [1745.16, 2075.02, 2600.59, 3417.64, 4116.71, 5123.12, 6366.40, 8270.25, 10387.8],
            'time': [0.0166, 0.0219, 0.0341, 0.0582, 0.0621, 0.0845, 0.0954, 0.3568, 0.2947]
        },
        'G6K': {
            'norm': [1702.51, 1893.22, 1943.41, 2168.73, 2391.95, 2535.04, 2661.28, 2738.13, 2881.41],
            'time': [0.3303, 1.2528, 1.2497, 3.3455, 5.5114, 20.582, 51.684, 112.83, 319.27]
        }
    }
    
    # Create output directory
    output_dir = "./table_plots"
    os.makedirs(output_dir, exist_ok=True)
    
    # Algorithm groups for easier color and line style assignment
    algorithm_groups = {
        'LLL Variants': ['LLL', 'deepLLL', 'potLLL'],
        'Dual LLL Variants': ['dualLLL', 'dualDeepLLL', 'dualPotLLL'],
        'BKZ Variants': ['BKZ', 'BKZ 2.0', 'deepBKZ', 'potBKZ', 'selfdualBKZ'],
        'Dual BKZ Variants': ['dualBKZ', 'dualDeepBKZ'],
        'Sieve Methods': ['Nguyen-Vidick sieve', 'Double Sieve', 'Gauss sieve'],
        'Others': ['L2', 'HKZ', 'Enum', 'G6K']
    }
    
    # Define color scheme
    colors = plt.cm.tab20(np.linspace(0, 1, len(algorithm_groups)))
    
    # Define line styles
    line_styles = ['-', '--', '-.', ':', '-', '--']
    
    # 1. Plot runtime chart
    plt.figure(figsize=(14, 9))
    
    # Assign colors to each algorithm group
    for i, (group_name, algorithms) in enumerate(algorithm_groups.items()):
        color = colors[i]
        
        # Assign different line styles to each algorithm within group
        for j, algo in enumerate(algorithms):
            if algo=='Enum' or algo=='HKZ':
                break
            if algo in data:
                times = data[algo]['time']
                # Handle None values
                valid_times = []
                valid_dims = []
                
                for idx, t in enumerate(times):
                    if t is not None:
                        valid_times.append(t)
                        valid_dims.append(dimensions[idx])
                
                if valid_times:
                    line_style = line_styles[j % len(line_styles)]
                    
                    # Show group name for first algorithm in group
                    if j == 0:
                        label = f'{group_name}:\n{algo}'
                    else:
                        label = algo
                    
                    plt.plot(valid_dims, valid_times,
                            label=label,
                            color=color,
                            linestyle=line_style,
                            linewidth=2.5,
                            marker='o' if 'sieve' not in algo else 's',
                            markersize=6)
    
    plt.xlabel('Dimension (d)', fontsize=25)
    plt.ylabel('Time (seconds)', fontsize=25)
    plt.title('Solving Time Comparison of Classical Lattice Solvers', fontsize=25, pad=20)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # Log scale
    
    # Place legend inside top-left corner of chart
    plt.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98), 
               fontsize=10, fancybox=True, framealpha=0.9, 
               ncol=2, borderaxespad=0.5, handlelength=2.5)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(output_dir, 'time_comparison_from_table.png'), 
                dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'time_comparison_from_table.pdf'), 
                bbox_inches='tight')
    plt.show()
    
    # 2. Plot output vector norm chart
    plt.figure(figsize=(14, 9))
    
    for i, (group_name, algorithms) in enumerate(algorithm_groups.items()):
        color = colors[i]
        
        for j, algo in enumerate(algorithms):
            if algo in data:
                norms = data[algo]['norm']
                # Handle None values
                valid_norms = []
                valid_dims = []
                
                for idx, n in enumerate(norms):
                    if n is not None:
                        valid_norms.append(n)
                        valid_dims.append(dimensions[idx])
                
                if valid_norms:
                    line_style = line_styles[j % len(line_styles)]
                    
                    if j == 0:
                        label = f'{group_name}:\n{algo}'
                    else:
                        label = algo
                    
                    plt.plot(valid_dims, valid_norms,
                            label=label,
                            color=color,
                            linestyle=line_style,
                            linewidth=2.5,
                            marker='o' if 'sieve' not in algo else 's',
                            markersize=6)
    
    plt.xlabel('Dimension (d)', fontsize=25)
    plt.ylabel('Output Vector Norm', fontsize=25)
    plt.title('Output Vector Norm Comparison of Classical Lattice Solvers', fontsize=25, pad=20)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # Log scale
    
    # Place legend inside top-left corner of chart
    plt.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98), 
               fontsize=10, fancybox=True, framealpha=0.9, 
               ncol=2, borderaxespad=0.5, handlelength=2.5)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(output_dir, 'norm_comparison_from_table.png'), 
                dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'norm_comparison_from_table.pdf'), 
                bbox_inches='tight')
    plt.show()
    
    # 3. Additional plot: Show only best performing algorithms (smallest norm)
    plt.figure(figsize=(12, 8))
    
    # Find algorithm with smallest norm for each dimension
    best_norms_per_dim = {}
    best_algo_per_dim = {}
    
    for dim_idx, dim in enumerate(dimensions):
        min_norm = float('inf')
        best_algo = None
        
        for algo in data:
            if (dim_idx < len(data[algo]['norm']) and 
                data[algo]['norm'][dim_idx] is not None and
                data[algo]['norm'][dim_idx] < min_norm):
                min_norm = data[algo]['norm'][dim_idx]
                best_algo = algo
        
        if best_algo:
            best_norms_per_dim[dim] = min_norm
            best_algo_per_dim[dim] = best_algo
    
    # Plot best algorithm curve
    if best_norms_per_dim:
        sorted_dims = sorted(best_norms_per_dim.keys())
        sorted_norms = [best_norms_per_dim[d] for d in sorted_dims]
        
        plt.plot(sorted_dims, sorted_norms, 'b-', linewidth=3, 
                label='Best Algorithm (Smallest Norm)', marker='o', markersize=8)
        
        # Annotate algorithm names on points
        for dim, norm, algo in zip(sorted_dims, sorted_norms, 
                                   [best_algo_per_dim[d] for d in sorted_dims]):
            plt.annotate(algo, (dim, norm), 
                        textcoords="offset points", 
                        xytext=(0,10), 
                        ha='center',
                        fontsize=8,
                        rotation=45)
    
    plt.xlabel('Dimension (d)', fontsize=14)
    plt.ylabel('Output Vector Norm', fontsize=14)
    plt.title('Best Performing Algorithm by Dimension (Smallest Norm)', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'best_algorithm_norm_from_table.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"All charts saved to {output_dir} directory")

# Run plotting function
if __name__ == "__main__":
    plot_from_table_data()