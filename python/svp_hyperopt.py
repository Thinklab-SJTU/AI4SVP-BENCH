import pandas as pd
import numpy as np
import time
import sys
import os
import json
import warnings
from datetime import datetime
from hebo.design_space.design_space import DesignSpace
from hebo.optimizers.hebo import HEBO
import matplotlib.pyplot as plt
import signal
import multiprocessing as mp
import traceback
from functools import wraps
try:
    from fpylll import IntegerMatrix, LLL, BKZ, GSO
    from fpylll.algorithms.bkz2 import BKZReduction as BKZ2
    from fpylll import Enumeration, EnumerationError
    from fpylll.tools.bkz_simulator import simulate
    FPLLL_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import fpylll: {e}")
    FPLLL_AVAILABLE = False
# Add library path
sys.path.append('../lib')
warnings.filterwarnings('ignore')

try:
    import lattice_env
    CPP_ENV_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import lattice_env: {e}")
    CPP_ENV_AVAILABLE = False
class TimeoutException(Exception):

    pass

def timeout_handler(signum, frame):

    raise TimeoutException("Evaluation timed out!")

def timeout_decorator(seconds):

    def decorator(func):
        def wrapper(*args, **kwargs):

            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result
        return wrapper
    return decorator
    
def evaluate_in_process(params_dict, dim, seed, algorithm, timeout_seconds):

    result_queue = mp.Queue()
    

    p = mp.Process(
        target=_evaluate_worker,
        args=(params_dict, dim, seed, algorithm, result_queue)
    )
    p.daemon = True
    p.start()
    

    p.join(timeout=timeout_seconds)
    
    if p.is_alive():

        print(f"    ? Process timeout after {timeout_seconds}s, terminating...")
        p.terminate()
        p.join(timeout=2)
        
        if p.is_alive():
            print(f"    ? Process not responding, killing...")
            p.kill()
            p.join()
        
        raise TimeoutException(f"Evaluation timed out after {timeout_seconds} seconds")
    

    if result_queue.empty():
        raise Exception("Evaluation failed to return results")

    status, result = result_queue.get()
    
    if status == 'success':
        return result
    else:
        raise Exception(f"Evaluation failed in subprocess: {result}")

def _evaluate_worker(params_dict, dim, seed, algorithm, result_queue):

    try:

        import lattice_env
        

        lat = lattice_env.create_lattice_int(dim, dim)
        lat.setSVPChallenge(dim, seed)
        lat.computeGSO()
        

        if algorithm in ['deepBKZ', 'BKZ', 'potBKZ', 'dualBKZ', 'dualDeepBKZ']:
            lat.setMaxLoop(100)
        

        initial_norm = lat.b1Norm()
        

        import time
        start_time = time.time()
        

        if algorithm == 'L2':
            lat.L2(params_dict['delta'], params_dict['eta'])
        elif algorithm in ['BKZ', 'deepBKZ', 'potBKZ', 'dualBKZ', 'dualDeepBKZ']:
            beta = int(params_dict['beta'])
            delta = params_dict['delta']
            
            if algorithm == 'BKZ':
                lat.BKZ(beta, delta)
            elif algorithm == 'deepBKZ':
                lat.deepBKZ(beta, delta)
            elif algorithm == 'potBKZ':
                lat.potBKZ(beta, delta)
            elif algorithm == 'dualBKZ':
                lat.dualBKZ(beta, delta)
            elif algorithm == 'dualDeepBKZ':
                lat.dualDeepBKZ(beta, delta)
        elif algorithm in ['LLL', 'deepLLL', 'potLLL', 'dualLLL',
                               'dualDeepLLL', 'dualPotLLL', 'HKZ']:
            delta = params_dict['delta']
            
            if algorithm == 'LLL':
                lat.LLL(delta)
            elif algorithm == 'deepLLL':
                lat.deepLLL(delta)
            elif algorithm == 'potLLL':
                lat.potLLL(delta)
            elif algorithm == 'dualLLL':
                lat.dualLLL(delta)
            elif algorithm == 'dualDeepLLL':
                lat.dualDeepLLL(delta)
            elif algorithm == 'dualPotLLL':
                lat.dualPotLLL(delta)
            elif algorithm == 'HKZ':
                lat.HKZ(delta)
        
        elif algorithm == 'ENUM':
            R = np.exp(params_dict['log_R'])
            print(R)
            coeff_vector = lat.ENUM(R)
            print("why")
            v = lat.mulVecBasis(coeff_vector)
            final_norm = np.linalg.norm(v)

            end_time = time.time()
            time_taken = end_time - start_time

            if final_norm < 1e-10 or np.isnan(final_norm):
                final_norm = float('inf')
            
            return time_taken, final_norm, initial_norm
        
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
        
        end_time = time.time()
        

        final_norm = lat.b1Norm()
        
        if isinstance(final_norm, float) and (np.isnan(final_norm) or final_norm < 0):
            final_norm = float('inf')
        
        time_taken = end_time - start_time
        
        result_queue.put(('success', (time_taken, final_norm, initial_norm)))
        
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        result_queue.put(('error', error_msg))

#fplll lattice creation function to read only first dim rows from SVP challenge file
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
        
        return A
        
    except FileNotFoundError:
        print(f"    Error: SVP challenge file {file_name} not found")
        print(f"    Falling back to random lattice for fplll")
        return IntegerMatrix.random(dim, "uniform", bits=30)
    except Exception as e:
        print(f"    Error reading SVP challenge file: {e}")
        return IntegerMatrix.random(dim, "uniform", bits=30)

class SVPHyperOptimizer:
    """
    SVP algorithm hyperparameter optimizer
    
    Supported algorithms and parameters:
    1. L2 algorithm: eta (0.5-0.99)
    2. BKZ series: beta (2-50), delta (0.5-0.99)
    3. fplll_BKZ2.0: beta (2-dim)
    4. fplll_self_dual_BKZ: beta (2-dim)
    5. General: delta (0.5-0.99)
    """
    
    def __init__(self, dim=30, seed=9, max_evaluations=50, 
                 algorithm='L2', obj_weight={'time': 0.3, 'norm': 0.7}, timeout_seconds=600):
        """
        Initialize optimizer
        
        Args:
            dim: Lattice dimension
            seed: Random seed
            max_evaluations: Maximum number of evaluations
            algorithm: Name of algorithm to optimize
            obj_weight: Objective weights {'time': w1, 'norm': w2}
        """
        if not CPP_ENV_AVAILABLE:
            raise ImportError("C++ lattice environment not available")
        
        self.dim = dim
        self.seed = seed
        self.max_evaluations = max_evaluations
        self.algorithm = algorithm
        self.obj_weights = obj_weight
        self.timeout_seconds = timeout_seconds
        self.timeout_decorator = timeout_decorator(self.timeout_seconds)
        mp.set_start_method('spawn', force=True)
        
        # Create results directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"./opt_results/{algorithm}_dim{dim}_{timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Define design space
        self.design_space = self._create_design_space()
        
        # Create optimizer
        self.optimizer = HEBO(self.design_space, 
                             model_name='gp',  # Gaussian Process model
                             rand_sample=5)    # Initial random samples
        
        # Store history
        self.history = {
            'params': [],
            'times': [],
            'norms': [],
            'objectives': [],
            'timestamps': []
        }
        
        print(f"\n{'='*70}")
        print(f"SVP Hyperparameter Optimization for {algorithm}")
        print(f"{'='*70}")
        print(f"Dimension: {dim}")
        print(f"Max evaluations: {max_evaluations}")
        print(f"Objective weights: {obj_weight}")
        print(f"Output directory: {self.output_dir}")
        print(f"{'='*70}\n")
    
    def _create_design_space(self):
        """Create design space based on algorithm"""
        space_config = []
        
        if self.algorithm == 'L2':
            # L2 algorithm: optimize eta and delta
            space_config.extend([
                {'name': 'eta', 'type': 'num', 'lb': 0.51, 'ub': 1},
                {'name': 'delta', 'type': 'num', 'lb': 0.5, 'ub': 1},
            ])
        
        elif self.algorithm in ['BKZ', 'deepBKZ', 'dualBKZ', 'dualDeepBKZ']:
            # BKZ series: optimize beta and delta
            space_config.extend([
                {'name': 'beta', 'type': 'int', 'lb': 2, 'ub': self.dim},
                {'name': 'delta', 'type': 'num', 'lb': 0.5, 'ub': 1},
            ])
            
        elif self.algorithm == 'potBKZ':
            # BKZ series: optimize beta and delta
            space_config.extend([
                {'name': 'beta', 'type': 'int', 'lb': 2, 'ub': self.dim},
                {'name': 'delta', 'type': 'num', 'lb': 0.981, 'ub': 1},
            ])
            
        elif self.algorithm in ['LLL', 'deepLLL', 'potLLL', 'dualLLL',
                               'dualDeepLLL', 'dualPotLLL', 'HKZ']:
            # General algorithms: only optimize delta
            space_config.append(
                {'name': 'delta', 'type': 'num', 'lb': 0.5, 'ub': 1}
            )
        
        elif self.algorithm == 'ENUM':
            space_config.extend([
                {'name': 'log_R', 'type': 'num', 'lb': np.log(3e6), 'ub': np.log(5e6)},
            ])
        
        elif self.algorithm == 'fplll_BKZ2.0':
            # fplll BKZ 2.0: only optimize beta
            space_config.extend([
                {'name': 'beta', 'type': 'int', 'lb': 2, 'ub': self.dim},
            ])
            
        elif self.algorithm == 'fplll_self_dual_BKZ':
            # fplll Self-Dual BKZ: only optimize beta
            space_config.extend([
                {'name': 'beta', 'type': 'int', 'lb': 2, 'ub': self.dim},
            ])
        
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
        
        return DesignSpace().parse(space_config)
    
    def evaluate_fplll_bkz20(self, params_dict):
        """Evaluate fplll BKZ 2.0 algorithm"""
        try:
            if not FPLLL_AVAILABLE:
                return float('inf'), float('inf'), float('nan')
            
            # Create lattice (from SVP challenge file)
            A = create_fplll_lattice_from_svp_challenge(self.dim, self.seed)
            
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
            
            # BKZ 2.0 parameters
            beta = int(params_dict['beta'])
            params = BKZ.Param(
                block_size=beta,
                strategies=BKZ.DEFAULT_STRATEGY,
                flags=BKZ.VERBOSE,
                max_loops=5,
                rerandomization_density=0
            )
            
            # Directly call BKZReduction object
            bkz(params)
            
            end_time = time.time()
            
            # Get final norm
            final_norm = np.linalg.norm([float(x) for x in A[0]])
            
            return end_time - start_time, final_norm, initial_norm
            
        except Exception as e:
            print(f"    Error in fplll BKZ 2.0: {e}")
            import traceback
            traceback.print_exc()
            return float('inf'), float('inf'), float('nan')
    
    def evaluate_fplll_self_dual_bkz(self, params_dict):
        """Evaluate fplll Self-Dual BKZ algorithm"""
        try:
            if not FPLLL_AVAILABLE:
                return float('inf'), float('inf'), float('nan')
            
            # Create lattice (from SVP challenge file)
            A = create_fplll_lattice_from_svp_challenge(self.dim, self.seed)
            
            # LLL reduce first
            M = GSO.Mat(A)
            L = LLL.Reduction(M)
            L()
            
            # Get initial norm
            initial_norm = np.linalg.norm([float(x) for x in A[0]])
            
            # Start timing
            start_time = time.time()
            
            # Self-Dual BKZ parameters
            beta = int(params_dict['beta'])
            
            try:
                # Try with SD flag first
                params = BKZ.Param(
                    block_size=beta,
                    strategies=BKZ.DEFAULT_STRATEGY,
                    flags=BKZ.VERBOSE | BKZ.SD,
                    max_loops=5,
                    auto_abort=True
                )
                
                # Use BKZ.reduction method
                BKZ.reduction(A, params)
                
            except AttributeError:
                # If BKZ.SD is not available, use alternative method
                print(f"    BKZ.SD flag not available, using alternative method")
                param = BKZ.EasyParam(
                    block_size=beta,
                    flags=BKZ.GH_BND,
                    max_loops=5
                )
                
                # Execute BKZ reduction
                BKZ.reduction(A, param)
            
            end_time = time.time()
            
            # Get final norm
            final_norm = np.linalg.norm([float(x) for x in A[0]])
            
            return end_time - start_time, final_norm, initial_norm
            
        except Exception as e:
            print(f"    Error in fplll self-dual BKZ: {e}")
            import traceback
            traceback.print_exc()
            return float('inf'), float('inf'), float('nan')
    
    def evaluate(self, params_dict):
        """Evaluate algorithm with given parameters"""
        
        # For fplll algorithms
        if self.algorithm == 'fplll_BKZ2.0':
            return self.evaluate_fplll_bkz20(params_dict)
        elif self.algorithm == 'fplll_self_dual_BKZ':
            return self.evaluate_fplll_self_dual_bkz(params_dict)
        
        # For other algorithms, use existing evaluate_in_process
        try:
            time_taken, final_norm, initial_norm = evaluate_in_process(
                params_dict, 
                self.dim, 
                self.seed, 
                self.algorithm, 
                self.timeout_seconds
            )
            
            if time_taken == float('inf') and final_norm == float('inf'):
                print(f" Process terminated due to timeout")
            
            return time_taken, final_norm, initial_norm
            
        except TimeoutException as e:
            print(f" {e}")
            return float('inf'), float('inf'), float('nan')
        except Exception as e:
            print(f" Evaluation failed: {str(e)[:100]}...")
            return float('inf'), float('inf'), float('nan')
            
            
    '''def _evaluate_core(self, params_dict):
        """
        Evaluate algorithm performance with given parameters
        
        Returns:
            tuple: (running_time, b1_norm)
        """
        try:
            # Create new lattice instance
            lat = lattice_env.create_lattice_int(self.dim, self.dim)
            lat.setSVPChallenge(self.dim, self.seed)
            #lat.setRandom(self.dim, self.dim, 100000, 1000000)
            lat.computeGSO()
            
            # Get initial norm
            initial_norm = lat.b1Norm()
            
            # Execute algorithm and time it
            start_time = time.time()
            
            # Call corresponding function based on algorithm and parameters
            if self.algorithm == 'L2':
                lat.L2(params_dict['delta'], params_dict['eta'])
            
            elif self.algorithm in ['BKZ', 'deepBKZ', 'potBKZ', 'dualBKZ',
                                   'dualDeepBKZ']:
                beta = int(params_dict['beta'])
                delta = params_dict['delta']
                
                lat.setMaxLoop(100)
                
                if self.algorithm == 'BKZ':
                    lat.BKZ(beta, delta)
                elif self.algorithm == 'deepBKZ':
                    lat.deepBKZ(beta, delta)
                elif self.algorithm == 'potBKZ':
                    lat.potBKZ(beta, delta)
                elif self.algorithm == 'dualBKZ':
                    lat.dualBKZ(beta, delta)
                elif self.algorithm == 'dualDeepBKZ':
                    lat.dualDeepBKZ(beta, delta)
            
            elif self.algorithm in ['LLL', 'deepLLL', 'potLLL', 'dualLLL',
                                   'dualDeepLLL', 'dualPotLLL', 'HKZ']:
                delta = params_dict['delta']
                
                if self.algorithm == 'LLL':
                    lat.LLL(delta)
                elif self.algorithm == 'deepLLL':
                    lat.deepLLL(delta)
                elif self.algorithm == 'potLLL':
                    lat.potLLL(delta)
                elif self.algorithm == 'dualLLL':
                    lat.dualLLL(delta)
                elif self.algorithm == 'dualDeepLLL':
                    lat.dualDeepLLL(delta)
                elif self.algorithm == 'dualPotLLL':
                    lat.dualPotLLL(delta)
                elif self.algorithm == 'HKZ':
                    lat.HKZ(delta)
            
            elif self.algorithm == 'ENUM':
                R = np.exp(params_dict['log_R'])

                coeff_vector = lat.ENUM(R)

                v = lat.mulVecBasis(coeff_vector)
                final_norm = np.linalg.norm(v)

                end_time = time.time()
                time_taken = end_time - start_time

                if final_norm < 1e-10 or np.isnan(final_norm):
                    final_norm = float('inf')
                
                return time_taken, final_norm, initial_norm
            
            else:
                raise ValueError(f"Unsupported algorithm: {self.algorithm}")
            
            end_time = time.time()
            
            # Get final norm
            final_norm = lat.b1Norm()
            
            # Check result validity
            if np.isnan(final_norm) or final_norm < 0:
                final_norm = float('inf')
            
            time_taken = end_time - start_time
            
            return time_taken, final_norm, initial_norm
            
        except Exception as e:
            print(f"    Evaluation failed: {e}")
            return float('inf'), float('inf'), float('nan')'''
    
    def objective_function(self, params_df):
        """
        Objective function: weighted composite metric (smaller is better)
        
        Composite metric = w1 * normalized_time + w2 * normalized_norm
        """
        objectives = []
        
        for _, row in params_df.iterrows():
            # Convert to dictionary
            params_dict = row.to_dict()
            
            print(f"  Evaluating with params: {params_dict}")
            
            # Evaluate performance
            time_taken, final_norm, initial_norm = self.evaluate(params_dict)
            
            if time_taken == float('inf') and final_norm == float('inf'):
                timeout_marker = "(TIMEOUT)"
            else:
                timeout_marker = ""
            
            print(f"    Result: time={time_taken:.4f}s{timeout_marker}, "
                  f"norm={initial_norm:.2f}->{final_norm:.2f}")
            
            # Record history
            self.history['params'].append(params_dict)
            self.history['times'].append(time_taken)
            self.history['norms'].append(final_norm)
            self.history['timestamps'].append(time.time())
            
            # Normalize (using min-max from current history)
            if len(self.history['times']) > 1:
                min_time = min(t for t in self.history['times'] if t != float('inf'))
                max_time = max(t for t in self.history['times'] if t != float('inf'))
                min_norm = min(n for n in self.history['norms'] if n != float('inf'))
                max_norm = max(n for n in self.history['norms'] if n != float('inf'))
                
                # Avoid division by zero
                if max_time == min_time:
                    norm_time = 1.0
                else:
                    norm_time = (time_taken - min_time) / (max_time - min_time)
                
                if max_norm == min_norm:
                    norm_norm = 1.0
                else:
                    norm_norm = (final_norm - min_norm) / (max_norm - min_norm)
            else:
                norm_time = 1.0 if time_taken != float('inf') else 2.0
                norm_norm = 1.0 if final_norm != float('inf') else 2.0
            
            # Calculate composite objective value
            objective = (self.obj_weights['time'] * norm_time + 
                        self.obj_weights['norm'] * norm_norm)
            
            # Penalize failed evaluations
            if time_taken == float('inf') or final_norm == float('inf'):
                objective = 10.0  # Penalty value
            
            self.history['objectives'].append(objective)
            objectives.append(objective)
            
            # Save current best result
            self._save_best_result()
        
        return np.array(objectives).reshape(-1, 1)
    
    def optimize(self, n_suggestions=4):
        """Execute optimization process"""
        print(f"\nStarting optimization for {self.algorithm}...")
        
        best_objective = float('inf')
        best_params = None
        best_time = None
        best_norm = None
        
        for i in range(self.max_evaluations // n_suggestions):
            print(f"\n{'='*60}")
            print(f"Iteration {i+1}")
            print(f"{'='*60}")
            
            # Get parameter suggestions
            rec_params = self.optimizer.suggest(n_suggestions=n_suggestions)
            
            # Evaluate parameters
            objectives = self.objective_function(rec_params)
            
            # Update optimizer
            self.optimizer.observe(rec_params, objectives)
            
            # Update best results
            current_best_idx = np.argmin(objectives)
            current_best_obj = objectives[current_best_idx][0]
            
            if current_best_obj < best_objective:
                best_objective = current_best_obj
                best_params = rec_params.iloc[current_best_idx].to_dict()
                
                # Get actual performance of best parameters
                best_time = self.history['times'][-n_suggestions + current_best_idx]
                best_norm = self.history['norms'][-n_suggestions + current_best_idx]
                
                print(f"\n? New best found!")
                print(f"   Params: {best_params}")
                print(f"   Time: {best_time:.4f}s, Norm: {best_norm:.2f}")
                print(f"   Objective: {best_objective:.4f}")
            
            print(f"\nProgress: {(i+1)*n_suggestions}/{self.max_evaluations} evaluations")
            print(f"Current best objective: {best_objective:.4f}")
            
            # Periodically save results
            if (i + 1) % 5 == 0:
                self._plot_progress()
                self._save_history()
        
        # Final analysis
        print(f"\n{'='*70}")
        print(f"OPTIMIZATION COMPLETE")
        print(f"{'='*70}")
        print(f"Algorithm: {self.algorithm}")
        print(f"Best parameters: {best_params}")
        print(f"Best time: {best_time:.4f}s")
        print(f"Best b1_norm: {best_norm:.2f}")
        print(f"Best objective: {best_objective:.4f}")
        
        # Generate final report
        self._generate_report(best_params, best_time, best_norm, best_objective)
        
        return best_params, best_objective
    
    def _save_best_result(self):
        """Save current best result"""
        if not self.history['objectives']:
            return
        
        # Find best result index
        best_idx = np.argmin(self.history['objectives'])
        
        best_result = {
            'algorithm': self.algorithm,
            'dimension': self.dim,
            'seed': self.seed,
            'timestamp': datetime.now().isoformat(),
            'best_params': self.history['params'][best_idx],
            'best_time': self.history['times'][best_idx],
            'best_norm': self.history['norms'][best_idx],
            'best_objective': self.history['objectives'][best_idx],
            'total_evaluations': len(self.history['objectives'])
        }
        
        with open(os.path.join(self.output_dir, 'best_result.json'), 'w') as f:
            json.dump(best_result, f, indent=2)
    
    def _save_history(self):
        """Save history records"""
        history_data = {
            'algorithm': self.algorithm,
            'dimension': self.dim,
            'seed': self.seed,
            'objective_weights': self.obj_weights,
            'evaluations': []
        }
        
        for i in range(len(self.history['objectives'])):
            eval_data = {
                'iteration': i + 1,
                'params': self.history['params'][i],
                'time': self.history['times'][i],
                'norm': self.history['norms'][i],
                'objective': self.history['objectives'][i],
                'timestamp': self.history['timestamps'][i]
            }
            history_data['evaluations'].append(eval_data)
        
        with open(os.path.join(self.output_dir, 'history.json'), 'w') as f:
            json.dump(history_data, f, indent=2, default=str)
    
    def _plot_progress(self):
        """Plot optimization progress"""
        if len(self.history['objectives']) < 2:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Objective value vs iteration
        ax1 = axes[0, 0]
        iterations = range(1, len(self.history['objectives']) + 1)
        ax1.plot(iterations, self.history['objectives'], 'b-', linewidth=2, marker='o')
        ax1.set_xlabel('Evaluation Number')
        ax1.set_ylabel('Objective Value')
        ax1.set_title('Objective Value vs Evaluation')
        ax1.grid(True, alpha=0.3)
        
        # Add best value marker
        best_idx = np.argmin(self.history['objectives'])
        ax1.plot(best_idx + 1, self.history['objectives'][best_idx], 
                'ro', markersize=10, label=f'Best: {self.history["objectives"][best_idx]:.4f}')
        ax1.legend()
        
        # Plot 2: Running time distribution
        ax2 = axes[0, 1]
        valid_times = [t for t in self.history['times'] if t != float('inf')]
        if valid_times:
            ax2.hist(valid_times, bins=20, alpha=0.7, color='green')
            ax2.set_xlabel('Time (seconds)')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Running Time Distribution')
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: b1_norm distribution
        ax3 = axes[1, 0]
        valid_norms = [n for n in self.history['norms'] if n != float('inf')]
        if valid_norms:
            ax3.hist(valid_norms, bins=20, alpha=0.7, color='orange')
            ax3.set_xlabel('b1_norm')
            ax3.set_ylabel('Frequency')
            ax3.set_title('b1_norm Distribution')
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: Parameter scatter plot (for L2 algorithm)
        ax4 = axes[1, 1]
        if self.algorithm == 'L2' and len(self.history['params']) > 0:
            etas = [p.get('eta', 0) for p in self.history['params']]
            deltas = [p.get('delta', 0) for p in self.history['params']]
            objectives = self.history['objectives']
            
            scatter = ax4.scatter(etas, deltas, c=objectives, 
                                 cmap='viridis', s=50, alpha=0.7)
            ax4.set_xlabel('eta')
            ax4.set_ylabel('delta')
            ax4.set_title(f'Parameter Space (Algorithm: {self.algorithm})')
            plt.colorbar(scatter, ax=ax4, label='Objective Value')
        
        elif self.algorithm == 'ENUM' and len(self.history['params']) > 0:

            log_Rs = [p.get('log_R', 0) for p in self.history['params']]
            Rs = [np.exp(r) for r in log_Rs]
            objectives = self.history['objectives']
            
            scatter = ax4.scatter(Rs, objectives, c=objectives, 
                                 cmap='viridis', s=50, alpha=0.7)
            ax4.set_xlabel('Radius R (log scale)')
            ax4.set_ylabel('Objective Value')
            ax4.set_title(f'ENUM Radius vs Objective')
            ax4.set_xscale('log')
            plt.colorbar(scatter, ax=ax4, label='Objective Value')
            

            best_idx = np.argmin(objectives)
            ax4.plot(Rs[best_idx], objectives[best_idx], 
                    'ro', markersize=10, label=f'Best R={Rs[best_idx]:.2e}')
            ax4.legend()
    
        elif self.algorithm in ['BKZ', 'deepBKZ', 'potBKZ'] and len(self.history['params']) > 0:
            betas = [p.get('beta', 0) for p in self.history['params']]
            deltas = [p.get('delta', 0) for p in self.history['params']]
            objectives = self.history['objectives']
            
            scatter = ax4.scatter(betas, deltas, c=objectives, 
                                 cmap='viridis', s=50, alpha=0.7)
            ax4.set_xlabel('beta')
            ax4.set_ylabel('delta')
            ax4.set_title(f'Parameter Space (Algorithm: {self.algorithm})')
            plt.colorbar(scatter, ax=ax4, label='Objective Value')
            
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'optimization_progress.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_report(self, best_params, best_time, best_norm, best_objective):
        """Generate optimization report"""
        report = f"""
{'='*70}
SVP Algorithm Hyperparameter Optimization Report
{'='*70}

Algorithm: {self.algorithm}
Dimension: {self.dim}
Random Seed: {self.seed}
Total Evaluations: {len(self.history['objectives'])}
Objective Weights: Time={self.obj_weights['time']}, Norm={self.obj_weights['norm']}

{'='*70}
BEST PARAMETERS FOUND:
{'='*70}
"""
        
        for param, value in best_params.items():
            if param == 'log_R' and self.algorithm == 'ENUM':
                actual_R = np.exp(value)
                report += f"  log_R: {value:.4f} (actual R = {actual_R:.2e})\n"
            else:
                report += f"  {param}: {value}\n"
        
        report += f"""
{'='*70}
PERFORMANCE RESULTS:
{'='*70}
Running Time: {best_time:.4f} seconds
b1_norm: {best_norm:.4f}
Objective Value: {best_objective:.4f}

{'='*70}
STATISTICAL SUMMARY:
{'='*70}
"""
        
        # Statistics for valid results
        valid_times = [t for t in self.history['times'] if t != float('inf')]
        valid_norms = [n for n in self.history['norms'] if n != float('inf')]
        
        if valid_times:
            report += f"Time Statistics:\n"
            report += f"  Min: {min(valid_times):.4f}s\n"
            report += f"  Max: {max(valid_times):.4f}s\n"
            report += f"  Mean: {np.mean(valid_times):.4f}s\n"
            report += f"  Std: {np.std(valid_times):.4f}s\n\n"
        
        if valid_norms:
            report += f"Norm Statistics:\n"
            report += f"  Min: {min(valid_norms):.4f}\n"
            report += f"  Max: {max(valid_norms):.4f}\n"
            report += f"  Mean: {np.mean(valid_norms):.4f}\n"
            report += f"  Std: {np.std(valid_norms):.4f}\n"
        
        report += f"""
{'='*70}
Optimization completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Results saved to: {self.output_dir}
{'='*70}
"""
        
        # Save report
        with open(os.path.join(self.output_dir, 'optimization_report.txt'), 'w') as f:
            f.write(report)
        
        # Print report
        print(report)

algorithms_to_optimize = [
        
    ]
def compare_algorithms_with_opt(dimensions=[20, 30, 40], seed=9, max_evals=40):
    """
    Compare performance of different algorithms after optimization
    """
    if not CPP_ENV_AVAILABLE:
        print("C++ lattice environment not available. Exiting.")
        return
    
    # List of algorithms to optimize (including fplll algorithms)
    algorithms_to_optimize = [
        #'fplll_BKZ2.0',
        #'fplll_self_dual_BKZ',
        'deepBKZ',
        'deepLLL',
        'potLLL',
        'dualDeepLLL',
        'dualPotLLL',
        'BKZ',
        #'deepBKZ',
        'potBKZ',
        'dualDeepBKZ'
        #'LLL','L2',
        #'dualBKZ','dualLLL',
        #'ENUM''HKZ'
    ]
    
    # Store optimization results for all algorithms
    all_results = {}
    
    for dim in dimensions:
        print(f"\n{'='*70}")
        print(f"OPTIMIZING FOR DIMENSION {dim}")
        print(f"{'='*70}")
        
        dim_results = {}
        
        for algo in algorithms_to_optimize:
            print(f"\n>>> Optimizing {algo} <<<")
            
            try:
                # Create optimizer
                if algo in ['fplll_BKZ2.0', 'fplll_self_dual_BKZ']:
                    # fplll algorithms use standard weights
                    optimizer = SVPHyperOptimizer(
                        dim=dim,
                        seed=seed,
                        max_evaluations=max_evals,
                        algorithm=algo,
                        obj_weight={'time': 0.1, 'norm': 0.9},
                        timeout_seconds=300
                    )
                elif algo == 'L2':
                    optimizer = SVPHyperOptimizer(
                        dim=dim,
                        seed=seed,
                        max_evaluations=max_evals,
                        algorithm=algo,
                        obj_weight={'time': 0.4, 'norm': 0.6}
                    )
                elif 'BKZ' in algo:
                    optimizer = SVPHyperOptimizer(
                        dim=dim,
                        seed=seed,
                        max_evaluations=max_evals,
                        algorithm=algo,
                        obj_weight={'time': 0.1, 'norm': 0.9}
                    )
                else:
                    optimizer = SVPHyperOptimizer(
                        dim=dim,
                        seed=seed,
                        max_evaluations=max_evals,
                        algorithm=algo,
                        obj_weight={'time': 0.1, 'norm': 0.9}
                    )
                
                # Execute optimization
                best_params, best_obj = optimizer.optimize(n_suggestions=4)
                
                dim_results[algo] = {
                    'best_params': best_params,
                    'best_objective': best_obj,
                    'output_dir': optimizer.output_dir
                }
                
            except Exception as e:
                print(f"Failed to optimize {algo}: {e}")
                dim_results[algo] = None
        
        all_results[dim] = dim_results
    
    # Generate comparison report
    generate_comparison_report(all_results)


def generate_comparison_report(all_results):
    """Generate algorithm comparison report"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = f"./comparison_reports/{timestamp}"
    os.makedirs(report_dir, exist_ok=True)
    
    report = f"""
{'='*80}
SVP Algorithms Hyperparameter Optimization Comparison Report
{'='*80}
Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

"""
    
    for dim, results in all_results.items():
        report += f"\n\n{'='*60}\n"
        report += f"DIMENSION {dim}\n"
        report += f"{'='*60}\n\n"
        
        # Create table
        report += f"{'Algorithm':<15} {'Best Parameters':<30} {'Objective':<12} {'Directory'}\n"
        report += f"{'-'*15} {'-'*30} {'-'*12} {'-'*40}\n"
        
        for algo, result in results.items():
            if result:
                params_str = ', '.join([f"{k}={v:.3f}" if isinstance(v, float) 
                                       else f"{k}={v}" 
                                       for k, v in result['best_params'].items()])
                report += f"{algo:<15} {params_str:<30} {result['best_objective']:<12.4f} {result['output_dir']}\n"
            else:
                report += f"{algo:<15} {'FAILED':<30} {'N/A':<12} {'N/A'}\n"
    
    # Save report
    report_path = os.path.join(report_dir, "comparison_report.txt")
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"\nComparison report saved to: {report_path}")
    print(report)
    
    return report_dir


# Example usage
if __name__ == "__main__":
    # Example 1: Optimize single algorithm
    print("Example 1: Optimize eta parameter for L2 algorithm")
    '''optimizer = SVPHyperOptimizer(
        dim=40,
        seed=9,
        max_evaluations=40,
        algorithm='potBKZ',
        obj_weight={'time': 0.7, 'norm': 0.3},
        timeout_seconds=200
    )
    best_params, best_obj = optimizer.optimize(n_suggestions=4)'''
    '''optimizer = SVPHyperOptimizer(
        dim=80,
        seed=0,
        max_evaluations=40,
        algorithm='dualBKZ',
        obj_weight={'time': 0.5, 'norm': 0.5},
        timeout_seconds=200
    )'''
    #best_params, best_obj = optimizer.optimize(n_suggestions=4)
    # Example 2: Compare multiple algorithms
    print("\n\nExample 2: Compare optimized performance of multiple algorithms")
    compare_algorithms_with_opt(
        dimensions=[80],
        seed=0,
        max_evals=40
    )