# main.py
import numpy as np
from numpy.linalg import norm
import random
import argparse
from typing import List
from config import Config
from data_collector import NVSieveDataCollector
from trainer import ModelTrainer
from ai_enhanced_sieve import AIEnhancedNVSieve
import sys
import os
sys.path.append('../lib')
try:
    import lattice_env
    CPP_ENV_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import lattice_env: {e}")
    CPP_ENV_AVAILABLE = False

def generate_random_lattice_vectors(dim: int, num_vectors: int) -> List[np.ndarray]:
    """
    Generate random lattice vectors list (for testing)
    """
    vectors = []
    for _ in range(num_vectors):
        # Generate random integer vector
        v = np.random.randint(-100, 100, size=dim)
        vectors.append(v)
    
    # Add some linear combinations to increase correlation
    for _ in range(num_vectors // 4):
        i, j = random.sample(range(len(vectors)), 2)
        v = vectors[i] + vectors[j]
        vectors.append(v)
    
    return vectors
def read_svp_challenge_file(dim: int, seed: int = 0) -> np.ndarray:
    """
    Read SVP Challenge format file
    
    Note: SVP Challenge files usually have shape (2*dim, dim),
    we need to extract the first dim rows as the basis matrix
    """
    file_name = f"../svp_challenge_list/svp_challenge_{dim}_{seed}.txt"
    
    if not os.path.exists(file_name):
        # If file doesn't exist, create simulated SVP Challenge data
        print(f"File {file_name} does not exist, creating simulated data")
        return generate_simulated_basis(dim)
    
    print(f"Reading file: {file_name}")
    
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
    # We need to extract the first dim rows
    if basis_full.shape[1] == dim and basis_full.shape[0] >= dim:
        # Extract first dim rows
        basis = basis_full[:dim, :]
        print(f"Shape after extraction: {basis.shape}")
        return basis
    elif basis_full.shape[0] == dim and basis_full.shape[1] == dim:
        # Already correct shape
        print(f"File shape correct: {basis_full.shape}")
        return basis_full
    else:
        print(f"Warning: Abnormal file shape {basis_full.shape}, expected (>={dim}, {dim})")
        
        # Try automatic adjustment
        if basis_full.shape[1] == dim and basis_full.shape[0] >= dim:
            # Take first dim rows
            basis = basis_full[:dim, :]
            print(f"Shape after automatic adjustment: {basis.shape}")
            return basis
        elif basis_full.shape[0] >= dim and basis_full.shape[1] >= dim:
            # Take first dim rows and first dim columns
            basis = basis_full[:dim, :dim]
            print(f"Shape after automatic adjustment: {basis.shape}")
            return basis
        
        print("Cannot adjust dimensions, using simulated data")
        return generate_simulated_basis(dim)

def generate_simulated_basis(dim: int) -> np.ndarray:
    """
    Generate simulated lattice basis (similar to SVP Challenge format)
    """
    np.random.seed(42)
    
    # Generate random orthogonal matrix
    Q, _ = np.linalg.qr(np.random.randn(dim, dim))
    
    # Generate diagonal elements, simulating lattice asymptotic properties
    diag = np.array([1.01 ** (dim - i) for i in range(dim)])
    
    # Construct basis matrix
    basis = Q * diag.reshape(1, -1)
    
    return basis

def generate_vectors_from_basis(basis: np.ndarray, num_vectors: int, 
                               coeff_range: int = 5) -> List[np.ndarray]:
    """
    Generate vector list from basis matrix
    """
    vectors = []
    dim = basis.shape[0]
    
    for _ in range(num_vectors):
        # Generate random integer coefficients
        coeffs = np.random.randint(-coeff_range, coeff_range + 1, size=dim)
        
        # Calculate linear combination
        v = np.dot(coeffs, basis)
        
        # Ensure not zero vector
        if np.linalg.norm(v) > 1e-10:
            vectors.append(v)
    
    return vectors

def lattice_int_to_vectors(lattice_obj, num_vectors: int = 300) -> List[np.ndarray]:
    """
    Convert lattice_env.LatticeInt object to vector list
    
    Note: This function assumes lattice_obj has some way to get basis vectors
    If lattice_env.LatticeInt doesn't provide a method to get basis vectors,
    we need to adjust according to actual situation
    """
    vectors = []
    
    # Method 1: Try to get basis matrix
    try:
        # Assume lattice_obj has basis attribute or method
        if hasattr(lattice_obj, 'basis'):
            basis = lattice_obj.m_basis
        elif hasattr(lattice_obj, 'get_basis'):
            basis = lattice_obj.get_basis()
        else:
            raise AttributeError("Cannot get basis matrix")
        
        # Generate vectors
        return generate_vectors_from_basis(basis, num_vectors)
        
    except Exception as e:
        print(f"Cannot get basis matrix from lattice object: {e}")
        
        # Method 2: Try to directly get vectors (if lattice_obj stores vectors)
        if hasattr(lattice_obj, 'vectors'):
            vectors = list(lattice_obj.vectors)
            if len(vectors) >= num_vectors:
                return vectors[:num_vectors]
        
        # Method 3: Generate random vectors as fallback
        print("Using fallback method to generate random vectors")
        dim = 40  # Default dimension
        if hasattr(lattice_obj, 'dim'):
            dim = lattice_obj.dim
        
        basis = generate_simulated_basis(dim)
        return generate_vectors_from_basis(basis, num_vectors)
def collect_training_data():
    """Collect training data"""
    print("Starting training data collection...")
    
    # Create save directory
    data_dir = "training_data"
    os.makedirs(data_dir, exist_ok=True)
    
    collector = NVSieveDataCollector(max_samples=20000)
    
    # Collect data on different dimensions
    dimensions = [40, 60, 80]
    
    for dim in dimensions:
        print(f"\nCollecting {dim}-dimensional data...")
        
        try:
            # Method 1: Use SVP Challenge file
            basis = read_svp_challenge_file(dim, seed=0)
            S = generate_vectors_from_basis(basis, num_vectors=400, coeff_range=3)
            
            print(f"Generated {len(S)} vectors from SVP Challenge file")
            
        except Exception as e:
            print(f"Failed to use SVP Challenge file: {e}")
            
            # Method 2: Use lattice_env module (if available)
            if HAVE_LATTICE_ENV:
                try:
                    lattice_obj = lattice_env.create_lattice_int(dim, dim)
                    lattice_obj.setSVPChallenge(dim, 0)
                    S = lattice_int_to_vectors(lattice_obj, num_vectors=400)
                    print(f"Generated {len(S)} vectors from lattice_env")
                except Exception as e2:
                    print(f"Failed to use lattice_env: {e2}")
                    # Method 3: Use simulated data
                    basis = generate_simulated_basis(dim)
                    S = generate_vectors_from_basis(basis, num_vectors=400)
                    print(f"Generated {len(S)} vectors using simulated data")
            else:
                # Method 3: Use simulated data
                basis = generate_simulated_basis(dim)
                S = generate_vectors_from_basis(basis, num_vectors=400)
                print(f"Generated {len(S)} vectors using simulated data")
        
        # Create sieve instance
        sieve = AIEnhancedNVSieve(
            use_ai=False,  # Don't use AI, use original algorithm to collect data
            collect_data=True,
            top_k=Config.TOP_K
        )
        
        # Set data collector
        sieve.data_collector = collector
        
        try:
            # Run sieve to collect data
            result = sieve.run(S, Config.GAMMA, max_iterations=30)
            print(f"  {dim}D data collection completed, samples: {collector.samples_collected}")
        except Exception as e:
            print(f"  {dim}D data collection error: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save dataset
    if collector.samples_collected > 0:
        data_file = os.path.join(data_dir, "training_data.pkl")
        collector.save_dataset(data_file)
        
        # Prepare training data
        features, labels = collector.get_training_dataset()
        
        # Print data statistics
        if features is not None and labels is not None:
            print(f"\nDataset statistics:")
            print(f"  Total samples: {len(labels)}")
            print(f"  Positive samples: {np.sum(labels)}")
            print(f"  Negative samples: {len(labels) - np.sum(labels)}")
            if len(labels) > 0:
                print(f"  Positive sample ratio: {np.sum(labels) / len(labels):.4f}")
        
        return features, labels
    else:
        print("No data collected")
        return None, None

def test_ai_sieve():
    """Test AI-enhanced sieve"""
    print("\nTesting AI-enhanced sieve...")
    
    dim = 60  # Use medium dimension for testing
    gamma = 0.85  # Use more aggressive gamma
    
    try:
        # Generate test data
        basis = read_svp_challenge_file(dim, seed=1)  # Use seed=1 to avoid overlap with training data
        S = generate_vectors_from_basis(basis, num_vectors=500, coeff_range=4)
        
        print(f"Generated {len(S)} test vectors")
        
    except Exception as e:
        print(f"Failed to generate test data: {e}")
        # Use simulated data
        basis = generate_simulated_basis(dim)
        S = generate_vectors_from_basis(basis, num_vectors=500, coeff_range=4)
        print(f"Using simulated test data: {len(S)} vectors")
    
    # Baseline test: Original algorithm
    print("\n1. Running original algorithm...")
    sieve_original = AIEnhancedNVSieve(
        use_ai=False, 
        collect_data=False,
        top_k=5
    )
    
    start_time = time.time()
    result_original = sieve_original.run(S, gamma, max_iterations=30)
    time_original = time.time() - start_time
    
    print(f"Original algorithm result: norm={norm(result_original):.4f}, time={time_original:.2f}s")
    
    # AI-enhanced algorithm
    print("\n2. Running AI-enhanced algorithm...")
    model_path = "center_match_model.pth"
    
    if not os.path.exists(model_path):
        print(f"Model file {model_path} does not exist, using heuristic rules")
        sieve_ai = AIEnhancedNVSieve(
            model_path=None,
            use_ai=False,  # No model, use heuristic
            collect_data=False,
            top_k=Config.TOP_K
        )
    else:
        sieve_ai = AIEnhancedNVSieve(
            model_path=model_path,
            use_ai=True,
            collect_data=False,
            top_k=Config.TOP_K
        )
    
    start_time = time.time()
    result_ai = sieve_ai.run(S, gamma, max_iterations=30)
    time_ai = time.time() - start_time
    
    print(f"AI-enhanced algorithm result: norm={norm(result_ai):.4f}, time={time_ai:.2f}s")
    
    # Performance comparison
    if time_original > 0 and time_ai > 0:
        speedup = time_original / time_ai
        print(f"\nPerformance comparison:")
        print(f"  Original algorithm time: {time_original:.2f}s")
        print(f"  AI-enhanced algorithm time: {time_ai:.2f}s")
        print(f"  Speedup ratio: {speedup:.2f}x")
        
        # Quality comparison
        norm_ratio = norm(result_ai) / norm(result_original)
        print(f"  Norm ratio (AI/Original): {norm_ratio:.4f}")
    else:
        print("Time measurement error")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Nguyen-Vidick Sieve AI Acceleration")
    parser.add_argument('--mode', type=str, default='all',
                       choices=['collect', 'train', 'test', 'all'],
                       help='Run mode: collect (collect data), train (train model), test (test), all (all)')
    parser.add_argument('--dim', type=int, default=Config.DIMENSION,
                       help='Lattice dimension')
    parser.add_argument('--top_k', type=int, default=Config.TOP_K,
                       help='Number of centers to check per vector')
    parser.add_argument('--gamma', type=float, default=Config.GAMMA,
                       help='Sieve parameter gamma')
    
    args = parser.parse_args()
    
    # Update configuration
    Config.DIMENSION = args.dim
    Config.TOP_K = args.top_k
    Config.GAMMA = args.gamma
    
    print(f"Configuration: dimension={Config.DIMENSION}, TOP_K={Config.TOP_K}, gamma={Config.GAMMA}")
    
    try:
        # Run based on mode
        if args.mode in ['collect', 'all']:
            features, labels = collect_training_data()
        
        if args.mode in ['train', 'all']:
            if 'features' not in locals() or features is None:
                # Try to load existing data
                data_file = "training_data/training_data.pkl"
                if os.path.exists(data_file):
                    collector = NVSieveDataCollector()
                    collector.load_dataset(data_file)
                    features, labels = collector.get_training_dataset()
                    print(f"Loaded {len(labels) if labels is not None else 0} samples from {data_file}")
                else:
                    print("Error: No training data found, please run collection mode first")
                    return
            
            if features is not None and labels is not None:
                model, history = train_model(features, labels)
            else:
                print("Error: Training data is empty")
        
        if args.mode in ['test', 'all']:
            test_ai_sieve()
        
        print("\nAll steps completed!")
        
    except KeyboardInterrupt:
        print("\nUser interrupted")
    except Exception as e:
        print(f"\nProgram error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()