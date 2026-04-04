import numpy as np
import time
import sys

sys.path.append('../lib')

try:
    import lattice_env
    CPP_ENV_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import lattice_env: {e}")
    CPP_ENV_AVAILABLE = False


import lattice_env
import numpy as np
import time

def vector_norm(v):
    """Compute the Euclidean norm of a vector (corresponds to vectorNorm in C++)"""
    sum_val = 0.0
    for x in v:
        sum_val += float(x) * float(x)
    return np.sqrt(sum_val)

def main():
    """Corresponds to the original C++ main function"""
    print("=== Lattice SVP Problem Test ===\n")
    
    # Corresponds to: Lattice<int> lat(70, 70);
    print("Creating 70x70 lattice...")
    lat =lattice_env.LatticeInt(40, 40)  # 70-dimensional full-rank lattice
    # Corresponds to: lat.setSVPChallenge(70, 9);
    print("Setting up SVP challenge...")
    lat.setSVPChallenge(40, 9)
    # Optional: Set as random lattice
    # lat.setRandom(10, 10, 1000, 10000);
    
    # Print lattice basis (optional)
    # print(lat);
    
    # Corresponds to: lat.computeGSO();
    print("Computing GSO information...")
    lat.computeGSO()
    
    # Corresponds to: std::vector<int> v = lat.mulVecBasis(lat.ENUM(4000000));
    print("Executing ENUM algorithm to search for shortest vector (radius=4000000)...")
    start_time = time.time()
    
    # Call ENUM algorithm
    coeff_vector = lat.ENUM(4000000.0)  # Note: radius is double type
    #lat.deepBKZ(35, 0.99);
    # Multiply coefficient vector with basis to get lattice vector
    v = lat.mulVecBasis(coeff_vector)
    
    end_time = time.time()
    
    print(f"ENUM search completed, time elapsed: {end_time - start_time:.2f} seconds")
    
    # Print vector v (optional)
    # print(v);
    
    # Optional: Lattice reduction algorithms
    # lat.L2(0.99, 0.51);
    # lat.LLL(0.99);
    # lat.dualBKZ(20, 0.99);
    # lat.deepBKZ(35, 0.99);
    # lat.dualDeepBKZ(30, 0.99);
    # lat.potBKZ(35, 0.99);
    # lat.BKZ(6, 0.99);
    # print("Euclidean norm: ", lat.b1Norm());
    
    # Corresponds to: std::cout << "Euclidean norm of vector v: " << vectorNorm(v) << std::endl;
    norm_v = vector_norm(v)
    print(f"Euclidean norm of vector v: {norm_v}")
    
    # Optional: Print lattice information
    # std::cout << lat;
    # printf("rhf = %Lf\n", lat.rhf());
    # printf("sl = %Lf\n", lat.sl());
    # std::cout << lat.volume(false) << std::endl;
    
    # Get additional information (for debugging only)
    try:
        b1_norm = lat.b1Norm()
        print(f"First basis vector norm: {b1_norm}")
    except:
        pass
    
    try:
        rhf = lat.rhf()
        print(f"Root Hermite factor: {rhf}")
    except:
        pass
    
    try:
        sl = lat.sl()
        print(f"Sequence length: {sl}")
    except:
        pass
    
    print("\nTest completed!")
    
    return {
        #'coeff_vector': coeff_vector,
        #'vector': v,
        #'norm': norm_v,
        'time': end_time - start_time
    }

if __name__ == "__main__":
    try:
        result = main()
        
        # Optional: Display more detailed information
        print("\n=== Detailed Information ===")
        #print(f"Coefficient vector length: {len(result['coeff_vector'])}")
       # print(f"Lattice vector length: {len(result['vector'])}")
        #print(f"First 5 elements of coefficient vector: {result['coeff_vector'][:5]}")
        #print(f"First 5 elements of lattice vector: {result['vector'][:5]}")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()