# debug_environment.py
import os
import torch
import numpy as np
import sys
import matplotlib.pyplot as plt
from collections import deque
from enum_environment import EnumEnvironment
from ppo_agent import ActorCritic, PPOAgent

# Try to import C++ lattice environment
sys.path.append('../lib')
try:
    import lattice_env
    CPP_ENV_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import lattice_env: {e}")
    CPP_ENV_AVAILABLE = False

def test_environment_setup():
    """Test if environment setup is correct"""
    print("=== Environment Setup Test ===")
    
    # Create lattice
    lat = lattice_env.create_lattice_int(40, 40)
    #lat.setRandom(5, 5, 10, 100)
    lat.setSVPChallenge(40, 9)
    print(f"1. Lattice object created successfully: ")
    
    # Set SVP challenge
    print("2. SVP challenge setup completed")
    
    # Compute GSO
    lat.computeGSO()
    print("3. GSO computation completed")
    
    # Test traditional ENUM
    print("\n=== Traditional ENUM Test ===")
    coeff = lat.ENUM(4000000)
    v = lat.mulVecBasis(coeff)
    norm = np.linalg.norm(v)
    print(f"Vector norm found by traditional ENUM: {norm}")
    
    # Check if lattice basis is reasonable
    try:
        b1_norm = lat.b1Norm()
        print(f"First basis vector norm: {b1_norm}")
        
        # Check lattice volume
        volume = lat.volume()
        print(f"Lattice volume: {volume}")
        
        # Check RHF
        rhf = lat.rhf()
        print(f"Root Hermite factor: {rhf}")
        
    except Exception as e:
        print(f"Cannot get lattice metrics: {e}")
    
    return lat, norm

def test_rl_wrapper(lat):
    """Test if RL wrapper works correctly"""
    print("\n=== RL Wrapper Test ===")
    
    #lat = lattice_env.create_lattice_int(5, 5)
    #lat.setRandom(5, 5, 10, 100)
    #print(lat)
    lat.computeGSO()
    try:
        # Check if necessary interfaces are exposed
        print("Checking RL wrapper interfaces...")
        
        # Create RL wrapper
        rl_wrapper = lattice_env.RL_ENUM_Wrapper(lat)
        print("1. RL wrapper created successfully")
        
        # Reset
        rl_wrapper.reset(4000000.0)
        print(rl_wrapper,"I beg you please")
        print("2. RL wrapper reset successful")
        for i in range(1000000):
            action = 0 
            reward, done, info = rl_wrapper.step(action)
            print(f"Step {i}: reward={reward}, done={done}, info={info}")
            if done:
                break
        print(rl_wrapper,"I beg you please")
        print("Best vector:", rl_wrapper.get_best_vector())
        # Get state
        state = rl_wrapper.get_state()
        print(f"3. State retrieved: k={state.current_k}, rho={state.current_rho}")
        
        # Execute one step
        action = 0  # Neutral action
        reward, done, info = rl_wrapper.step(action)
        print(f"4. Step executed: reward={reward}, done={done}, info={info}")
        
        return True
        
    except Exception as e:
        print(f"RL wrapper test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Test environment setup
    lat, norm = test_environment_setup()
    
    # Test RL wrapper
    rl_success = test_rl_wrapper(lat)
    
    if rl_success and norm < 2000:  # Reasonable norm
        print("\n? Environment setup correct, ready to start training")
    else:
        print("\n? Environment setup has issues, needs fixing")