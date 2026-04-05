#!/usr/bin/env python3
"""
Python test script - verify lattice_env module
"""

import sys
import os

# Add module path
current_dir = os.path.dirname(os.path.abspath(__file__))
build_dir = os.path.join(current_dir, "..", "build", "lib")
sys.path.insert(0, build_dir)

def test_import():
    """Test module import"""
    print("=" * 50)
    print("Testing lattice_env module import")
    print("=" * 50)
    
    try:
        import lattice_env
        print(f"? Successfully imported lattice_env version: {lattice_env.__version__}")
        return lattice_env
    except ImportError as e:
        print(f"? Import failed: {e}")
        print("\nPossible causes:")
        print("1. Module not compiled - please run build.sh first")
        print("2. Python version mismatch")
        print("3. Module path incorrect")
        return None

def test_create_lattice(module):
    """Test creating Lattice object"""
    print("\n" + "=" * 50)
    print("Test creating Lattice object")
    print("=" * 50)
    
    try:
        # Create lattice object
        lattice = module.create_lattice(40, 40)
        print("? Successfully created Lattice object")
        
        # Note: need to call some methods to initialize the lattice
        # e.g.: lattice.setSVPChallenge(40, 0)
        # but these methods need to be exposed on the C++ side first
        
        return lattice
    except Exception as e:
        print(f"? Creation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_create_env(module, lattice):
    """Test creating environment"""
    print("\n" + "=" * 50)
    print("Test creating environment")
    print("=" * 50)
    
    try:
        env = module.LatticeEnv(lattice)
        print("? Successfully created LatticeEnv object")
        
        # Test configuration
        config = module.Config()
        config.max_dimension = 40
        config.action_range = 5.0
        config.max_steps = 2000
        env.set_config(config)
        print("? Successfully set configuration")
        
        return env
    except Exception as e:
        print(f"? Environment creation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_reset_and_step(env):
    """Test reset and step"""
    print("\n" + "=" * 50)
    print("Test environment interaction")
    print("=" * 50)
    
    try:
        # Reset environment
        state = env.reset(R=100.0)
        print(f"? Reset successful, state dimension: {len(state)}")
        print(f"   Environment dimension: {env.dimension}")
        
        # Execute a few steps
        for i in range(5):
            # Random action (-5 to 5)
            action = 0  # Test action 0 first
            
            state, reward, done, info = env.step(action)
            print(f"  Step {i+1}: reward={reward:.4f}, done={done}, info={info}")
            
            if done:
                print(f"  Early termination: {info}")
                break
        
        print(f"\n? Environment interaction test complete")
        print(f"   Current k: {env.current_k}")
        print(f"   Current rho: {env.current_rho}")
        print(f"   Solved: {env.solved}")
        
    except Exception as e:
        print(f"? Interaction test failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main test function"""
    print("Starting lattice_env module tests")
    print(f"Python version: {sys.version}")
    print(f"Current directory: {os.getcwd()}")
    print(f"Build directory: {build_dir}")
    
    # 1. Test import
    module = test_import()
    if not module:
        return
    
    # 2. Test creating Lattice
    lattice = test_create_lattice(module)
    if not lattice:
        return
    
    # 3. Test creating environment
    env = test_create_env(module, lattice)
    if not env:
        return
    
    # 4. Test interaction
    test_reset_and_step(env)
    
    print("\n" + "=" * 50)
    print("? All tests complete!")
    print("=" * 50)
    print("\nNext steps:")
    print("1. Modify lattice.h to add friend class")
    print("2. Expose more Lattice methods in C++")
    print("3. Implement full step-by-step ENUM execution")

if __name__ == "__main__":
    main()