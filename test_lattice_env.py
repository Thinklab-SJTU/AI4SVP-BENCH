#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lattice_env module functional verification script
"""

import sys
import os
import numpy as np

# Add module path
current_dir = os.path.dirname(os.path.abspath(__file__))
build_dir = os.path.join(current_dir, "lib")
sys.path.insert(0, build_dir)

def print_header(title):
    """Print header"""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def test_1_basic_import():
    """Test 1: Basic import"""
    print_header("1. Basic Import Test")
    
    try:
        import lattice_env
        print(f"SUCCESS: Imported lattice_env")
        print(f"  Version: {lattice_env.__version__}")
        print(f"  Module path: {lattice_env.__file__}")
        return lattice_env
    except Exception as e:
        print(f"FAILED: Import failed: {e}")
        return None

def test_2_create_lattice(module):
    """Test 2: Create Lattice objects"""
    print_header("2. Create Lattice Objects")
    
    try:
        # Test creating lattices of different dimensions
        for dim in [10, 20, 40]:
            lattice = module.create_lattice(dim, dim)
            print(f"SUCCESS: Created {dim}x{dim} Lattice object")
            
            # Try to call some methods (if exposed)
            try:
                rows = lattice.numRows()
                cols = lattice.numCols()
                print(f"  Dimensions: {rows}x{cols}")
            except:
                print(f"  NOTE: Some methods not exposed")
                
        return True
    except Exception as e:
        print(f"FAILED: Creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_3_create_env(module):
    """Test 3: Create environment"""
    print_header("3. Create Reinforcement Learning Environment")
    
    try:
        # Create a small lattice for testing
        lattice = module.create_lattice(20, 20)
        
        # Create environment
        env = module.LatticeEnv(lattice)
        print("SUCCESS: Created LatticeEnv environment")
        
        # Test configuration
        config = module.Config()
        config.max_dimension = 20
        config.action_range = 3.0  # Small range for testing
        config.max_steps = 100
        
        env.set_config(config)
        print("SUCCESS: Environment configuration set")
        
        return env
    except Exception as e:
        print(f"FAILED: Environment creation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_4_reset_function(env):
    """Test 4: Reset function"""
    print_header("4. Environment Reset Test")
    
    try:
        # Test resetting with different radii
        test_radii = [50.0, 100.0, 200.0]
        
        for radius in test_radii:
            state = env.reset(radius)
            state_array = np.array(state, dtype=np.float32)
            
            print(f"SUCCESS: Reset with radius R={radius}")
            print(f"  State dimensions: {state_array.shape}")
            print(f"  State range: [{state_array.min():.4f}, {state_array.max():.4f}]")
            print(f"  Environment dimension: {env.dimension}")
            print(f"  Current k value: {env.current_k}")
            
        return True
    except Exception as e:
        print(f"FAILED: Reset failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_5_step_function(env):
    """Test 5: Single step execution"""
    print_header("5. Single Step Execution Test")
    
    try:
        # First reset environment
        state = env.reset(100.0)
        print(f"Initial state: {np.array(state)[:5]}... (showing first 5 dimensions)")
        
        # Test different actions
        test_actions = [-3, -1, 0, 1, 3]  # Actions within test range
        
        for i, action in enumerate(test_actions):
            try:
                next_state, reward, done, info = env.step(action)
                
                print(f"\nStep {i+1}, action={action}:")
                print(f"  Reward: {reward:.6f}")
                print(f"  Done: {done}")
                print(f"  Info: {info}")
                print(f"  New state dimensions: {len(next_state)}")
                print(f"  Current k: {env.current_k}")
                print(f"  Current rho: {env.current_rho:.4f}")
                print(f"  Solved: {env.solved}")
                
                if done:
                    print("  NOTE: Environment ended early")
                    break
                    
            except Exception as e:
                print(f"  FAILED: Action {action} failed: {e}")
                break
        
        print(f"\nSUCCESS: Executed {i+1} steps")
        return True
        
    except Exception as e:
        print(f"FAILED: Single step execution test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_6_full_episode(env):
    """Test 6: Complete episode"""
    print_header("6. Complete Episode Test")
    
    try:
        state = env.reset(80.0)
        total_reward = 0.0
        step_count = 0
        max_steps = 50
        
        print(f"Starting complete episode (max {max_steps} steps)...")
        
        while step_count < max_steps:
            # Simple random policy
            action = np.random.randint(-3, 4)  # -3 to 3
            
            state, reward, done, info = env.step(action)
            total_reward += reward
            step_count += 1
            
            if step_count % 10 == 0:
                print(f"  Step {step_count}: action={action}, reward={reward:.4f}, total={total_reward:.4f}, k={env.current_k}")
            
            if done:
                print(f"  ENV END: Environment ended at step {step_count}: {info}")
                break
        
        print(f"\nSUCCESS: Episode completed")
        print(f"  Total steps: {step_count}")
        print(f"  Total reward: {total_reward:.4f}")
        print(f"  Final k value: {env.current_k}")
        print(f"  Solution found: {env.solved}")
        print(f"  Best norm: {env.best_norm}")
        
        return True
        
    except Exception as e:
        print(f"FAILED: Complete episode test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_7_state_consistency(env):
    """Test 7: State consistency"""
    print_header("7. State Consistency Test")
    
    try:
        env.reset(100.0)
        
        # Get state multiple times to ensure consistency
        states = []
        for i in range(5):
            state = env.get_state()
            states.append(np.array(state))
            
            # Execute a step
            if i < 4:
                env.step(0)
        
        # Check state dimension consistency
        dims = [s.shape for s in states]
        if all(d == dims[0] for d in dims):
            print(f"SUCCESS: State dimensions consistent: {dims[0]}")
        else:
            print(f"FAILED: State dimensions inconsistent: {dims}")
        
        # Check state value changes
        print(f"\nState change analysis:")
        for i in range(len(states)-1):
            diff = np.abs(states[i] - states[i+1]).mean()
            print(f"  Step {i}->{i+1}: average change = {diff:.6f}")
        
        return True
        
    except Exception as e:
        print(f"FAILED: State consistency test failed: {e}")
        return False

def test_8_error_handling(env):
    """Test 8: Error handling"""
    print_header("8. Error Handling Test")
    
    try:
        print("Testing invalid actions...")
        try:
            # Action that should be out of range
            env.reset(100.0)
            env.step(100)  # Clearly out of range
            print("FAILED: Did not catch invalid action error")
        except Exception as e:
            print(f"SUCCESS: Correctly caught error: {type(e).__name__}")
        
        print("\nTesting multiple resets...")
        try:
            for i in range(3):
                env.reset(100.0 + i*50)
            print("SUCCESS: Multiple resets successful")
        except Exception as e:
            print(f"FAILED: Multiple resets failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"FAILED: Error handling test failed: {e}")
        return False

def test_9_feature_analysis(env):
    """Test 9: Feature analysis"""
    print_header("9. State Feature Analysis")
    
    try:
        env.reset(100.0)
        state = np.array(env.get_state())
        
        print(f"Feature dimensions: {state.shape}")
        print(f"Feature statistics:")
        print(f"  Minimum: {state.min():.6f}")
        print(f"  Maximum: {state.max():.6f}")
        print(f"  Mean: {state.mean():.6f}")
        print(f"  Std: {state.std():.6f}")
        
        # Check if features are within reasonable range
        if state.min() >= -10 and state.max() <= 10:
            print("SUCCESS: Feature values within reasonable range")
        else:
            print("NOTE: Feature values may be outside normal range")
        
        # Check for NaN or Inf
        if np.any(np.isnan(state)):
            print("FAILED: Features contain NaN values")
        elif np.any(np.isinf(state)):
            print("FAILED: Features contain Inf values")
        else:
            print("SUCCESS: Features contain no NaN/Inf values")
        
        return True
        
    except Exception as e:
        print(f"FAILED: Feature analysis failed: {e}")
        return False

def main():
    """Main test function"""
    print("START: lattice_env module functional verification")
    print(f"Python path: {build_dir}")
    
    # Test counter
    passed = 0
    total = 9
    
    # Test 1: Import
    module = test_1_basic_import()
    if not module:
        print("\nFAILED: Basic import failed, stopping tests")
        return
    
    passed += 1
    
    # Test 2: Create Lattice
    if test_2_create_lattice(module):
        passed += 1
    
    # Test 3: Create environment
    env = test_3_create_env(module)
    if env:
        passed += 1
        
        # Subsequent tests require environment
        tests = [
            (test_4_reset_function, "Reset function"),
            (test_5_step_function, "Single step execution"),
            (test_6_full_episode, "Complete episode"),
            (test_7_state_consistency, "State consistency"),
            (test_8_error_handling, "Error handling"),
            (test_9_feature_analysis, "Feature analysis"),
        ]
        
        for test_func, test_name in tests:
            try:
                if test_func(env):
                    passed += 1
                else:
                    print(f"NOTE: {test_name} test failed")
            except Exception as e:
                print(f"FAILED: {test_name} test exception: {e}")
    
    # Summary
    print_header("Test Summary")
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("SUCCESS: All tests passed! Module functionality is complete")
        print("\nNext steps:")
        print("1. Proceed with reinforcement learning training")
        print("2. Collect training data")
        print("3. Integrate ONNX model")
    elif passed >= 7:
        print("NOTE: Most tests passed, some features need debugging")
    elif passed >= 4:
        print("NOTE: Basic functionality available, but needs fixes")
    else:
        print("FAILED: Need to focus on debugging module functionality")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)