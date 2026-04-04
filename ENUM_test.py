#!/usr/bin/env python3
"""
ENUM Algorithm Debugging Script
"""

import sys
import os
sys.path.append('lib')

import lattice_env
import numpy as np

def test_enum_logic():
    """Test ENUM logic"""
    print("="*60)
    print("ENUM Algorithm Logic Test")
    print("="*60)
    
    # Create environment
    lattice = lattice_env.create_lattice(10, 10)  # Small dimension for debugging
    
    # Initialize lattice (important!)
    print("Initializing lattice...")
    try:
        lattice.setRandom(10, 10, -100, 100)  # Random initialization
        lattice.computeGSO()  # Compute GSO
        print("SUCCESS: Lattice initialized successfully")
    except Exception as e:
        print(f"WARNING: Initialization failed: {e}")
    
    env = lattice_env.LatticeEnv(lattice)
    
    # Set configuration
    config = lattice_env.Config()
    config.max_dimension = 10
    config.action_range = 2.0
    config.max_steps = 100
    env.set_config(config)
    
    # Reset
    print("\nResetting environment...")
    state = env.reset(R=500.0)  # Large radius to ensure solution can be found
    
    # Print debug information
    print("Initial state:")
    print(f"  Dimension: {env.dimension}")
    print(f"  State features: {np.array(state)[:5]}...")
    
    # Run a few steps, observe state changes
    print("\nRunning steps...")
    for i in range(20):
        action = 0  # Neutral action
        next_state, reward, done, info = env.step(action)
        
        print(f"\nStep {i+1}:")
        print(f"  Action: {action}")
        print(f"  Reward: {reward:.6f}")
        print(f"  k value: {env.current_k}")
        print(f"  rho: {env.current_rho}")
        print(f"  Solution found: {env.solved}")
        print(f"  Info: {info}")
        
        if done:
            print(f"  Termination reason: {info}")
            break
    
    print("\n" + "="*60)
    print("Test completed")
    print("="*60)
    
    # Print final state
    print(f"Final k value: {env.current_k}")
    print(f"Final solution status: {env.solved}")
    print(f"Best norm: {env.best_norm}")

def test_different_actions():
    """Test the impact of different actions"""
    print("\n" + "="*60)
    print("Different Actions Test")
    print("="*60)
    
    lattice = lattice_env.create_lattice(8, 8)
    lattice.setRandom(8, 8, -50, 50)
    lattice.computeGSO()
    
    env = lattice_env.LatticeEnv(lattice)
    env.reset(R=300.0)
    
    # Test different actions
    actions = [-2, -1, 0, 1, 2]
    
    for action in actions:
        # Restart each time
        env.reset(R=300.0)
        
        steps = 0
        total_reward = 0
        
        for _ in range(10):
            state, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        print(f"Action {action:2d}: Steps={steps:2d}, Total reward={total_reward:7.4f}, "
              f"Final k={env.current_k}, Solved={env.solved}")

def test_with_real_svp():
    """Test using SVP challenge"""
    print("\n" + "="*60)
    print("SVP Challenge Test")
    print("="*60)
    
    try:
        lattice = lattice_env.create_lattice(20, 20)
        
        # Try to load SVP challenge
        lattice.setSVPChallenge(20, 1)
        lattice.computeGSO()
        
        env = lattice_env.LatticeEnv(lattice)
        env.reset(R=200.0)
        
        print("SVP challenge loaded successfully")
        print(f"Lattice dimension: {env.dimension}")
        
        # Run some steps
        for i in range(15):
            action = 0
            state, reward, done, info = env.step(action)
            
            if i % 5 == 0:
                print(f"Step {i}: k={env.current_k}, rho={env.current_rho:.2f}, "
                      f"reward={reward:.4f}")
            
            if done:
                print(f"Ended at step {i}: {info}")
                break
        
    except Exception as e:
        print(f"SVP test failed: {e}")
        print("setSVPChallenge method may not be exposed")

if __name__ == "__main__":
    print(f"Python path: {sys.path[-1]}")
    
    # Run tests
    test_enum_logic()
    test_different_actions()
    test_with_real_svp()  # Optional
    
    print("\nDEBUGGING SUGGESTIONS:")
    print("1. Check if GSO is computed correctly")
    print("2. Observe if k value changes normally (0->n->0->n)")
    print("3. Check if rho value is reasonable (should gradually decrease)")
    print("4. Ensure it doesn't 'find solution' at the first step")