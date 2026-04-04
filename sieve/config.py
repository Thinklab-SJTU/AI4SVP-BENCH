# config.py
import numpy as np

class Config:
    # Sieve parameters
    DIMENSION = 100  # Lattice dimension
    GAMMA = 0.8    # Sieve parameter
    
    # AI model parameters
    INPUT_DIM = 6    # Input feature dimension
    HIDDEN_DIM = 64  # Hidden layer dimension
    TOP_K = 10        # Number of centers to check each time
    
    # Training parameters
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    EPOCHS = 50
    
    # Data collection
    MAX_TRAINING_SAMPLES = 30000
    TRAIN_TEST_SPLIT = 0.8