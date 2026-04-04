import numpy as np
from numpy.linalg import norm
from multiprocessing import Pool
from itertools import combinations
from math import comb
from random import randint, sample
from utils import remove_zeros, par_remove_zeros

def double_sieve(S, gamma, minkowski_bound):
    """
        The double sieve. This method iteratively calls the sieve step until we have no
        more vectors to reduce or we reach the Minkowski bound. Vectors in the sieve
        are reduced at each step based on the condition: 
        For v, w in S, 
            Promote v +- w if norm(v +- w) <= gamma * R
        for appropriate gamma and R

        Parameters:
            S:                  The original set
            gamma:              The reduction factor
            minkowski_bound:    Upper bound on size of shortest vector

        Returns:
            v:                  Vector with norm less than minkowski bound.

    """
    S_0 = S
    marked_list = []
    avg_list = []
    
    # Run sieve until we run out of vectors
    while len(S) > 0:
        S_0 = S
        # Run a sieve step and get the set for the next step
        S, marked, avg_length = lattice_sieve_two(S, gamma)
        
        #S = lattice_sieve(S, gamma)

        # Keep track of the number of reducible pairs at each step
        marked_list.append(marked)
        avg_list.append(avg_length)

        # Print the min norm at the current step
        if len(S):
            min_norm = norm(min(S, key=lambda v: norm(v)))
            print("\r" + str(len(S)) + ", Min norm in S: " + str(min_norm), end="\r")
            
            # If the min norm is less than the upper bound, we can stop
            if min_norm < minkowski_bound:
                print(marked_list)
                print(avg_list)
                return min(S, key=lambda v: norm(v))
            
    # Return the vector we found
    
    print(marked_list)
    print(avg_list)
    return min(S_0, key=lambda v: norm(v))

def lattice_sieve(S, gamma):
    """
        One step of the sieve. The loop in this method only runs until we have enough
        vectors ~ 2^(0.415d) for the next step of the sieve.

        Parameters:
            S:      The starting set of vectors
            gamma:  The reduction factor, typically 0.99

        Returns:
            S_p:    The set for the next step of the sieve
    """
    
    # Start with an empty sieve and compute R as the mean norm of the vectors in 
    # the input set
    S_p = []
    #R = norm(max(S, key=lambda x: norm(x)))
    R = sum(norm(v) for v in S)/len(S)
    gR = gamma*R
    N = len(S)
    
    # Counter for the number of vectors that make it through the sieve
    num_next_sieve = 0
    
    # Set to keep track of which pairs of vectors we have already looked at
    already_generated = set()
    # Function that generates random indices
    gen_indices = lambda: (randint(0, N-1), randint(0, N-1))
    
    # Main sieving loop that runs until we have enough vectors in the sieve
    while num_next_sieve < N:
        # Generate random indices to look at
        new = False
        while not new:
            i, j = gen_indices()
            if i != j and (i, j) not in already_generated:
                new = True
                already_generated.add((i, j))
                already_generated.add((j, i))
        
        # See if we can reduce this pair of vectors
        v, w = S[i], S[j]
        if np.count_nonzero(v - w) and norm(v - w) <= gR:
            S_p.append(v - w)
            num_next_sieve += 1
        elif np.count_nonzero(v + w) and norm(v + w) <= gR:
            S_p.append(v + w)
            num_next_sieve += 1
    
    # Return our new set
    return S_p

def lattice_sieve_two(S, gamma):
    """
        One step of the sieve. This method is the same as lattice_sieve(), except we don't stop the
        loop once we have enough vectors. This is used for instrumentation. We do truncate the sieve
        to size ~2^(0.415d), however we let the loop run in order to compute the number of pairs
        that are reducible at each sieve step.

        Parameters:
            S:              The starting set of vectors
            gamma:          The reduction factor, typically 0.99

        Returns:
            S_p:            The set for the next step of the sieve
            num_next_sieve: The number of pairs that were reducible in the previous set
    """
    
    # Start with an empty sieve and compute R as the mean norm of the vectors in 
    # the input set
    S_p = []
    R = sum(norm(v) for v in S)/len(S)
    gR = gamma*R
    N = len(S)
    
    # Counter for the number of pairs that make it to the next step
    num_next_sieve = 0
    
    # Loop over all combinations of vectors and reduce
    for i, j in combinations(range(N), 2):
        v, w = S[i], S[j]
        if np.count_nonzero(v - w) and norm(v - w) <= gR:
            S_p.append(v - w)
            num_next_sieve += 1
        elif np.count_nonzero(v + w) and norm(v + w) <= gR:
            S_p.append(v + w)
            num_next_sieve += 1
    
    # Only promote N vectors to the next step
    avg_length = sum(norm(v) for v in S_p)/len(S_p)
    return sample(S_p, N), num_next_sieve, avg_length

def run_loop(S, gR, i, j):
    """
        Helper method for parallel execution of the double sieve
        Checks if the two given vectors can be reduced

        Parameters:
            S:              Set of vectors
            gR:             Reduction factor
            i:              First index for pair
            j:              Second index for pair

        Returns:
            v +- w or None: The reduced pair, or None
    """
    # Get the pair
    v, w  = S[i], S[j]
    
    # Check if they can be reduced and return appropriate 
    # sum or difference
    if norm(v - w) <= gR:
        return v - w
    elif norm(v + w) <= gR:
        return v + w

def parallel_lattice_sieve(S, gamma):
    """
        Method for parallel execution of the double sieve

        Parameters:
            S:      Set of vectors
            gamma:  Reduction factor

        Returns:
            S_p:    The next sieve set
            marked: Number of pairs from previous set that were reduced
    """
    # Start with empty set
    S_p = []
    
    # Compute the value of R as the mean of the vectors in S
    R = sum(norm(v) for v in S)/len(S)
    gR = gamma*R
    N = len(S)

    # Number of combinations to try
    num_repeat = comb(N, 2)

    # Start a pool of 16 processes since we have 16 pprocessors
    with Pool(16) as p:
        # Get the results in an array
        S_p = p.starmap(run_loop, [(S, gR, i, j) for i, j in combinations(range(N), 2)])

    # The retured list might have None values and zero vectors, so remove them
    S_p = par_remove_zeros(S_p)
    
    # Get the number of reduced vectors
    marked = len(S_p)
    
    # Randomly pick N vectors to move to the next step of the sieve and return
    return sample(S_p, N), marked
