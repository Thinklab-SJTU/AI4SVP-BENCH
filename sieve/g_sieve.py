import numpy as np
from numpy.linalg import norm
from sample import sample_vec

def gauss_sieve(basis, c):
    """
        The Gauss sieve. 

        Parameters:
            basis:  Basis for the lattice we want to run the sieve on
            c:      Number of collisions before we stop the sieve

        Returns:
            v:      Shortest vector found in the sieve
    """
    
    # Dimension of the lattice
    d = len(basis) 
    
    # Parameters from Ajtai generator
    n = sum(1 for i in range(d) if basis[i][i] == 1)
    r = d - n
    q = int(basis[n][n])
    
    # Compute and print the Minkowski bound
    minkowski_bound = (d**(0.5))*(q**r)**(1/d)
    print("Minkowski_bound = " + str(minkowski_bound))
    
    L = []
    S = []
    K = 0

    # Since the sieve could run for a long time, we catch a Ctrl+C
    # and return the shortest vector found so far
    try:
        # Run while we have fewer than c collisions
        while K < c:
            # Draw from the top of S if S is not empty, otherwise sample a new vector
            v_new = S.pop() if len(S) else sample_vec(basis, 1, n, r, q)[0]
            
            # Reduce it
            v_new = gauss_reduce(v_new, L, S)
            
            # If v_new is 0, we have a collision
            if np.count_nonzero(v_new) == 0:
                K += 1
            else:
                L.append(v_new)
                print("\r Min norm in L: " + str(norm(min(L, key=lambda v: norm(v)))) , end="\r")
    # Got Ctrl+C
    except KeyboardInterrupt:
        # Return the minimum so far
        return min(L, key=lambda v: norm(v)).reshape(d)
    
    # Return minimum once the loop ends
    return min(S, key=lambda v: norm(v))


def gauss_reduce(p, L, S):
    """
        Helper method in the gauss sieve algorithm. Takes a vector p
        and reduces it using all the other vectors in L

        Parameters:
            p:  The vector to reduce
            L:  The list of vectors to check against
            S:  Stack of vectors to draw p from in the next step

        Returns:
            p:  Reduced vector
    """
    
    # Subroutine to check if we can find a smaller vector
    def check_shorter_vec(p, L):
        for v in L:
            if np.count_nonzero(v) > 0:
                if (norm(v) <= norm(p)) and (norm(p-v) <= norm(p)):
                    return (True, v)
        return (False, -1)
    
    shorter_found = True
    # Loop until a smaller vector exists and reduce p by it
    while shorter_found:
        shorter_found, v = check_shorter_vec(p, L)
        if shorter_found: 
            p -= v

    # L_indices stores the indices of v_i's that match the condition below
    L_indices = []
    
    # When deleting elements from a list, all the indices shift to the left after each operation,
    # so we use a counter to account for that offset
    ct = 0

    # Loop over L and see if we can find v_i such that ||v_i|| > ||p||
    # and ||v_i - p|| <= ||v_i||
    for i in range(len(L)):
        v = L[i]
        if (norm(v) > norm(p)) and (norm(v - p) <= norm(v)):
            L_indices.append(i - ct)
            ct += 1
    
    # Remove matched v_i's from the list and append v_i - p to S
    for i in L_indices:
        v = L.pop(i)
        S.append(v - p)

    return p
