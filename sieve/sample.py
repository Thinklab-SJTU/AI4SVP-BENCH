import numpy as np
from numpy.random import default_rng

def sample_vec(basis, N, n, r, q):
    """
        Given a basis for a lattice, sample N lattice vectors.
        We draw from a Gaussian distribution, but round the points to
        the nearest integer, to effectively draw from a 'Discrete Gaussian'

        Parameters:
            basis:  The lattice basis
            N:      Number of lattice points to sample
            n:      Number of vectors for Ajtai generator
            r:      Dimension of vectors for Ajtai generator
            q:      Prime modulus for Ajtai generator

        Returns:
            S:      Set containing the generated lattice vectors
    """
    S = []
    rng = default_rng()
    
    for _ in range(N):
        # Draw from a normal distribution
        x = rng.normal(0, 2*q, n+r)

        # Round off the coordinates to the nearest integer
        x = np.array([int(v) for v in x]).reshape(n+r, 1)
        
        # Compute lattice vector by multiplying it with the basis
        S.append(basis.dot(x).reshape(n+r))
    
    # Return the vectors
    return S