import numpy as np
from multiprocessing import Pool
from itertools import repeat

def remove_zeros(S):
    """
        Removes zero vectors from the given list of vectors

        Parameters:
            S: List of vectors

        Returns: 
            S: The list with zero vectors removed
    """
    # Maintain a counter so we can account for shift in indices
    # that happen due to the 'del' operation on a list
    ct = 0
    zero_indices = []
    
    # Loop and check for zero vectors
    for i in range(len(S)):
        if np.count_nonzero(S[i]) == 0:
            zero_indices.append(i - ct)
            ct += 1

    # Loop over the 'corrected' indices and delete zero elements
    for i in zero_indices:
        del S[i]
    
    # Return the list
    return S

def par_remove_zeros_loop(S, i):
    """
        Helper method for parallel execution of remove_zeros()
        Simply checks the given index for the zero vector or 
        value None

        Parameters:
            S: The set of vectors
            i: Index to check

        Returns:
            True if given index is 0 or None, else False
    """
    return S[i] is None or np.count_nonzero(S[i]) == 0

def par_remove_zeros(S):
    """
        Removes zero vectors and None elements from the given 
        list of vectors

        Parameters:
            S: List of vectors

        Returns: 
            S: The list with zero vectors and None elements removed
    """
    # Maintain a counter so we can account for shift in indices
    # that happen due to the 'del' operation on a list
    ct = 0
    N = len(S)

    # Start 16 processes to check which vectors are 0 or None
    with Pool(16) as p:
        res = p.starmap(par_remove_zeros_loop, zip(repeat(S, N), range(N)))

    # Loop over the results and delete the right elements
    for idx, val in enumerate(res):
        if val:
            del S[idx - ct]
            ct += 1

    # Return the new list    
    return S