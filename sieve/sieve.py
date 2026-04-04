#!/usr/bin/python3

import argparse
from numpy.linalg import norm
from sample import sample_vec
from ajtai_generator import gen_basis
from nv_sieve import nguyen_vidick_sieve
from g_sieve import gauss_sieve
from k_sieve import double_sieve

def main(args):
    """
        Main method for running a sieving algorithm. This method prints
        the shortest vector found using the specified sieving algorithm

        Parameters:
            args:   Contains a Namespace object with arguments necessary for
                    running a specific sieve

    """
    
    # Get Ajtai generator parameters
    n, r, q = args.n[0], args.r[0], args.q[0]
    basis, w = gen_basis(n, r, q)
    d = n + r
    
    # Compute the Minkowski bound
    minkowski_bound = (d**(0.5))*(q**r)**(1/d)
    
    # Run the Nguyen-Vidick sieve
    if args.subparser_name == "nv":
        N,  gamma = args.N[0], args.gamma[0]
        S = sample_vec(basis, N, n, r, q)
        sieve = nguyen_vidick_sieve
        arguments = [S, gamma]
    
    # Run the Gauss sieve
    elif args.subparser_name == "gauss":
        c = args.c[0]
        sieve = gauss_sieve
        arguments = [basis, c]

    # Run the Double sieve
    elif args.subparser_name == "double":
        gamma = args.gamma[0]
        d = n+r
        N = args.N[0] if args.N is not None else int(2**(0.208*d))
        S = sample_vec(basis, N, n, r, q)
        sieve = double_sieve
        arguments = [S, gamma, minkowski_bound]

    # Get the shortest vector found
    v = sieve(*arguments)

    # Print it along with its norm
    print(v, norm(v))
    


if __name__ == "__main__":
    # Make a parser
    parser = argparse.ArgumentParser(description='Module containing lattice sieving algorithms')
    
    # Add Ajtai basis generation arguments
    ajtai_group = parser.add_argument_group('Ajtai basis generation parameters')
    ajtai_group.add_argument('-n', metavar='n', nargs=1, type=int, help='Number of vectors for Ajtai basis', required=True)
    ajtai_group.add_argument('-r', metavar='r', nargs=1, type=int, help='Dimension of vectors for Ajtai basis', required=True)
    ajtai_group.add_argument('-q', metavar='q', nargs=1, type=int, help='Prime modulus', required=True)
    
    # Make subparsers for the sieving algorithms
    subparsers = parser.add_subparsers(dest='subparser_name', title='Available sieving algorithms')

    # Ngyuen-Vidick sieve
    parser_nv = subparsers.add_parser('nv', help='The Nguyen-Vidick sieve')
    nv_group = parser_nv.add_argument_group('Arguments to NV sieve')
    nv_group.add_argument('-N', metavar='N', nargs=1, type=int, help='Number of samples to draw', required=True)
    nv_group.add_argument('-gamma', metavar='gamma', nargs=1, type=float, help='Constant used in norm reduction step', required=True)
    
    # Gauss sieve
    parser_gauss = subparsers.add_parser('gauss', help='The Gauss sieve')
    gauss_group = parser_gauss.add_argument_group('Arguments to Gauss sieve')
    gauss_group.add_argument('-c', metavar='c', nargs=1, type=int, help='Number of collisions', required=True)


    # Double sieve
    parser_double = subparsers.add_parser('double', help='The Double Sieve')
    double_group = parser_double.add_argument_group('Arguments to the Double sieve')
    double_group.add_argument('-N', metavar='N', nargs=1, type=int, help='Number of samples to draw (optional). Default is 2^(0.415*d).')
    double_group.add_argument('-gamma', metavar='gamma', nargs=1, type=float, help='Constant used in norm reduction step', required=True)


    # Parse the args and call main
    args = parser.parse_args()
    parser.print_help() if args.subparser_name == None else main(args)
    