#!/usr/bin/python3

import numpy
import random # Warning standand RNGs cannot use to generate lattice problems: fix this!
import sys

def gen_basis(n, r, q):
    U = [ [ random.randrange(int(q)) for i in range(n-1) ] + [0] for j in range(r) ]
    v = [ random.randrange(3) - 1 for i in range(n-1) ]
    for j in range(r):
        for i in range(n-1):
            U[j][n-1] += 2*q - v[i]*U[j][i]
        U[j][n-1] = U[j][n-1] % q

    B = [ [ 0 for i in range(n+r) ] for j in range(n+r) ]
    for i in range(n):
        B[i][i] = 1
    for i in range(n,n+r):
        B[i][i] = q
        for j in range(n):
            B[i][j] = U[i-n][j]
    w = v + [1] + ([ 0 ] * r) 

    A = numpy.array(B)
    return (A,w)

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: %s <n> <r> <q>" % sys.argv[0])
        print("  n = number of vectors")
        print("  r = dimension of vectors (want c2*r*log(r) <= n <= r^c3)")
        print("  q = prime modulus (want r^c1 < q < 2r^c1 with q)")
        sys.exit(2)

    n = int(sys.argv[1])
    r = int(sys.argv[2])
    q = int(sys.argv[3])

    A, w = gen_basis(n, r, q)
    print (A)
    print ("", w)
