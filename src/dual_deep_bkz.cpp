#include "lattice.h"

#include "core.h"

#include <stdio.h>

#include <algorithm>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <vector>

template<class T>
void Lattice<T>::dualDeepBKZ(const long beta, const double delta, const bool compute_gso)
{
    try
    {
        if ((delta < 0.25) || (delta > 1))
        {
            throw std::out_of_range("[WARNING]The reduction parameter must be in [0.25, 1.0].");
        }
    }
    catch (const std::exception &ex)
    {
        std::cerr << ex.what() << "@ function " << __FUNCTION__ << std::endl;
    }

    if ((beta < 2) || (beta > m_num_rows))
    {
        char err_s[ERR_STR_LEN];
        sprintf(err_s, "The blocksize is %ld. The blocksize must be in [2, %ld]. @ function dualDeepBKZ.", beta, m_num_rows);
        throw std::out_of_range(err_s);
    }

    dualDeepLLL(delta, compute_gso);

    long h, d;
    std::vector<long> x;

    for(long flag = 1, j; flag >= 1;)
    {
        flag = 0;
        for(j = m_num_rows - 1; j >= 1; --j)
        {
            h = std::max(j - beta, static_cast<long>(0));
        }
        d = j - h + 1;

        computeDualGSO();
        
        if (dualENUM_(x, 0.99 / m_B[h], h, j + 1))
        {
            ++flag;

            insertToDualBasis(x, d);
            dualDeepLLL(0.99, true);
        }
    }
}

template class Lattice<int>;
template class Lattice<long>;
template class Lattice<long long>;
template class Lattice<float>;
template class Lattice<double>;
