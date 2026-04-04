#include "lattice.h"

#include "core.h"

#include <algorithm>
#include <exception>
#include <iostream>
#include <stdexcept>

template <class T>
void Lattice<T>::deepLLL(const double delta, const bool compute_gso, long start_, long end_, const long h)
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

    if ((end_ <= -2) || (end_ == 0))
    {
        throw std::out_of_range("The parameter end_ must be a positive integer or -1. @ function deepLLL");
    }
    else if (end_ == -1)
    {
        if ((start_ <= -1) || (start_ >= m_num_rows))
        {
            throw std::out_of_range("The parameter start_ is out of index. @ function deepLLL");
        }
    }
    else
    {
        if ((start_ <= -1) || (start_ >= m_num_rows))
        {
            throw std::out_of_range("The parameter start_ is out of index. @ function deepLLL");
        }
        if (start_ >= end_)
        {
            throw std::invalid_argument("The parameter start_ must be less than the parameter end_. @ function deepLLL");
        }
        if (end_ > m_num_rows)
        {
            throw std::out_of_range("The parameter end_ is out of index. @ function deepLLL");
        }
    }

    long start = start_;
    long end;

    if (end_ == -1)
    {
        end = m_num_rows;
    }
    else
    {
        end = end_;
    }

    double C;

    if (compute_gso)
    {
        computeGSO();
    }

    for (long k = h, j, i, t, l; k < end;)
    {
        for (j = k - 1; j >= start; --j)
        {
            sizeReduce(k, j);
        }

        C = static_cast<double>(dot(m_basis[k], m_basis[k]));

        for (i = start; i < k;)
        {
            if (C >= delta * m_B[i])
            {
                C -= m_mu[k][i] * m_mu[k][i] * m_B[i];
                ++i;
            }
            else
            {
                deepInsertion(i, k);
                updateDeepInsGSO(i, k, start, end);

                k = std::max(i - 1, static_cast<long>(0));
            }
        }
        ++k;
    }
}

template class Lattice<int>;
template class Lattice<long>;
template class Lattice<long long>;
template class Lattice<float>;
template class Lattice<double>;
