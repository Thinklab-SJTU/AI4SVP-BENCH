#include "lattice.h"

#include "core.h"

#include <exception>
#include <iostream>
#include <stdexcept>

template <class T>
void Lattice<T>::LLL(const double delta, const bool compute_gso, long start_, long end_, long h)
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
        throw std::out_of_range("The parameter end_ must be a positive integer or -1. @ function LLL");
    }
    else if (end_ == -1)
    {
        if ((start_ <= -1) || (start_ >= m_num_rows))
        {
            throw std::out_of_range("The parameter start_ is out of index. @ function LLL");
        }
    }
    else
    {
        if ((start_ <= -1) || (start_ >= m_num_rows))
        {
            throw std::out_of_range("The parameter start_ is out of index. @ function LLL");
        }
        if (start_ >= end_)
        {
            throw std::invalid_argument("The parameter start_ must be less than the parameter end_. @ function LLL");
        }
        if (end_ > m_num_rows)
        {
            throw std::out_of_range("The parameter end_ is out of index. @ function LLL");
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

    double nu, B, t;
    T tmp;

    if (compute_gso)
    {
        computeGSO();
    }

    for (long k = h, i, j; k < end;)
    {
        for (j = k - 1; j > -1; --j)
        {
            sizeReduce(k, j);
        }

        if ((k > start) && (m_B[k] < (delta - m_mu[k][k - 1] * m_mu[k][k - 1]) * m_B[k - 1]))
        {
            for (i = 0; i < m_num_cols; ++i)
            {
                tmp = m_basis[k - 1][i];
                m_basis[k - 1][i] = m_basis[k][i];
                m_basis[k][i] = tmp;
            }

            nu = m_mu[k][k - 1];
            B = m_B[k] + nu * nu * m_B[k - 1];
            m_mu[k][k - 1] = nu * m_B[k - 1] / B;
            m_B[k] *= m_B[k - 1] / B;
            m_B[k - 1] = B;

            for (i = 0; i < k - 1; ++i)
            {
                t = m_mu[k - 1][i];
                m_mu[k - 1][i] = m_mu[k][i];
                m_mu[k][i] = t;
            }
            for (i = k + 1; i < end; ++i)
            {
                t = m_mu[i][k];
                m_mu[i][k] = m_mu[i][k - 1] - nu * t;
                m_mu[i][k - 1] = t + m_mu[k][k - 1] * m_mu[i][k];
            }

            --k;
        }
        else
        {
            ++k;
        }
    }
}

template class Lattice<int>;
template class Lattice<long>;
template class Lattice<long long>;
template class Lattice<float>;
template class Lattice<double>;