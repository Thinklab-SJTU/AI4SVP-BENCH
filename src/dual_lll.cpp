#include "lattice.h"

#include "core.h"

#include <math.h>

#include <exception>
#include <iostream>
#include <stdexcept>
#include <vector>

template <class T>
void Lattice<T>::dualLLL(const double delta, const bool compute_gso)
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

    long j, i;
    double q;
    T tmp;

    if (compute_gso)
    {
        computeGSO();
    }

    m_dual_mu[m_num_rows - 1][m_num_rows - 1] = 1.0;

    for (long k = m_num_rows - 2; k >= 0;)
    {
        m_dual_mu[k][k] = 1.0;

        for (j = k + 1; j < m_num_rows; ++j)
        {
            m_dual_mu[k][j] = 0;
            for (i = k; i < j; ++i)
            {
                m_dual_mu[k][j] -= m_mu[j][i] * m_dual_mu[k][i];
            }

            if (m_dual_mu[k][j] > 0.5 || m_dual_mu[k][j] < -0.5)
            {
                q = round(m_dual_mu[k][j]);
                for (i = 0; i < m_num_cols; ++i)
                {
                    m_basis[j][i] += static_cast<T>(q) * m_basis[k][i];
                }
                for (i = j; i < m_num_rows; ++i)
                {
                    m_dual_mu[k][i] -= q * m_dual_mu[j][i];
                }
                for (i = 0; i <= k; ++i)
                {
                    m_mu[j][i] += q * m_mu[k][i];
                }
            }
        }

        if (k < m_num_rows - 1 && (delta - m_dual_mu[k][k + 1] * m_dual_mu[k][k + 1]) * m_B[k] > m_B[k + 1])
        {
            for (i = 0; i < m_num_cols; ++i)
            {
                tmp = m_basis[k + 1][i];
                m_basis[k + 1][i] = m_basis[k][i];
                m_basis[k][i] = tmp;
            }
            updateDeepInsGSO(k, k + 1, 0, m_num_rows);
            ++k;
        }
        else
        {
            --k;
        }
    }
}

template class Lattice<int>;
template class Lattice<long>;
template class Lattice<long long>;
template class Lattice<float>;
template class Lattice<double>;
