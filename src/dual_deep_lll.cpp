#include "lattice.h"

#include "core.h"

#include <math.h>

#include <algorithm>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <vector>

template <class T>
void Lattice<T>::dualDeepLLL(const double delta, const bool compute_gso)
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

    long j, i, l, h;
    double q, d, D;
    std::vector<double> dual_D(m_num_rows);

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
            m_dual_mu[k][j] = 0.0;
            for (i = k; i < j; ++i)
            {
                m_dual_mu[k][j] -= m_mu[j][i] * m_dual_mu[k][i];
            }

            if ((m_dual_mu[k][j] > 0.5) || (m_dual_mu[k][j] < -0.5))
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

        d = 0.0;
        l = m_num_rows - 1;
        for (j = k; j < m_num_rows; ++j)
        {
            d += m_dual_mu[k][j] * m_dual_mu[k][j] / m_B[j];
        }

        while (l > k)
        {
            if (m_B[l] * d < delta)
            {
                D = 1.0 / m_B[k];

                std::fill(dual_D.begin(), dual_D.end(), 0.0);
                dual_D[k] = D;
                for (h = k + 1; h < m_num_rows; ++h)
                {
                    D += m_dual_mu[k][h] * m_dual_mu[k][h] / m_B[h];
                    dual_D[h] = D;
                }

                dualDeepInsertion(k, l);
                updateDualDeepInsGSO(k, l, dual_D);

                if (l < m_num_rows - 2)
                {
                    k = l + 1;
                }
                else
                {
                    k = m_num_rows - 1;
                }
            }
            else
            {
                d -= m_dual_mu[k][l] * m_dual_mu[k][l] / m_B[l];
                --l;
            }
        }
        --k;
    }
}

template class Lattice<int>;
template class Lattice<long>;
template class Lattice<long long>;
template class Lattice<float>;
template class Lattice<double>;
