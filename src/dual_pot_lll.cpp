#include "lattice.h"

#include "core.h"

#include <math.h>

#include <algorithm>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <vector>

template <class T>
void Lattice<T>::dualPotLLL(const double delta, const bool compute_gso)
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

    double P;
    double P_min;
    double s;
    double D;
    std::vector<double> dual_D(m_num_rows);

    LLL(delta, compute_gso);

    for (long k = m_num_rows - 1, j, i, l, q, h; k >= 0;)
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
                q = static_cast<long>(round(m_dual_mu[k][j]));
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

        P = 1.0;
        P_min = 1.0;
        l = m_num_rows - 1;
        for (j = k + 1; j < m_num_rows; ++j)
        {
            s = 0.0;
            for (i = k; i <= j; ++i)
            {
                s += m_dual_mu[k][i] * m_dual_mu[k][i] / m_B[i];
            }
            P *= m_B[j];
            P *= s;

            if (P < P_min)
            {
                l = j;
                P_min = P;
            }
        }

        if (delta > P_min)
        {
            D = 1.0 / m_B[k];
            std::fill(dual_D.begin(), dual_D.end(), 0);

            dual_D[k] = D;
            for (h = k + 1; h < m_num_rows; ++h)
            {
                D += m_dual_mu[k][h] * m_dual_mu[k][h] / m_B[h];
                dual_D[h] = D;
            }

            dualDeepInsertion(k, l);
            updateDualDeepInsGSO(k, l, dual_D);

            k = l;
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
