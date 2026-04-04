#include "lattice.h"

#include "core.h"

#include <exception>
#include <iostream>
#include <stdexcept>

template <class T>
void Lattice<T>::potLLL(const double delta, const bool compute_gso)
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

    double P, P_min, S;

    LLL(delta, compute_gso);

    for (long l = 0, j, i, k; l < m_num_rows;)
    {
        for (j = l - 1; j > -1; --j)
        {
            sizeReduce(l, j);
        }

        P = P_min = 1.0;
        k = 0;
        for (j = l - 1; j >= 0; --j)
        {
            S = 0;
            for (i = j; i < l; ++i)
            {
                S += m_mu[l][i] * m_mu[l][i] * m_B[i];
            }
            P *= (m_B[l] + S) / m_B[j];

            if (P < P_min)
            {
                k = j;
                P_min = P;
            }
        }

        if (delta > P_min)
        {
            deepInsertion(k, l);
            updateDeepInsGSO(k, l, 0, m_num_rows);
            l = k;
        }
        else
        {
            ++l;
        }
    }
}

template class Lattice<int>;
template class Lattice<long>;
template class Lattice<long long>;
template class Lattice<float>;
template class Lattice<double>;