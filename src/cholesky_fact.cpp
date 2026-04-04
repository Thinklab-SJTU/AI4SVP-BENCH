#include "lattice.h"

#include "core.h"

#include <iostream>

template <class T>
void Lattice<T>::choleskyFact()
{
    for (long i = 0, j, k; i < m_num_rows; ++i)
    {
        for (j = 0; j < i; ++j)
        {
            m_r[i][j] = dot(m_basis[i], m_basis[j]);
            for (k = 0; k < j; ++k)
            {
                m_r[i][j] -= m_r[i][k] * m_mu[j][k];
            }
            m_mu[i][j] = m_r[i][j] / m_r[j][j];
            m_s[0] = dot(m_basis[i], m_basis[i]);
            for (k = 1; k <= i; ++k)
            {
                m_s[k] = m_s[k - 1] - m_mu[i][k - 1] * m_r[i][k - 1];
            }
            m_r[i][i] = m_s[i];
        }
    }
}

template class Lattice<int>;
template class Lattice<long>;
template class Lattice<long long>;
template class Lattice<float>;
template class Lattice<double>;
