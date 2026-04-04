#include "lattice.h"

#include "core.h"

#include <math.h>

template <class T>
void Lattice<T>::sizeReduce(const long i, const long j)
{
    if ((m_mu[i][j] > 0.5) || (m_mu[i][j] < -0.5))
    {
        long k;
        const long q = static_cast<long>(round(m_mu[i][j]));

        for (k = 0; k < m_num_cols; ++k)
        {
            m_basis[i][k] -= static_cast<T>(q) * m_basis[j][k];
        }
        for (k = 0; k <= j; ++k)
        {
            m_mu[i][k] -= m_mu[j][k] * static_cast<double>(q);
        }
    }
}

template <class T>
void Lattice<T>::sizeReduce(const bool compute_gso)
{
    if (compute_gso)
    {
        computeGSO();
    }

    for (long i = 1, j; i < m_num_rows; ++i)
    {
        for (j = i - 1; j >= 0; --j)
        {
            sizeReduce(i, j);
        }
    }
}

template class Lattice<int>;
template class Lattice<long>;
template class Lattice<long long>;
template class Lattice<float>;
template class Lattice<double>;