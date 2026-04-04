#include "lattice.h"

#include <iostream>
#include <cmath>

template <class T>
long double Lattice<T>::sl() const
{
    long double sum1 = 0, sum2 = 0;
    for (int i = 0; i < m_num_rows; ++i)
    {
        sum1 += (i + 1) * logl(m_B[i]);
        sum2 += logl(m_B[i]);
    }
    return 12 * (sum1 - 0.5 * (m_num_rows + 1) * sum2) / (m_num_rows * (m_num_rows * m_num_rows - 1));
}

template class Lattice<int>;
template class Lattice<long>;
template class Lattice<long long>;
template class Lattice<float>;
template class Lattice<double>;
