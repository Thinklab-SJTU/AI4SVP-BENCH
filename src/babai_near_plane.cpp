#include "lattice.h"

#include "core.h"

#include <stdio.h>

#include <iostream>
#include <stdexcept>
#include <vector>

template <class T>
std::vector<T> Lattice<T>::babaiNearPlane(const std::vector<double> target)
{
    if (target.size() != m_num_cols)
    {
        char err_s[ERR_STR_LEN];
        sprintf(err_s, "The size of a target vector %ld is not equal to the size of the lattice vector %ld. @ function babaiNearPlane.", target.size(), m_num_cols);
        throw std::invalid_argument(err_s);
    }

    T c;
    std::vector<T> a(m_num_cols);
    std::vector<double> t = target;

    computeGSO();

    for(long i = 0; i < m_num_cols; ++i)
    {
        a[i] = target[i];
    }

    for (long i = m_num_rows - 1, j; i >= 0; --i)
    {
        c = round(dot(t, m_b_star[i]) / dot(m_b_star[i], m_b_star[i]));
        for (j = 0; j < m_num_cols; ++j)
        {
            t[j] -= c * m_basis[i][j];
        }
    }

    for (long i = 0; i < m_num_cols; ++i)
    {
        a[i] -= t[i];
    }

    return a;
}

template class Lattice<int>;
template class Lattice<long>;
template class Lattice<long long>;
template class Lattice<float>;
template class Lattice<double>;
