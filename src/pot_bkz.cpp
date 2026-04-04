#include "lattice.h"

#include "core.h"

#include <stdio.h>

#include <algorithm>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <vector>

template <class T>
void Lattice<T>::potBKZ(const long beta, const double delta, const bool compute_gso)
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

    if ((beta < 2) || (beta > m_num_rows))
    {
        char err_s[ERR_STR_LEN];
        sprintf(err_s, "The blocksize is %ld. The blocksize must be in [2, %ld]. @ function deepBKZ.", beta, m_num_rows);
        throw std::out_of_range(err_s);
    }

    std::vector<long> v;
    std::vector<T> w(m_num_cols, 0);

    if (compute_gso)
    {
        computeGSO();
    }

    for (long z = 0, j = 0, i, k, l, d; z < m_num_rows - 1;)
    {
        if (j == m_num_rows - 2)
        {
            j = 0;
        }
        ++j;

        k = std::min(j + beta - 1, m_num_rows - 1);

        d = k - j + 1;
        v.resize(d);

        v = potENUM(j - 1, d);
        if (!isZero(v))
        {
            z = 0;

            for (i = 0; i < m_num_cols; ++i)
            {
                w[i] = 0;
                for (l = 0; l < d; ++l)
                {
                    w[i] += static_cast<T>(v[l]) * m_basis[l + j - 1][i];
                }
            }

            for (i = d - 1; i >= 0; --i)
            {
                if ((v[i] == 1) || (v[i] == -1))
                {
                    for (l = 0; l < m_num_cols; ++l)
                    {
                        m_basis[i + j - 1][l] = w[l];
                    }

                    deepInsertion(j, i + j - 1);
                    break;
                }
            }

            potLLL(delta);
        }
        else
        {
            ++z;
        }
    }
}

template class Lattice<int>;
template class Lattice<long>;
template class Lattice<long long>;
template class Lattice<float>;
template class Lattice<double>;