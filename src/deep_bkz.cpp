#include "lattice.h"

#include "core.h"

#include <stdio.h>

#include <algorithm>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <vector>

template <class T>
void Lattice<T>::deepBKZ(const long beta, const double delta, const bool compute_gso)
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

    std::vector<T> v(m_num_cols);
    std::vector<long> w(m_num_rows);

    deepLLL(delta, compute_gso);

    for (long z = 0, j, t, num_tour = 0, i, k = 0, h, d, l; z < m_num_rows - 1;)
    {
        if (num_tour >= m_max_loop)
        {
            break;
        }

        if (k == m_num_rows - 1)
        {
            k = 0;
            ++num_tour;
        }
        ++k;
        l = std::min(k + beta - 1, m_num_rows);
        h = std::min(l + 1, m_num_rows);
        d = l - k + 1;

        if (ENUM_(w, 0.99 * m_B[k - 1], k - 1, l))
        {
            z = 0;

            for (i = 0; i < m_num_cols; ++i)
            {
                v[i] = 0;
                for (j = 0; j < d; ++j)
                {
                    v[i] += w[j] * m_basis[j + k - 1][i];
                }
            }

            for (i = d - 1; i >= 0; --i)
            {
                if ((w[i] == 1) || (w[i] == -1))
                {
                    for (j = 0; j < m_num_cols; ++j)
                    {
                        m_basis[i + k - 1][j] = v[j];
                    }

                    deepInsertion(k - 1, i + k - 1);
                    break;
                }
            }

            deepLLL(delta, true, 0, h, k - 1);
        }
        else
        {
            ++z;
            deepLLL(delta, false, 0, h, h - 2);
        }
    }
}

template class Lattice<int>;
template class Lattice<long>;
template class Lattice<long long>;
template class Lattice<float>;
template class Lattice<double>;
