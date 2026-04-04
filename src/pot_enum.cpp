#include "lattice.h"

#include "core.h"

#include <math.h>

#include <algorithm>
#include <stdexcept>
#include <vector>

template <class T>
std::vector<long> Lattice<T>::potENUM(const long start, const long n)
{
    if ((start < 0) || (start >= m_num_rows))
    {
        throw std::out_of_range("The argument start is out of index. @ function potENUM.");
    }
    if ((n <= 0) || (n > m_num_rows))
    {
        throw std::invalid_argument("The argument n is invalid. @ function potENUM.");
    }

    long i, r[n + 1];
    double R = log(m_B[start]), P = 0, temp;
    std::vector<long> w(n, 0), v(n, 0);
    std::vector<double> c(n, 0), D(n + 1, 0);
    std::vector<std::vector<double>> sigma(n + 1, std::vector<double>(n, 0));

    v[0] = 1;

    for (i = 0; i <= n; ++i)
    {
        r[i] = i;
    }

    for (long k = 0, last_nonzero = 0;;)
    {
        temp = static_cast<double>(v[k]) - c[k];
        temp *= temp;
        D[k] = D[k + 1] + temp * m_B[k + start];

        if ((k + 1) * log(D[k]) + P < (k + 1) * log(0.99) + R)
        {
            if (k == 0)
            {
                return v;
            }
            else
            {
                P += log(D[k]);
                --k;

                if (r[k] <= r[k + 1])
                {
                    r[k] = r[k + 1];
                }

                for (i = r[k]; i > k; --i)
                {
                    sigma[i][k] = sigma[i + 1][k] + m_mu[i + start][k + start] * v[i];
                }
                c[k] = -sigma[k + 1][k];
                v[k] = static_cast<long>(round(c[k]));
                w[k] = 1;
            }
        }
        else
        {
            ++k;
            if (k == n)
            {
                std::fill(v.begin(), v.end(), 0);
                return v;
            }
            else
            {
                r[k - 1] = k;
                if (k >= last_nonzero)
                {
                    last_nonzero = k;
                    ++v[k];
#if 1
                    if (v[last_nonzero] >= 2)
                    {
                        ++k;
                        if (k == n)
                        {
                            std::fill(v.begin(), v.end(), 0);
                            return v;
                        }
                        else
                        {
                            r[k - 1] = k;
                            last_nonzero = k;
                            v[last_nonzero] = 1;
                        }
                    }
#endif
                    P = 0;
                    R = 0;
                    for (i = 0; i <= last_nonzero; ++i)
                    {
                        R += log(m_B[i + start]);
                    }
                }
                else
                {
                    if (v[k] > c[k])
                    {
                        v[k] -= w[k];
                    }
                    else
                    {
                        v[k] += w[k];
                    }

                    ++w[k];
                    P -= log(D[k]);
                }
            }
        }
    }
}

template class Lattice<int>;
template class Lattice<long>;
template class Lattice<long long>;
template class Lattice<float>;
template class Lattice<double>;