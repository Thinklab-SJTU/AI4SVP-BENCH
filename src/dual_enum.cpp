#include "lattice.h"

#include "core.h"

#include <math.h>

#include <stdexcept>
#include <vector>

template <class T>
bool Lattice<T>::dualENUM_(std::vector<long>& coeff_vector, double R, const long start, const long end)
{
    if ((start < 0) || (start >= m_num_rows))
    {
        throw std::out_of_range("The argument start is out of index. @ function dualENUM_.");
    }
    if ((end < 0) || (end > m_num_rows))
    {
        throw std::out_of_range("The argument end is out of index. @ function dualENUM_.");
    }
    if (start >= end)
    {
        throw std::invalid_argument("The arguments start and end is invalid. @ function dualENUM_.");
    }

    if (R <= 0)
    {
        return false;
    }

    bool has_solution = false;
    long n = end - start;
    long i, r[n + 1];
    long last_nonzero = 0; // index of last non-zero elements
    double temp;
    std::vector<long> weight(n, 0);
    std::vector<long> temp_vec(n, 0);
    std::vector<double> center(n, 0);
    std::vector<std::vector<double>> sigma(n + 1, std::vector<double>(n, 0));
    std::vector<double> rho(n + 1, 0);

    coeff_vector.resize(n);
    temp_vec[0] = 1;
    for (i = 0; i < n; ++i)
    {
        r[i] = i;
    }

    for (long k = 0;;)
    {
        temp = static_cast<double>(temp_vec[k]) - center[k];
        temp *= temp;
        rho[k] = rho[k + 1] + temp / m_B[k + start]; // rho[k]=∥πₖ(shortest_vec)∥
        if (rho[k] <= R)
        {
            if (k == 0)
            {
                for(i = 0; i < n; ++i)
                {
                    coeff_vector[i] = temp_vec[i];
                }
                has_solution = true;
                R = fmin(0.99 * rho[0], R);
            }
            else
            {
                --k;
                if (r[k + 1] >= r[k])
                {
                    r[k] = r[k + 1];
                }
                for (i = r[k]; i > k; --i)
                {
                    sigma[i][k] = sigma[i + 1][k] + m_dual_mu[i + start][k + start] * temp_vec[i];
                }
                center[k] = -sigma[k + 1][k];
                temp_vec[k] = round(center[k]);
                weight[k] = 1;
            }
        }
        else
        {
            ++k;
            if (k == n)
            { // no solution
                return has_solution;
            }
            else
            {
                r[k] = k;
                if (k >= last_nonzero)
                {
                    last_nonzero = k;
                    ++temp_vec[k];
                }
                else
                {
                    if (temp_vec[k] > center[k])
                    {
                        temp_vec[k] -= weight[k];
                    }
                    else
                    {
                        temp_vec[k] += weight[k];
                    }

                    ++weight[k];
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
