#include "lattice.h"

#include "core.h"

#include <exception>
#include <iostream>
#include <stdexcept>

template <class T>
void Lattice<T>::sizeReduceL2(const double eta, const long k)
{
    try
    {
        if ((eta < 0.5) || (eta > 1))
        {
            throw std::out_of_range("[WARNING]The reduction parameter must be in [0.5, 1.0].");
        }
    }
    catch (const std::exception &ex)
    {
        std::cerr << ex.what() << "@ function " << __FUNCTION__ << std::endl;
    }

    const double eta_bar = (eta + eta + 1) * 0.25;
    double max;

    m_r[0][0] = dot(m_basis[0], m_basis[0]);
    m_mu[0][0] = 1.0;

    for (long i, j, x, h, count = 0;;)
    {
        for (j = 0; j <= k; ++j)
        {
            m_r[k][j] = dot(m_basis[k], m_basis[j]);
            for (h = 0; h < j; ++h)
            {
                m_r[k][j] -= m_r[k][h] * m_mu[j][h];
            }
            m_mu[k][j] = m_r[k][j] / m_r[j][j];
        }

        m_s[0] = dot(m_basis[k], m_basis[k]);
        for (j = 1; j <= k; ++j)
        {
            m_s[j] = m_s[j - 1] - m_mu[k][j - 1] * m_r[k][j - 1];
        }
        m_r[k][k] = m_s[k];

        max = -1;
        for (i = 0; i < k; ++i)
        {
            if ((m_mu[k][i] > max) or (m_mu[k][i] < -max))
            {
                max = fabs(m_mu[k][i]);
            }
        }

        if (max > eta_bar)
        {
            for (i = k - 1; i >= 0; --i)
            {
                x = static_cast<long>(round(m_mu[k][i]));
                for (j = 0; j < m_num_cols; ++j)
                {
                    m_basis[k][j] -= x * m_basis[i][j];
                }
                for (j = 0; j <= i; ++j)
                {
                    m_mu[k][j] -= static_cast<long double>(x) * m_mu[i][j];
                }
            }
        }
        else
        {
            break;
        }
    }
}

template <class T>
void Lattice<T>::L2(const double delta, const double eta)
{
    try
    {
        if ((eta < 0.5) || (eta > sqrt(delta)))
        {
            char err_s[ERR_STR_LEN];
            sprintf(err_s, "[WARNING]The reduction parameter eta must be in [0.5, %lf].", sqrt(delta));
            throw std::out_of_range(err_s);
        }

        if ((eta < 0.5) || (eta > 1))
        {
            throw std::out_of_range("[WARNING]The reduction parameter delta must be in [0.25, 1.0].");
        }
    }
    catch (const std::exception &ex)
    {
        std::cerr << ex.what() << "@ function " << __FUNCTION__ << std::endl;
    }

    T tmp;
    const double delta_bar = (delta + 1) * 0.5;
    m_r[0][0] = dot(m_basis[0], m_basis[0]);

    long count = 0;

    for (long k = 1, k_, j; k < m_num_rows;)
    {
        sizeReduceL2(eta, k);

        //printf("k = %ld\n", k);

        k_ = k;
        while ((k >= 1) and (delta_bar * m_r[k - 1][k - 1] >= m_s[k - 1]))
        {
            for (j = 0; j < m_num_rows; ++j)
            {
                tmp = m_basis[k - 1][j];
                m_basis[k - 1][j] = m_basis[k][j];
                m_basis[k][j] = tmp;
            }
            --k;
        }

        for (j = 0; j < k; ++j)
        {
            m_mu[k][j] = m_mu[k_][j];
            m_r[k][j] = m_r[k_][j];
        }
        m_r[k][k] = m_s[k];

        ++k;
    }
}

template class Lattice<int>;
template class Lattice<long>;
template class Lattice<long long>;
template class Lattice<float>;
template class Lattice<double>;
