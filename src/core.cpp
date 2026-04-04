#include "core.h"

#include <math.h>
#include <stdio.h>

#include <iostream>
#include <stdexcept>
#include <vector>

template <class T>
void print(const std::vector<T> v)
{
    std::cout << "[";
    for (const T w : v)
    {
        std::cout << w << ", ";
    }
    std::cout << "\b\b" << "]\n";
}

template <class T>
void print(const std::vector<std::vector<T>> mat)
{
    puts("[");
    for (const std::vector<T> v : mat)
    {
        print(v);
    }
    puts("]");
}

long prime(const long n)
{
    if (n <= 0)
    {
        char err_s[ERR_STR_LEN];
        sprintf(err_s, "The %ld-th prime number cannot be defined.", n);
        throw std::invalid_argument(err_s);
    }

    long S = 0, T, U;
    for (long k = 2, j, i; k <= floor(2 * n * log(n) + 2); ++k)
    {
        T = 0;
        for (j = 2; j <= k; ++j)
        {
            U = 0;
            for (i = 1; i <= j; ++i)
            {
                U += floor(static_cast<double>(j) / i) - floor(static_cast<double>(j - 1) / i);
            }
            T += 1 + floor((2.0 - U) / j);
        }
        S += 1 - floor(static_cast<double>(T) / n);
    }

    return S + 2;
}

template <class U, class V>
V dot(const std::vector<U> x, const std::vector<V> y)
{
    if (x.size() != y.size())
    {
        char err_s[ERR_STR_LEN];
        sprintf(err_s, "An inner product of %ld-th vector and %ld-th vector cannot be defined.", x.size(), y.size());
        throw std::invalid_argument(err_s);
    }

    V S = 0;
    for (int i = 0; i < x.size(); ++i)
    {
        S += static_cast<V>(x[i]) * y[i];
    }
    return S;
}

template <class T>
bool isZero(const std::vector<T> v)
{
    for (const T w : v)
    {
        if (w != 0)
        {
            return false;
        }
    }
    return true;
}

template void print<int>(const std::vector<int>);
template void print<long>(const std::vector<long>);
template void print<int>(const std::vector<std::vector<int>>);
template void print<long>(const std::vector<std::vector<long>>);
template void print<double>(const std::vector<std::vector<double>>);
template int dot<int, int>(std::vector<int>, std::vector<int>);
template double dot<int, double>(std::vector<int>, std::vector<double>);
template double dot<long, double>(std::vector<long>, std::vector<double>);
template double dot<long long, double>(std::vector<long long>, std::vector<double>);
template double dot<float, double>(std::vector<float>, std::vector<double>);
template double dot<double, double>(std::vector<double>, std::vector<double>);
template long dot<long, long>(std::vector<long>, std::vector<long>);
template long long dot<long long, long long>(std::vector<long long>, std::vector<long long>);
template float dot<float, float>(std::vector<float>, std::vector<float>);
template bool isZero<int>(const std::vector<int>);
template bool isZero<long>(const std::vector<long>);
