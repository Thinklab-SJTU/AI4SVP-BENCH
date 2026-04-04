#ifndef CORE_H_
#define CORE_H_

#include <vector>

#define ERR_STR_LEN 200

template<class T>
void print(const std::vector<T> v);

template<class T>
void print(const std::vector<std::vector<T>> mat);

long prime(const long n);


template <class U, class V>
V dot(const std::vector<U> x, const std::vector<V> y);


template<class T>
bool isZero(const std::vector<T> v);

#endif // !CORE_H_
