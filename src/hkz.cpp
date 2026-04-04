#include "lattice.h"

#include "core.h"

template <class T>
void Lattice<T>::HKZ(const double delta, const bool compute_gso)
{
    BKZ(m_num_rows, delta, compute_gso);
}

template class Lattice<int>;
template class Lattice<long>;
template class Lattice<long long>;
template class Lattice<float>;
template class Lattice<double>;
