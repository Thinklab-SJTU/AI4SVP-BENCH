#include <iostream>

#include "lattice.h"
template<typename T>
double vectorNorm(const std::vector<T>& v) {
    double sum = 0.0;
    for (const auto& x : v) {
        sum += static_cast<double>(x) * static_cast<double>(x);
    }
    return std::sqrt(sum);
}
int main()
{
    Lattice<int> lat(120, 120); // 40-demensional full-rank lattice
    lat.setSVPChallenge(120, 0);
    // set as a random lattice
    // lat.setRandom(10, 10, 1000, 10000);

    // print the lattice basis
    //std::cout << lat;

    // compute GSO-information
    lat.computeGSO();

    // compute the shortest vector
    //std::vector<int> v = lat.mulVecBasis(lat.ENUM(4000000));
    //print(v);

    // lattice basis reduction
    // lat.L2(0.99, 0.51);
    //lat.dualPotLLL(0.99);
    //lat.dualBKZ(40, 0.999);
    lat.deepBKZ(2, 0.99);
    // lat.dualDeepBKZ(30, 0.99);
    //lat.potBKZ(13, 0.99253);
    //lat.dualBKZ(50, 0.98458);
    // lat.BKZ(6, 0.99);
    std::cout << "欧几里得范数: " << lat.b1Norm() << std::endl;
    //std::cout << "向量v的欧几里得范数: " << vectorNorm(v) << std::endl;
    //std::cout << lat;
    //printf("rhf = %Lf\n", lat.rhf());
    //printf("sl = %Lf\n", lat.sl());
    // std::cout << lat.volume(false) << std::endl;

    return 0;
}
