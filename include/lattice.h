#ifndef LATTICE_H_
#define LATTICE_H_

#include <time.h>

#include <iostream>
#include <vector>
#include <random>

#include "core.h"

extern bool output_rhf;    // Output root of hermite factor to csv-file or not
extern bool output_gsa_sl; // Output GSA-slope to csv-file or not


template <class T>
class Lattice
{
private:
    long m_num_rows = 0;
    long m_num_cols = 0;
    T m_vol;
    long m_max_loop = 99999;
    std::vector<std::vector<T>> m_basis;
    std::vector<double> m_B;
    std::vector<double> m_dual_B;
    std::vector<double> m_s;
    std::vector<std::vector<double>> m_b_star;
    std::vector<std::vector<double>> m_dual_b_star;
    std::vector<std::vector<double>> m_r;
    std::vector<std::vector<double>> m_mu;
    std::vector<std::vector<double>> m_dual_mu;
    std::mt19937_64 m_mt64 = std::mt19937_64((unsigned int)time(nullptr));
    std::uniform_real_distribution<> m_get_rand_uni;

    bool ENUM_(std::vector<long> &coeff_vector, double R, const long start, const long end);

    bool dualENUM_(std::vector<long> &coeff_vector, double R, const long start, const long end);

    bool isBasis();

public:

    Lattice(const long n = 0, const long m = 0);

    friend std::ostream &operator<<(std::ostream &os, const Lattice<T> &lat)
    {
        os << "[" << std::endl;
        for (std::vector<T> bb : lat.m_basis)
        {
            os << "[";
            for (T b : bb)
            {
                os << b << ", ";
            }
            os << "\b\b]" << std::endl;
        }
        os << "]" << std::endl;
        return os;
    }
    friend class PyLatticeEnv;
    friend class RL_ENUM_Wrapper;

    void setMaxLoop(const long max_loop);
    long numRows() const;

    long numCols() const;

    void setDims(const long n, const long m);

    long double b1Norm();

    void setSVPChallenge(const long dim, const long seed);

    void setRandom(const long n, const long m, const T min, const T max);

    void setBasis(const std::vector<std::vector<T>> basis_mat);

    void setGoldesteinMayerLattice(const T p, const T q);

    void setSchnorrLattice(const long N, const double c);

    T volume(const bool compute_gso = true);

    double potential(const bool compute_gso = true);

    double logPotential(const bool compute_gso = true);

    std::vector<T> mulVecBasis(const std::vector<long> v);

    void deepInsertion(const long k, const long l);

    void dualDeepInsertion(const long k, const long l);

    void choleskyFact();

    void computeGSO();

    void computeDualGSO();

    void updateDeepInsGSO(const long i, const long k, const long start, const long end);

    void updateDualDeepInsGSO(const long k, const long l, const std::vector<double> dual_D);

    long double rhf() const;

    long double sl() const;

    void sizeReduce(const long i, const long j);

    void sizeReduce(const bool compute_gso = true);

    void LLL(const double delta = 0.75, const bool compute_gso = true, const long start_ = 0, const long end_ = -1, long h = 0);

    void sizeReduceL2(const double eta, const long k);

    void L2(const double delta = 0.75, const double eta = 0.51);

    void deepLLL(const double delta = 0.75, const bool compute_gso = true, const long start_ = 0, const long end_ = -1, const long h = 0);

    void potLLL(const double delta = 0.75, const bool compute_gso = true);

    std::vector<long> ENUM(double R);

    void BKZ(const long beta, const double delta = 0.75, const bool compute_gso = true);

    void HKZ(const double delta = 0.75, const bool compute_gso = true);

    void deepBKZ(const long beta, const double delta = 0.75, const bool compute_gso = true);

    std::vector<long> potENUM(const long start, const long n);

    void potBKZ(const long beta, const double delta = 0.75, const bool compute_gso = true);

    void dualLLL(const double delta = 0.75, const bool compute_gso = true);

    void dualDeepLLL(const double delta = 0.75, const bool compute_gso = true);

    void dualPotLLL(const double delta = 0.75, const bool compute_gso = true);

    void insertToDualBasis(const std::vector<long> x, const long dim);

    void dualBKZ(const long beta, const double delta = 0.75, const bool compute_gso = true);

    void dualDeepBKZ(const long beta, const double delta = 0.75, const bool compute_gso = true);

    std::vector<T> babaiNearPlane(const std::vector<double> target);
};

#endif // !LATTICE_H_
