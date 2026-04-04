// 在 lattice.h 或单独的枚举状态头文件中添加

#ifndef ENUM_STATE_H
#define ENUM_STATE_H

#include "lattice.h"
#include "enum_state.h"
#include <functional>
#include <vector>
#include <cstdint>
#include <string>
#include <memory>

struct EnumState {
    // 基本状态信息
    long current_k;                     // 当前层
    double current_rho;                // 当前rho值
    double current_center;             // 当前中心值
    double radius;                     // 当前搜索半径R
    long num_rows;                     // 格维度
    bool has_solution;                 // 是否已找到解
    
    // GSO信息
    std::vector<double> gs_norms;      // m_B值 (Gram-Schmidt范数)
    std::vector<double> mu_values;     // 当前层的mu值 (m_mu[*][current_k])
    
    // 搜索历史
    std::vector<long> tried_coeffs;    // 最近尝试的系数
    std::vector<long> current_coeffs;  // 当前系数向量temp_vec
    std::vector<long> best_coeffs;     // 最佳系数向量coeff_vector
    double best_norm;                  // 最佳找到的范数
    
    // 算法内部状态
    std::vector<long> r;               // r数组
    std::vector<long> weight;          // weight数组
    std::vector<double> center_array;  // center数组
    std::vector<double> rho_array;     // rho数组
    long last_nonzero;                 // 最后非零元素索引
    
    // 算法终止信息
    bool terminated;                   // 是否已终止
    bool found_solution;               // 是否找到解
    
    // 构造函数
    EnumState() : current_k(0), current_rho(0.0), current_center(0.0),
                  radius(0.0), num_rows(0), has_solution(false),
                  best_norm(0.0), last_nonzero(0),
                  terminated(false), found_solution(false) {}
};

#endif // ENUM_STATE_H