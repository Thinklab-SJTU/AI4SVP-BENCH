#include "RL_ENUM_Wrapper.h"
#include "enum_state.h"
#include <cmath>
#include <algorithm>
#include <sstream>
#include <iostream>

RL_ENUM_Wrapper::RL_ENUM_Wrapper(std::shared_ptr<Lattice<int>> lattice)
    : m_lattice(lattice), m_num_rows(lattice->numRows()),
      m_current_R(0.0), m_has_solution(false),
      m_last_nonzero(0), m_temp(0.0),
      m_total_steps(0), prev_k(-1),
      backtrack_count(0), solution_count(0) {
    
    if (!m_lattice) {
        throw std::invalid_argument("Lattice pointer cannot be null");
    }
    //std::cout << m_lattice;
    // 初始化数组大小
    m_num_rows = m_lattice->numRows();
    m_r = std::make_unique<long[]>(m_num_rows + 1);
    m_weight.resize(m_num_rows, 0);
    m_coeff_vector.resize(m_num_rows, 0);
    m_temp_vec.resize(m_num_rows, 0);
    m_center.resize(m_num_rows, 0.0);
    m_sigma.resize(m_num_rows + 1, std::vector<double>(m_num_rows, 0.0));
    m_rho.resize(m_num_rows + 1, 0.0);
    
    // 初始化当前状态
    m_current_state.num_rows = m_num_rows;
    m_current_state.best_norm = std::numeric_limits<double>::max();
}
void RL_ENUM_Wrapper::print_current_vectors() const {
    std::cout << "=== 当前向量信息 ===" << std::endl;
    
    // 打印系数向量
    std::cout << "系数向量temp_vec: [";
    for (long i = 0; i < std::min(10L, m_num_rows); ++i) {
        std::cout << m_temp_vec[i];
        if (i < m_num_rows - 1 && i < 9) std::cout << ", ";
    }
    if (m_num_rows > 10) std::cout << ", ...";
    std::cout << "]" << std::endl;
    
    // 计算并打印格向量
    if (m_lattice) {
        auto lattice_vector = m_lattice->mulVecBasis(m_temp_vec);
        std::cout << "格向量: [";
        for (size_t i = 0; i < std::min((size_t)10, lattice_vector.size()); ++i) {
            std::cout << lattice_vector[i];
            if (i < lattice_vector.size() - 1 && i < 9) std::cout << ", ";
        }
        if (lattice_vector.size() > 10) std::cout << ", ...";
        std::cout << "]" << std::endl;
        
        // 计算范数
        double norm_sq = 0.0;
        for (const auto& x : lattice_vector) {
            norm_sq += static_cast<double>(x) * static_cast<double>(x);
        }
        std::cout << "向量范数: " << std::sqrt(norm_sq) << std::endl;
    }
}
void RL_ENUM_Wrapper::reset(double R) {
    m_current_R = R;
    m_has_solution = false;
    m_last_nonzero = 0;
    m_temp = 0.0;
    m_total_steps = 0;
    //std::cout << m_lattice;
    // 重置数组
    std::fill(m_weight.begin(), m_weight.end(), 0);
    std::fill(m_coeff_vector.begin(), m_coeff_vector.end(), 0);
    std::fill(m_temp_vec.begin(), m_temp_vec.end(), 0);
    std::fill(m_center.begin(), m_center.end(), 0.0);
    
    for (auto& row : m_sigma) {
        std::fill(row.begin(), row.end(), 0.0);
    }
    
    std::fill(m_rho.begin(), m_rho.end(), 0.0);
    
    // 初始化r数组
    for (long i = 0; i < m_num_rows; ++i) {
        m_r[i] = i;
    }
    
    // 关键修复：ENUM从最底层开始，所以k应该初始化为m_num_rows-1
    // 而不是从0开始
    m_current_state.current_k = m_num_rows - 1;
    
    // 关键修复：初始化temp_vec[m_num_rows-1] = 1（而不是temp_vec[0] = 1）
    // 因为ENUM算法从最底层开始搜索
    if (m_num_rows > 0) {
        m_temp_vec[m_num_rows - 1] = 1;
    }
    
    // 重置当前状态
    m_current_state = EnumState();
    m_current_state.num_rows = m_num_rows;
    m_current_state.radius = R;
    m_current_state.best_norm = std::numeric_limits<double>::max();
    m_current_state.current_k = m_num_rows - 1;  // 重要：从底层开始
    m_current_state.current_rho = 0.0;
    m_current_state.current_center = 0.0;
    m_current_state.has_solution = false;
    
    m_tried_coeffs_history.clear();
    
    /*std::cout << "RL_ENUM_Wrapper重置完成: R=" << R 
              << ", k=" << m_current_state.current_k 
              << ", num_rows=" << m_num_rows << std::endl;*/
}

// 新增：decode_action实现
long RL_ENUM_Wrapper::decode_action(long action, double center) const {
    // 动作解码：将离散动作映射到系数值
    // action范围通常是[-5, 5]，对应11个离散动作
    long base_coeff = static_cast<long>(std::round(center));
    
    // 确保动作在合理范围内
    // 这里我们假设action是已经偏移过的值（-5到+5）
    // 如果需要将[0, 10]映射到[-5, 5]，可以这样处理：
    // long offset = action - 5;
    // long chosen_coeff = base_coeff + offset;
    
    // 直接使用action作为偏移
    long chosen_coeff = base_coeff + action;
    
    return chosen_coeff;
}
double RL_ENUM_Wrapper::calculate_immediate_reward(double prev_rho) {
    double reward = 0.0;
    
    // 1. 找到解的奖励（大奖励）
    if (m_current_state.found_solution) {
        // 奖励与解的质量成反比（范数越小奖励越大）
        double quality_bonus = 1000.0 / (m_current_state.best_norm + 1.0);
        reward += quality_bonus;
    }
    /*if (!m_current_state.found_solution) {
        // 奖励与解的质量成反比（范数越小奖励越大）
        double quality_bonus =m_current_state.best_norm/(1e+308);
        reward -= quality_bonus;
    }*/
    
    // 2. rho值减少的奖励
    if (m_current_state.current_rho < prev_rho) {
        double reduction = prev_rho - m_current_state.current_rho;
        double reduction_ratio = reduction / prev_rho;
        reward += 10.0 * reduction_ratio;
    }
    
    // 3. 进入更深层的奖励（探索）
    if (m_current_state.current_k < m_num_rows / 2) {
        // 进入较深层，奖励探索
        double depth_bonus = 5.0 * (1.0 - static_cast<double>(m_current_state.current_k) / m_num_rows);
        reward += depth_bonus;
    }
    
    // 4. 惩罚：rho值超过半径
    if (m_current_state.current_rho > m_current_state.radius) {
        double excess_ratio = (m_current_state.current_rho - m_current_state.radius) / m_current_state.radius;
        reward -= 2.0 * excess_ratio;
    }
    
    // 5. 惩罚：步数消耗（鼓励效率）
    reward -= 0.05;
    
    // 6. 惩罚：频繁回溯
    if (m_current_state.current_k > prev_k) {  // 需要记录prev_k
        // 发生了回溯
        reward -= 1.0;
    }
    
    return reward;
}
bool RL_ENUM_Wrapper::check_termination() const {
    // 检查终止条件
    
    // 1. 达到最大步数（如果在配置中设置了）
    // if (m_total_steps >= max_steps) return true;
    
    // 2. 已经找到解且达到精度要求
    if (m_has_solution) {
        // 如果当前最佳范数已经足够小
        if (m_current_state.best_norm < m_current_R * 0.01) {
            return true;
        }
    }
    
    // 3. 搜索空间已经穷尽（当k == m_num_rows时）
    if (m_current_state.current_k >= m_num_rows) {
        return true;
    }
    
    // 4. rho值持续过大，表明当前区域无解
    if (m_current_state.current_rho > m_current_R * 00.0) {
        // 如果连续多次rho都很大，考虑终止
        // 这里简化处理：单次过大就终止（可能需要调整）
        return true;
    }
    
    // 5. 搜索停滞：连续多次没有进展
    // 可以添加计数器跟踪无进展的步数
    
    return false;
}

std::tuple<double, bool, std::string> RL_ENUM_Wrapper::step(long action) {
    if (m_current_state.terminated) {
        return std::make_tuple(0.0, true, "Already terminated");
    }
    
    m_total_steps++;
    
    // 记录上一步的rho值，用于计算变化
    double prev_rho = m_current_state.current_rho;
    
    // 执行一步ENUM核心逻辑
    bool should_terminate = execute_enum_step(action);
    
    // 更新当前状态
    update_state_record();
    
    // 计算奖励（使用实际的即时奖励计算）
    double reward = calculate_immediate_reward(prev_rho);
    
    // 检查是否终止
    bool done = should_terminate || check_termination();
    m_current_state.terminated = done;
    
    // 生成信息字符串
    std::stringstream info;
    info << "Step: " << m_total_steps 
         << ", k: " << m_current_state.current_k
         << ", rho: " << m_current_state.current_rho
         << "/" << m_current_state.radius
         << ", center: " << m_current_state.current_center
         << ", found_solution: " << m_current_state.found_solution
         << ", best_norm: " << m_current_state.best_norm;
    
    return std::make_tuple(reward, done, info.str());
}

bool RL_ENUM_Wrapper::execute_enum_step(long action) {
    long k = m_current_state.current_k;
    
    /*std::cout << "=== ENUM Step Debug ===" << std::endl;
    std::cout << "k = " << k << std::endl;
    std::cout << "temp_vec[" << k << "] = " << m_temp_vec[k] << std::endl;
    std::cout << "center[" << k << "] = " << m_center[k] << std::endl;*/
    
    // 计算差值
    double diff = static_cast<double>(m_temp_vec[k]) - m_center[k];
    /*std::cout << "diff = " << diff << std::endl;
    std::cout << "diff^2 = " << diff * diff << std::endl;*/
    
    // 检查m_B[k]
    double B_k = m_lattice->m_B[k];
    //std::cout << "m_B[" << k << "] = " << B_k << std::endl;
    
    // 计算当前rho
    m_temp = diff * diff;
    m_rho[k] = m_rho[k + 1] + m_temp * B_k;
    //std::cout << "rho[" << k+1 << "] = " << m_rho[k+1] << std::endl;
    //std::cout << "rho[" << k << "] = " << m_rho[k] << std::endl;
    
    // 检查sigma和center的计算
    /*if (k > 0) {
        std::cout << "sigma[" << k+1 << "][" << k << "] = " << m_sigma[k+1][k] << std::endl;
    }*/
    
    // 检查是否满足半径条件
    if (m_rho[k] <= m_current_R) {
        if (k == 0) {
            // 找到解
            m_has_solution = true;
            m_current_state.found_solution = true;
            
            // 更新最佳系数
            for (long i = 0; i < m_num_rows; ++i) {
                m_coeff_vector[i] = m_temp_vec[i];
            }
            
            // 更新最佳范数
            auto v = m_lattice->mulVecBasis(m_coeff_vector);
            double norm_sq = 0.0;
            for (const auto& x : v) {
                norm_sq += static_cast<double>(x) * static_cast<double>(x);
            }
            m_current_state.best_norm = std::sqrt(norm_sq);
            
            // 缩小半径（保持比当前最佳解稍小）
            if (m_rho[0] > 0) {
                m_current_R = std::fmin(0.99 * m_rho[0], m_current_R);
            }
            
            return false; // 继续搜索（可能还有更好的解）
        } else {
            // 向下走一层
            k--;
            m_current_state.current_k = k;
            
            if (m_r[k + 1] >= m_r[k]) {
                m_r[k] = m_r[k + 1];
            }
            
            // 更新sigma
            update_sigma(k);
            
            // 更新中心值
            update_center(k);
            
            // 使用RL动作选择系数
            double center_val = m_center[k];
            long chosen_coeff = decode_action(action, center_val);
            
            m_temp_vec[k] = chosen_coeff;
            m_weight[k] = 1;
            
            // 记录尝试的系数
            m_tried_coeffs_history.push_back(chosen_coeff);
            if (m_tried_coeffs_history.size() > 100) {
                m_tried_coeffs_history.erase(m_tried_coeffs_history.begin());
            }
            
            return false;
        }
    } else {
        // 回溯
        k++;
        m_current_state.current_k = k;
        
        if (k == m_num_rows) {
            // 搜索完成
            if (!m_has_solution) {
                std::fill(m_coeff_vector.begin(), m_coeff_vector.end(), 0);
            }
            return true; // 终止
        } else {
            m_r[k] = k;
            
            if (k >= m_last_nonzero) {
                m_last_nonzero = k;
                m_temp_vec[k]++;
            } else {
                // 使用原始的回溯策略
                if (m_temp_vec[k] > m_center[k]) {
                    m_temp_vec[k] -= m_weight[k];
                } else {
                    m_temp_vec[k] += m_weight[k];
                }
                m_weight[k]++;
            }
            
            // 记录尝试的系数
            m_tried_coeffs_history.push_back(m_temp_vec[k]);
            if (m_tried_coeffs_history.size() > 100) {
                m_tried_coeffs_history.erase(m_tried_coeffs_history.begin());
            }
            
            return false;
        }
    }
}

void RL_ENUM_Wrapper::update_sigma(long k) {
    /*std::cout << "update_sigma: k=" << k << ", r[" << k << "]=" << m_r[k] << std::endl;*/
    
    for (long i = m_r[k]; i > k; --i) {
        double old_sigma = m_sigma[i + 1][k];
        double mu = m_lattice->m_mu[i][k];
        double temp_vec_i = m_temp_vec[i];
        double new_sigma = old_sigma + mu * temp_vec_i;
        
        /*std::cout << "  i=" << i << ": sigma=" << old_sigma 
                  << " + mu=" << mu << " * temp_vec=" << temp_vec_i
                  << " = " << new_sigma << std::endl;*/
                  
        m_sigma[i][k] = new_sigma;
    }
}

void RL_ENUM_Wrapper::update_center(long k) {
    m_center[k] = -m_sigma[k + 1][k];
    /*std::cout << "update_center: k=" << k 
              << ", sigma[" << k+1 << "][" << k << "]=" << m_sigma[k+1][k]
              << ", center=" << m_center[k] << std::endl;*/
}

// 新增：计算即时奖励（使用之前rho值进行比较）


// 修正：这个函数不再需要，因为我们已经有了calculate_immediate_reward(double prev_rho)
// double RL_ENUM_Wrapper::calculate_reward(bool found_solution, bool backtrack) const {
//     // 这个函数已弃用，使用calculate_immediate_reward代替
//     return 0.0;
// }

void RL_ENUM_Wrapper::update_state_record() {
    long k = m_current_state.current_k;
    
    // 更新当前状态
    m_current_state.current_k = k;
    m_current_state.current_rho = m_rho[k];
    m_current_state.current_center = m_center[k];
    m_current_state.radius = m_current_R;
    m_current_state.has_solution = m_has_solution;
    
    // 更新GSO信息
    m_current_state.gs_norms.clear();
    // 注意：这里假设m_lattice的m_B是public或通过getter访问
    // 实际实现需要根据Lattice类的设计调整
    for (long i = 0; i < m_num_rows; ++i) {
        m_current_state.gs_norms.push_back(m_lattice->m_B[i]);
    }
    
    // 更新当前层的mu值
    m_current_state.mu_values.clear();
    for (long i = 0; i < m_num_rows; ++i) {
        m_current_state.mu_values.push_back(m_lattice->m_mu[i][k]);
    }
    
    // 更新尝试的系数历史
    m_current_state.tried_coeffs = m_tried_coeffs_history;
    
    // 更新当前系数
    m_current_state.current_coeffs = m_temp_vec;
    
    // 更新最佳系数
    m_current_state.best_coeffs = m_coeff_vector;
    
    // 更新数组状态
    m_current_state.r.assign(m_r.get(), m_r.get() + m_num_rows);
    m_current_state.weight = m_weight;
    m_current_state.center_array = m_center;
    m_current_state.rho_array = m_rho;
    m_current_state.last_nonzero = m_last_nonzero;
}

EnumState RL_ENUM_Wrapper::get_state() const {
    EnumState state = m_current_state;
    
    // 确保状态是最新的
    state.current_k = m_current_state.current_k;
    state.current_rho = m_rho[state.current_k];
    state.current_center = m_center[state.current_k];
    state.radius = m_current_R;
    state.has_solution = m_has_solution;
    state.best_norm = m_current_state.best_norm;
    
    // 填充GSO信息
    state.gs_norms.clear();
    for (long i = 0; i < m_num_rows; ++i) {
        state.gs_norms.push_back(m_lattice->m_B[i]);
    }
    
    // 填充当前层的mu值
    state.mu_values.clear();
    long k = state.current_k;
    if (k >= 0 && k < m_num_rows) {
        for (long i = 0; i < m_num_rows; ++i) {
            state.mu_values.push_back(m_lattice->m_mu[i][k]);
        }
    }
    
    // 填充尝试的系数
    state.tried_coeffs = m_tried_coeffs_history;
    
    // 填充当前系数
    state.current_coeffs = m_temp_vec;
    
    return state;
}

std::vector<long> RL_ENUM_Wrapper::get_best_coeffs() const {
    return m_coeff_vector;
}

// 新增：get_best_vector实现
std::vector<int> RL_ENUM_Wrapper::get_best_vector() const {
    // 调用lattice的mulVecBasis方法
    return m_lattice->mulVecBasis(m_coeff_vector);
}

bool RL_ENUM_Wrapper::is_terminated() const {
    return m_current_state.terminated;
}

// 新增：get_statistics实现
RL_ENUM_Wrapper::Statistics RL_ENUM_Wrapper::get_statistics() const {
    Statistics stats;
    stats.total_steps = m_total_steps;
    stats.best_norm = m_current_state.best_norm;
    
    // 计算回溯次数（简单估计：k值增加的次数）
    // 需要记录历史，这里简化处理
    stats.backtracks = 0;  // 实际实现需要跟踪
    
    stats.solutions_found = m_has_solution ? 1 : 0;
    
    // rho历史（记录最近一些值）
    // 实际实现需要维护rho历史数组
    
    return stats;
}

// 新增：compute_rho实现（如果需要）
double RL_ENUM_Wrapper::compute_rho(long k) const {
    if (k < 0 || k >= m_num_rows) return 0.0;
    
    double temp_val = static_cast<double>(m_temp_vec[k]) - m_center[k];
    temp_val *= temp_val;
    return m_rho[k + 1] + temp_val * m_lattice->m_B[k];
}