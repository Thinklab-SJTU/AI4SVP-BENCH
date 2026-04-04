#include "PyLatticeEnv.h"
#include <cmath>
#include "../include/lattice.h"
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <numeric>

// ============ 辅助函数：更好的特征归一化 ============
static inline double safe_log(double x) {
    return std::log(std::max(x, 1e-20));
}

static inline double safe_log1p(double x) {
    return std::log1p(std::max(x, -0.9999));
}

static inline double clamp(double x, double min_val, double max_val) {
    return std::max(min_val, std::min(max_val, x));
}

static inline double normalize_to_range(double x, double min_val, double max_val, double target_min = -1.0, double target_max = 1.0) {
    if (std::abs(max_val - min_val) < 1e-12) return 0.0;
    double normalized = (x - min_val) / (max_val - min_val);
    return target_min + normalized * (target_max - target_min);
}

// ============ EnumState::reset ============
void PyLatticeEnv::EnumState::reset(long n, double R) {
    k = 0;
    temp_vec.assign(n, 0);
    center.assign(n, 0.0);
    rho.assign(n + 1, 0.0);
    weight.assign(n, 0);
    sigma.assign(n + 1, std::vector<double>(n, 0.0));
    r.resize(n + 1);
    last_nonzero = 0;
    has_solution = false;
    current_R = R;
    current_P = 0.0;  // 添加P值初始化
    
    if (n > 0) {
        temp_vec[0] = 1;
        for (long i = 0; i <= n; ++i) {
            r[i] = i;
        }
    }
}

// ============ 构造函数 ============
PyLatticeEnv::PyLatticeEnv(std::shared_ptr<Lattice<double>> lattice) 
    : lattice_(lattice),
      dimension_(lattice->numRows()),
      initial_R_(0.0),
      best_norm_(1e9),
      solved_(false),
      step_count_(0) {
    
    config_.max_dimension = dimension_;
    config_.action_range = 5.0;
    config_.use_pruning = true;
    config_.max_steps = dimension_ * 100;
}

// ============ reset ============
std::vector<double> PyLatticeEnv::reset(double R) {
    if (!lattice_) {
        throw std::runtime_error("Lattice not initialized");
    }
    
    // 确保GSO已计算
    int max_attempts = 3;
    for (int attempt = 0; attempt < max_attempts; ++attempt) {
        try {
            lattice_->computeGSO();
            
            // 检查GSO结果有效性
            bool gso_valid = true;
            if (lattice_->m_B.empty()) {
                std::cerr << "ERROR: m_B is empty after computeGSO()" << std::endl;
                gso_valid = false;
            } else {
                // 检查并修正B值
                for (size_t i = 0; i < lattice_->m_B.size(); ++i) {
                    if (lattice_->m_B[i] <= 1e-10) {
                        std::cout << "Fixing small B[" << i << "] = " << lattice_->m_B[i] 
                                  << " -> 1.0" << std::endl;
                        lattice_->m_B[i] = 1.0;
                    }
                }
            }
            
            if (gso_valid) {
                std::cout << "Reset: GSO computed, m_B size = " << lattice_->m_B.size() 
                          << ", m_B[0] = " << lattice_->m_B[0] 
                          << ", m_mu rows = " << lattice_->m_mu.size() << std::endl;
                
                // 计算初始半径（标准ENUM使用 log(m_B[0])）
                double log_B0 = safe_log(lattice_->m_B[0]);
                std::cout << "log(B[0]) = " << log_B0 
                          << ", exp(log(B[0])) = " << std::exp(log_B0) << std::endl;
                break;
            } else if (attempt < max_attempts - 1) {
                std::cout << "Attempt " << (attempt + 1) << ": Regenerating lattice..." << std::endl;
                lattice_->setRandom(dimension_, dimension_, 10, 100);  // 使用更大的范围
            }
            
        } catch (const std::exception& e) {
            std::cerr << "GSO error (attempt " << (attempt + 1) << "): " << e.what() << std::endl;
            if (attempt == max_attempts - 1) {
                throw;
            }
        }
    }
    
    // 标准ENUM使用 log(B[0]) 作为初始半径
    double initial_R = lattice_->m_B.empty() ? safe_log(R) : safe_log(lattice_->m_B[0]);
    
    initial_R_ = R;  // 保存输入的R
    best_norm_ = 1e9;
    solved_ = false;
    step_count_ = 0;
    
    state_.reset(dimension_, initial_R);
    
    std::cout << "Environment reset: dim=" << dimension_ 
              << ", input_R=" << R
              << ", enum_R(log)=" << initial_R
              << ", exp(enum_R)=" << std::exp(initial_R)
              << ", k=" << state_.k << std::endl;
    
    return extract_features();
}

// ============ extract_features (改进版本) ============
std::vector<double> PyLatticeEnv::extract_features() const {
    std::vector<double> features;
    long n = dimension_;
    
    if (n <= 0) {
        return std::vector<double>(15, 0.0);
    }
    
    // === 1. 基础层信息 (3维) ===
    features.push_back(static_cast<double>(state_.k) / std::max(static_cast<double>(n), 1.0));
    
    double center_val = 0.0;
    double rho_val = 0.0;
    
    if (state_.k >= 0 && state_.k < n && 
        state_.k < static_cast<long>(state_.center.size()) &&
        state_.k < static_cast<long>(state_.rho.size())) {
        center_val = state_.center[state_.k];
        rho_val = state_.rho[state_.k];
    }
    
    // 改进归一化：使用tanh限制范围
    features.push_back(std::tanh(center_val / 100.0));  // center归一化到[-1,1]
    
    // D[k]的对数相对R归一化（关键改进！）
    if (rho_val > 0 && state_.current_R > -1e10) {
        double log_rho = safe_log(rho_val);
        double relative_log = (log_rho - state_.current_R) / 20.0;  // 除以20缩小范围
        features.push_back(std::tanh(relative_log));  // 限制到[-1,1]
    } else {
        features.push_back(0.0);
    }
    
    // === 2. 局部GSO信息 (5维) ===
    if (lattice_ && !lattice_->m_B.empty()) {
        // 计算局部B值的统计信息用于归一化
        std::vector<double> local_B;
        for (int offset = -2; offset <= 2; ++offset) {
            int idx = state_.k + offset;
            if (idx >= 0 && idx < static_cast<int>(lattice_->m_B.size())) {
                local_B.push_back(lattice_->m_B[idx]);
            }
        }
        
        if (!local_B.empty()) {
            double max_B = *std::max_element(local_B.begin(), local_B.end());
            double min_B = *std::min_element(local_B.begin(), local_B.end());
            
            for (int offset = -2; offset <= 2; ++offset) {
                int idx = state_.k + offset;
                if (idx >= 0 && idx < static_cast<int>(lattice_->m_B.size())) {
                    double b_val = lattice_->m_B[idx];
                    // 对数归一化，然后限制范围
                    double log_b = safe_log(b_val);
                    double norm_log = (log_b - safe_log(min_B)) / (safe_log(max_B) - safe_log(min_B) + 1e-12);
                    features.push_back(2.0 * norm_log - 1.0);  // 映射到[-1,1]
                } else {
                    features.push_back(0.0);
                }
            }
        } else {
            for (int i = 0; i < 5; ++i) features.push_back(0.0);
        }
    } else {
        for (int i = 0; i < 5; ++i) features.push_back(0.0);
    }
    
    // === 3. 正交化系数特征 (3维) ===
    if (lattice_ && state_.k >= 0 && state_.k < static_cast<long>(lattice_->m_mu.size())) {
        const auto& mu_k = lattice_->m_mu[state_.k];
        for (int j = 0; j < 3; ++j) {
            long idx = state_.k - j - 1;
            if (idx >= 0 && idx < static_cast<long>(mu_k.size())) {
                double mu_val = mu_k[idx];
                features.push_back(std::tanh(mu_val));  // mu值通常很小，直接tanh
            } else {
                features.push_back(0.0);
            }
        }
    } else {
        for (int j = 0; j < 3; ++j) features.push_back(0.0);
    }
    
    // === 4. 历史决策统计 (2维) ===
    if (state_.k > 0) {
        double sum_abs = 0.0;
        double max_abs = 0.0;
        int count = 0;
        
        for (long i = 0; i < state_.k && i < static_cast<long>(state_.temp_vec.size()); ++i) {
            double abs_val = std::abs(static_cast<double>(state_.temp_vec[i]));
            sum_abs += abs_val;
            max_abs = std::max(max_abs, abs_val);
            count++;
        }
        
        if (count > 0) {
            // 改进归一化：假设系数通常在0-10范围内
            features.push_back(std::tanh((sum_abs / count) / 10.0));
            features.push_back(std::tanh(max_abs / 20.0));
        } else {
            features.push_back(0.0);
            features.push_back(0.0);
        }
    } else {
        features.push_back(0.0);
        features.push_back(0.0);
    }
    
    // === 5. 搜索进度特征 (2维) ===
    features.push_back(static_cast<double>(n - state_.k) / std::max(static_cast<double>(n), 1.0));
    features.push_back(state_.last_nonzero / std::max(static_cast<double>(n), 1.0));
    
    // === 检查和限制特征范围 ===
    for (auto& f : features) {
        if (std::isnan(f) || std::isinf(f)) {
            f = 0.0;
        }
        f = clamp(f, -1.0, 1.0);  // 严格限制到[-1,1]
    }
    
    return features;  // 总共15维
}

// ============ step ============
std::tuple<std::vector<double>, double, bool, std::string> 
PyLatticeEnv::step(int action) {
    if (!lattice_) {
        return {extract_features(), -10.0, true, "Lattice not initialized"};
    }
    
    // 限制动作范围
    int clamped_action = clamp(action, -5, 5);
    
    if (solved_ || step_count_ >= config_.max_steps) {
        return {extract_features(), 0.0, true, "Episode already finished"};
    }
    
    step_count_++;
    
    bool step_success = execute_enum_step(clamped_action);
    bool found_solution = state_.has_solution;
    
    if (found_solution) {
        solved_ = true;
        // 计算实际向量范数
        std::vector<long> coeff(state_.temp_vec.begin(), state_.temp_vec.begin() + dimension_);
        auto vector = lattice_->mulVecBasis(coeff);
        double norm_sq = 0.0;
        for (auto val : vector) norm_sq += static_cast<double>(val) * val;
        best_norm_ = std::sqrt(norm_sq);
    }
    
    double reward = calculate_reward(step_success, found_solution);
    bool done = solved_ || (state_.k == dimension_) || step_count_ >= config_.max_steps;
    
    std::string info;
    if (done) {
        if (solved_) info = "Found solution";
        else if (state_.k == dimension_) info = "Reached max depth";
        else info = "Max steps reached";
    } else {
        info = "Continue";
    }
    
    return {extract_features(), reward, done, info};
}

// ============ execute_enum_step (关键改进) ============
bool PyLatticeEnv::execute_enum_step(int action) {
    long n = dimension_;
    long& k = state_.k;
    auto& temp_vec = state_.temp_vec;
    auto& center = state_.center;
    auto& rho = state_.rho;  // 对应D[k]
    auto& weight = state_.weight;
    auto& sigma = state_.sigma;
    auto& r = state_.r;
    auto& last_nonzero = state_.last_nonzero;
    auto& has_solution = state_.has_solution;
    double& current_R = state_.current_R;  // 注意：这是log值！
    double& current_P = state_.current_P;
    
    // 检查边界
    if (k < 0 || k >= n) {
        std::cerr << "ERROR: k out of range: " << k << std::endl;
        k = 0;
        return false;
    }
    
    // 获取B值
    if (k >= static_cast<long>(lattice_->m_B.size())) {
        std::cerr << "ERROR: k >= m_B size" << std::endl;
        return false;
    }
    
    double B_value = lattice_->m_B[k];
    if (B_value <= 0) {
        B_value = 1.0;  // 安全值
    }
    
    // 计算D[k] = D[k+1] + (v[k]-c[k])^2 * B[k]
    double temp_val = static_cast<double>(temp_vec[k]);
    double center_val = center[k];
    double diff = temp_val - center_val;
    
    // 确保rho数组大小
    if (rho.size() < static_cast<size_t>(k + 2)) {
        rho.resize(k + 2, 0.0);
    }
    
    rho[k] = rho[k + 1] + diff * diff * B_value;
    
    // === 关键：使用标准ENUM条件判断 ===
    bool should_go_deeper = false;
    if (rho[k] > 0) {
        double log_Dk = safe_log(rho[k]);
        double left_side = (k + 1) * log_Dk + current_P;
        double right_side = (k + 1) * std::log(0.99) + current_R;
        
        should_go_deeper = (left_side < right_side);
        
        // 调试输出
        std::cout << "Step " << step_count_ << ": k=" << k 
                  << ", v=" << temp_val
                  << ", c=" << center_val
                  << ", D=" << rho[k]
                  << ", logD=" << log_Dk
                  << ", P=" << current_P
                  << ", R=" << current_R
                  << ", left=" << left_side
                  << ", right=" << right_side
                  << ", go_deeper=" << (should_go_deeper ? "YES" : "NO")
                  << ", action=" << action << std::endl;
    }
    
    if (should_go_deeper) {
        // 向下探索
        if (k == 0) {
            // 找到解
            has_solution = true;
            
            // 验证解
            std::vector<long> coeff(temp_vec.begin(), temp_vec.begin() + n);
            auto vector = lattice_->mulVecBasis(coeff);
            double norm_sq = 0.0;
            for (auto val : vector) norm_sq += static_cast<double>(val) * val;
            best_norm_ = std::sqrt(norm_sq);
            
            // 收缩半径（可选）
            current_R = safe_log(rho[0] * 0.95);
            
            std::cout << "SOLUTION FOUND! norm=" << best_norm_ 
                      << ", new R=" << current_R << std::endl;
            
            // 继续搜索：回溯
            k = 1;
            if (r.size() > 0) r[0] = 1;
            center[0] = -sigma[1][0];
            temp_vec[0] = static_cast<long>(std::round(center[0])) + action;
            weight[0] = 1;
            
            return true;
        } else {
            // 向更深层移动
            current_P += safe_log(rho[k]);
            k--;
            
            // 更新r数组
            if (k + 1 < static_cast<long>(r.size())) {
                if (r[k + 1] > r[k]) {
                    r[k] = r[k + 1];
                }
            }
            
            // 更新sigma和center
            update_sigma_matrix();
            
            // 应用动作（ML或传统）
            long base_value = static_cast<long>(std::round(center[k]));
            temp_vec[k] = base_value + action;  // 直接使用动作作为偏移
            weight[k] = 1;
            
            return true;
        }
    } else {
        // 需要回溯
        k++;
        
        if (k == n) {
            std::cout << "SEARCH FAILED: reached top level" << std::endl;
            return false;
        }
        
        // 标准ENUM回溯逻辑
        if (k - 1 < static_cast<long>(r.size())) {
            r[k - 1] = k;
        }
        
        if (k > last_nonzero) {
            // 新层
            last_nonzero = k;
            center[k] = 0.0;
            temp_vec[k] = action;  // 使用动作初始化
            weight[k] = 1;
            
            // 重新计算半径和P
            current_P = 0.0;
            current_R = 0.0;
            for (long i = 0; i <= last_nonzero; ++i) {
                if (i < static_cast<long>(lattice_->m_B.size())) {
                    current_R += safe_log(lattice_->m_B[i]);
                }
            }
            
            std::cout << "NEW LAYER: k=" << k 
                      << ", new R=" << current_R << std::endl;
        } else {
            // 已访问层，使用Schnorr-Euchner策略
            if (action == 0) {
                // 传统策略
                if (temp_vec[k] > center[k]) {
                    temp_vec[k] -= weight[k];
                } else {
                    temp_vec[k] += weight[k];
                }
                weight[k]++;
            } else {
                // ML策略：直接使用动作
                temp_vec[k] = static_cast<long>(std::round(center[k])) + action;
            }
            
            current_P -= safe_log(rho[k - 1]);
        }
        
        return true;
    }
}

// ============ update_sigma_matrix ============
void PyLatticeEnv::update_sigma_matrix() {
    long n = dimension_;
    long k = state_.k;
    
    if (k < 0 || k >= n) return;
    
    if (state_.sigma.size() < static_cast<size_t>(n + 1)) {
        state_.sigma.resize(n + 1, std::vector<double>(n, 0.0));
    }
    
    // 更新sigma矩阵
    for (long i = state_.r[k]; i > k; --i) {
        if (i >= n || k >= n) continue;
        if (i >= lattice_->m_mu.size() || k >= lattice_->m_mu[i].size()) continue;
        
        double mu_value = lattice_->m_mu[i][k];
        double temp_i = static_cast<double>(state_.temp_vec[i]);
        state_.sigma[i][k] = state_.sigma[i + 1][k] + mu_value * temp_i;
    }
    
    // 更新center[k]
    if (k + 1 < static_cast<long>(state_.sigma.size()) && 
        k < static_cast<long>(state_.center.size())) {
        state_.center[k] = -state_.sigma[k + 1][k];
    }
}

// ============ calculate_reward ============
double PyLatticeEnv::calculate_reward(bool step_success, bool found_solution) const {
    if (!step_success) return -0.1;
    if (found_solution) {
        if (best_norm_ < 1e-8) return 0.0;
        return 1.0 / (best_norm_ + 0.1);  // 避免除零
    }
    
    double reward = 0.001;  // 基础奖励
    
    // 深度奖励
    if (state_.k > 0) {
        reward += 0.01 * (dimension_ - state_.k) / dimension_;
    }
    
    // D值减少的奖励
    if (state_.k < dimension_ && state_.k > 0) {
        if (state_.rho[state_.k] < state_.rho[state_.k + 1]) {
            reward += 0.02;
        }
    }
    
    return clamp(reward, -1.0, 1.0);
}

// ============ get_state ============
std::vector<double> PyLatticeEnv::get_state() const {
    return extract_features();
}

// ============ 新增：调试函数 ============
/*void PyLatticeEnv::print_debug_info() const {
    std::cout << "\n=== PyLatticeEnv Debug ===" << std::endl;
    std::cout << "Dimension: " << dimension_ << std::endl;
    std::cout << "Current k: " << state_.k << std::endl;
    std::cout << "Last nonzero: " << state_.last_nonzero << std::endl;
    std::cout << "Has solution: " << state_.has_solution << std::endl;
    std::cout << "Current R(log): " << state_.current_R << std::endl;
    std::cout << "Current P: " << state_.current_P << std::endl;
    
    if (state_.k >= 0 && state_.k < static_cast<long>(state_.rho.size())) {
        std::cout << "D[" << state_.k << "]: " << state_.rho[state_.k] 
                  << " (log=" << safe_log(state_.rho[state_.k]) << ")" << std::endl;
    }
    
    std::cout << "=========================\n" << std::endl;
}*/