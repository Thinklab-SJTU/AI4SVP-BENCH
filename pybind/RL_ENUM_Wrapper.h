// RL_ENUM_Wrapper.h
#ifndef RL_ENUM_WRAPPER_H
#define RL_ENUM_WRAPPER_H

#include "lattice.h"
#include "enum_state.h"
#include <functional>
#include <memory>

class RL_ENUM_Wrapper {
private:
    std::shared_ptr<Lattice<int>> m_lattice;;  // 原始格对象
    
    // ENUM算法内部状态变量（与原始ENUM函数对应）
    long m_num_rows;
    double m_current_R;
    bool m_has_solution;
    long m_last_nonzero;
    double m_temp;
    long prev_k;  // 记录上一步的k值，用于判断回溯
    std::vector<double> rho_history;  // rho值历史记录
    
    // 统计信息
    long backtrack_count;
    long solution_count;
    std::vector<double> recent_rho_values;
    
    // 状态数组
    std::unique_ptr<long[]> m_r;                            // r数组
    std::vector<long> m_weight;                             // weight数组
    std::vector<long> m_coeff_vector;                       // coeff_vector数组
    std::vector<long> m_temp_vec;                           // temp_vec数组
    std::vector<double> m_center;                           // center数组
    std::vector<std::vector<double>> m_sigma;               // sigma数组
    std::vector<double> m_rho;                              // rho数组
    
    // RL相关状态
    EnumState m_current_state;                              // 当前状态
    long m_total_steps;                                     // 总步数
    std::vector<long> m_tried_coeffs_history;               // 历史尝试系数
    
public:
    // 构造函数
    RL_ENUM_Wrapper(std::shared_ptr<Lattice<int>> lattice);
    
    // 重置ENUM算法状态
    void reset(double R);
    void print_current_vectors() const;
    // 执行一步ENUM（RL控制）
    // 参数: action - RL选择的系数偏移量
    // 返回: (reward, done, info)
    std::tuple<double, bool, std::string> step(long action);
    
    // 获取当前状态
    EnumState get_state() const;
    
    // 获取最佳找到的向量
    std::vector<long> get_best_coeffs() const;
    std::vector<int> get_best_vector() const;
    
    // 检查是否终止
    bool is_terminated() const;
    
    // 计算即时奖励（供外部调用）
    //double calculate_immediate_reward() const;
    double calculate_immediate_reward(double prev_rho);
    
    // 获取统计信息
    struct Statistics {
        long total_steps;
        long backtracks;
        long solutions_found;
        double best_norm;
        std::vector<double> rho_history;
    };
    
    Statistics get_statistics() const;
    
private:
    // 内部执行一步ENUM核心逻辑
    bool execute_enum_step(long action);
    
    // 更新sigma数组
    void update_sigma(long k);
    
    // 更新中心值
    void update_center(long k);
    
    // 计算rho值
    double compute_rho(long k) const;
    
    // 检查终止条件
    bool check_termination() const;
    
    // 从动作解码系数值
    long decode_action(long action, double center) const;
    
    // 更新状态记录
    void update_state_record();
    
    // RL奖励计算
    double calculate_reward(bool found_solution, bool backtrack) const;
};

#endif // RL_ENUM_WRAPPER_H