#ifndef PYLATTICE_ENV_H
#define PYLATTICE_ENV_H

#include "../include/lattice.h"
#include <vector>
#include <string>
#include <memory>

template <class T> class Lattice;

class PyLatticeEnv {
public:
    struct EnumState {
        long k;                              // 当前深度
        std::vector<long> temp_vec;          // 当前系数向量
        std::vector<double> center;          // 中心值
        std::vector<double> rho;             // 投影长度
        std::vector<long> weight;            // 搜索权重
        std::vector<std::vector<double>> sigma;  // 中间矩阵
        std::vector<long> r;                 // 索引数组
        long last_nonzero;                   // 最后一个非零索引
        bool has_solution;                   // 是否找到解
        double current_R;                    // 当前搜索半径
        double current_P;
        
        void reset(long n, double R);
    };
    
    struct Config {
        int max_dimension = 100;
        double action_range = 5.0;
        bool use_pruning = true;
        int max_steps = 10000;
    };
    
    // 重要：由于Lattice是模板类，我们需要获取double特化的实例
    PyLatticeEnv(std::shared_ptr<Lattice<double>> lattice);
    
    std::vector<double> reset(double R = 100.0);
    
    std::tuple<std::vector<double>, double, bool, std::string> 
    step(int action);
    
    std::vector<double> get_state() const;
    
    int get_dimension() const { return dimension_; }
    double get_best_norm() const { return best_norm_; }
    bool is_solved() const { return solved_; }
    int get_current_k() const { return state_.k; }
    double get_current_rho() const { 
        return (state_.k < static_cast<long>(state_.rho.size())) ? state_.rho[state_.k] : 0.0; 
    }
    
    void set_config(const Config& config) { config_ = config; }
    
    void print_debug_info() const {
        std::cout << "\n=== ENUM State Debug ===" << std::endl;
        std::cout << "k: " << state_.k << "/" << dimension_ << std::endl;
        std::cout << "temp_vec: ";
        for (int i = 0; i < std::min(5, (int)state_.temp_vec.size()); ++i) {
            std::cout << state_.temp_vec[i] << " ";
        }
        std::cout << std::endl;
        std::cout << "center: ";
        for (int i = 0; i < std::min(5, (int)state_.center.size()); ++i) {
            std::cout << state_.center[i] << " ";
        }
        std::cout << std::endl;
        std::cout << "rho: ";
        for (int i = 0; i < std::min(5, (int)state_.rho.size()); ++i) {
            std::cout << state_.rho[i] << " ";
        }
        std::cout << std::endl;
        std::cout << "has_solution: " << state_.has_solution << std::endl;
        std::cout << "current_R: " << state_.current_R << std::endl;
        std::cout << "best_norm: " << best_norm_ << std::endl;
        
        // 检查lattice数据
        if (lattice_) {
            std::cout << "Lattice m_B size: " << lattice_->m_B.size() << std::endl;
            if (!lattice_->m_B.empty()) {
                std::cout << "m_B[0]: " << lattice_->m_B[0] << std::endl;
            }
        }
    }
private:
    std::shared_ptr<Lattice<double>> lattice_;  // 使用shared_ptr管理
    EnumState state_;
    Config config_;
    
    int dimension_;
    double initial_R_;
    double best_norm_;
    bool solved_;
    int step_count_;
    
    void initialize_state(double R);
    std::vector<double> extract_features() const;
    double calculate_reward(bool step_success, bool found_solution) const;
    bool execute_enum_step(int action);
    void update_sigma_matrix();
};

#endif