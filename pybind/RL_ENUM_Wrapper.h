// RL_ENUM_Wrapper.h
#ifndef RL_ENUM_WRAPPER_H
#define RL_ENUM_WRAPPER_H

#include "lattice.h"
#include "enum_state.h"
#include <functional>
#include <memory>

class RL_ENUM_Wrapper {
private:
    std::shared_ptr<Lattice<int>> m_lattice;;  // ԭʼ�����
    
    // ENUM�㷨�ڲ�״̬��������ԭʼENUM������Ӧ��
    long m_num_rows;
    double m_current_R;
    bool m_has_solution;
    long m_last_nonzero;
    double m_temp;
    long prev_k;  // ��¼��һ����kֵ�������жϻ���
    std::vector<double> rho_history;  // rhoֵ��ʷ��¼
    
    // ͳ����Ϣ
    long backtrack_count;
    long solution_count;
    std::vector<double> recent_rho_values;
    double m_episode_best_norm;  // Track best norm per episode for delta reward
    
    // ״̬����
    std::unique_ptr<long[]> m_r;                            // r����
    std::vector<long> m_weight;                             // weight����
    std::vector<long> m_coeff_vector;                       // coeff_vector����
    std::vector<long> m_temp_vec;                           // temp_vec����
    std::vector<double> m_center;                           // center����
    std::vector<std::vector<double>> m_sigma;               // sigma����
    std::vector<double> m_rho;                              // rho����
    
    // RL���״̬
    EnumState m_current_state;                              // ��ǰ״̬
    long m_total_steps;                                     // �ܲ���
    std::vector<long> m_tried_coeffs_history;               // ��ʷ����ϵ��
    
public:
    // ���캯��
    RL_ENUM_Wrapper(std::shared_ptr<Lattice<int>> lattice);
    
    // ����ENUM�㷨״̬
    void reset(double R);
    void print_current_vectors() const;
    // ִ��һ��ENUM��RL���ƣ�
    // ����: action - RLѡ���ϵ��ƫ����
    // ����: (reward, done, info)
    std::tuple<double, bool, std::string> step(long action);
    
    // ��ȡ��ǰ״̬
    EnumState get_state() const;
    
    // ��ȡ����ҵ�������
    std::vector<long> get_best_coeffs() const;
    std::vector<int> get_best_vector() const;
    
    // ����Ƿ���ֹ
    bool is_terminated() const;
    
    // ���㼴ʱ���������ⲿ���ã�
    //double calculate_immediate_reward() const;
    double calculate_immediate_reward(double prev_rho);
    
    // ��ȡͳ����Ϣ
    struct Statistics {
        long total_steps;
        long backtracks;
        long solutions_found;
        double best_norm;
        std::vector<double> rho_history;
    };
    
    Statistics get_statistics() const;
    
private:
    // �ڲ�ִ��һ��ENUM�����߼�
    bool execute_enum_step(long action);
    
    // ����sigma����
    void update_sigma(long k);
    
    // ��������ֵ
    void update_center(long k);
    
    // ����rhoֵ
    double compute_rho(long k) const;
    
    // �����ֹ����
    bool check_termination() const;
    
    // �Ӷ�������ϵ��ֵ
    long decode_action(long action, double center) const;
    
    // ����״̬��¼
    void update_state_record();
    
    // RL��������
    double calculate_reward(bool found_solution, bool backtrack) const;
};

#endif // RL_ENUM_WRAPPER_H